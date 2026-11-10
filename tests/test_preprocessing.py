"""Decoding, normals and downsampling against toothseg3d."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, teeth3ds_scans


def _decode(path: Path, merge: bool = True):
    from algorithm.scan_decoder.scan_decoder import ScanDecoder, ScanDecoderInput, ScanDecoderOutput
    dec, out = ScanDecoder(), ScanDecoderOutput()
    dec.set_strategy_for(path)
    assert dec.execute(ScanDecoderInput(path, merge), out)
    return out


def test_obj_decoder_matches_read_obj() -> None:
    from t3ds_io import read_obj
    path = teeth3ds_scans("upper", 1)[0]
    V, F, C = read_obj(str(path))
    out = _decode(path)
    assert np.array_equal(out.positions.numpy(), V) and np.array_equal(out.faces.numpy(), F)
    assert np.allclose(out.colours.numpy(), C)


def test_ply_round_trip(tmp_path: Path) -> None:
    """Our PLY writer -> our PLY reader keeps positions, faces and instance colours."""
    from algorithm.result_encoder.common.palette import instance_colours
    from algorithm.result_encoder.result_encoder import (
        ResultEncoder, ResultEncoderInput, ResultEncoderOutput, ResultEncoderType)
    scan = _decode(teeth3ds_scans("lower", 1)[0])
    ins = torch.randint(0, 20, (scan.positions.shape[0],))
    enc = ResultEncoder()
    enc.set_strategy(ResultEncoderType.PLY_CPU)
    enc.execute(ResultEncoderInput(tmp_path / "a.ply", "a", "lower", scan.positions, ins, scan.faces),
                ResultEncoderOutput())
    back = _decode(tmp_path / "a.ply")
    assert torch.equal(back.positions, scan.positions) and torch.equal(back.faces, scan.faces)
    assert np.allclose(back.colours.numpy() * 255, instance_colours(ins.numpy()), atol=1e-3)


def test_ascii_ply_and_stl(tmp_path: Path) -> None:
    V = np.random.RandomState(0).rand(50, 3).astype(np.float32)
    N = V / np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
    lines = ["ply", "format ascii 1.0", f"element vertex {len(V)}", "property float x", "property float y",
             "property float z", "property float nx", "property float ny", "property float nz",
             "property uchar red", "property uchar green", "property uchar blue",
             f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
    lines += [f"{a} {b} {c} {d} {e} {f} 10 20 30" for (a, b, c), (d, e, f) in zip(V, N)]
    lines += [f"3 {a} {b} {c}" for a, b, c in F]
    (tmp_path / "p.ply").write_text("\n".join(lines) + "\n")
    out = _decode(tmp_path / "p.ply")
    assert np.allclose(out.positions.numpy(), V, atol=1e-6) and np.array_equal(out.faces.numpy(), F)
    assert np.allclose(out.normals.numpy(), N, atol=1e-6)
    assert np.allclose(out.colours.numpy()[0], [10 / 255, 20 / 255, 30 / 255])

    # a point cloud written as a mesh with "element face 0" (Medit compound exports)
    cloud = [ln if not ln.startswith("element face") else "element face 0" for ln in lines[:15 + len(V)]]
    (tmp_path / "c.ply").write_text("\n".join(cloud) + "\n")
    out = _decode(tmp_path / "c.ply")
    assert out.faces is None and np.allclose(out.positions.numpy(), V, atol=1e-6)

    # binary STL: two triangles sharing an edge -> 4 welded vertices
    tri = np.array([[V[0], V[1], V[2]], [V[2], V[1], V[3]]], np.float32)
    rec = np.zeros(2, [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tri
    (tmp_path / "s.stl").write_bytes(b"solid but binary".ljust(80, b" ") + np.uint32(2).tobytes() + rec.tobytes())
    out = _decode(tmp_path / "s.stl")
    assert out.positions.shape[0] == 4
    assert torch.equal(out.positions[out.faces], torch.from_numpy(tri))
    assert _decode(tmp_path / "s.stl", merge=False).positions.shape[0] == 6


@pytest.mark.parametrize("device", DEVICES)
def test_mesh_normals_match(device: str) -> None:
    from algorithm.normal_estimation.common.normals import mesh_normals
    from t3ds_io import read_obj, vertex_normals
    V, F, _ = read_obj(str(teeth3ds_scans("lower", 1)[0]))
    got = mesh_normals(torch.from_numpy(V).to(device), torch.from_numpy(F.astype(np.int64)).to(device))
    assert np.abs(got.cpu().numpy() - vertex_normals(V, F)).max() < 1e-5


def test_pca_normals_orientation_matches() -> None:
    """Same normals up to the arbitrary eigenvector sign, oriented the same way."""
    from algorithm.normal_estimation.common.normals import pca_normals
    from infer import pca_normals as legacy_pca
    from t3ds_io import read_obj, vertex_normals
    V, F, _ = read_obj(str(teeth3ds_scans("lower", 1)[0]))
    V = V[::4].copy()
    want = legacy_pca(V)
    got = pca_normals(torch.from_numpy(V)).numpy()
    dot = (want * got).sum(1)
    assert np.abs(dot).min() > 0.99            # the same axes
    assert (dot > 0).mean() > 0.999            # and the same sign walk
    mesh = vertex_normals(*read_obj(str(teeth3ds_scans("lower", 1)[0]))[:2])[::4]
    assert ((mesh * got).sum(1) > 0).mean() > 0.95   # mostly outward, as the original


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_downsample_groups_match(device: str) -> None:
    from algorithm.voxel_downsample.common.grid import voxel_downsample
    from t3ds_io import read_obj, voxel_downsample as legacy
    for path in teeth3ds_scans("lower", 2):
        V, _, _ = read_obj(str(path))
        rep_l, _, inv_l = legacy(V, 0.25, return_inverse=True)
        rep, inv, cells = voxel_downsample(torch.from_numpy(V).to(device), 0.25)
        rep, inv = rep.cpu().numpy(), inv.cpu().numpy()
        assert np.array_equal(inv, inv_l)                       # same voxels, same order
        assert np.array_equal(inv[rep], np.arange(len(rep)))    # each rep inside its voxel
        assert (rep == rep_l).mean() > 0.999                    # float32 tie-breaks, as the original
        grid = np.floor((V - V.min(0)) / 0.25).astype(np.int64)
        assert np.array_equal(cells.cpu().numpy(), grid[rep])
