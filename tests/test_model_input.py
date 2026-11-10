"""Network input against toothseg3d: integer arrays bit for bit, floats to rounding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, teeth3ds_scans

ENC, DEC = (1, 1, 2, 4, 2), (1, 1, 1, 1)
EYE = torch.eye(3, dtype=torch.float64)[None]


def _l0(path):
    from t3ds_io import read_obj, vertex_normals, voxel_downsample
    V, F, _ = read_obj(str(path))
    rep, _, _ = voxel_downsample(V, 0.25, return_inverse=True)
    p = (V[rep] - V[rep].mean(0)).astype(np.float32)
    n = vertex_normals(V, F)[rep]
    n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    cells = np.floor((V[rep] - V.min(0)) / 0.25).astype(np.int64)
    return p, n, cells


def _legacy_make_input():
    """toothseg3d/algorithm/make_input/make_input.py, by path: that `algorithm`
    folder is a namespace package our own `algorithm` shadows."""
    import importlib.util
    from tests.conftest import LEGACY
    spec = importlib.util.spec_from_file_location(
        "legacy_make_input", LEGACY / "algorithm/make_input/make_input.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(p, n, typ, rotations=EYE, cells=None, colour=None):
    from algorithm.model_input.model_input import ModelInput, ModelInputInput, ModelInputOutput
    alg, out = ModelInput(), ModelInputOutput()
    alg.set_strategy(typ)
    assert alg.execute(ModelInputInput(torch.from_numpy(p), torch.from_numpy(n), False, colour, rotations,
                                       0.25, cells=None if cells is None else torch.from_numpy(cells)), out)
    return out.features, out.geometry.to(torch.device("cpu"))


def _assert_same(ref: dict, geo) -> None:
    """ref: the original's geo_to_torch view.  Integers exact, floats close."""
    for L, lv in enumerate(ref["levels"]):
        got = geo.levels[L]
        assert lv["n"] == got.n and list(lv["counts"]) == got.counts
        if L:
            assert torch.equal(lv["inv"], got.inv), f"L{L} inv"
            assert torch.equal(lv["cnt"], got.cnt), f"L{L} cnt"
        assert (lv["coord"] - got.coord).abs().max() < 1e-5, f"L{L} coord"
    assert ref["layouts"].keys() == geo.layouts.keys()
    for key, (sel, inv_sel, rel) in ref["layouts"].items():
        lay = geo.layouts[key]
        assert torch.equal(sel, lay.sel), f"{key} sel"
        assert torch.equal(inv_sel, lay.inv_sel), f"{key} inv_sel"
        assert (rel - lay.rel).abs().max() < 1e-4, f"{key} rel"     # float32 mean order
    assert torch.equal(ref["pad"][0], geo.pad_index) and torch.equal(ref["pad"][1], geo.pad_valid)
    assert torch.equal(ref["seed"], geo.seed)


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_geometry_bit_exact(device: str) -> None:
    import geometry as G
    from algorithm.model_input.model_input import ModelInputType
    typ = ModelInputType.LEGACY_GPU if device == "cuda" else ModelInputType.LEGACY_CPU
    for path in teeth3ds_scans("lower", 2) + teeth3ds_scans("upper", 1):
        p, n, _ = _l0(path)
        ref = G.geo_to_torch(G.pack_geometry(G.collate_geometry(
            [G.build_geometry(p, 0.25, ENC, DEC, 1024)]), pad_level=1, n_query=32), "cpu")
        feat, geo = _build(p, n, typ)
        _assert_same(ref, geo)
        want = np.concatenate([p / 35.0, n, np.zeros((len(p), 1), np.float32)], 1)
        if device == "cpu":
            assert np.array_equal(feat.numpy(), want)
        else:   # CUDA divides by a scalar as a multiply by its reciprocal: 1 ulp
            assert np.abs(feat.cpu().numpy() - want).max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_morton3d_geometry_matches_make_input(device: str) -> None:
    import geometry as G
    from algorithm.model_input.model_input import ModelInputType
    MI = _legacy_make_input()
    typ = ModelInputType.MORTON3D_GPU if device == "cuda" else ModelInputType.MORTON3D_CPU
    p, n, cells = _l0(teeth3ds_scans("lower", 1)[0])
    ref = G.geo_to_torch(G.pack_geometry(G.collate_geometry(
        [MI.build_geometry(p, cells, ENC, DEC, 1024)]), pad_level=1, n_query=32), "cpu")
    _, geo = _build(p, n, typ, cells=cells)
    _assert_same(ref, geo)


def test_poses_collate_like_the_original() -> None:
    """Three poses in one packed batch == the original's collate of three scans."""
    import geometry as G
    from algorithm.model_input.model_input import ModelInputType
    from pipeline.rotations import tta_rotations
    p, n, _ = _l0(teeth3ds_scans("lower", 1)[0])
    rots = tta_rotations(3)
    poses = [p] + [(p @ r.numpy().T).astype(np.float32) for r in rots[1:]]
    ref = G.geo_to_torch(G.pack_geometry(G.collate_geometry(
        [G.build_geometry(q, 0.25, ENC, DEC, 1024) for q in poses]), pad_level=1, n_query=32), "cpu")
    feat, geo = _build(p, n, ModelInputType.LEGACY_CPU, rotations=rots)
    _assert_same(ref, geo)
    assert np.allclose(feat[len(p):2 * len(p), :3].numpy(), poses[1] / 35.0)


def test_rotations_reproduce_the_original_sequence() -> None:
    from dataset import Teeth3DS
    from pipeline.rotations import tta_rotations
    rs = np.random.RandomState(0)
    want = [np.eye(3)]
    for _ in range(7):
        np.random.seed(rs.randint(1 << 30))
        want.append(Teeth3DS.random_so3())
    assert np.array_equal(tta_rotations(8).numpy(), np.stack(want))


def test_colour_standardisation() -> None:
    from algorithm.model_input.common.features import standardise_colour
    from dataset import standardise_colour as legacy
    rgb = np.random.RandomState(1).rand(1000, 3).astype(np.float32)
    assert np.allclose(standardise_colour(torch.from_numpy(rgb)).numpy(), legacy(rgb), atol=1e-5)
    flat = np.full((100, 3), 0.5, np.float32)
    assert (standardise_colour(torch.from_numpy(flat)) == 0).all()
