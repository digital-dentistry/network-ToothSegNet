"""The network port: strict checkpoint loading, bit-identical forward."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import teeth3ds_scans

ENC, DEC = (1, 1, 2, 4, 2), (1, 1, 1, 1)


@pytest.fixture(scope="module")
def legacy_input() -> tuple[torch.Tensor, dict, np.ndarray, np.ndarray]:
    import geometry as G
    from t3ds_io import read_obj, vertex_normals, voxel_downsample
    V, F, _ = read_obj(str(teeth3ds_scans("lower", 1)[0]))
    rep, _ = voxel_downsample(V, 0.25)
    p = (V[rep] - V[rep].mean(0)).astype(np.float32)
    n = vertex_normals(V, F)[rep]
    n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    feat = torch.from_numpy(np.concatenate([p / 35.0, n, np.zeros((len(p), 1), np.float32)], 1))
    geo = G.geo_to_torch(G.pack_geometry(G.collate_geometry(
        [G.build_geometry(p, 0.25, ENC, DEC, 1024)]), pad_level=1, n_query=32), "cpu")
    return feat, geo, p, n


@pytest.fixture(scope="module")
def nets() -> tuple[torch.nn.Module, torch.nn.Module]:
    from layer.checkpoint import ModelSpec, build_network
    from train_inst import build_net
    torch.manual_seed(0)
    old = build_net(32, "cpu", 7, 128, 4).eval()
    new = build_network(ModelSpec())
    new.load_state_dict(old.state_dict(), strict=True)       # every key, same shapes
    return old, new.eval()


def _as_geometry(ref: dict):
    from algorithm.model_input.common.geometry import Geometry, Layout, Level
    levels = [Level(coord=lv["coord"], n=lv["n"], counts=list(lv["counts"]),
                    inv=lv.get("inv"), cnt=lv.get("cnt")) for lv in ref["levels"]]
    return Geometry(levels, {k: Layout(*v) for k, v in ref["layouts"].items()}, 1, ref["seed"])


def test_forward_bit_identical(legacy_input, nets) -> None:
    feat, ref, _, _ = legacy_input
    old, new = nets
    with torch.inference_mode():
        a, b = old(feat, ref), new(feat, _as_geometry(ref))
    assert torch.equal(a["cls"], b["cls"]) and torch.equal(a["mask"], b["mask"])


def test_full_port_close(legacy_input, nets) -> None:
    """Own geometry + features: differs only by the patch-mean rounding."""
    from algorithm.model_input.model_input import ModelInput, ModelInputInput, ModelInputOutput, ModelInputType
    feat, ref, p, n = legacy_input
    old, new = nets
    mi, out = ModelInput(), ModelInputOutput()
    mi.set_strategy(ModelInputType.LEGACY_CPU)
    mi.execute(ModelInputInput(torch.from_numpy(p), torch.from_numpy(n), False, None,
                               torch.eye(3, dtype=torch.float64)[None], 0.25), out)
    with torch.inference_mode():
        a, b = old(feat, ref), new(out.features, out.geometry)
    assert (a["cls"] - b["cls"]).abs().max() < 1e-3
    assert (a["mask"] - b["mask"]).abs().max() < 1e-2


def test_batched_poses_equal_single_runs(legacy_input, nets) -> None:
    """TTA as one packed batch == each pose alone (the padding fix in InstanceHead)."""
    from algorithm.model_input.model_input import ModelInput, ModelInputInput, ModelInputOutput, ModelInputType
    from pipeline.rotations import tta_rotations
    _, _, p, n = legacy_input
    _, new = nets
    mi = ModelInput()
    mi.set_strategy(ModelInputType.LEGACY_CPU)

    def run(rots: torch.Tensor) -> dict:
        out = ModelInputOutput()
        mi.execute(ModelInputInput(torch.from_numpy(p), torch.from_numpy(n), False, None, rots, 0.25), out)
        with torch.inference_mode():
            return new(out.features, out.geometry)

    rots = tta_rotations(3)
    both = run(rots)
    for t in range(3):
        one = run(rots[t:t + 1])
        assert (both["cls"][t] - one["cls"][0]).abs().max() < 1e-4
        assert (both["mask"][t * len(p):(t + 1) * len(p)] - one["mask"]).abs().max() < 1e-4
