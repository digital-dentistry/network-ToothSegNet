"""The whole pipeline against `infer.segment_inst`, with an oracle in place of the network.

There is no trained checkpoint on this machine, and random weights claim no
teeth, which would make a comparison vacuous.  The oracle turns ground-truth
instances into mask / class logits with noise, and shuffles which query slot
holds which tooth in every pose -- so TTA has to line the poses up by mask IoU,
exactly what it does with a real model.  Both pipelines see the same logits for
the same pose and point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, teeth3ds_scans

K = 32


def _oracle_logits(gt_l0: np.ndarray, pose: int) -> tuple[np.ndarray, np.ndarray]:
    """[K,2] class logits and [N,K] mask logits for one pose."""
    rs = np.random.RandomState(1000 + pose)
    slot = rs.permutation(K)                         # tooth g -> query slot[g - 1]
    n_teeth = int(gt_l0.max())
    mask = (-4.0 + 1.5 * rs.randn(len(gt_l0), K)).astype(np.float32)
    tooth = gt_l0 > 0
    mask[np.flatnonzero(tooth), slot[gt_l0[tooth] - 1]] += 8.0
    cls = np.tile(np.float32([-3.0, 3.0]), (K, 1))
    cls[slot[:n_teeth]] = [3.0, -3.0]
    cls += 0.3 * rs.randn(K, 2).astype(np.float32)
    return cls, mask


class _LegacyOracle(torch.nn.Module):
    """Stands in for the network inside `infer.segment_inst`: one call per pose."""

    def __init__(self, gt_l0: np.ndarray) -> None:
        super().__init__()
        self._gt, self._pose = gt_l0, 0

    def forward(self, feat: torch.Tensor, geo: dict) -> dict:
        cls, mask = _oracle_logits(self._gt, self._pose)
        self._pose += 1
        return {"cls": torch.from_numpy(cls)[None], "mask": torch.from_numpy(mask)}


class _Oracle:
    """Stands in for the InstanceNetwork strategy: a batch holds several poses."""

    def __init__(self, gt_l0: np.ndarray) -> None:
        self._gt, self._pose = gt_l0, 0

    def execute(self, inp, out) -> bool:
        cls, mask = [], []
        for _ in range(inp.geometry.batch):
            c, m = _oracle_logits(self._gt, self._pose)
            cls.append(torch.from_numpy(c))
            mask.append(torch.from_numpy(m))
            self._pose += 1
        out.cls_logits, out.mask_logits = torch.stack(cls), torch.cat(mask)
        return True


@pytest.fixture(scope="module")
def case() -> tuple[Path, np.ndarray]:
    path = teeth3ds_scans("lower", 1)[0]
    raw = np.asarray(json.loads(path.with_suffix(".json").read_text())["instances"])
    return path, np.unique(raw, return_inverse=True)[1]      # 0 = gingiva, 1..G


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tta,erode", [(4, 0.0), (1, 0.2)])
def test_matches_segment_inst(case, device: str, tta: int, erode: float) -> None:
    from infer import segment_inst
    from t3ds_io import read_obj, vertex_normals, voxel_downsample
    from algorithm.voxel_downsample.common.grid import voxel_downsample as new_downsample
    from pipeline.config import Backend, SegmentationConfig
    from pipeline.segmenter import Segmenter

    path, gt = case
    V, F, _ = read_obj(str(path))
    rep_old, _ = voxel_downsample(V, 0.25)
    want_ins, want_sc, want_conf = segment_inst(
        _LegacyOracle(gt[rep_old]), V, vertex_normals(V, F), "lower", 0.25, K, "cpu",
        return_conf=True, tta=tta, erode_mm=erode)

    seg = Segmenter(SegmentationConfig(backend=Backend.GPU if device == "cuda" else Backend.CPU,
                                       tta=tta, erode_mm=erode))
    rep_new = new_downsample(torch.from_numpy(V), 0.25)[0].numpy()
    seg._infer._network._current = _Oracle(gt[rep_new])        # test-only: swap the forward
    got = seg.segment_file(path)

    assert got.n_teeth == len(want_sc) == int(gt.max())
    agree = (got.instances == want_ins).mean()
    assert agree > 0.999, f"instance agreement {agree:.5f}"
    assert np.allclose(got.scores, want_sc, atol=2e-3)
    assert np.percentile(np.abs(got.vertex_confidence - want_conf), 99.9) < 1e-2
    # and the oracle is recovered, so this is not a comparison of two failures.
    # Ids follow pose 0's query slots, not the GT numbering: match by best IoU.
    best = [max(((got.instances == j) & (gt == g)).sum() / ((got.instances == j) | (gt == g)).sum()
                for j in range(1, got.n_teeth + 1)) for g in range(1, int(gt.max()) + 1)]
    assert min(best) > 0.8          # 0.2 mm erosion alone costs ~0.1 IoU
