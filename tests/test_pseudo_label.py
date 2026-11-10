"""Pseudo-labels: which in-house files are labelled, and that a written label
trains like a Teeth3DS scan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import DATA


def test_select_keeps_one_copy_and_holds_out_copies(tmp_path: Path) -> None:
    from training.pseudo_label import select
    for name, body in (("case_0001_unknown", b"a"), ("case_0002_upper", b"a"), ("case_0003_lower", b"b"),
                       ("case_0004_lower", b"c"), ("case_0005_unknown", b"c")):
        (tmp_path / f"{name}.ply").write_bytes(body)
    keep, skipped = select(sorted(tmp_path.iterdir()), {"case_0004_lower"})
    assert [p.stem for p in keep] == ["case_0002_upper", "case_0003_lower"]      # the copy that knows its jaw
    assert skipped == {"case_0001_unknown": "duplicate of case_0002_upper",
                       "case_0004_lower": "holdout", "case_0005_unknown": "holdout"}


@pytest.fixture(scope="module")
def label(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A coloured in-house scan with four made-up teeth under scattered ids."""
    from pipeline.config import Backend, Placement, SegmentationConfig
    from pipeline.stages import LoadStage
    from pipeline.types import SegmentationResult
    from training.pseudo_label import write_label
    path = DATA / "ios/inhouse/raw/INHOUSE0003_lower.ply"
    if not path.exists():
        pytest.skip("in-house scans not found under data/ios/inhouse/raw")
    cfg = SegmentationConfig(backend=Backend.CPU)
    scan = LoadStage(cfg, Placement.resolve(cfg)).run(path)
    assert scan.colours is not None
    p = scan.positions.numpy()
    top = p[:, 2] > np.median(p[:, 2])
    band = np.digitize(p[:, 0], np.quantile(p[:, 0], [0.25, 0.5, 0.75]))
    ins = np.where(top, 3 + 2 * band, 0).astype(np.int64)                       # ids 3, 5, 7, 9
    out = tmp_path_factory.mktemp("pseudo")
    write_label(scan, SegmentationResult(path=path, name=scan.name, jaw=scan.jaw, n_vertices=len(ins),
                                         instances=ins), out / f"{scan.name}.npz")
    return out, scan.name


def test_label_is_the_cache_format(label) -> None:
    out, name = label
    d = np.load(out / f"{name}.npz")
    assert set(d.files) == {"xyz", "nrm", "cls", "ins", "jaw", "rgb"}
    assert (d["xyz"].dtype, d["nrm"].dtype, d["cls"].dtype, d["ins"].dtype, d["rgb"].dtype) == (
        np.float32, np.float16, np.uint8, np.uint8, np.uint8)
    assert set(np.unique(d["ins"])) == {0, 1, 2, 3, 4}                          # ids packed to 1..K
    assert np.array_equal(d["cls"] == 17, d["ins"] > 0)
    assert int(d["jaw"]) == 0 and len(d["rgb"]) == len(d["xyz"])


def test_label_trains_like_teeth3ds(label) -> None:
    """Through the training dataset, augmented and not: ids survive, no arch
    position is claimed for a tooth, and the real colour reaches the features."""
    from training.dataset import DatasetConfig, TeethDataset
    out, name = label
    torch.manual_seed(0)
    for _ in range(4):
        it = TeethDataset([out], [name], DatasetConfig())[0]
        tooth = it.instances > 0
        assert tooth.any() and set(it.classes.unique().tolist()) <= {0, 17}
        assert bool((it.arch_positions[tooth] == -1).all()) and bool((it.arch_positions[~tooth] == 0).all())
        assert it.features.shape[1] == 10 and bool(torch.isfinite(it.features).all())
    it = TeethDataset([out], [name], DatasetConfig(train=False, colour_force=True))[0]
    assert set(it.instances.unique().tolist()) == {0, 1, 2, 3, 4}
    assert float(it.features[:, 7:].abs().sum()) > 0                            # real colour, not zeros
