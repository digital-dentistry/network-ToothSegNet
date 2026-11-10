"""The training path: network training outputs against toothseg3d, the cache
format, the dataset, batching, and a short end-to-end run of both stages."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, ROOT, teeth3ds_scans
from tests.test_instance_network import _as_geometry, legacy_input, nets  # noqa: F401  (fixtures)

PY = sys.executable


def _no_drop_path(*nets: torch.nn.Module) -> None:
    for net in nets:
        for m in net.modules():
            if hasattr(m, "dp"):
                m.dp = 0.0


def test_training_outputs_match_toothseg3d(legacy_input, nets) -> None:  # noqa: F811
    """Train mode adds the decoder's coarse masks and the auxiliary heads, equal
    to what toothseg3d's training forward returns (one scan: no padding)."""
    feat, ref, _, _ = legacy_input
    old, new = nets
    _no_drop_path(old, new)
    old.train(), new.train()
    try:
        with torch.no_grad():
            a, b = old(feat, ref), new(feat, _as_geometry(ref))
    finally:
        old.eval(), new.eval()
    assert len(a["aux"]) == len(b["aux"]) == 4
    for x, y in zip(a["aux"], b["aux"]):
        assert torch.equal(x, y)
    for k in ("cls", "mask", "sem", "off"):
        assert torch.equal(a[k], b[k]), k
    with torch.no_grad():
        assert set(new(feat, _as_geometry(ref))) == {"cls", "mask"}          # inference unchanged


def test_preprocess_matches_toothseg3d(tmp_path: Path) -> None:
    import preprocess as legacy
    from training.preprocess import cache_scan
    obj = teeth3ds_scans("lower", 1)[0]
    assert cache_scan(obj, obj.with_suffix(".json"), tmp_path / "ours.npz") is None
    _, err = legacy.process((str(obj), str(obj.with_suffix(".json")), str(tmp_path / "ref.npz")))
    assert err is None
    a, b = np.load(tmp_path / "ours.npz"), np.load(tmp_path / "ref.npz")
    for k in ("xyz", "cls", "ins", "jaw"):
        assert np.array_equal(a[k], b[k]), k
    assert np.abs(a["nrm"].astype(np.float32) - b["nrm"].astype(np.float32)).max() < 1e-3   # 1 float16 ulp


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    from training.preprocess import cache_scan
    out = tmp_path_factory.mktemp("cache")
    names = []
    for jaw in ("upper", "lower"):
        for obj in teeth3ds_scans(jaw, 4):
            assert cache_scan(obj, obj.with_suffix(".json"), out / f"{obj.stem}.npz") is None
            names.append(obj.stem)
    return out, names


def test_validation_item_matches_toothseg3d(cache) -> None:
    """Without augmentation the item is toothseg3d's: same points, labels, targets."""
    from algorithm.model_input.common.builder import CurveKey
    from dataset import POS_MAP, Teeth3DS
    from training.dataset import DatasetConfig, TeethDataset
    d, names = cache
    ours = TeethDataset([d], names[:2], DatasetConfig(train=False, max_points=None, in_ch=7,
                                                       curve=CurveKey.LEGACY))
    ref = Teeth3DS(str(d), names[:2], 0.25, train=False, mirror_p=0.0, in_ch=7)
    for i in range(2):
        a, b = ours[i], ref[i]
        assert a.features.shape[0] == b["n"]
        assert torch.equal(a.classes, torch.from_numpy(b["cls"]))
        # toothseg3d renumbers the surviving teeth 1..G; the grouping is what matters
        assert torch.equal(a.instances > 0, torch.from_numpy(b["ins"] > 0))
        assert torch.equal(a.arch_positions, torch.from_numpy(POS_MAP[b["cls"]]))
        # a few voxels pick a different member as representative (the float
        # tie-break of test_preprocessing: 99.98 % agree), which moves that
        # point by up to a voxel; everything else matches
        same_off = (a.offset_targets - torch.from_numpy(b["off"])).abs().max(1).values < 1e-3
        same_feat = (a.features - torch.from_numpy(b["feat"])).abs().max(1).values < 1e-4
        assert same_off.float().mean() > 0.999 and same_feat.float().mean() > 0.999


def test_batch_equals_scans_alone(cache) -> None:
    """A collated batch gives each scan the outputs it gets on its own."""
    from layer.checkpoint import ModelSpec, build_network
    from training.dataset import DatasetConfig, TeethDataset, collate
    d, names = cache
    ds = TeethDataset([d], names[:3], DatasetConfig(train=False, max_points=None))
    items = [ds[i] for i in range(3)]
    torch.manual_seed(0)
    net = build_network(ModelSpec(in_ch=10)).eval()
    batch = collate(items)
    with torch.no_grad():
        both = net(batch.features, batch.geometry)
        start = 0
        for b, it in enumerate(items):
            one = net(it.features, it.geometry)
            n = it.features.shape[0]
            assert (both["cls"][b] - one["cls"][0]).abs().max() < 1e-4
            assert (both["mask"][start:start + n] - one["mask"]).abs().max() < 1e-4
            start += n


def test_training_items_are_augmented(cache) -> None:
    from training.dataset import DatasetConfig, TeethDataset
    d, names = cache
    ds = TeethDataset([d], names[:1], DatasetConfig(train=True))
    torch.manual_seed(0)
    a, b = ds[0], ds[0]
    assert not torch.equal(a.features[:, :3], b.features[:, :3]) or a.features.shape != b.features.shape
    assert a.features.shape[1] == 10 and torch.isfinite(a.features).all()


@pytest.mark.skipif("cuda" not in DEVICES, reason="training needs CUDA")
def test_both_stages_run(cache, tmp_path: Path) -> None:
    """Semantic, then instance from its checkpoint, then the pipeline on the result."""
    d, names = cache
    names_file = tmp_path / "names.txt"
    names_file.write_text("\n".join(names))
    common = ["--cache", str(d), "--names", str(names_file), "--epochs", "1", "--batch", "2",
              "--workers", "0", "--val-every", "1"]
    run = lambda *args: subprocess.run([PY, str(ROOT / "training/train.py"), *args, *common],
                                       capture_output=True, text=True, timeout=600)
    sem = run("--stage", "semantic", "--out", str(tmp_path / "sem"))
    assert sem.returncode == 0, sem.stderr[-2000:]
    inst = run("--stage", "instance", "--out", str(tmp_path / "inst"), "--val-crop", "bridge",
               "--init-from", str(tmp_path / "sem/best.pt"))
    assert inst.returncode == 0, inst.stderr[-2000:]
    assert "widened" in inst.stdout                    # the 7-channel stem grew to 10

    from algorithm.model_input.common.builder import CurveKey
    from layer.checkpoint import load_checkpoint
    from pipeline.config import SegmentationConfig
    from pipeline.segmenter import Segmenter
    model = load_checkpoint(tmp_path / "inst/best.pt")
    assert model.spec.curve is CurveKey.MORTON3D and model.spec.in_ch == 10
    r = Segmenter(SegmentationConfig(checkpoint=tmp_path / "inst/best.pt", tta=1)).segment_file(
        teeth3ds_scans("upper", 1)[0])
    assert r.error is None
