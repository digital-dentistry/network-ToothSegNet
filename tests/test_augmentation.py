"""Augmentation against `toothseg3d/dataset.py`: the deterministic parts exactly,
the random parts by the properties they must keep."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, teeth3ds_scans


@pytest.fixture(scope="module")
def scan() -> dict[str, np.ndarray]:
    from t3ds_io import fdi_to_class, read_labels, read_obj, vertex_normals
    path = teeth3ds_scans("lower", 1)[0]
    V, F, _ = read_obj(str(path))
    lab, ins, jaw = read_labels(str(path)[:-4] + ".json")
    return {"xyz": V.astype(np.float64), "nrm": vertex_normals(V, F).astype(np.float32),
            "cls": fdi_to_class(lab, jaw).astype(np.int64), "ins": ins.astype(np.int64)}


def _sample(scan: dict[str, np.ndarray], device: str = "cpu", colour: bool = False):
    from algorithm.augmentation.common.sample import AugmentationSample
    n = len(scan["xyz"])
    rgb = torch.rand(n, 3, generator=torch.Generator().manual_seed(1)) if colour else None
    return AugmentationSample(torch.from_numpy(scan["xyz"]).float(), torch.from_numpy(scan["nrm"]),
                              torch.from_numpy(scan["cls"]), torch.from_numpy(scan["ins"]), rgb).to(device)


def _legacy_dataset():
    from dataset import Teeth3DS
    return Teeth3DS.__new__(Teeth3DS)


def _run(typ, sample, params=None, seed: int = 0):
    from algorithm.augmentation.augmentation import (Augmentation, AugmentationInput,
                                                     AugmentationOutput)
    from algorithm.augmentation.common.params import AugmentationParams
    alg, out = Augmentation(), AugmentationOutput()
    alg.set_strategy(typ)
    assert alg.execute(AugmentationInput(sample, params or AugmentationParams(),
                                         torch.Generator().manual_seed(seed)), out)
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reach", [2.5, 6.0, 14.0])
def test_trim_matches_original(scan, device: str, reach: float) -> None:
    from algorithm.augmentation.common.crop import trim_mask
    ref = _legacy_dataset()._trim_gingiva(scan["xyz"], scan["ins"], reach)
    s = _sample(scan, device)
    got = trim_mask(s.positions, s.instances, reach).cpu().numpy()
    # float32 positions here, float64 in the original: at most a tie at the reach
    assert (got != ref).sum() <= 2, f"{(got != ref).sum()} vertices differ"


@pytest.mark.parametrize("seed", range(4))
def test_wedge_matches_original(scan, seed: int) -> None:
    from algorithm.augmentation.common.crop import wedge_mask
    keep = 0.3 + 0.15 * seed
    np.random.seed(seed)
    start = np.random.uniform(-np.pi, np.pi)
    np.random.seed(seed)
    ref = _legacy_dataset()._cut(scan["xyz"], keep)
    got = wedge_mask(torch.from_numpy(scan["xyz"]), keep, start).numpy()
    assert (got != ref).sum() <= 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["bridge", "quadrant", "half"])
def test_segment_crop_keeps_whole_teeth(scan, device: str, mode: str) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    from algorithm.augmentation.common.distance import nearest_distance
    from algorithm.augmentation.common.params import AugmentationParams
    s = _sample(scan, device)
    typ = AugmentationType.SEGMENT_CROP_GPU if device == "cuda" else AugmentationType.SEGMENT_CROP_CPU
    params = AugmentationParams(crop_mode=mode)
    for seed in range(5):
        out = _run(typ, s, params, seed)
        assert out.applied
        c = out.sample
        before = torch.bincount(s.instances)
        after = torch.bincount(c.instances, minlength=before.numel())
        kept = torch.nonzero(after[1:]).squeeze(1) + 1
        assert torch.equal(after[kept], before[kept]), "a tooth was cut"
        gum = c.positions[c.instances == 0]
        teeth = c.positions[c.instances > 0]
        assert float(nearest_distance(gum, teeth).max()) < params.crop_margin_mm[mode][1]


def test_crop_declines_with_too_few_teeth(scan) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    s = _sample(scan)
    two = s.select(s.instances <= int(torch.unique(s.instances)[2]))       # gingiva + 2 teeth
    out = _run(AugmentationType.SEGMENT_CROP_CPU, two)
    assert not out.applied and out.sample.n == two.n


@pytest.mark.parametrize("device", DEVICES)
def test_mirror(scan, device: str) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    typ = AugmentationType.MIRROR_GPU if device == "cuda" else AugmentationType.MIRROR_CPU
    s = _sample(scan, device)
    once = _run(typ, s).sample
    from dataset import MIRROR_MAP
    assert torch.equal(once.classes.cpu(), torch.from_numpy(MIRROR_MAP[scan["cls"]]))
    assert torch.equal(once.positions[:, 1:], s.positions[:, 1:])
    twice = _run(typ, once).sample
    assert torch.equal(twice.positions, s.positions) and torch.equal(twice.classes, s.classes)


@pytest.mark.parametrize("device", DEVICES)
def test_rotations_are_rigid(scan, device: str) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    from algorithm.augmentation.common.params import AugmentationParams
    from algorithm.augmentation.common.rotation import random_so3, small_rotation
    for r in (random_so3(torch.Generator().manual_seed(3)), small_rotation(None, 10.0, 25.0)):
        assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
        assert abs(float(torch.linalg.det(r)) - 1) < 1e-12
    s = _sample(scan, device)
    rigid = AugmentationParams(scale=0.0, jitter_mm=0.0)
    for typ in (AugmentationType.RANDOM_SO3_CPU, AugmentationType.SMALL_ROTATION_CPU):
        if device == "cuda":
            typ = AugmentationType(typ.value + 1)
        r = _run(typ, s, rigid).sample
        idx = torch.arange(0, s.n, 997, device=device)
        a, b = r.positions[idx].double(), s.positions[idx].double()
        assert torch.allclose(torch.cdist(a, a), torch.cdist(b, b), atol=1e-4)
        assert torch.allclose(r.normals.norm(dim=1), s.normals.norm(dim=1), atol=1e-5)


def test_small_rotation_stays_small() -> None:
    from algorithm.augmentation.common.rotation import small_rotation
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        r = small_rotation(g, 10.0, 25.0)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(r)) - 1) / 2))))
        assert angle <= math.sqrt(10 ** 2 + 10 ** 2 + 25 ** 2) + 1e-6


def test_synth_colour(scan) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    s = _sample(scan)
    for seed in range(5):
        c = _run(AugmentationType.SYNTH_COLOUR_CPU, s, seed=seed).sample.colour
        assert c.shape == (s.n, 3) and float(c.min()) >= 0 and float(c.max()) <= 1
        assert float(c.std(0).mean()) > 0.01            # not flat: standardise_colour must not zero it


def test_recipe_colour_policy(scan) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    from algorithm.augmentation.common.params import AugmentationParams
    plain = AugmentationParams(mirror_p=0, crop_p=0, trim_p=0, jitter_mm=0)
    real = _sample(scan, colour=True)
    out = _run(AugmentationType.TRAIN_RECIPE_CPU, real, AugmentationParams(**{**plain.__dict__, "colour_p": 0}))
    assert out.sample.colour is None                     # colourless, real colour included
    out = _run(AugmentationType.TRAIN_RECIPE_CPU, real, AugmentationParams(**{**plain.__dict__, "colour_p": 1}))
    assert torch.equal(out.sample.colour, real.colour)   # real colour wins over painting
    out = _run(AugmentationType.TRAIN_RECIPE_CPU, _sample(scan), AugmentationParams(**{**plain.__dict__, "colour_p": 1}))
    assert out.sample.colour is not None


@pytest.mark.skipif("cuda" not in DEVICES, reason="needs CUDA")
def test_recipe_draws_the_same_on_cpu_and_gpu(scan) -> None:
    """Every scalar comes from the host generator, so a seed crops, trims and
    mirrors the same on both devices; only the per-vertex jitter differs."""
    from algorithm.augmentation.augmentation import AugmentationType
    from algorithm.augmentation.common.params import AugmentationParams
    params = AugmentationParams(mirror_p=0.5, crop_p=0.8, trim_p=0.8)
    for seed in range(6):
        cpu = _run(AugmentationType.TRAIN_RECIPE_CPU, _sample(scan), params, seed).sample
        gpu = _run(AugmentationType.TRAIN_RECIPE_GPU, _sample(scan), params, seed).sample
        assert cpu.n == gpu.n
        assert torch.equal(cpu.instances, gpu.instances.cpu()) and torch.equal(cpu.classes, gpu.classes.cpu())
        assert float((cpu.positions - gpu.positions.cpu()).abs().max()) < 0.5    # jitter only


def test_same_seed_same_sample(scan) -> None:
    from algorithm.augmentation.augmentation import AugmentationType
    a = _run(AugmentationType.TRAIN_RECIPE_CPU, _sample(scan), seed=7).sample
    b = _run(AugmentationType.TRAIN_RECIPE_CPU, _sample(scan), seed=7).sample
    assert torch.equal(a.positions, b.positions) and torch.equal(a.instances, b.instances)
