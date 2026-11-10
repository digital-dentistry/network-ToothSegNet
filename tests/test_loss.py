"""The loss against `inst_head.InstanceCriterion` and `train_inst.py`'s training step."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import DEVICES

COUNTS = [400, 300]
COARSE = [100, 80]
TEETH = [3, 2]
K = 8


def _batch(device: str, seed: int = 0) -> dict:
    """Random network outputs and labels for two scans, in both id conventions."""
    from algorithm.model_input.common.geometry import pad_index
    g = torch.Generator().manual_seed(seed)
    n = sum(COUNTS)
    local = torch.cat([torch.randint(0, t + 1, (c,), generator=g) for c, t in zip(COUNTS, TEETH)])
    # toothseg3d's collate: ids run on across the batch
    batch_ids, base = local.clone(), 0
    start = 0
    for c, t in zip(COUNTS, TEETH):
        seg = batch_ids[start:start + c]
        seg[seg > 0] += base
        base += t
        start += c
    inverse = torch.cat([torch.randint(0, m, (c,), generator=g) + sum(COARSE[:i])
                         for i, (c, m) in enumerate(zip(COUNTS, COARSE))])
    pidx, pvalid = pad_index(COARSE, torch.device("cpu"))
    cls_ = torch.randint(0, 18, (n,), generator=g)
    out = {
        "cls": torch.randn(len(COUNTS), K, 2, generator=g),
        "mask": torch.randn(n, K, generator=g) * 2,
        "aux": [torch.randn(len(COUNTS), K, max(COARSE), generator=g) * 2 for _ in range(3)],
        "sem": torch.randn(n, 9, generator=g),
        "off": torch.randn(n, 3, generator=g),
    }
    from dataset import POS_MAP
    pos = torch.from_numpy(POS_MAP[cls_.numpy()])
    offt = torch.randn(n, 3, generator=g)
    move = lambda t: t.to(device)
    return {"out": {k: (list(map(move, v)) if isinstance(v, list) else move(v)) for k, v in out.items()},
            "local": move(local), "batch_ids": move(batch_ids), "inverse": move(inverse),
            "pidx": move(pidx), "pvalid": move(pvalid), "pos": move(pos), "cls": move(cls_), "offt": move(offt)}


def _legacy(b: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from inst_head import InstanceCriterion
    from train_inst import coarse_targets, make_targets
    dev = b["local"].device
    crit = InstanceCriterion(w_cls=0.5, w_bce=5.0, w_dice=2.0, no_obj=0.1).to(dev)
    tg = make_targets(b["batch_ids"], np.array(COUNTS), TEETH, dev)
    ctg = coarse_targets(tg, b["inverse"], np.array(COUNTS), b["pidx"], b["pvalid"])
    l_bce, l_dice, l_cls, l_aux = crit(b["out"], tg, np.array(COUNTS), ctg, b["pvalid"])
    l_sem = F.cross_entropy(b["out"]["sem"].float(), b["pos"], ignore_index=-1)
    tm = b["pos"] > 0
    l_off = (b["out"]["off"].float()[tm] - b["offt"][tm]).abs().sum(1).mean()
    total = l_bce + l_dice + l_cls + 0.5 * l_aux + 0.5 * l_sem + 0.05 * l_off
    return total, [l_bce, l_dice, l_cls, l_aux, l_sem, l_off]


def _ours(b: dict):
    from algorithm.loss.loss import Loss, LossInput, LossOutput, LossType
    alg, out = Loss(), LossOutput()
    alg.set_strategy(LossType.MASK_TRANSFORMER)
    o = b["out"]
    assert alg.execute(LossInput(
        b["local"], COUNTS, cls_logits=o["cls"], mask_logits=o["mask"], aux_mask_logits=o["aux"],
        coarse_inverse=b["inverse"], pad_index=b["pidx"], pad_valid=b["pvalid"],
        position_logits=o["sem"], arch_positions=b["pos"], offsets=o["off"], offset_targets=b["offt"]), out)
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(3))
def test_matches_instance_criterion(device: str, seed: int) -> None:
    b = _batch(device, seed)
    ref_total, ref_terms = _legacy(b)
    got = _ours(b)
    for name, ref in zip(["bce", "dice", "cls", "aux", "position", "offset"], ref_terms):
        assert torch.allclose(got.terms[name], ref.detach(), rtol=1e-5, atol=1e-6), name
    assert torch.allclose(got.total, ref_total, rtol=1e-5, atol=1e-6)


def test_gradients_match() -> None:
    b = _batch("cpu", 5)
    for v in (b["out"]["mask"], b["out"]["cls"], *b["out"]["aux"]):
        v.requires_grad_(True)
    ref_total, _ = _legacy(b)
    ref = torch.autograd.grad(ref_total, [b["out"]["mask"], b["out"]["cls"], b["out"]["aux"][0]])
    got = torch.autograd.grad(_ours(b).total, [b["out"]["mask"], b["out"]["cls"], b["out"]["aux"][0]])
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, rtol=1e-4, atol=1e-7)


def test_scan_without_teeth() -> None:
    b = _batch("cpu", 1)
    b["local"][:COUNTS[0]] = 0                     # scan 0: gingiva only
    got = _ours(b)
    assert got.matches[0] is None and got.matches[1] is not None
    assert torch.isfinite(got.total)


def test_targets() -> None:
    from algorithm.loss.common.targets import arch_position, coarse_inverse, offset_targets
    from dataset import POS_MAP
    cls_ = torch.arange(18)
    assert torch.equal(arch_position(cls_), torch.from_numpy(POS_MAP))

    g = torch.Generator().manual_seed(0)
    p = torch.randn(700, 3, generator=g) * 20
    ins = torch.randint(0, 5, (700,), generator=g)
    got = offset_targets(p, ins, COUNTS)
    start = 0
    for n in COUNTS:
        pp, ii = p[start:start + n].numpy(), ins[start:start + n].numpy()
        for j in range(1, 5):
            m = ii == j
            assert np.allclose(got[start:start + n][m].numpy(), pp[m].mean(0) - pp[m], atol=1e-4)
        assert (got[start:start + n][ii == 0] == 0).all()
        start += n

    class _Lv:
        def __init__(self, inv: torch.Tensor) -> None:
            self.inv = inv
    l1, l2 = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([0, 1, 1])
    assert torch.equal(coarse_inverse([None, _Lv(l1), _Lv(l2)], 2), torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(coarse_inverse([None, _Lv(l1), _Lv(l2)], 1), l1)


@pytest.mark.parametrize("device", DEVICES)
def test_semantic_matches_train_py(device: str) -> None:
    """SEMANTIC against the loss lines of toothseg3d/train.py's training loop."""
    import train as legacy
    from algorithm.loss.common.targets import class_weights
    from algorithm.loss.loss import Loss, LossInput, LossOutput, LossType, LossWeights
    g = torch.Generator().manual_seed(0)
    n = 900
    sem, off = torch.randn(n, 17, generator=g).to(device), torch.randn(n, 3, generator=g).to(device)
    cls_ = torch.randint(0, 17, (n,), generator=g).to(device)
    offt = torch.randn(n, 3, generator=g).to(device)
    freq = torch.rand(17, generator=g)
    freq /= freq.sum()
    w = 1.0 / np.sqrt(np.maximum(freq.numpy().astype(np.float64), 1e-6))
    wt = torch.tensor(np.clip(w / w.mean(), 0.2, 6.0), dtype=torch.float32, device=device)
    assert torch.allclose(class_weights(freq).to(device), wt)

    l_ce = F.cross_entropy(sem, cls_, weight=wt)
    l_di = legacy.dice_loss(sem, cls_, 17)
    l_of, l_dr = legacy.offset_loss(off, offt, cls_ > 0)
    ref = l_ce + 1.0 * l_di + 0.1 * l_of + 0.2 * l_dr

    alg, out = Loss(), LossOutput()
    alg.set_strategy(LossType.SEMANTIC)
    assert alg.execute(LossInput(cls_ * 0, [n], offsets=off, offset_targets=offt, class_logits=sem,
                                 classes=cls_, class_weights=wt, weights=LossWeights.semantic()), out)
    for name, r in zip(["ce", "class_dice", "offset", "direction"], [l_ce, l_di, l_of, l_dr]):
        assert torch.allclose(out.terms[name], r, rtol=1e-5, atol=1e-6), name
    assert torch.allclose(out.total, ref, rtol=1e-5, atol=1e-6)

    # class 17 (unknown position) is left out rather than crashing one_hot
    cls_[:50] = 17
    assert alg.execute(LossInput(cls_ * 0, [n], class_logits=sem, classes=cls_), out)
    assert torch.isfinite(out.total)
