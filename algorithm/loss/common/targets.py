"""Training targets built from per-point labels (ports of `train_inst.make_targets`,
`train_inst.coarse_targets` and the target half of `Teeth3DS.__getitem__`)."""
from __future__ import annotations

from typing import Protocol

import torch

# Position along the arch, ignoring the side of the mouth: 0 gingiva, 1 central
# incisor .. 8 third molar.  Under SO(3) augmentation the two sides cannot be
# told apart, so the 17-class FDI target would be noise.  Class 17 (a tooth of
# unknown position, the in-house scans) maps to -1, which cross_entropy ignores.
ARCH_POSITION = torch.tensor([0] + list(range(1, 9)) * 2 + [-1])
NUM_ARCH_POSITIONS = 9


class _Pooled(Protocol):
    inv: torch.Tensor                # [points of the level below] -> point of this level


def class_weights(frequency: torch.Tensor) -> torch.Tensor:
    """Cross-entropy weights from class frequencies: inverse square root, mean 1,
    clipped to [0.2, 6] (`train.py`).  Third molars are in ~6 % of scans; plain
    CE would ignore them."""
    w = 1.0 / frequency.double().clamp_min(1e-6).sqrt()
    return (w / w.mean()).clamp(0.2, 6.0).float()


def arch_position(classes: torch.Tensor) -> torch.Tensor:
    """[N] class 0..17 -> [N] arch position 0..8, -1 unknown."""
    return ARCH_POSITION.to(classes.device)[classes]


def instance_targets(instances: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    """Per scan, a [G,n] float mask per tooth, teeth in ascending id order."""
    out = []
    for v in torch.split(instances, counts):
        ids = torch.unique(v)
        ids = ids[ids > 0]
        out.append((v[None] == ids[:, None]).float())
    return out


def offset_targets(positions: torch.Tensor, instances: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """[N,3] offset from every tooth point to its tooth's centroid (0 on the gingiva)."""
    out = torch.zeros_like(positions)
    start = 0
    for n in counts:
        p, v = positions[start:start + n], instances[start:start + n]
        tooth = torch.nonzero(v > 0).squeeze(1)
        ids, slot = torch.unique(v[tooth], return_inverse=True)
        centroid = torch.zeros(ids.numel(), 3, dtype=torch.float64, device=p.device)
        centroid.index_add_(0, slot, p[tooth].double())
        centroid /= torch.bincount(slot, minlength=ids.numel())[:, None]
        out[start + tooth] = (centroid[slot] - p[tooth].double()).to(out.dtype)
        start += n
    return out


def coarse_inverse(levels: list[_Pooled], level: int) -> torch.Tensor:
    """[N] L0 point -> its point on `level`, composing each pooling's inverse."""
    inv = levels[1].inv
    for lv in levels[2:level + 1]:
        inv = lv.inv[inv]
    return inv


def coarse_targets(targets: list[torch.Tensor], inverse: torch.Tensor, counts: list[int],
                   pad_index: torch.Tensor) -> list[torch.Tensor]:
    """Pool every scan's [G,n] masks onto the cross-attention level -> [G,M] (padded).

    A coarse point belongs to a tooth when more than half of the L0 points
    pooled into it do.  Deep supervision compares each decoder layer's coarse
    mask with these.
    """
    m = int(pad_index.max()) + 1 if pad_index.numel() else 0
    out, start = [], 0
    for b, n in enumerate(counts):
        t = targets[b]
        if t.numel() == 0:
            out.append(t.new_zeros(0, pad_index.shape[1]))
            start += n
            continue
        iv = inverse[start:start + n]
        acc = t.new_zeros(t.shape[0], m).index_add_(1, iv, t)
        cnt = t.new_zeros(1, m).index_add_(1, iv, torch.ones_like(t[:1]))
        out.append(((acc / cnt.clamp_min(1))[:, pad_index[b]] > 0.5).float())
        start += n
    return out
