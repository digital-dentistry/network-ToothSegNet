"""Partial-scan crops and the gingiva trim (port of `Teeth3DS._cut`,
`_crop_segment` and `_trim_gingiva`)."""
from __future__ import annotations

import math

import torch

from .distance import nearest_distance
from .params import CROP_MODES, AugmentationParams
from .rng import choice, randint, uniform
from .sample import AugmentationSample

MIN_PICKED_POINTS = 200      # fewer and the picked teeth are too small to crop around


# ---------------------------------------------------------------------- wedge
def wedge_mask(positions: torch.Tensor, keep_frac: float, start_angle: float) -> torch.Tensor:
    """Keep a contiguous angular wedge of the arch about its xy centre.

    Angles are taken in xy, where Teeth3DS puts the occlusal plane, so a crop
    has to run before the scan is rotated.
    """
    c = positions[:, :2] - positions[:, :2].mean(0)
    a = torch.atan2(c[:, 1], c[:, 0])
    d = torch.remainder(a - start_angle + math.pi, 2 * math.pi) - math.pi
    return d.abs() <= math.pi * keep_frac


def wedge_crop(sample: AugmentationSample, params: AugmentationParams,
               gen: torch.Generator | None) -> AugmentationSample | None:
    keep_frac = uniform(gen, *params.wedge_keep)
    m = wedge_mask(sample.positions, keep_frac, uniform(gen, -math.pi, math.pi))
    return sample.select(m) if int(m.sum()) > params.crop_min_points else None


# -------------------------------------------------------------------- segment
def tooth_centroids(positions: torch.Tensor, instances: torch.Tensor,
                    ids: torch.Tensor) -> torch.Tensor:
    """[G,3] float64 centroid of every tooth in `ids` (sorted)."""
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    acc = torch.zeros(ids.numel(), 3, dtype=torch.float64, device=positions.device)
    acc.index_add_(0, slot, positions[tooth].double())
    return acc / torch.bincount(slot, minlength=ids.numel())[:, None]


def arch_order(centroids: torch.Tensor) -> list[int]:
    """Walk the arch from one end to the other, nearest neighbour to nearest neighbour.

    Sorting by angle about the arch centre looks simpler but puts atan2's branch
    cut somewhere along the arch, which splits a run in two; chaining from the
    far end makes no assumption about the frame.
    """
    d = torch.cdist(centroids, centroids).cpu()
    n = d.shape[0]
    first = int(torch.argmax(d)) // n                    # one end of the longest span
    dl = d.tolist()
    order, left = [first], set(range(n)) - {first}
    while left:
        last = order[-1]
        nxt = min(left, key=lambda j: dl[last][j])
        order.append(nxt)
        left.discard(nxt)
    return order


def segment_mask(positions: torch.Tensor, instances: torch.Tensor, mode: str,
                 params: AugmentationParams, gen: torch.Generator | None) -> torch.Tensor | None:
    """Keep a contiguous run of teeth plus the gingiva around them, or None.

    A wedge of the arch is not what a partial scan looks like.  A bridge prep is
    two to five neighbouring teeth and a collar of gingiva; a quadrant is a
    quarter to a half of the arch.  Teeth are picked along the arch and a
    margin is grown outward from them.
    """
    ids = torch.unique(instances)
    ids = ids[ids > 0]
    n = ids.numel()
    if n < 3:
        return None
    order = arch_order(tooth_centroids(positions, instances, ids))
    if mode == "bridge":
        k = randint(gen, 2, min(6, n + 1))
    elif mode == "quadrant":
        k = max(2, round(n * uniform(gen, 0.25, 0.5)))
    else:                                                # half
        k = max(3, round(n * uniform(gen, 0.45, 0.6)))
    # the arch is an open U, not a ring: wrapping past the last molar would
    # jump across the mouth opening and leave two disconnected pieces
    k = min(k, n)
    start = randint(gen, 0, n - k + 1)
    pick = ids[torch.tensor(order[start:start + k], device=ids.device)]

    sel = torch.isin(instances, pick)
    if int(sel.sum()) < MIN_PICKED_POINTS:
        return None
    margin = uniform(gen, *params.crop_margin_mm[mode])
    keep = nearest_distance(positions, positions[sel]) < margin

    # a tooth is either in the scan or out of it -- no slivers to label.  The
    # picked teeth are set back to whole by the `| sel` below.
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    kept = torch.zeros(n, dtype=torch.float64, device=positions.device)
    kept.index_add_(0, slot, keep[tooth].double())
    whole = kept / torch.bincount(slot, minlength=n) > 0.5
    keep[tooth] = whole[slot]
    return keep | sel


def segment_crop(sample: AugmentationSample, params: AugmentationParams,
                 gen: torch.Generator | None) -> AugmentationSample | None:
    mode = params.crop_mode or CROP_MODES[choice(gen, params.crop_mix)]
    m = segment_mask(sample.positions, sample.instances, mode, params, gen)
    if m is None or int(m.sum()) <= params.crop_min_points:
        return None
    return sample.select(m)


# ----------------------------------------------------------------------- trim
def trim_mask(positions: torch.Tensor, instances: torch.Tensor, reach_mm: float) -> torch.Tensor | None:
    """Drop gingiva farther than `reach_mm` from the nearest tooth, or None.

    Teeth3DS keeps the palate and a broad gingival skirt -- its gingiva reaches
    11.9 mm past the teeth (p95).  The in-house scanner trims to 4.6 mm.
    Without this the model never sees a scan shaped like the ones it runs on.
    """
    tooth = instances != 0
    if not bool(tooth.any()) or bool(tooth.all()):
        return None
    gum = ~tooth
    keep = torch.ones_like(tooth)
    keep[gum] = nearest_distance(positions[gum], positions[tooth]) <= reach_mm
    return keep


def gingiva_trim(sample: AugmentationSample, params: AugmentationParams,
                 gen: torch.Generator | None) -> AugmentationSample | None:
    m = trim_mask(sample.positions, sample.instances, uniform(gen, *params.trim_reach_mm))
    if m is None or int(m.sum()) <= params.trim_min_points:
        return None
    return sample.select(m)
