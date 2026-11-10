"""Full-resolution soft masks -> one instance id per vertex.

Port of the tail of `infer.segment_inst`: argmax over the active queries,
optional erosion, speck removal, renumbering, and the mask-confidence score.
The original's per-instance Python loop is a bincount / index_add here.
"""
from __future__ import annotations

import torch

from common.spatial import knn_search

MIN_INSTANCE_POINTS = 30


def assign(confidence: torch.Tensor, active: torch.Tensor, n_vertex: int,
           score_thresh: float) -> torch.Tensor:
    """Each vertex goes to its most confident active query (id + 1), or 0 if none clears."""
    ins = torch.zeros(n_vertex, dtype=torch.int64, device=confidence.device)
    if active.numel():
        best, arg = confidence.max(1)               # first maximum, as numpy's argmax
        claimed = best > score_thresh
        ins[claimed] = active[arg[claimed]] + 1
    return ins


def erode(ins: torch.Tensor, positions: torch.Tensor, erode_mm: float) -> torch.Tensor:
    """Pull every tooth back from the gingiva by `erode_mm`.

    Annotation conventions differ on exactly where a crown stops; this matched
    the in-house labels to Teeth3DS without retraining.  The models fine-tuned
    on in-house labels no longer need it (README §20: 78.7 -> 70.0 with it).
    """
    gum = torch.nonzero(ins == 0).squeeze(1)
    tooth = torch.nonzero(ins > 0).squeeze(1)
    if erode_mm <= 0 or gum.numel() == 0 or tooth.numel() == 0:
        return ins
    dist, _ = knn_search(positions[tooth], positions[gum], 1, cell=erode_mm, max_radius=erode_mm)
    out = ins.clone()
    out[tooth[dist[:, 0] < erode_mm]] = 0
    return out


def renumber(ins: torch.Tensor, n_query: int, min_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop specks and renumber the survivors 1..G in query order.

    A real tooth is 2-15 % of a scan's vertices; out-of-domain scans grow a few
    sub-1 % slivers.  Returns (instances [V], kept query ids [G]).
    """
    floor = max(MIN_INSTANCE_POINTS, int(min_frac * ins.numel()))
    sizes = torch.bincount(ins, minlength=n_query + 1)
    keep = sizes >= floor
    keep[0] = False
    remap = torch.cumsum(keep, 0) * keep
    return remap[ins], torch.nonzero(keep).squeeze(1) - 1


def instance_scores(ins: torch.Tensor, kept: torch.Tensor, query_scores: torch.Tensor,
                    confidence: torch.Tensor, active: torch.Tensor, mask_confidence: bool) -> torch.Tensor:
    """Score per kept instance: P(tooth), times the mean confidence inside its mask.

    AP ranks predictions by score, so the score should say how good the *mask*
    is, not just whether a tooth was found (Mask Scoring R-CNN; +1.0 mAP).
    `ins` is already renumbered, so instance g belongs to query kept[g-1].
    """
    scores = query_scores[kept].float()
    if not mask_confidence or active.numel() == 0 or kept.numel() == 0:
        return scores
    column = torch.full((int(query_scores.numel()),), -1, dtype=torch.int64, device=ins.device)
    column[active] = torch.arange(active.numel(), device=ins.device)
    tooth = torch.nonzero(ins > 0).squeeze(1)
    g = ins[tooth] - 1
    vals = confidence[tooth, column[kept[g]]]
    total = torch.zeros(kept.numel(), dtype=torch.float64, device=ins.device).index_add_(0, g, vals.double())
    count = torch.bincount(g, minlength=kept.numel())
    return scores * (total / count).float()
