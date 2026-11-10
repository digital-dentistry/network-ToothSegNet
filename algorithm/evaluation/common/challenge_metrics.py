"""3DTeethSeg'22 challenge metrics TSA / TLA / TIR (port of `evaluate.jaw_metrics`).

Kept quirks of the challenge's evaluation.py, for comparability with the
leaderboard: a tooth's "size" is sqrt(sum over its vertices of the squared
deviation), per axis -- it grows with the vertex count -- and an unmatched
tooth costs TLA an un-normalised 5 * |size|.  The centroid error in mm and the
detection counts are reported beside them as something readable.
"""
from __future__ import annotations

import math

import torch

from .instance_metrics import tooth_dice


def _per_instance(ins: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (ids [I], slot of every tooth vertex [T], centroids [I,3] float64)."""
    tooth = ins > 0
    ids, slot = torch.unique(ins[tooth], return_inverse=True)
    cen = torch.zeros(ids.numel(), 3, dtype=torch.float64, device=positions.device)
    cen.index_add_(0, slot, positions[tooth].double())
    cen /= torch.bincount(slot, minlength=ids.numel())[:, None]
    return ids, slot, cen


def jaw_metrics(positions: torch.Tensor, gt_classes: torch.Tensor, gt_instances: torch.Tensor,
                pred_instances: torch.Tensor, pred_classes: torch.Tensor | None) -> dict[str, float]:
    """One jaw.  `pred_classes[i - 1]` is the label of predicted instance i; None
    (an identity-free model) reports TIR and label accuracy as NaN."""
    tsa = tooth_dice(pred_instances != 0, gt_instances != 0)

    g_ids, g_slot, g_cen = _per_instance(gt_instances, positions)
    gt_tooth = gt_instances > 0
    dev2 = (positions[gt_tooth].double() - g_cen[g_slot]).square()
    g_size = torch.zeros_like(g_cen).index_add_(0, g_slot, dev2).sqrt()            # [G,3]
    # majority label per tooth; ties go to the smaller label, as np.argmax does
    n_lab = int(gt_classes.max()) + 1
    votes = torch.bincount(g_slot * n_lab + gt_classes[gt_tooth], minlength=g_ids.numel() * n_lab)
    g_lab = votes.view(g_ids.numel(), n_lab).argmax(1)

    p_ids, _, p_cen = _per_instance(pred_instances, positions)
    p_lab = None
    if pred_classes is not None:
        # an instance id past the end of pred_classes has label 0, as the
        # original's `pr_label_of.get(i, 0)`
        lut = torch.zeros(int(p_ids.max()) + 1 if p_ids.numel() else 1, dtype=torch.int64, device=p_ids.device)
        k = min(pred_classes.numel(), lut.numel() - 1)
        lut[1:k + 1] = pred_classes[:k].to(lut)
        p_lab = lut[p_ids]

    n_gt, n_pred = g_ids.numel(), p_ids.numel()
    unknown = math.nan if p_lab is None else 0.0          # identity metrics without labels
    base = {"tsa": float(tsa), "centroid_mm": 0.0, "label_acc": unknown, "missed": 0, "extra": 0,
            "n_gt": n_gt, "n_pred": n_pred}
    if n_gt == 0:
        return {**base, "tla": 0.0, "tir": 1.0 if p_lab is not None else unknown, "extra": n_pred}
    if n_pred == 0:
        tla = float((5 * torch.linalg.vector_norm(g_size, dim=1)).mean())
        return {**base, "tla": tla, "tir": unknown, "missed": n_gt}

    from scipy.optimize import linear_sum_assignment
    dist = torch.cdist(g_cen, p_cen).cpu().numpy()
    r, c = linear_sum_assignment(dist)
    g_cen, p_cen, g_size, g_lab = g_cen.cpu(), p_cen.cpu(), g_size.cpu(), g_lab.cpu()
    p_lab = None if p_lab is None else p_lab.cpu()
    matched = dict(zip(r.tolist(), c.tolist()))

    tla, tir, mm, lab_ok = 0.0, 0, 0.0, 0
    for g in range(n_gt):
        if g not in matched:
            tla += 5 * float(torch.linalg.vector_norm(g_size[g]))   # the challenge's miss penalty
            continue
        p = matched[g]
        d = float(torch.linalg.vector_norm((g_cen[g] - p_cen[p]) / g_size[g]))
        tla += d
        mm += float(torch.linalg.vector_norm(g_cen[g] - p_cen[p]))
        if p_lab is not None and int(g_lab[g]) == int(p_lab[p]):
            lab_ok += 1
            tir += d < 0.5
    n_match = len(matched)
    return {**base, "tla": tla / n_gt, "centroid_mm": mm / max(n_match, 1),
            "tir": unknown if p_lab is None else tir / max(n_match, 1),
            "label_acc": unknown if p_lab is None else lab_ok / max(n_match, 1),
            "missed": n_gt - n_match, "extra": max(0, n_pred - n_match)}
