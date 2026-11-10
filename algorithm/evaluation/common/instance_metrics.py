"""Per-scan instance metrics from two label maps (port of `inst_metrics.iou_matrix`,
`masks_from_labels` and `APMeter.add`'s per-scan summary).

Masks are never materialised: one joint histogram of (predicted id, true id)
gives every intersection at once.
"""
from __future__ import annotations

import torch


def instance_iou(pred: torch.Tensor, gt: torch.Tensor, min_pred_points: int,
                 min_gt_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IoU of every predicted instance against every true one -> (iou [P,G] float64,
    pred ids [P], gt ids [G]), ids ascending, id 0 and instances under the
    point floor left out."""
    pid, pinv, pcnt = torch.unique(pred, return_inverse=True, return_counts=True)
    gid, ginv, gcnt = torch.unique(gt, return_inverse=True, return_counts=True)
    joint = torch.bincount(pinv * gid.numel() + ginv, minlength=pid.numel() * gid.numel())
    joint = joint.view(pid.numel(), gid.numel())
    kp = (pid > 0) & (pcnt >= min_pred_points)
    kg = (gid > 0) & (gcnt >= min_gt_points)
    inter = joint[kp][:, kg].double()
    union = pcnt[kp][:, None] + gcnt[kg][None] - inter
    return inter / union.clamp_min(1), pid[kp], gid[kg]


def tooth_dice(pred_tooth: torch.Tensor, gt_tooth: torch.Tensor) -> float:
    """TSA: micro F1 of the binary tooth / gingiva split, over vertices."""
    tp = int((pred_tooth & gt_tooth).sum())
    return 2 * tp / max(int(pred_tooth.sum()) + int(gt_tooth.sum()), 1)


def matched_summary(iou: torch.Tensor) -> dict[str, float]:
    """Hungarian match on IoU, then the mean IoU of the matched pairs."""
    n_pred, n_gt = iou.shape
    if n_pred == 0 or n_gt == 0:
        return {"mean_iou": 0.0, "matched50": 0, "n_pred": n_pred, "n_gt": n_gt}
    from scipy.optimize import linear_sum_assignment
    m = iou.cpu().numpy()
    r, c = linear_sum_assignment(-m)
    matched = m[r, c]
    return {"mean_iou": float(matched.mean()), "matched50": int((matched >= 0.5).sum()),
            "n_pred": n_pred, "n_gt": n_gt}
