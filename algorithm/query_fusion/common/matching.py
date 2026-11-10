"""TTA fusion: average the poses' per-point confidences after lining up their queries.

Query k of one pose is not query k of another -- each pass hands out its 32
slots afresh -- so before averaging, every pose's queries are Hungarian-matched
to pose 0's by soft mask IoU (toothseg3d README §12.1: TTA x8 is +3 mAP).
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def query_confidence(cls_logits: torch.Tensor, mask_logits: torch.Tensor,
                     n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cls [T,K,2], mask [T*N,K] -> (conf [T,N,K] = P(point in mask) * P(tooth), P(tooth) [T,K])."""
    prob = cls_logits.float().softmax(-1)[..., 0]
    mask = mask_logits.float().sigmoid().view(cls_logits.shape[0], n_points, -1)
    return mask * prob[:, None, :], prob


def fuse_poses(conf: torch.Tensor) -> torch.Tensor:
    """conf [T,N,K] -> [N,K], each later pose aligned to pose 0 then averaged.

    Matching is against pose 0 alone, not the running mean, as the original
    does.  The 32 x 32 assignment runs in scipy -- torch has no linear
    assignment solver, and the matrix is tiny next to the IoU matmul feeding it.
    """
    base = conf[0]
    acc = base.clone()
    base_sum = base.sum(0)
    for t in range(1, conf.shape[0]):
        other = conf[t]
        inter = base.T @ other                                     # [K,K]
        union = base_sum[:, None] + other.sum(0)[None] - inter
        iou = inter / union.clamp_min(1e-6)
        rows, cols = linear_sum_assignment(-iou.cpu().numpy())
        acc[:, torch.as_tensor(rows, device=acc.device)] += other[:, torch.as_tensor(cols, device=acc.device)]
    return acc / conf.shape[0]
