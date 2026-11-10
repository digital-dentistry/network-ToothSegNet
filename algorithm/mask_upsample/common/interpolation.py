"""Soft masks at the downsampled resolution -> the full mesh, before any decision.

Deciding at 250 um and copying each voxel's label to all its vertices freezes a
boundary voxel's whole population onto one tooth; interpolating the soft masks
first and deciding per vertex is +1.5 mAP (toothseg3d README §11.4).
"""
from __future__ import annotations

import torch

from common.spatial import knn_search


def active_queries(conf: torch.Tensor, score_thresh: float) -> torch.Tensor:
    """Queries that clear the threshold somewhere.  The rest can never win the
    argmax, and they are most of what the interpolation would cost."""
    return torch.nonzero(conf.amax(0) > score_thresh).squeeze(1)


def knn_interpolate(conf: torch.Tensor, points: torch.Tensor, query: torch.Tensor,
                    k: int, voxel_mm: float) -> torch.Tensor:
    """Inverse-distance-weighted mean of the k nearest points' confidences.

    conf [N,A], points [N,3], query [V,3] -> [V,A].  A 2-voxel cell holds ~8
    downsampled points, so most vertices find their 5 neighbours in one pass.
    """
    dist, nb = knn_search(query, points, k, cell=2.0 * voxel_mm)
    w = 1.0 / dist.clamp_min(1e-6)                 # a missing neighbour (inf) weighs 0
    w = w / w.sum(1, keepdim=True)
    nb = nb.clamp_min(0)
    # one neighbour at a time: never materialises the [V, k, A] gather
    full = w[:, :1] * conf[nb[:, 0]]
    for j in range(1, k):
        full.addcmul_(w[:, j:j + 1], conf[nb[:, j]])
    return full
