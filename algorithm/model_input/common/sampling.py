from __future__ import annotations

import torch


def farthest_point_seeds(coord: torch.Tensor, index: torch.Tensor, valid: torch.Tensor,
                         k: int) -> torch.Tensor:
    """k farthest-point seeds per scan, for the whole packed batch at once.

    Port of `geometry.farthest_point_sample`, which seeds the instance queries
    at real, spread-out points (Mask3D's non-parametric queries).  The original
    loops over scans; here every scan runs in the same k passes over the padded
    [B, M] view, with padding at -inf so it is never picked.

    The squared distance is summed x, y, z in that order in float32, as the
    original's np.sum over three elements is, so the argmax ties break the same.
    Returns [B,k] offsets into each scan's own padded row.
    """
    pts = coord[index]                                     # [B,M,3]
    b_n, m = index.shape
    rows = torch.arange(b_n, device=coord.device)
    seeds = torch.empty(b_n, k, dtype=torch.int64, device=coord.device)
    if m == 0 or k == 0:
        return seeds.zero_()
    seeds[:, 0] = valid.sum(1) // 2
    dist = _sq_dist(pts, pts[rows, seeds[:, 0]])
    dist.masked_fill_(~valid, -torch.inf)
    for i in range(1, k):
        seeds[:, i] = dist.argmax(1)                       # first maximum, as numpy
        dist = torch.minimum(dist, _sq_dist(pts, pts[rows, seeds[:, i]]))
    return seeds


def _sq_dist(pts: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    d = pts - centre[:, None, :]
    return d[..., 0].square() + d[..., 1].square() + d[..., 2].square()
