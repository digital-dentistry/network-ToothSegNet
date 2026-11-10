"""Grid downsampling: one representative vertex per occupied voxel.

Port of `toothseg3d/t3ds_io.voxel_downsample`.  The voxel groups and their
order are integer arithmetic and match the original exactly.

Which member represents a voxel is "nearest to the voxel centroid", and that
is mostly a float tie-break: in a two-member voxel the centroid is the exact
midpoint, so rounding alone picks the member.  The original rounds in float32,
so this does too -- centroid, difference, sqrt of x^2 + y^2 + z^2 in that
order -- and picks the lowest vertex index among equals, which is its stable
sort's first hit.  On Teeth3DS scans 99.98 % of voxels pick the same vertex on
the CPU and 100 % on the GPU (float64 here would give 88 %; the C++ port, 98.9 %).
"""
from __future__ import annotations

import torch


def voxel_downsample(positions: torch.Tensor,
                     voxel_mm: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """positions [V,3] float32 -> (representative [N], inverse [V], cells [N,3]).

    `representative` is ordered by ascending voxel key, as the original is.
    `inverse` maps every vertex to its voxel, which pushes results back onto
    the full mesh.  `cells` are the voxels' integer grid coordinates.
    """
    n_vert = positions.shape[0]
    dev = positions.device
    cell = torch.floor((positions - positions.amin(0)) / voxel_mm).long()
    dims = cell.amax(0) + 1
    key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    _, inverse, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
    n_vox = counts.numel()

    centroid = torch.zeros(n_vox, 3, dtype=positions.dtype, device=dev)
    centroid.index_add_(0, inverse, positions).div_(counts[:, None].to(positions.dtype))
    d = positions - centroid[inverse]
    dist = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]).sqrt()
    nearest = torch.full((n_vox,), torch.inf, dtype=dist.dtype, device=dev)
    nearest.scatter_reduce_(0, inverse, dist, "amin")
    candidate = torch.where(dist == nearest[inverse], torch.arange(n_vert, device=dev), n_vert)
    representative = torch.full((n_vox,), n_vert, dtype=torch.int64, device=dev)
    representative.scatter_reduce_(0, inverse, candidate, "amin")
    return representative, inverse, cell[representative]
