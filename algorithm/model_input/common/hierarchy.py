"""Voxel hierarchies: L0 is the downsampled cloud, each next level doubles the voxel."""
from __future__ import annotations

import torch

from .geometry import Level
from .morton import legacy_key

# The legacy key fits in 34 bits; the scan (TTA pose) index goes above it, so
# one sort orders a whole batch scan by scan, then by key -- the packed order.
SCAN_SHIFT = 40


def legacy_hierarchy(coords: list[torch.Tensor], grid_mm: float, n_levels: int) -> list[Level]:
    """Port of `geometry.build_hierarchy` + `collate_geometry`, for a whole batch at once.

    Every scan of the batch (every TTA pose) is grouped by the same rule, and
    prefixing the key with the scan index makes a single `unique` per level do
    what the original does with one `unique` per scan and a collate after.
    Index for index the same as the original, including two details that
    decide bit-exactness:

    * the grid is re-derived from the centred coordinates (per scan), not taken
      from the downsampling;
    * a group's grid cell is its *last* member's (`cg[inv] = g` in numpy is
      last-write-wins).  With the colliding legacy key the members do not
      share a cell, and taking any other member diverges from level 2 on (the
      C++ port hit this).
    """
    dev = coords[0].device
    counts = [c.shape[0] for c in coords]
    grid = torch.cat([g - g.amin(0) for g in (torch.floor(c / grid_mm).long() for c in coords)])
    scan = torch.repeat_interleave(torch.arange(len(coords), device=dev),
                                   torch.tensor(counts, device=dev))
    levels = [Level(coord=torch.cat(coords), n=sum(counts), counts=counts, grid=grid, scan=scan)]
    for _ in range(n_levels - 1):
        fine = levels[-1]
        g = fine.grid >> 1
        key = (fine.scan << SCAN_SHIFT) | legacy_key(g)
        uniq, inv, cnt = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
        m = cnt.numel()
        last = torch.full((m,), -1, dtype=torch.int64, device=dev)
        last.scatter_reduce_(0, inv, torch.arange(inv.numel(), device=dev), "amax")
        scan = uniq >> SCAN_SHIFT
        levels.append(Level(coord=_centroid(fine.coord, inv, cnt), n=m,
                            counts=torch.bincount(scan, minlength=len(coords)).tolist(),
                            inv=inv, cnt=cnt.float(), grid=g[last], scan=scan))
    return levels


def morton3d_hierarchy(coord: torch.Tensor, grid0: torch.Tensor, n_levels: int,
                       key: torch.Tensor) -> tuple[list[Level], torch.Tensor]:
    """Port of `make_input.build_levels`: every level from one sort.

    With a real Morton key a parent cell's key is its child's key >> 3, so once
    L0 is sorted by key every coarser level's groups are contiguous runs of
    that same order -- no unique per level.  L0 keeps its input order; each
    coarser level is numbered in Morton order.  Returns the levels and the L0
    Morton order, which is also L0's variant-0 curve order.
    """
    n0 = coord.shape[0]
    order0 = torch.argsort(key, stable=True)
    ks = key[order0]
    ps = coord[order0].double()
    dev = coord.device
    levels = [Level(coord=coord, n=n0, counts=[n0], grid=grid0)]
    prev_starts = None                   # first sorted L0 index of each group one level up
    for L in range(1, n_levels):
        k = ks >> (3 * L)
        first = torch.ones(n0, dtype=torch.bool, device=dev)
        first[1:] = k[1:] != k[:-1]
        starts = torch.nonzero(first).squeeze(1)
        gid = torch.cumsum(first, 0) - 1           # sorted L0 point -> its group here
        if L == 1:
            inv = torch.empty(n0, dtype=torch.int64, device=dev)
            inv[order0] = gid
        else:
            inv = gid[prev_starts]                 # group one level up -> its parent here
        m = starts.numel()
        size = torch.diff(starts, append=torch.tensor([n0], device=dev))
        # centroid of every L0 point in the voxel, not the mean of the child
        # centroids -- a child holding one point should not weigh like eight
        cen = torch.zeros(m, 3, dtype=torch.float64, device=dev).index_add_(0, gid, ps)
        cen = (cen / size[:, None]).float()
        cnt = torch.bincount(inv, minlength=m).float()
        levels.append(Level(coord=cen, n=m, counts=[m], inv=inv, cnt=cnt,
                            grid=grid0[order0[starts]] >> L))
        prev_starts = starts
    return levels, order0


def _centroid(coord: torch.Tensor, inv: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """Mean of each group, accumulated in float64 as the original's np.add.at is."""
    cen = torch.zeros(cnt.numel(), 3, dtype=torch.float64, device=coord.device)
    cen.index_add_(0, inv, coord.double())
    return (cen / cnt[:, None]).float()
