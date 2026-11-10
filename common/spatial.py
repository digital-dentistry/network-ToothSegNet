"""Exact k-nearest-neighbour search in plain torch ops, on any device.

torch has no spatial index, and a brute-force `cdist` against a whole scan is
quadratic (1 M vertices x 45 k points is 45 G distances).  A uniform grid fixes
that: hash the reference points into cells of size `cell`, and gather each
query's candidates from the 3 x 3 x 3 block of cells around it.

The block reaches at least one cell in every direction, so every reference
point closer than `cell` is in it.  A query whose k-th candidate is within
`cell` therefore has its exact answer.  The rest -- edges of the scan, small
islands -- are retried with a cell four times as large, and whatever is still
left after that falls back to brute force.  The result matches a KD-tree
up to the order of exactly tied distances.

Two things make it fast enough to stand in for scipy's cKDTree (93 k queries
into 42 k points, k = 5: ~35 ms on the CPU where cKDTree takes 16 ms, ~4 ms on
the GPU):

* Candidate lists are built once per occupied *query cell*, not per query --
  every query in a cell shares the same 27-cell block, and there are ~7x fewer
  cells than queries.
* Queries are processed sorted by how many candidates they have, so a chunk is
  only as wide as its widest row instead of the widest row in the scan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_OFFSETS = torch.stack(torch.meshgrid(*(torch.arange(-1, 2),) * 3, indexing="ij"), -1).reshape(-1, 3)

_QUERY_CHUNK = {"cpu": 1 << 14, "cuda": 1 << 16}   # queries per distance / top-k chunk
_CELL_BUDGET = 1 << 22          # [query cells x candidates] per candidate-list block
_BRUTE_BUDGET = 1 << 22         # [queries x reference points] per brute-force chunk
_MAX_RETRIES = 2
_GROWTH = 4.0                   # cell growth per retry: 0.5 mm -> 2 mm -> 8 mm


def knn_search(query: torch.Tensor, ref: torch.Tensor, k: int, cell: float,
               max_radius: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest `ref` points of every `query` point.

    query [Q,3], ref [R,3], same device, float32.  Returns (dist [Q,k], idx
    [Q,k] int64), nearest first.  Slots with no neighbour -- fewer than k
    reference points, or nothing within `max_radius` -- get dist = inf and
    idx = -1.  With `max_radius`, neighbours beyond it are dropped and no query
    is retried past it, so a radius test is a single grid pass when
    `cell >= max_radius`.
    """
    q_n = query.shape[0]
    dist = torch.full((q_n, k), math.inf, dtype=query.dtype, device=query.device)
    idx = torch.full((q_n, k), -1, dtype=torch.int64, device=query.device)
    if q_n == 0 or ref.shape[0] == 0 or k <= 0:
        return dist, idx

    origin = torch.minimum(query.amin(0), ref.amin(0))
    todo = torch.arange(q_n, device=query.device)
    for _ in range(_MAX_RETRIES + 1):
        d2, nb, exact = _CellGrid.build(ref, cell, origin).query(query[todo], k)
        if max_radius is not None and cell >= max_radius:
            exact = torch.ones_like(exact)          # anything nearer is in the block
        dist[todo[exact]] = d2[exact]
        idx[todo[exact]] = nb[exact]
        todo = todo[~exact]
        if todo.numel() == 0:
            break
        cell *= _GROWTH
    if todo.numel():
        dist[todo], idx[todo] = _brute_force(query[todo], ref, k)

    dist = dist.sqrt()
    if max_radius is not None:
        far = dist > max_radius
        dist = dist.masked_fill(far, math.inf)
        idx = idx.masked_fill(far, -1)
    return dist, idx


@dataclass
class _CellGrid:
    """Reference points bucketed by cell: sorted cell keys plus each cell's run."""

    ref: torch.Tensor
    cell: float
    origin: torch.Tensor
    dims: torch.Tensor
    order: torch.Tensor          # sorted position -> reference index
    occupied: torch.Tensor       # sorted unique cell keys
    counts: torch.Tensor         # points per occupied cell
    starts: torch.Tensor         # first sorted position of each occupied cell

    @staticmethod
    def build(ref: torch.Tensor, cell: float, origin: torch.Tensor) -> _CellGrid:
        # +1 so the block's -1 offset never leaves the grid, and +2 on the far
        # side for its +1 offset.  Queries are clamped into the same box.
        ref_cell = _cells(ref, origin, cell)
        dims = ref_cell.amax(0) + 2
        key = _linear(ref_cell, dims)
        order = torch.argsort(key)
        occupied, counts = torch.unique_consecutive(key[order], return_counts=True)
        return _CellGrid(ref, cell, origin, dims, order, occupied, counts,
                         torch.cumsum(counts, 0) - counts)

    def query(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Squared distances and indices from the 27-cell block, plus which are exact."""
        dev = query.device
        q_n, r_n = query.shape[0], self.ref.shape[0]
        # A query outside the reference box is clamped to the nearest edge cell.
        # Every reference point within `cell` of it is still inside that cell's
        # block, so the exactness test below stays valid.
        qc = torch.minimum(_cells(query, self.origin, self.cell), self.dims - 2).clamp_min(1)
        ukey, qinv = torch.unique(_linear(qc, self.dims), return_inverse=True)
        cand, valid, total = self._block_candidates(ukey)          # per query cell

        d2_out = torch.full((q_n, k), math.inf, dtype=query.dtype, device=dev)
        nb_out = torch.full((q_n, k), -1, dtype=torch.int64, device=dev)
        q_total = total[qinv]
        q_order = torch.argsort(q_total)
        widths = q_total[q_order].cpu()
        rx, ry, rz = (self.ref[:, i].contiguous() for i in range(3))
        chunk = _QUERY_CHUNK.get(dev.type, _QUERY_CHUNK["cuda"])
        for lo in range(0, q_n, chunk):
            hi = min(lo + chunk, q_n)
            w = int(widths[hi - 1])                 # sorted: the chunk's widest row
            if w == 0:
                continue
            sel = q_order[lo:hi]
            cell_of = qinv[sel]
            c, v, q = cand[cell_of, :w], valid[cell_of, :w], query[sel]
            # per axis, in place: a [n, w, 3] difference tensor costs 2x the time
            d2 = (q[:, 0:1] - rx[c]).square_()
            d2 += (q[:, 1:2] - ry[c]).square_()
            d2 += (q[:, 2:3] - rz[c]).square_()
            d2.masked_fill_(~v, math.inf)
            kk = min(k, w)
            top, j = torch.topk(d2, kk, dim=1, largest=False, sorted=True)
            d2_out[sel, :kk] = top
            nb_out[sel, :kk] = torch.where(torch.isinf(top), -1, c.gather(1, j))

        exact = (q_total >= min(k, r_n)) & (
            (d2_out[:, -1] <= self.cell * self.cell) | (q_total >= r_n))
        return d2_out, nb_out, exact

    def _block_candidates(self, ukey: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference indices in the 27-cell block of every query cell, as a padded matrix."""
        dev = ukey.device
        d1, d2 = self.dims[1], self.dims[2]
        uc = torch.stack([ukey // (d1 * d2), (ukey // d2) % d1, ukey % d2], 1)
        nb_key = _linear(uc[:, None, :] + _OFFSETS.to(dev)[None], self.dims)          # [U,27]
        pos = torch.searchsorted(self.occupied, nb_key).clamp_max(self.occupied.numel() - 1)
        cnt = torch.where(self.occupied[pos] == nb_key, self.counts[pos], 0)
        first = self.starts[pos]
        total = cnt.sum(1)
        u_n, w = ukey.numel(), max(int(total.max()), 1)

        cand = torch.zeros((u_n, w), dtype=torch.int64, device=dev)
        valid = torch.zeros((u_n, w), dtype=torch.bool, device=dev)
        slot_all = torch.arange(w, device=dev)
        step = max(1, _CELL_BUDGET // w)
        for lo in range(0, u_n, step):
            hi = min(lo + step, u_n)
            c = cnt[lo:hi]
            # slot j of a row is filled from the cell whose running count first
            # exceeds j, at offset j - (points in the cells before it)
            cum = torch.cumsum(c, 1)
            slot = slot_all.expand(hi - lo, w).contiguous()
            which = torch.searchsorted(cum, slot, right=True).clamp_max(26)
            within = slot - (cum - c).gather(1, which)
            v = slot < total[lo:hi, None]
            sorted_pos = (first[lo:hi].gather(1, which) + within).masked_fill_(~v, 0)
            cand[lo:hi] = self.order[sorted_pos]
            valid[lo:hi] = v
        return cand, valid, total


def _brute_force(query: torch.Tensor, ref: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    kk = min(k, ref.shape[0])
    d2_out = torch.full((query.shape[0], k), math.inf, dtype=query.dtype, device=query.device)
    nb_out = torch.full((query.shape[0], k), -1, dtype=torch.int64, device=query.device)
    step = max(1, _BRUTE_BUDGET // ref.shape[0])
    for lo in range(0, query.shape[0], step):
        d2 = (query[lo:lo + step, None, :] - ref[None]).square().sum(-1)
        top, j = torch.topk(d2, kk, dim=1, largest=False, sorted=True)
        d2_out[lo:lo + step, :kk], nb_out[lo:lo + step, :kk] = top, j
    return d2_out, nb_out


def _cells(points: torch.Tensor, origin: torch.Tensor, cell: float) -> torch.Tensor:
    return torch.floor((points - origin) / cell).long() + 1


def _linear(cell: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    return (cell[..., 0] * dims[1] + cell[..., 1]) * dims[2] + cell[..., 2]
