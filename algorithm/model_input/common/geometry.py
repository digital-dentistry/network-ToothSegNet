"""The geometry tensors the network reads, and how a batch of them is packed.

Everything that depends only on *where* the points are -- the pooling
hierarchy, the curve orders, the patch layouts, the query seeds -- is computed
before the forward, so the forward is a fixed sequence of gathers and matmuls
with no device syncs (toothseg3d README §5: 28 ms -> 7.6 ms).

Points are packed: a batch is the concatenation of its scans, level by level,
with every index shifted by where its scan starts.  Each TTA pose is one scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Level:
    coord: torch.Tensor                  # [n,3] float32
    n: int
    counts: list[int]                    # points per scan, host side: no syncs
    inv: torch.Tensor | None = None      # [n of the level above] -> index here
    cnt: torch.Tensor | None = None      # [n] float32 children per point
    grid: torch.Tensor | None = None     # [n,3] int64, used while building only
    scan: torch.Tensor | None = None     # [n] which scan of the batch, building only


@dataclass
class Layout:
    """Gather / scatter indices for patch attention over one level and curve."""

    sel: torch.Tensor                    # [slots] int64: slot -> point
    inv_sel: torch.Tensor                # [n] int64: point -> its canonical slot
    rel: torch.Tensor                    # [slots,3] float32: offset from the patch centre


@dataclass
class Geometry:
    levels: list[Level]
    layouts: dict[tuple[int, int], Layout]
    pad_level: int
    seed: torch.Tensor                   # [B,K] int64 query seeds, offsets into each padded row
    pad_index: torch.Tensor = field(init=False)
    pad_valid: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.pad_index, self.pad_valid = pad_index(self.levels[self.pad_level].counts,
                                                   self.seed.device)

    @property
    def device(self) -> torch.device:
        return self.levels[0].coord.device

    @property
    def batch(self) -> int:
        return len(self.levels[0].counts)

    def to(self, device: torch.device, non_blocking: bool = True) -> Geometry:
        """Move to `device` in two transfers instead of ~50.

        One small copy per tensor is ~23 ms of pageable transfers for a batch;
        packed into one int64 and one float32 buffer and pinned, the whole
        geometry crosses the bus in two async copies (README §5: -> 0.28 ms).
        The padded index is rebuilt on the device from the host-side counts.
        """
        if device == self.device:
            return self
        ints, floats = self._tensors()
        i_buf = _pack([t for t in ints], torch.int64, device, non_blocking)
        f_buf = _pack([t for t in floats], torch.float32, device, non_blocking)
        it = iter(_unpack(i_buf, ints))
        ft = iter(_unpack(f_buf, floats))

        levels = []
        for lv in self.levels:
            levels.append(Level(coord=next(ft), n=lv.n, counts=lv.counts,
                                inv=None if lv.inv is None else next(it),
                                cnt=None if lv.cnt is None else next(ft)))
        layouts = {key: Layout(next(it), next(it), next(ft)) for key in self.layouts}
        return Geometry(levels, layouts, self.pad_level, next(it))

    def _tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """(int64 tensors, float32 tensors) in the order `to` rebuilds them."""
        ints, floats = [], []
        for lv in self.levels:
            floats.append(lv.coord)
            if lv.inv is not None:
                ints.append(lv.inv)
            if lv.cnt is not None:
                floats.append(lv.cnt)
        for lay in self.layouts.values():
            ints += [lay.sel, lay.inv_sel]
            floats.append(lay.rel)
        ints.append(self.seed)
        return ints, floats


def pad_index(counts: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense [B, Nmax] gather index + validity mask for one packed level.

    The instance decoder cross-attends a fixed set of queries to each scan's
    points, which wants a rectangular tensor.  Padding slots point at their
    scan's first point and are masked out.
    """
    c = torch.tensor(counts, dtype=torch.int64, device=device)
    starts = torch.cumsum(c, 0) - c
    width = torch.arange(max(counts, default=0), device=device)
    valid = width[None] < c[:, None]
    index = torch.where(valid, starts[:, None] + width[None], starts[:, None])
    return index, valid


def collate(scans: list[tuple[list[Level], dict[tuple[int, int], Layout]]]
            ) -> tuple[list[Level], dict[tuple[int, int], Layout]]:
    """Merge per-scan levels and layouts into one packed batch.

    Port of `geometry.collate_geometry`: points are concatenated scan by scan
    and each scan's indices shift by where that scan starts.
    """
    if len(scans) == 1:
        return scans[0]
    n_levels = len(scans[0][0])
    levels, bases = [], []
    for L in range(n_levels):
        ns = [lv[L].n for lv, _ in scans]
        base = [sum(ns[:i]) for i in range(len(ns))]
        bases.append(base)
        lv = Level(coord=torch.cat([s[0][L].coord for s in scans]), n=sum(ns), counts=ns)
        if L > 0:
            # inv holds indices *into this level*, one per point of the level
            # above, so it shifts by this level's base
            lv.inv = torch.cat([s[0][L].inv + base[i] for i, s in enumerate(scans)])
            lv.cnt = torch.cat([s[0][L].cnt for s in scans])
        levels.append(lv)

    layouts = {}
    for key in scans[0][1]:
        base = bases[key[0]]
        sels, invs, rels, slot = [], [], [], 0
        for i, (_, lays) in enumerate(scans):
            lay = lays[key]
            sels.append(lay.sel + base[i])
            invs.append(lay.inv_sel + slot)
            rels.append(lay.rel)
            slot += lay.sel.numel()
        layouts[key] = Layout(torch.cat(sels), torch.cat(invs), torch.cat(rels))
    return levels, layouts


def _pack(tensors: list[torch.Tensor], dtype: torch.dtype, device: torch.device,
          non_blocking: bool) -> torch.Tensor:
    if not tensors:
        return torch.empty(0, dtype=dtype, device=device)
    buf = torch.cat([t.reshape(-1).to(dtype) for t in tensors])
    if device.type == "cuda" and buf.device.type == "cpu":
        buf = buf.pin_memory()
    return buf.to(device, non_blocking=non_blocking)


def _unpack(buf: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    parts = torch.split(buf, [t.numel() for t in like])
    return [p.view(t.shape) for p, t in zip(parts, like)]
