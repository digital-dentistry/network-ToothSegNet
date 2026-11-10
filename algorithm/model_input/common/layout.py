"""Patch layouts: which points attend together, for each (level, curve) a block uses."""
from __future__ import annotations

from typing import Callable

import torch

from .geometry import Layout, Level
from .hierarchy import SCAN_SHIFT
from .morton import CURVE_VARIANTS, variant_grid

KeyFn = Callable[[torch.Tensor], torch.Tensor]


def block_plan(enc_depths: tuple[int, ...], dec_depths: tuple[int, ...]) -> list[tuple[int, int]]:
    """(level, curve variant) of every block, in the model's stage order."""
    plan, n_stage, n_var = [], len(enc_depths), len(CURVE_VARIANTS)
    for i, d in enumerate(enc_depths):
        for j in range(d):
            plan.append((i, (i + j) % n_var))
    for i, d in enumerate(dec_depths):
        lvl = n_stage - 2 - i
        for j in range(d):
            plan.append((lvl, (i + 1 + j) % n_var))
    return plan


def curve_order(grid: torch.Tensor, variant: int, key_fn: KeyFn,
                scan: torch.Tensor | None = None) -> torch.Tensor:
    """Points sorted along curve `variant`, scan by scan when `scan` is given.

    Stable: with the legacy key many points share a key, and the tie order is
    part of what the checkpoints were trained on.  The scan prefix keeps every
    scan's points in its own block, each in the order a lone sort would give.
    """
    key = key_fn(variant_grid(grid, variant))
    if scan is not None:
        key = (scan << SCAN_SHIFT) | key
    return torch.argsort(key, stable=True)


def slots(order: torch.Tensor, counts: list[int], patch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`sel` gathers each scan's points into ceil(n/patch)*patch slots laid out
    as whole patches; `inv_sel` gathers each point back from its canonical slot.

    The overhang slides the last patch back over real points so no patch is
    mostly padding; a scan smaller than one patch wraps instead.  `order` lists
    the scans' points scan by scan, as `curve_order` returns them; the result
    is what the original's per-scan layout plus collate produce.
    """
    dev = order.device
    n = torch.tensor(counts, dtype=torch.int64, device=dev)
    npad = (n + patch - 1) // patch * patch
    base, slot_base = torch.cumsum(n, 0) - n, torch.cumsum(npad, 0) - npad
    scans = torch.arange(len(counts), device=dev)
    slot_scan = torch.repeat_interleave(scans, npad, output_size=sum(-(-c // patch) * patch for c in counts))
    j = torch.arange(slot_scan.numel(), device=dev) - slot_base[slot_scan]
    nt, pt = n[slot_scan], npad[slot_scan]
    local = torch.where(nt >= patch, torch.where(j < nt, j, j - (pt - nt)), j % nt.clamp_min(1))
    sel = order[base[slot_scan] + local]

    point_scan = torch.repeat_interleave(scans, n, output_size=order.numel())
    inv_sel = torch.empty_like(order)
    inv_sel[order] = slot_base[point_scan] + torch.arange(order.numel(), device=dev) - base[point_scan]
    return sel, inv_sel


def patch_rel_coord(coord: torch.Tensor, sel: torch.Tensor, patch: int) -> torch.Tensor:
    """Per-slot coordinate relative to its patch centre (the CPE input)."""
    c = coord[sel].view(-1, patch, 3)
    return (c - c.mean(1, keepdim=True)).view(-1, 3)


def build_layouts(levels: list[Level], plan: list[tuple[int, int]], patch: int,
                  order_fn: Callable[[int, int], torch.Tensor]) -> dict[tuple[int, int], Layout]:
    """One layout per distinct (level, variant) in the plan."""
    layouts: dict[tuple[int, int], Layout] = {}
    for lvl, var in plan:
        if (lvl, var) in layouts:
            continue
        sel, inv_sel = slots(order_fn(lvl, var), levels[lvl].counts, patch)
        layouts[(lvl, var)] = Layout(sel, inv_sel, patch_rel_coord(levels[lvl].coord, sel, patch))
    return layouts
