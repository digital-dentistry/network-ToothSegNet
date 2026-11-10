"""Space-filling-curve keys, as int64 tensor bit operations.

Two keys, because the trained checkpoints and the correct key disagree:

* `legacy_key` is `toothseg3d/geometry.morton` exactly.  Its bit spreader
  (`0x5555...` mask) is the 2D one, so x's bit i lands on z's bit i+1 and
  distinct cells share a key.  Every existing checkpoint was trained on the
  orders this produces, so it is reproduced, not fixed (cpp/README.md: "재현할
  것, 개선하지 말 것").
* `morton3d_key` is a real 3D interleave (`algorithm/make_input/make_input.py`):
  one key per cell, and a parent cell's key is its child's key >> 3.  It needs
  a retrained model to pay off.
"""
from __future__ import annotations

import torch

# (axis a, axis b, axis c, mirror a): the curve variants blocks alternate
# between so that patch boundaries move.  Must match model._ORDERS.
CURVE_VARIANTS: tuple[tuple[int, int, int, bool], ...] = (
    (0, 1, 2, False), (1, 2, 0, False), (2, 0, 1, False), (0, 1, 2, True))


def _spread_legacy(x: torch.Tensor) -> torch.Tensor:
    x = x & 0xFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def _spread3(x: torch.Tensor) -> torch.Tensor:
    """Low 21 bits of x, two places apart: bit i -> bit 3i."""
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def legacy_key(grid: torch.Tensor) -> torch.Tensor:
    return (_spread_legacy(grid[:, 0]) << 2) | (_spread_legacy(grid[:, 1]) << 1) | _spread_legacy(grid[:, 2])


def morton3d_key(grid: torch.Tensor) -> torch.Tensor:
    return (_spread3(grid[:, 0]) << 2) | (_spread3(grid[:, 1]) << 1) | _spread3(grid[:, 2])


def variant_grid(grid: torch.Tensor, variant: int) -> torch.Tensor:
    """The axis permutation / mirror that turns the base curve into `variant`."""
    a, b, c, flip = CURVE_VARIANTS[variant % len(CURVE_VARIANTS)]
    g = grid[:, [a, b, c]]
    if flip:
        g[:, 0] = 65535 - g[:, 0]
    return g
