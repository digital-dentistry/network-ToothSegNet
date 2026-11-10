"""Random draws from one CPU `torch.Generator`.

Scalars are always drawn on the host, so a GPU strategy draws the same crop,
reach and rotation as the CPU one for the same seed.  Per-vertex noise is drawn
on the vertices' device from a generator seeded off the host one.  `None`
draws from torch's global RNG instead.
"""
from __future__ import annotations

import torch

from common.device import CPU


def uniform(gen: torch.Generator | None, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand((), generator=gen, dtype=torch.float64))


def randint(gen: torch.Generator | None, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi)."""
    return int(torch.randint(lo, hi, (), generator=gen))


def chance(gen: torch.Generator | None, p: float) -> bool:
    return p > 0 and uniform(gen, 0.0, 1.0) < p


def choice(gen: torch.Generator | None, weights: tuple[float, ...]) -> int:
    w = torch.tensor(weights, dtype=torch.float64)
    return int(torch.multinomial(w, 1, generator=gen))


def device_generator(gen: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    if gen is None or device.type == "cpu":
        return gen
    out = torch.Generator(device)
    out.manual_seed(int(torch.randint(0, 2 ** 62, (), generator=gen)))
    return out


def normal(gen: torch.Generator | None, shape: tuple[int, ...], std: float,
           device: torch.device = CPU, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(shape, generator=device_generator(gen, device), device=device, dtype=dtype) * std
