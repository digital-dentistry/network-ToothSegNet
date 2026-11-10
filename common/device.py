from __future__ import annotations

import time

import torch

CPU = torch.device("cpu")


def cuda_device() -> torch.device:
    """The device every GPU strategy runs on; fails early and clearly without CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("a GPU strategy was selected but torch.cuda.is_available() is False")
    return torch.device("cuda", torch.cuda.current_device())


def gpu_available() -> bool:
    return torch.cuda.is_available()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    """Wall-clock per named step.  Syncs the device first, so GPU work is counted
    where it was launched rather than wherever the next sync happens to fall."""

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self._device = device
        self._enabled = enabled
        self._t = time.perf_counter()
        self.ms: dict[str, float] = {}

    def mark(self, name: str) -> None:
        if not self._enabled:
            return
        synchronize(self._device)
        now = time.perf_counter()
        self.ms[name] = self.ms.get(name, 0.0) + (now - self._t) * 1000.0
        self._t = now
