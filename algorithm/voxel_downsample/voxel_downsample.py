from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class VoxelDownsampleInput:
    positions: torch.Tensor          # [V,3] float32, any device
    voxel_mm: float


@dataclass
class VoxelDownsampleOutput:
    representative: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    inverse: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    cells: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3, dtype=torch.int64))


class VoxelDownsampleType(Enum):
    GRID_CPU = 0
    GRID_GPU = 1


VoxelDownsampleStrategy = IStrategy[VoxelDownsampleInput, VoxelDownsampleOutput]


def _make_grid_cpu() -> VoxelDownsampleStrategy:
    from .strategy.voxel_grid_cpu import VoxelGridCPU
    return VoxelGridCPU()


def _make_grid_gpu() -> VoxelDownsampleStrategy:
    from .strategy.voxel_grid_gpu import VoxelGridGPU
    return VoxelGridGPU()


class VoxelDownsample(StrategyAlgorithm[VoxelDownsampleInput, VoxelDownsampleOutput, VoxelDownsampleType]):
    """Full-resolution scan -> one point per occupied voxel (the network's L0)."""

    def __init__(self) -> None:
        super().__init__({
            VoxelDownsampleType.GRID_CPU: _make_grid_cpu,
            VoxelDownsampleType.GRID_GPU: _make_grid_gpu,
        })
