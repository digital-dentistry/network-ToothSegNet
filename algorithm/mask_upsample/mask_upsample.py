from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class MaskUpsampleInput:
    confidence: torch.Tensor         # [N,K] fused per-point confidence at L0
    points: torch.Tensor             # [N,3] L0 coordinates, centred
    query: torch.Tensor              # [V,3] full-resolution vertices, same centring
    inverse: torch.Tensor            # [V] vertex -> its L0 point (voxel)
    voxel_mm: float
    score_thresh: float = 0.5
    k: int = 5


@dataclass
class MaskUpsampleOutput:
    active: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))   # [A] query ids
    confidence: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0))              # [V,A]


class MaskUpsampleType(Enum):
    VOXEL_CPU = 0               # copy the voxel's confidence: fastest, -1.5 mAP
    VOXEL_GPU = 1
    KNN_CPU = 2                 # inverse-distance kNN: the default
    KNN_GPU = 3


MaskUpsampleStrategy = IStrategy[MaskUpsampleInput, MaskUpsampleOutput]


def _make_voxel_cpu() -> MaskUpsampleStrategy:
    from .strategy.voxel_upsample_cpu import VoxelUpsampleCPU
    return VoxelUpsampleCPU()


def _make_voxel_gpu() -> MaskUpsampleStrategy:
    from .strategy.voxel_upsample_gpu import VoxelUpsampleGPU
    return VoxelUpsampleGPU()


def _make_knn_cpu() -> MaskUpsampleStrategy:
    from .strategy.knn_upsample_cpu import KnnUpsampleCPU
    return KnnUpsampleCPU()


def _make_knn_gpu() -> MaskUpsampleStrategy:
    from .strategy.knn_upsample_gpu import KnnUpsampleGPU
    return KnnUpsampleGPU()


class MaskUpsample(StrategyAlgorithm[MaskUpsampleInput, MaskUpsampleOutput, MaskUpsampleType]):
    """L0 soft masks -> full-resolution soft masks of the active queries."""

    def __init__(self) -> None:
        super().__init__({
            MaskUpsampleType.VOXEL_CPU: _make_voxel_cpu,
            MaskUpsampleType.VOXEL_GPU: _make_voxel_gpu,
            MaskUpsampleType.KNN_CPU: _make_knn_cpu,
            MaskUpsampleType.KNN_GPU: _make_knn_gpu,
        })
