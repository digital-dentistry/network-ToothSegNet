from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class QueryFusionInput:
    cls_logits: torch.Tensor         # [T,K,2], one row per pose
    mask_logits: torch.Tensor        # [T*N,K], pose by pose
    n_points: int                    # N, the same cloud in every pose


@dataclass
class QueryFusionOutput:
    confidence: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0))   # [N,K]
    scores: torch.Tensor = field(default_factory=lambda: torch.empty(0))          # [K] P(tooth)


class QueryFusionType(Enum):
    HUNGARIAN_CPU = 0
    HUNGARIAN_GPU = 1


QueryFusionStrategy = IStrategy[QueryFusionInput, QueryFusionOutput]


def _make_hungarian_cpu() -> QueryFusionStrategy:
    from .strategy.hungarian_fusion_cpu import HungarianFusionCPU
    return HungarianFusionCPU()


def _make_hungarian_gpu() -> QueryFusionStrategy:
    from .strategy.hungarian_fusion_gpu import HungarianFusionGPU
    return HungarianFusionGPU()


class QueryFusion(StrategyAlgorithm[QueryFusionInput, QueryFusionOutput, QueryFusionType]):
    """Network logits for T poses -> one per-point confidence per query."""

    def __init__(self) -> None:
        super().__init__({
            QueryFusionType.HUNGARIAN_CPU: _make_hungarian_cpu,
            QueryFusionType.HUNGARIAN_GPU: _make_hungarian_gpu,
        })
