from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class InstanceExtractionInput:
    confidence: torch.Tensor         # [V,A] full-resolution soft masks of the active queries
    active: torch.Tensor             # [A] query ids
    query_scores: torch.Tensor       # [K] P(tooth) per query
    positions: torch.Tensor          # [V,3] for erosion
    score_thresh: float = 0.5
    min_frac: float = 0.005
    erode_mm: float = 0.0
    mask_confidence: bool = True     # score = P(tooth) * mean mask confidence


@dataclass
class InstanceExtractionOutput:
    # [V] 0 = gingiva, 1..G teeth
    instances: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    scores: torch.Tensor = field(default_factory=lambda: torch.empty(0))                        # [G]
    vertex_confidence: torch.Tensor = field(default_factory=lambda: torch.empty(0))             # [V]


class InstanceExtractionType(Enum):
    ARGMAX_CPU = 0
    ARGMAX_GPU = 1


InstanceExtractionStrategy = IStrategy[InstanceExtractionInput, InstanceExtractionOutput]


def _make_argmax_cpu() -> InstanceExtractionStrategy:
    from .strategy.argmax_extraction_cpu import ArgmaxExtractionCPU
    return ArgmaxExtractionCPU()


def _make_argmax_gpu() -> InstanceExtractionStrategy:
    from .strategy.argmax_extraction_gpu import ArgmaxExtractionGPU
    return ArgmaxExtractionGPU()


class InstanceExtraction(StrategyAlgorithm[InstanceExtractionInput, InstanceExtractionOutput,
                                           InstanceExtractionType]):
    """Soft masks -> instance ids, instance scores, per-vertex confidence."""

    def __init__(self) -> None:
        super().__init__({
            InstanceExtractionType.ARGMAX_CPU: _make_argmax_cpu,
            InstanceExtractionType.ARGMAX_GPU: _make_argmax_gpu,
        })
