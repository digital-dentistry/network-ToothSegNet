from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from algorithm.model_input.common.geometry import Geometry
from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy
from layer.checkpoint import LoadedModel


@dataclass
class InstanceNetworkInput:
    features: torch.Tensor           # [M, in_ch], packed over the poses
    geometry: Geometry


@dataclass
class InstanceNetworkOutput:
    cls_logits: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0, 2))   # [B,K,2] float32
    mask_logits: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0))     # [M,K] float32


class InstanceNetworkType(Enum):
    TORCH_CPU = 0
    TORCH_GPU = 1


InstanceNetworkStrategy = IStrategy[InstanceNetworkInput, InstanceNetworkOutput]


class InstanceNetwork(StrategyAlgorithm[InstanceNetworkInput, InstanceNetworkOutput, InstanceNetworkType]):
    """The mask-transformer forward: per-pose query scores and per-point mask logits."""

    def __init__(self, model: LoadedModel) -> None:
        def make_cpu() -> InstanceNetworkStrategy:
            from .strategy.torch_net_cpu import TorchNetCPU
            return TorchNetCPU(model)

        def make_gpu() -> InstanceNetworkStrategy:
            from .strategy.torch_net_gpu import TorchNetGPU
            return TorchNetGPU(model)

        super().__init__({
            InstanceNetworkType.TORCH_CPU: make_cpu,
            InstanceNetworkType.TORCH_GPU: make_gpu,
        })
