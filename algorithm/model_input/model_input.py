from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy

from .common.builder import CurveKey
from .common.geometry import Geometry


@dataclass(frozen=True)
class GeometrySpec:
    """The architecture facts the geometry has to agree with.  Defaults are the
    trained toothinst / toothseg3d_ios models (toothseg3d/train.py)."""

    enc_depths: tuple[int, ...] = (1, 1, 2, 4, 2)
    dec_depths: tuple[int, ...] = (1, 1, 1, 1)
    patch: int = 1024
    pad_level: int = 1               # the level the instance queries cross-attend to
    n_query: int = 32


@dataclass
class ModelInputInput:
    coord: torch.Tensor              # [N,3] float32 L0 points, centred on the origin
    normals: torch.Tensor            # [N,3] float32, unit length
    jaw_upper: bool
    colour: torch.Tensor | None      # [N,3] standardised RGB, None for a 7-channel model
    rotations: torch.Tensor          # [T,3,3] float64 TTA poses; identity for none
    voxel_mm: float
    spec: GeometrySpec = field(default_factory=GeometrySpec)
    cells: torch.Tensor | None = None   # [N,3] downsampling voxel of each point (MORTON3D)


@dataclass
class ModelInputOutput:
    features: torch.Tensor = field(default_factory=lambda: torch.empty(0, 7))   # [T*N, C]
    geometry: Geometry | None = None


class ModelInputType(Enum):
    LEGACY_CPU = 0
    LEGACY_GPU = 1
    MORTON3D_CPU = 2
    MORTON3D_GPU = 3


ModelInputStrategy = IStrategy[ModelInputInput, ModelInputOutput]


def _make_legacy_cpu() -> ModelInputStrategy:
    from .strategy.legacy_curve_cpu import LegacyCurveCPU
    return LegacyCurveCPU()


def _make_legacy_gpu() -> ModelInputStrategy:
    from .strategy.legacy_curve_gpu import LegacyCurveGPU
    return LegacyCurveGPU()


def _make_morton3d_cpu() -> ModelInputStrategy:
    from .strategy.morton3d_curve_cpu import Morton3DCurveCPU
    return Morton3DCurveCPU()


def _make_morton3d_gpu() -> ModelInputStrategy:
    from .strategy.morton3d_curve_gpu import Morton3DCurveGPU
    return Morton3DCurveGPU()


class ModelInput(StrategyAlgorithm[ModelInputInput, ModelInputOutput, ModelInputType]):
    """L0 cloud -> everything the network reads: features plus the geometry.

    The strategies differ in the curve key (LEGACY for the trained checkpoints,
    MORTON3D for the fixed hierarchy) and in where the geometry is built.
    """

    def __init__(self) -> None:
        super().__init__({
            ModelInputType.LEGACY_CPU: _make_legacy_cpu,
            ModelInputType.LEGACY_GPU: _make_legacy_gpu,
            ModelInputType.MORTON3D_CPU: _make_morton3d_cpu,
            ModelInputType.MORTON3D_GPU: _make_morton3d_gpu,
        })


def model_input_type(curve: CurveKey, gpu: bool) -> ModelInputType:
    if curve is CurveKey.LEGACY:
        return ModelInputType.LEGACY_GPU if gpu else ModelInputType.LEGACY_CPU
    return ModelInputType.MORTON3D_GPU if gpu else ModelInputType.MORTON3D_CPU
