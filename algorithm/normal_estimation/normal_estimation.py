from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class NormalEstimationInput:
    positions: torch.Tensor                  # [V,3] float32
    faces: torch.Tensor | None = None        # [F,3] int64; the MESH strategies need it
    k: int = 24                              # PCA neighbourhood size


@dataclass
class NormalEstimationOutput:
    normals: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))   # [V,3] unit


class NormalEstimationType(Enum):
    MESH_CPU = 0            # area-weighted from faces -- use whenever there are faces
    MESH_GPU = 1
    PCA_CPU = 2             # local PCA + MST sign propagation, for bare point clouds
    PCA_GPU = 3


NormalEstimationStrategy = IStrategy[NormalEstimationInput, NormalEstimationOutput]


def _make_mesh_cpu() -> NormalEstimationStrategy:
    from .strategy.mesh_normals_cpu import MeshNormalsCPU
    return MeshNormalsCPU()


def _make_mesh_gpu() -> NormalEstimationStrategy:
    from .strategy.mesh_normals_gpu import MeshNormalsGPU
    return MeshNormalsGPU()


def _make_pca_cpu() -> NormalEstimationStrategy:
    from .strategy.pca_normals_cpu import PcaNormalsCPU
    return PcaNormalsCPU()


def _make_pca_gpu() -> NormalEstimationStrategy:
    from .strategy.pca_normals_gpu import PcaNormalsGPU
    return PcaNormalsGPU()


class NormalEstimation(StrategyAlgorithm[NormalEstimationInput, NormalEstimationOutput,
                                         NormalEstimationType]):
    """Vertex normals for a scan that does not carry its own.

    Prefer, in order: the file's own normals, the faces, and only then PCA --
    on one compound point cloud the PCA guess got the orientation wrong badly
    enough that the model found zero teeth, where the scanner's normals gave 14.
    """

    def __init__(self) -> None:
        super().__init__({
            NormalEstimationType.MESH_CPU: _make_mesh_cpu,
            NormalEstimationType.MESH_GPU: _make_mesh_gpu,
            NormalEstimationType.PCA_CPU: _make_pca_cpu,
            NormalEstimationType.PCA_GPU: _make_pca_gpu,
        })
