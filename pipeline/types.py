"""What travels between the pipeline stages (and between the workflow threads)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from algorithm.model_input.model_input import ModelInputOutput


@dataclass
class Scan:
    path: Path
    name: str
    jaw: str
    positions: torch.Tensor              # [V,3] float32
    normals: torch.Tensor                # [V,3] unit
    faces: torch.Tensor | None = None    # [F,3] int64
    colours: torch.Tensor | None = None  # [V,3] 0..1
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class PreparedScan:
    scan: Scan
    points: torch.Tensor                 # [N,3] L0 cloud, centred
    centre: torch.Tensor                 # [3] subtracted from the scan's coordinates
    inverse: torch.Tensor                # [V] vertex -> L0 point
    voxel_mm: float
    batches: list[ModelInputOutput]      # network inputs, up to tta_batch poses each


@dataclass
class InferredScan:
    prepared: PreparedScan
    confidence: torch.Tensor             # [N,K] fused over the poses
    scores: torch.Tensor                 # [K] P(tooth)


@dataclass
class SegmentationResult:
    path: Path
    name: str
    jaw: str = ""
    n_vertices: int = 0
    instances: np.ndarray | None = None          # [V] int64, 0 = gingiva, 1..G teeth
    scores: list[float] = field(default_factory=list)
    vertex_confidence: np.ndarray | None = None  # [V]
    timing: dict[str, float] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def n_teeth(self) -> int:
        return len(self.scores)

    @property
    def tooth_fraction(self) -> float:
        return float((self.instances > 0).mean()) if self.instances is not None and self.n_vertices else 0.0

    @staticmethod
    def failed(path: Path, error: BaseException) -> SegmentationResult:
        return SegmentationResult(path=path, name=path.stem, error=f"{type(error).__name__}: {error}")
