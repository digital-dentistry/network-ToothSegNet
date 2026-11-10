from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from algorithm.model_input.common.builder import CurveKey
from common.device import gpu_available


class Backend(Enum):
    AUTO = "auto"           # GPU where it pays off, if there is one
    CPU = "cpu"
    GPU = "gpu"


class UpsampleMode(Enum):
    KNN = "knn"             # interpolate soft masks onto the mesh (+1.5 mAP)
    VOXEL = "voxel"         # copy each voxel's masks to its vertices


@dataclass(frozen=True)
class SegmentationConfig:
    """Defaults follow toothseg3d/batch_infer.py."""

    checkpoint: Path | None = None       # None: random weights, for shapes and timing only
    backend: Backend = Backend.AUTO
    voxel_mm: float | None = None        # None: the checkpoint's training resolution
    tta: int = 4                         # poses averaged; x8 is the accuracy plateau (README §12.1)
    tta_batch: int = 8                   # poses per forward
    score_thresh: float = 0.5
    upsample: UpsampleMode = UpsampleMode.KNN
    knn: int = 5
    min_frac: float = 0.005              # drop instances under this fraction of the scan
    erode_mm: float = 0.0                # 0 for the in-house fine-tuned models (README §20)
    mask_confidence: bool = True
    curve: CurveKey | None = None        # None: the curve the checkpoint was trained on
    jaw: str | None = None               # None: guess from the file name
    merge_stl_vertices: bool = True
    geometry_on_gpu: bool | None = None  # None: on the GPU when tta > 1 (see Placement)
    timing: bool = True


@dataclass(frozen=True)
class Placement:
    """Which device each algorithm runs on.

    Everything that scales with the full-resolution vertex count (normals,
    downsampling, upsampling, extraction) goes to the GPU when there is one.
    The geometry is sync-bound on the GPU -- every `unique` waits for the
    device -- so it costs ~8 ms there whatever the pose count, against ~5 ms
    on the host plus ~1.5 ms per pose (TTA x8: 9 ms vs 18 ms).  Without TTA
    the host wins, and its work overlaps the previous scan's forward in the
    threaded pipeline.
    """

    normals: bool
    downsample: bool
    geometry: bool
    network: bool
    fusion: bool
    upsample: bool
    extraction: bool

    @staticmethod
    def resolve(config: SegmentationConfig) -> Placement:
        if config.backend is Backend.GPU and not gpu_available():
            raise RuntimeError("backend=gpu but CUDA is not available")
        gpu = config.backend is Backend.GPU or (config.backend is Backend.AUTO and gpu_available())
        geometry = config.geometry_on_gpu if config.geometry_on_gpu is not None else config.tta > 1
        return Placement(normals=gpu, downsample=gpu, geometry=gpu and geometry,
                         network=gpu, fusion=gpu, upsample=gpu, extraction=gpu)
