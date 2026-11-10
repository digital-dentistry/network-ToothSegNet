from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class ScanDecoderInput:
    path: Path
    merge_vertices: bool = True      # STL stores three private vertices per triangle


@dataclass
class ScanDecoderOutput:
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))    # [V,3] float32 mm
    faces: torch.Tensor | None = None                                             # [F,3] int64
    colours: torch.Tensor | None = None                                           # [V,3] float32 in 0..1
    normals: torch.Tensor | None = None                                           # [V,3] float32, as stored


class ScanDecoderType(Enum):
    OBJ_CPU = 0
    PLY_CPU = 1
    STL_CPU = 2
    ARRAY_CPU = 3            # .npy / .xyz / .txt / .csv point arrays


EXTENSIONS: dict[str, ScanDecoderType] = {
    ".obj": ScanDecoderType.OBJ_CPU,
    ".ply": ScanDecoderType.PLY_CPU,
    ".stl": ScanDecoderType.STL_CPU,
    ".npy": ScanDecoderType.ARRAY_CPU,
    ".xyz": ScanDecoderType.ARRAY_CPU,
    ".txt": ScanDecoderType.ARRAY_CPU,
    ".csv": ScanDecoderType.ARRAY_CPU,
}

ScanDecoderStrategy = IStrategy[ScanDecoderInput, ScanDecoderOutput]


def _make_obj() -> ScanDecoderStrategy:
    from .strategy.obj_decoder_cpu import ObjDecoderCPU
    return ObjDecoderCPU()


def _make_ply() -> ScanDecoderStrategy:
    from .strategy.ply_decoder_cpu import PlyDecoderCPU
    return PlyDecoderCPU()


def _make_stl() -> ScanDecoderStrategy:
    from .strategy.stl_decoder_cpu import StlDecoderCPU
    return StlDecoderCPU()


def _make_array() -> ScanDecoderStrategy:
    from .strategy.array_decoder_cpu import ArrayDecoderCPU
    return ArrayDecoderCPU()


class ScanDecoder(StrategyAlgorithm[ScanDecoderInput, ScanDecoderOutput, ScanDecoderType]):
    """Scan file -> vertices (+ faces, colours, normals when the file has them).

    Decoding is file parsing, so there are CPU strategies only; the tensors it
    returns are host tensors that the next stage moves where it needs them.
    """

    def __init__(self) -> None:
        super().__init__({
            ScanDecoderType.OBJ_CPU: _make_obj,
            ScanDecoderType.PLY_CPU: _make_ply,
            ScanDecoderType.STL_CPU: _make_stl,
            ScanDecoderType.ARRAY_CPU: _make_array,
        })

    def set_strategy_for(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext not in EXTENSIONS:
            raise ValueError(f"no decoder for {ext!r} ({path.name})")
        self.set_strategy(EXTENSIONS[ext])
