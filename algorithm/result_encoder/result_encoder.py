from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class ResultEncoderInput:
    path: Path                       # file to write
    name: str
    jaw: str
    positions: torch.Tensor          # [V,3]
    instances: torch.Tensor          # [V] int64, 0 = gingiva
    faces: torch.Tensor | None = None
    colours: np.ndarray | None = None    # [V,3] uint8: PLY colours in place of the instance palette


@dataclass
class ResultEncoderOutput:
    written: Path | None = None


class ResultEncoderType(Enum):
    PLY_CPU = 0              # the scan coloured by instance, faces kept
    JSON_CPU = 1             # Teeth3DS label format: per-vertex instances + tooth flag


ResultEncoderStrategy = IStrategy[ResultEncoderInput, ResultEncoderOutput]


def _make_ply() -> ResultEncoderStrategy:
    from .strategy.ply_encoder_cpu import PlyEncoderCPU
    return PlyEncoderCPU()


def _make_json() -> ResultEncoderStrategy:
    from .strategy.json_encoder_cpu import JsonEncoderCPU
    return JsonEncoderCPU()


class ResultEncoder(StrategyAlgorithm[ResultEncoderInput, ResultEncoderOutput, ResultEncoderType]):
    def __init__(self) -> None:
        super().__init__({
            ResultEncoderType.PLY_CPU: _make_ply,
            ResultEncoderType.JSON_CPU: _make_json,
        })
