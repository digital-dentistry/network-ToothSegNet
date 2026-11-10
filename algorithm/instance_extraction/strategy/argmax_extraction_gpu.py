from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.extract import extract
from ..instance_extraction import InstanceExtractionInput, InstanceExtractionOutput


class ArgmaxExtractionGPU(IStrategy[InstanceExtractionInput, InstanceExtractionOutput]):
    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: InstanceExtractionInput, out: InstanceExtractionOutput) -> bool:
        extract(inp, out, self._device)
        return True
