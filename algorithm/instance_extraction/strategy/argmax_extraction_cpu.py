from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.extract import extract
from ..instance_extraction import InstanceExtractionInput, InstanceExtractionOutput


class ArgmaxExtractionCPU(IStrategy[InstanceExtractionInput, InstanceExtractionOutput]):
    def execute(self, inp: InstanceExtractionInput, out: InstanceExtractionOutput) -> bool:
        extract(inp, out, CPU)
        return True
