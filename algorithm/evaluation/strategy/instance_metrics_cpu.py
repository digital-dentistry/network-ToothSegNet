from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.run import instance_metrics
from ..evaluation import EvaluationInput, EvaluationOutput


class InstanceMetricsCPU(IStrategy[EvaluationInput, EvaluationOutput]):
    def execute(self, inp: EvaluationInput, out: EvaluationOutput) -> bool:
        return instance_metrics(inp, out, CPU)
