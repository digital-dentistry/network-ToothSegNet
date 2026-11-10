from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.run import semantic_metrics
from ..evaluation import EvaluationInput, EvaluationOutput


class SemanticMetricsCPU(IStrategy[EvaluationInput, EvaluationOutput]):
    def execute(self, inp: EvaluationInput, out: EvaluationOutput) -> bool:
        return semantic_metrics(inp, out, CPU)
