from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.run import challenge_metrics
from ..evaluation import EvaluationInput, EvaluationOutput


class ChallengeMetricsCPU(IStrategy[EvaluationInput, EvaluationOutput]):
    def execute(self, inp: EvaluationInput, out: EvaluationOutput) -> bool:
        return challenge_metrics(inp, out, CPU)
