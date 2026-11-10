from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.run import challenge_metrics
from ..evaluation import EvaluationInput, EvaluationOutput


class ChallengeMetricsGPU(IStrategy[EvaluationInput, EvaluationOutput]):
    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: EvaluationInput, out: EvaluationOutput) -> bool:
        return challenge_metrics(inp, out, self._device)
