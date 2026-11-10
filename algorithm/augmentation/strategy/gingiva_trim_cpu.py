from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..augmentation import AugmentationInput, AugmentationOutput
from ..common.apply import apply_op
from ..common.crop import gingiva_trim


class GingivaTrimCPU(IStrategy[AugmentationInput, AugmentationOutput]):
    def execute(self, inp: AugmentationInput, out: AugmentationOutput) -> bool:
        return apply_op(gingiva_trim, inp, out, CPU)
