from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..augmentation import AugmentationInput, AugmentationOutput
from ..common.apply import apply_op
from ..common.rotation import rotate_so3


class RandomSO3CPU(IStrategy[AugmentationInput, AugmentationOutput]):
    def execute(self, inp: AugmentationInput, out: AugmentationOutput) -> bool:
        return apply_op(rotate_so3, inp, out, CPU)
