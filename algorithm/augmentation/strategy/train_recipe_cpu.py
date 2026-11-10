from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..augmentation import AugmentationInput, AugmentationOutput
from ..common.apply import apply_op
from ..common.recipe import train_recipe


class TrainRecipeCPU(IStrategy[AugmentationInput, AugmentationOutput]):
    def execute(self, inp: AugmentationInput, out: AugmentationOutput) -> bool:
        return apply_op(train_recipe, inp, out, CPU)
