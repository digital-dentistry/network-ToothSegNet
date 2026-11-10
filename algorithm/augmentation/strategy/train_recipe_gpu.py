from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..augmentation import AugmentationInput, AugmentationOutput
from ..common.apply import apply_op
from ..common.recipe import train_recipe


class TrainRecipeGPU(IStrategy[AugmentationInput, AugmentationOutput]):
    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: AugmentationInput, out: AugmentationOutput) -> bool:
        return apply_op(train_recipe, inp, out, self._device)
