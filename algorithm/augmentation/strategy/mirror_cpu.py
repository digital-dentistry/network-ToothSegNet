from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..augmentation import AugmentationInput, AugmentationOutput
from ..common.apply import apply_op
from ..common.mirror import mirror


class MirrorCPU(IStrategy[AugmentationInput, AugmentationOutput]):
    def execute(self, inp: AugmentationInput, out: AugmentationOutput) -> bool:
        return apply_op(mirror, inp, out, CPU)
