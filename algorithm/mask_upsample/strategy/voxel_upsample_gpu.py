from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.interpolation import active_queries
from ..mask_upsample import MaskUpsampleInput, MaskUpsampleOutput


class VoxelUpsampleGPU(IStrategy[MaskUpsampleInput, MaskUpsampleOutput]):
    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: MaskUpsampleInput, out: MaskUpsampleOutput) -> bool:
        conf = inp.confidence.to(self._device)
        out.active = active_queries(conf, inp.score_thresh)
        out.confidence = conf[:, out.active][inp.inverse.to(self._device)]
        return True
