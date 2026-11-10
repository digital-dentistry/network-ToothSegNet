from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.interpolation import active_queries
from ..mask_upsample import MaskUpsampleInput, MaskUpsampleOutput


class VoxelUpsampleCPU(IStrategy[MaskUpsampleInput, MaskUpsampleOutput]):
    def execute(self, inp: MaskUpsampleInput, out: MaskUpsampleOutput) -> bool:
        conf = inp.confidence.to(CPU)
        out.active = active_queries(conf, inp.score_thresh)
        out.confidence = conf[:, out.active][inp.inverse.to(CPU)]
        return True
