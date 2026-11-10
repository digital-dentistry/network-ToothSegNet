from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.interpolation import active_queries, knn_interpolate
from ..mask_upsample import MaskUpsampleInput, MaskUpsampleOutput


class KnnUpsampleCPU(IStrategy[MaskUpsampleInput, MaskUpsampleOutput]):
    """Grid-hash kNN in torch on the host (~2x scipy cKDTree at 93 k vertices)."""

    def execute(self, inp: MaskUpsampleInput, out: MaskUpsampleOutput) -> bool:
        conf = inp.confidence.to(CPU)
        out.active = active_queries(conf, inp.score_thresh)
        if out.active.numel() == 0:
            out.confidence = conf.new_zeros(inp.query.shape[0], 0)
            return True
        out.confidence = knn_interpolate(conf[:, out.active], inp.points.to(CPU),
                                         inp.query.to(CPU), inp.k, inp.voxel_mm)
        return True
