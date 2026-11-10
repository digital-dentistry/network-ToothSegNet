from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.normals import pca_normals
from ..normal_estimation import NormalEstimationInput, NormalEstimationOutput


class PcaNormalsCPU(IStrategy[NormalEstimationInput, NormalEstimationOutput]):
    def execute(self, inp: NormalEstimationInput, out: NormalEstimationOutput) -> bool:
        if inp.positions.shape[0] < 3:
            return False
        out.normals = pca_normals(inp.positions.to(CPU), inp.k)
        return True
