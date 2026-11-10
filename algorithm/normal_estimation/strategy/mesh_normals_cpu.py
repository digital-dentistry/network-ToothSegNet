from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.normals import mesh_normals
from ..normal_estimation import NormalEstimationInput, NormalEstimationOutput


class MeshNormalsCPU(IStrategy[NormalEstimationInput, NormalEstimationOutput]):
    def execute(self, inp: NormalEstimationInput, out: NormalEstimationOutput) -> bool:
        if inp.faces is None or inp.faces.shape[0] == 0:
            return False
        out.normals = mesh_normals(inp.positions.to(CPU), inp.faces.to(CPU))
        return True
