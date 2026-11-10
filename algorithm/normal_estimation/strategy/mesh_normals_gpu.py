from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.normals import mesh_normals
from ..normal_estimation import NormalEstimationInput, NormalEstimationOutput


class MeshNormalsGPU(IStrategy[NormalEstimationInput, NormalEstimationOutput]):
    """Face normals scattered with atomics: same sums, summed in any order."""

    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: NormalEstimationInput, out: NormalEstimationOutput) -> bool:
        if inp.faces is None or inp.faces.shape[0] == 0:
            return False
        out.normals = mesh_normals(inp.positions.to(self._device), inp.faces.to(self._device))
        return True
