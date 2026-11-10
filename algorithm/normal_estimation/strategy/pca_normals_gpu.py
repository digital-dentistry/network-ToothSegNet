from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.normals import pca_normals
from ..normal_estimation import NormalEstimationInput, NormalEstimationOutput


class PcaNormalsGPU(IStrategy[NormalEstimationInput, NormalEstimationOutput]):
    """kNN and the batched 3x3 eigh on the device; the spanning tree on the host."""

    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: NormalEstimationInput, out: NormalEstimationOutput) -> bool:
        if inp.positions.shape[0] < 3:
            return False
        out.normals = pca_normals(inp.positions.to(self._device), inp.k)
        return True
