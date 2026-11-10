from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.grid import voxel_downsample
from ..voxel_downsample import VoxelDownsampleInput, VoxelDownsampleOutput


class VoxelGridGPU(IStrategy[VoxelDownsampleInput, VoxelDownsampleOutput]):
    """Same grid on the GPU: one radix sort replaces numpy's argsort, which is
    most of the cost on a million-vertex scan."""

    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: VoxelDownsampleInput, out: VoxelDownsampleOutput) -> bool:
        if inp.positions.shape[0] == 0:
            return False
        out.representative, out.inverse, out.cells = voxel_downsample(
            inp.positions.to(self._device, non_blocking=True), inp.voxel_mm)
        return True
