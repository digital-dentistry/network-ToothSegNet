from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.grid import voxel_downsample
from ..voxel_downsample import VoxelDownsampleInput, VoxelDownsampleOutput


class VoxelGridCPU(IStrategy[VoxelDownsampleInput, VoxelDownsampleOutput]):
    def execute(self, inp: VoxelDownsampleInput, out: VoxelDownsampleOutput) -> bool:
        if inp.positions.shape[0] == 0:
            return False
        out.representative, out.inverse, out.cells = voxel_downsample(
            inp.positions.to(CPU), inp.voxel_mm)
        return True
