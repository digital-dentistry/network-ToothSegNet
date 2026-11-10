from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.builder import CurveKey, build_model_input
from ..model_input import ModelInputInput, ModelInputOutput


class LegacyCurveGPU(IStrategy[ModelInputInput, ModelInputOutput]):
    """Legacy key, built where the network runs.  Saves the upload; every
    `unique` is a host sync, so on small clouds this is not faster than the CPU."""

    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: ModelInputInput, out: ModelInputOutput) -> bool:
        if inp.coord.shape[0] == 0:
            return False
        s = inp.spec
        out.features, out.geometry = build_model_input(
            inp.coord, inp.normals, inp.jaw_upper, inp.colour, inp.rotations, inp.cells,
            inp.voxel_mm, s.enc_depths, s.dec_depths, s.patch, s.pad_level, s.n_query,
            CurveKey.LEGACY, self._device)
        return True
