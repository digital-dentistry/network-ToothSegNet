from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.builder import CurveKey, build_model_input
from ..model_input import ModelInputInput, ModelInputOutput


class LegacyCurveCPU(IStrategy[ModelInputInput, ModelInputOutput]):
    """Legacy key, built on the host.  The geometry is a few dozen sorts and
    gathers over ~45 k points, which the CPU does without a device sync; the
    network strategy uploads it in two packed transfers."""

    def execute(self, inp: ModelInputInput, out: ModelInputOutput) -> bool:
        if inp.coord.shape[0] == 0:
            return False
        s = inp.spec
        out.features, out.geometry = build_model_input(
            inp.coord, inp.normals, inp.jaw_upper, inp.colour, inp.rotations, inp.cells,
            inp.voxel_mm, s.enc_depths, s.dec_depths, s.patch, s.pad_level, s.n_query,
            CurveKey.LEGACY, CPU)
        return True
