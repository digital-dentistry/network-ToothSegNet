from __future__ import annotations

import numpy as np

from interface.strategy import IStrategy

from ..common.arrays import fill_output
from ..scan_decoder import ScanDecoderInput, ScanDecoderOutput


class ArrayDecoderCPU(IStrategy[ScanDecoderInput, ScanDecoderOutput]):
    """Bare point arrays: columns x y z [r g b] (.npy, or whitespace / comma text)."""

    def execute(self, inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
        ext = inp.path.suffix.lower()
        if ext == ".npy":
            a = np.load(inp.path)
        else:
            a = np.loadtxt(inp.path, delimiter="," if ext == ".csv" else None, ndmin=2)
        if a.ndim != 2 or a.shape[1] < 3 or len(a) == 0:
            return False
        fill_output(out, a[:, :3], None, a[:, 3:6] if a.shape[1] >= 6 else None, None)
        return True
