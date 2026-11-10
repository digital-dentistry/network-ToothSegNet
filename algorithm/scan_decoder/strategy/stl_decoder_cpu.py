from __future__ import annotations

import re

import numpy as np
import torch

from interface.strategy import IStrategy

from ..common.arrays import fill_output, merge_duplicates
from ..scan_decoder import ScanDecoderInput, ScanDecoderOutput

_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)")


class StlDecoderCPU(IStrategy[ScanDecoderInput, ScanDecoderOutput]):
    """STL, binary or ascii.  No colour, no shared vertices -- they are welded
    unless `merge_vertices` is off, which keeps the file's own vertex count
    (what toothseg3d's trimesh(process=False) path produced)."""

    def execute(self, inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
        buf = inp.path.read_bytes()
        tri = _binary(buf)
        if tri is None:
            tri = _ascii(buf)
        if tri is None or len(tri) == 0:
            return False
        positions = torch.from_numpy(np.ascontiguousarray(tri.reshape(-1, 3), np.float32))
        faces = torch.arange(positions.shape[0], dtype=torch.int64).view(-1, 3)
        if inp.merge_vertices:
            positions, faces = merge_duplicates(positions, faces)
        fill_output(out, positions.numpy(), faces.numpy(), None, None)
        return True


def _binary(buf: bytes) -> np.ndarray | None:
    """[F,3,3] float32, or None when the size does not add up (an ascii file --
    some binary headers also begin with 'solid', so the size decides)."""
    if len(buf) < 84:
        return None
    count = int(np.frombuffer(buf, "<u4", 1, 80)[0])
    if len(buf) != 84 + 50 * count:
        return None
    rec = np.frombuffer(buf, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count, 84)
    return rec["v"]


def _ascii(buf: bytes) -> np.ndarray | None:
    if not buf.lstrip().startswith(b"solid"):
        return None
    v = np.array(_VERTEX.findall(buf), dtype=np.float32)
    return v.reshape(-1, 3, 3) if len(v) % 3 == 0 else None
