from __future__ import annotations

import numpy as np

from interface.strategy import IStrategy

from ..common.arrays import fill_output
from ..scan_decoder import ScanDecoderInput, ScanDecoderOutput


class ObjDecoderCPU(IStrategy[ScanDecoderInput, ScanDecoderOutput]):
    """Wavefront OBJ, port of `t3ds_io.read_obj`.

    Teeth3DS files are machine-written and perfectly regular -- 'v x y z r g b'
    lines then 'f i j k' -- so each block is joined once and parsed by numpy's
    C tokenizer (`fromstring` with a separator: half the time of splitting into
    Python bytes first).  'f v/vt/vn' keeps the vertex index; 'vn' / 'vt'
    lines are ignored.
    """

    def execute(self, inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
        buf = inp.path.read_bytes()
        k = buf.find(b"\nf ")
        vb, fb = (buf, b"") if k < 0 else (buf[:k], buf[k + 1:])

        vlines = [ln for ln in vb.split(b"\n") if ln[:2] == b"v "]
        if not vlines:
            return False
        ncol = len(vlines[0].split())
        flat = np.fromstring(b" ".join(ln[2:] for ln in vlines), dtype=np.float32, sep=" ")
        flat = flat.reshape(len(vlines), ncol - 1)
        colours = flat[:, 3:6] if flat.shape[1] >= 6 else None

        faces = None
        if fb:
            flines = [ln for ln in fb.split(b"\n") if ln[:2] == b"f "]
            body = b" ".join(ln[2:] for ln in flines)
            if b"/" in body:
                body = b" ".join(t.split(b"/", 1)[0] for t in body.split())
            faces = np.fromstring(body, dtype=np.int64, sep=" ").reshape(len(flines), -1)[:, :3] - 1
        fill_output(out, flat[:, :3], faces, colours, None)
        return True
