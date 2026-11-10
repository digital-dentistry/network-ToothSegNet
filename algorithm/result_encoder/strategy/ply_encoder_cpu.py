from __future__ import annotations

import numpy as np

from interface.strategy import IStrategy

from ..common.palette import instance_colours
from ..result_encoder import ResultEncoderInput, ResultEncoderOutput


class PlyEncoderCPU(IStrategy[ResultEncoderInput, ResultEncoderOutput]):
    """Binary PLY coloured by instance.  Keeps the faces when the input had
    them, so the result opens as a coloured mesh rather than a cloud of dots."""

    def execute(self, inp: ResultEncoderInput, out: ResultEncoderOutput) -> bool:
        pos = inp.positions.cpu().numpy()
        rgb = inp.colours if inp.colours is not None else instance_colours(inp.instances.cpu().numpy())
        faces = None if inp.faces is None else inp.faces.cpu().numpy()
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pos)}",
                  "property float x", "property float y", "property float z",
                  "property uchar red", "property uchar green", "property uchar blue"]
        if faces is not None and len(faces):
            header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
        header.append("end_header\n")

        vert = np.empty(len(pos), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
        vert["x"], vert["y"], vert["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        vert["r"], vert["g"], vert["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        inp.path.parent.mkdir(parents=True, exist_ok=True)
        with open(inp.path, "wb") as fh:
            fh.write("\n".join(header).encode())
            fh.write(vert.tobytes())
            if faces is not None and len(faces):
                rec = np.empty(len(faces), [("n", "u1"), ("i", "<i4", 3)])
                rec["n"], rec["i"] = 3, faces
                fh.write(rec.tobytes())
        out.written = inp.path
        return True
