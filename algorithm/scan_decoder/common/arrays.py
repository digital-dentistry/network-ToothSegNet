from __future__ import annotations

import numpy as np
import torch

from ..scan_decoder import ScanDecoderOutput


def fill_output(out: ScanDecoderOutput, positions: np.ndarray, faces: np.ndarray | None,
                colours: np.ndarray | None, normals: np.ndarray | None) -> None:
    """numpy arrays from a parser -> the decoder's tensors (zero-copy where possible)."""
    out.positions = torch.from_numpy(np.ascontiguousarray(positions, np.float32))
    out.faces = (None if faces is None or len(faces) == 0
                 else torch.from_numpy(np.ascontiguousarray(faces, np.int64)))
    out.colours = None
    if colours is not None and len(colours) == len(positions):
        c = np.ascontiguousarray(colours, np.float32)
        if c.size and float(c.max()) > 1.001:            # 0..255 stored as numbers
            c = c / 255.0
        out.colours = torch.from_numpy(c)
    out.normals = (torch.from_numpy(np.ascontiguousarray(normals, np.float32))
                   if normals is not None and len(normals) == len(positions) else None)


def merge_duplicates(positions: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weld bit-identical vertices.  STL writes three private vertices per
    triangle; welded, the area-weighted normals are smooth again, as they are on
    the meshes the model was trained on, and the in-house label files count
    welded vertices (toothseg3d README §8).

    Bit-identical means the float bits can stand in for the values: x|y packs
    into one int64 key and z into another, and two stable sorts order the rows
    -- ~10x faster than `torch.unique(dim=0)` on float rows.  Vertices come out
    in first-appearance order, so a welded file keeps its own vertex order.
    """
    bits = positions.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    xy, z = (bits[:, 0] << 32) | bits[:, 1], bits[:, 2]
    order = torch.argsort(z, stable=True)
    order = order[torch.argsort(xy[order], stable=True)]
    new = torch.ones(order.numel(), dtype=torch.bool)
    new[1:] = (xy[order][1:] != xy[order][:-1]) | (z[order][1:] != z[order][:-1])
    group = torch.cumsum(new, 0) - 1
    first = torch.full((int(group[-1]) + 1,), order.numel(), dtype=torch.int64)
    first.scatter_reduce_(0, group, order, "amin")         # earliest vertex of each group
    rank = torch.empty_like(first)
    rank[torch.argsort(first)] = torch.arange(first.numel())
    inverse = torch.empty_like(order)
    inverse[order] = rank[group]
    return positions[torch.sort(first).values], inverse[faces]
