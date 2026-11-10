from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interface.strategy import IStrategy

from ..common.arrays import fill_output
from ..scan_decoder import ScanDecoderInput, ScanDecoderOutput

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class _Element:
    name: str
    count: int
    props: list[tuple[str, str]] = field(default_factory=list)      # (name, numpy type)
    lists: list[tuple[str, str, str]] = field(default_factory=list)  # (name, count type, item type)


class PlyDecoderCPU(IStrategy[ScanDecoderInput, ScanDecoderOutput]):
    """Stanford PLY: binary (either endianness) or ascii, mesh or point cloud.

    Covers what the scanners here write -- Medit / L-SCAN binary meshes with
    colour and normals, CloudCompare ascii point clouds -- without trimesh.
    A PLY point cloud's own normals matter: on one compound scan the PCA
    fallback found zero teeth where the stored normals gave 14.
    """

    def execute(self, inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
        buf = inp.path.read_bytes()
        end = buf.find(b"end_header")
        if not buf.startswith(b"ply") or end < 0:
            return False
        body_start = buf.index(b"\n", end) + 1
        fmt, elements = _parse_header(buf[:end].decode("ascii", "replace"))
        if fmt == "ascii":
            tables = _read_ascii(buf[body_start:], elements)
        else:
            tables = _read_binary(buf, body_start, elements, "<" if fmt == "binary_little_endian" else ">")

        vert = tables.get("vertex")
        if vert is None or len(vert) == 0:
            return False
        names = vert.dtype.names
        pos = np.stack([vert["x"], vert["y"], vert["z"]], 1)
        normals = (np.stack([vert["nx"], vert["ny"], vert["nz"]], 1)
                   if all(k in names for k in ("nx", "ny", "nz")) else None)
        colours = None
        for r, g, b in (("red", "green", "blue"), ("r", "g", "b"),
                        ("diffuse_red", "diffuse_green", "diffuse_blue")):
            if all(k in names for k in (r, g, b)):
                c = np.stack([vert[r], vert[g], vert[b]], 1).astype(np.float32)
                colours = c / 255.0 if vert[r].dtype.kind in "iu" else c
                break
        fill_output(out, pos, tables.get("face"), colours, normals)
        return True


def _parse_header(text: str) -> tuple[str, list[_Element]]:
    fmt, elements = "", []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append(_Element(tok[1], int(tok[2])))
        elif tok[0] == "property" and elements:
            if tok[1] == "list":
                elements[-1].lists.append((tok[4], _TYPES[tok[2]], _TYPES[tok[3]]))
            else:
                elements[-1].props.append((tok[2], _TYPES[tok[1]]))
    return fmt, elements


def _read_binary(buf: bytes, offset: int, elements: list[_Element], endian: str) -> dict[str, np.ndarray]:
    tables: dict[str, np.ndarray] = {}
    for el in elements:
        if not el.lists:
            dt = np.dtype([(n, endian + t) for n, t in el.props])
            tables[el.name] = np.frombuffer(buf, dt, el.count, offset)
            offset += dt.itemsize * el.count
            continue
        if el.name != "face" or el.props or len(el.lists) != 1:
            break                     # an exotic element; nothing after it is needed
        _, ctype, itype = el.lists[0]
        # fast path: every face a triangle, so each record has a fixed size
        dt = np.dtype([("n", endian + ctype), ("i", endian + itype, 3)])
        rec = np.frombuffer(buf, dt, el.count, offset)
        if np.all(rec["n"] == 3):
            tables["face"] = rec["i"].astype(np.int64)
            offset += dt.itemsize * el.count
        else:
            tables["face"], offset = _polygons_binary(buf, offset, el.count, endian + ctype, endian + itype)
    return tables


def _polygons_binary(buf: bytes, offset: int, count: int, ctype: str, itype: str) -> tuple[np.ndarray, int]:
    """Mixed polygon sizes: walk the records and fan-triangulate."""
    csize, isize = np.dtype(ctype).itemsize, np.dtype(itype).itemsize
    tris = []
    for _ in range(count):
        n = int(np.frombuffer(buf, ctype, 1, offset)[0])
        offset += csize
        idx = np.frombuffer(buf, itype, n, offset).astype(np.int64)
        offset += isize * n
        tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, n - 1)]
    return np.asarray(tris, np.int64).reshape(-1, 3), offset


def _read_ascii(body: bytes, elements: list[_Element]) -> dict[str, np.ndarray]:
    lines = body.split(b"\n")
    tables: dict[str, np.ndarray] = {}
    at = 0
    for el in elements:
        rows = lines[at:at + el.count]
        at += el.count
        if not el.lists:
            vals = np.fromstring(b" ".join(rows), dtype=np.float64, sep=" ").reshape(el.count, len(el.props))
            dt = np.dtype([(n, t) for n, t in el.props])
            tab = np.empty(el.count, dt)
            for j, (n, t) in enumerate(el.props):
                tab[n] = vals[:, j].astype(t)
            tables[el.name] = tab
        elif el.name == "face" and len(el.lists) == 1 and not el.props:
            tok = [r.split() for r in rows]
            if all(len(t) == 4 for t in tok):
                # reshape: a point cloud saved with "element face 0" has no rows at all
                tables["face"] = np.array(tok, dtype=np.int64).reshape(-1, 4)[:, 1:]
            else:
                tris = []
                for t in tok:
                    idx = [int(v) for v in t[1:1 + int(t[0])]]
                    tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1)]
                tables["face"] = np.asarray(tris, np.int64).reshape(-1, 3)
    return tables
