"""Latency of toothseg3d's `infer.segment_inst` against this package, stage by stage.

    python tools/benchmark.py --n 10 --tta 8

Random weights on both sides (the forward costs the same); file loading and
normals are excluded, as in toothseg3d README §7.2.  The original's "network"
includes building every TTA pose's geometry, which it does inside the loop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "toothseg3d"))

from layer.checkpoint import ModelSpec, random_model  # noqa: E402
from pipeline.config import Backend, SegmentationConfig  # noqa: E402
from pipeline.segmenter import Segmenter  # noqa: E402


def _legacy(paths: list[Path], tta: int, dev: str) -> list[dict[str, float]]:
    from infer import segment_inst
    from t3ds_io import read_obj, vertex_normals
    from train_inst import build_net
    torch.manual_seed(0)
    net = build_net(32, dev, 7, 128, 4).eval()
    rows = []
    for i, p in enumerate(paths):
        V, F, _ = read_obj(str(p))
        nrm = vertex_normals(V, F)
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        segment_inst(net, V, nrm, "lower", 0.25, 32, dev, timing, tta=tta)
        timing["total"] = (time.perf_counter() - t0) * 1000
        if i:                                    # first scan warms up cuDNN / allocator
            rows.append(timing)
    return rows


def _ours(paths: list[Path], tta: int, backend: Backend) -> list[dict[str, float]]:
    seg = Segmenter(SegmentationConfig(backend=backend, tta=tta), random_model(ModelSpec()))
    rows = []
    for i, p in enumerate(paths):
        scan = seg.load(p)
        t0 = time.perf_counter()
        r = seg.segment(scan)
        timing = {k: v for k, v in r.timing.items() if k not in ("load", "normals")}
        timing["total"] = (time.perf_counter() - t0) * 1000
        if i:
            rows.append(timing)
    return rows


def _report(name: str, rows: list[dict[str, float]]) -> None:
    keys = list(rows[0])
    med = {k: float(np.median([r.get(k, 0.0) for r in rows])) for k in keys}
    print(f"{name:<26}" + " | ".join(f"{k} {v:6.1f}" for k, v in med.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="also time both on the CPU (slow)")
    a = ap.parse_args()
    paths = sorted((ROOT.parent / "data/open/teeth3ds/raw").glob("*/lower/*/*_lower.obj"))[:a.n + 1]
    print(f"{len(paths) - 1} Teeth3DS scans (+1 warm-up), TTA x{a.tta}, median ms\n")
    if torch.cuda.is_available():
        _report("toothseg3d (GPU)", _legacy(paths, a.tta, "cuda"))
        _report("segmentation3d (GPU)", _ours(paths, a.tta, Backend.GPU))
    if a.cpu or not torch.cuda.is_available():
        _report("toothseg3d (CPU)", _legacy(paths, a.tta, "cpu"))
        _report("segmentation3d (CPU)", _ours(paths, a.tta, Backend.CPU))


if __name__ == "__main__":
    main()
