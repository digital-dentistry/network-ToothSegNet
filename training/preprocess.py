"""Cache Teeth3DS scans as compact npz (port of `toothseg3d/preprocess.py`).

Every original vertex is kept -- scoring needs the full-resolution ground
truth -- with its normal and labels.  Downsampling happens in the dataset, so
the training resolution can change without re-running this.  The files are
the format toothseg3d writes, so either cache works for either trainer.

    python training/preprocess.py --root ../data/open/teeth3ds/raw --out ../data/open/teeth3ds/cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fdi_to_class(labels: np.ndarray, jaw: str) -> np.ndarray:
    """FDI number -> 0 gingiva, 1..8 first quadrant, 9..16 second (lower jaw: minus 20)."""
    lab = labels.astype(np.int64).copy()
    if jaw == "lower":
        lab[lab > 0] -= 20
    out = np.zeros_like(lab)
    q1, q2 = lab // 10 == 1, lab // 10 == 2
    out[q1] = lab[q1] % 10
    out[q2] = lab[q2] % 10 + 8
    return out


def cache_scan(obj_path: Path, json_path: Path, out_path: Path) -> str | None:
    """One scan -> one npz.  Returns an error message, or None."""
    import torch
    torch.set_num_threads(1)
    from algorithm.normal_estimation.normal_estimation import (
        NormalEstimation, NormalEstimationInput, NormalEstimationOutput, NormalEstimationType)
    from algorithm.scan_decoder.scan_decoder import ScanDecoder, ScanDecoderInput, ScanDecoderOutput
    try:
        dec, mesh = ScanDecoder(), ScanDecoderOutput()
        dec.set_strategy_for(obj_path)
        if not dec.execute(ScanDecoderInput(obj_path, True), mesh) or mesh.faces is None:
            return "could not decode a mesh"
        d = json.loads(json_path.read_text(encoding="utf-8"))
        labels, ins, jaw = np.asarray(d["labels"], np.int64), np.asarray(d["instances"], np.int64), d["jaw"]
        if len(labels) != mesh.positions.shape[0]:
            return f"label/vertex mismatch {len(labels)} vs {mesh.positions.shape[0]}"
        est = NormalEstimation()
        est.set_strategy(NormalEstimationType.MESH_CPU)
        nrm = NormalEstimationOutput()
        est.execute(NormalEstimationInput(mesh.positions, mesh.faces), nrm)
        cls = fdi_to_class(labels, jaw)

        # instance ids -> 1..K ordered by their (unique) class, so ids are reproducible
        teeth = [i for i in np.unique(ins) if i != 0]
        teeth.sort(key=lambda i: int(np.bincount(cls[ins == i]).argmax()))
        ins_out = np.zeros(len(ins), np.uint8)
        for k, i in enumerate(teeth, start=1):
            ins_out[ins == i] = k
        np.savez(out_path, xyz=mesh.positions.numpy().astype(np.float32),
                 nrm=nrm.normals.numpy().astype(np.float16), cls=cls.astype(np.uint8), ins=ins_out,
                 jaw=np.uint8(1 if jaw == "upper" else 0))
        return None
    except Exception as e:                       # one broken scan must not stop the batch
        return repr(e)


def _task(args: tuple[Path, Path, Path]) -> tuple[Path, str | None]:
    return args[2], cache_scan(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT.parent / "data/open/teeth3ds/raw")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "data/open/teeth3ds/cache")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scans (a smoke test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    objs = sorted(a.root.glob("data_part_*/*/*/*.obj"))
    tasks = [(o, o.with_suffix(".json"), a.out / f"{o.stem}.npz") for o in objs
             if o.with_suffix(".json").exists() and (a.force or not (a.out / f"{o.stem}.npz").exists())]
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"{len(objs)} obj files, {len(tasks)} to process -> {a.out}", flush=True)
    t0, errs = time.time(), []
    with Pool(a.workers) as pool:
        for k, (dst, err) in enumerate(pool.imap_unordered(_task, tasks, chunksize=4), 1):
            if err:
                errs.append((dst, err))
            if k % 100 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[{k}/{len(tasks)}] {el / 60:.1f} min  eta {(len(tasks) - k) * el / k / 60:.1f} min  "
                      f"errs {len(errs)}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(errs)} errors")
    for d, e in errs[:20]:
        print("  ", d.name, e)


if __name__ == "__main__":
    main()
