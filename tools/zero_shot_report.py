"""Zero-shot feasibility on unlabelled scans, calibrated on labelled Teeth3DS.

Without labels nothing measures accuracy directly, so three label-free signals
are taken on both sets:

* anatomy -- teeth per scan (<= 16) and the tooth share of the vertices;
* confidence -- the instance scores the model gives itself;
* stability -- the scan is segmented again in a random SO(3) frame and the two
  labellings are matched instance for instance (mean IoU, count change).  The
  model was trained to be indifferent to the frame, so where it disagrees with
  itself it is guessing.

On Teeth3DS testing the same signals sit next to the true mAP / TSA, which
says how far each signal can be trusted when it is all there is.

    python tools/zero_shot_report.py --ckpt runs/instance/best.pt --dir ../data/ios/inhouse/raw \\
                                     --out runs/zero_shot

After pseudo-label training only the held-out scans still say something:

    python tools/zero_shot_report.py --ckpt runs/instance_r1/best.pt \\
                                     --names ../data/ios/inhouse/splits/holdout.txt --out runs/zero_shot_r1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm.augmentation.common.rotation import random_so3  # noqa: E402
from algorithm.evaluation.common.average_precision import InstanceAPMeter  # noqa: E402
from algorithm.evaluation.evaluation import (Evaluation, EvaluationInput, EvaluationOutput,  # noqa: E402
                                             EvaluationType)
from pipeline.config import SegmentationConfig  # noqa: E402
from pipeline.segmenter import Segmenter  # noqa: E402
from pipeline.types import Scan, SegmentationResult  # noqa: E402

EXTENSIONS = {".ply", ".stl", ".obj"}
MAX_TEETH = 16
TOOTH_SHARE = (0.20, 0.75)          # toothseg3d import_pseudo.py's plausible range


def rotated(scan: Scan, rot: torch.Tensor) -> Scan:
    r = rot.float()
    return Scan(path=scan.path, name=scan.name, jaw=scan.jaw, positions=scan.positions @ r.to(scan.positions.device).T,
                normals=scan.normals @ r.to(scan.normals.device).T, faces=scan.faces, colours=scan.colours)


def agreement(ev: Evaluation, a: SegmentationResult, b: SegmentationResult) -> dict[str, float]:
    """b against a as if a were the truth: matched mean IoU and AP-free counts."""
    out = EvaluationOutput()
    ev.execute(EvaluationInput(torch.from_numpy(b.instances), torch.from_numpy(a.instances), min_pred_points=1), out)
    m = out.metrics
    return {"stab_iou": m["mean_iou"], "stab_matched50": m["matched50"] / max(m["n_gt"], m["n_pred"], 1),
            "stab_count_diff": abs(m["n_pred"] - m["n_gt"])}


def label_free(seg: Segmenter, ev: Evaluation, scan: Scan, seed: int) -> tuple[SegmentationResult, dict]:
    t0 = time.perf_counter()
    r = seg.segment(scan)
    ms = (time.perf_counter() - t0) * 1000
    rot = random_so3(torch.Generator().manual_seed(seed))
    r2 = seg.segment(rotated(scan, rot))
    scores = np.asarray(r.scores) if r.scores else np.zeros(0)
    row = {"name": scan.name, "jaw": scan.jaw, "vertices": r.n_vertices, "colour": scan.colours is not None,
           "teeth": r.n_teeth, "tooth_share": r.tooth_fraction,
           "score_mean": float(scores.mean()) if scores.size else 0.0,
           "score_min": float(scores.min()) if scores.size else 0.0, "ms": ms, **agreement(ev, r, r2)}
    row["plausible"] = bool(0 < row["teeth"] <= MAX_TEETH and TOOTH_SHARE[0] <= row["tooth_share"] <= TOOTH_SHARE[1])
    return r, row


def teeth3ds_testing(n: int) -> list[Path]:
    split = ROOT.parent / "data/open/teeth3ds/splits/Teeth3DS_train_test_split"
    names = [ln.strip() for f in ("testing_upper.txt", "testing_lower.txt")
             for ln in (split / f).read_text().splitlines() if ln.strip()]
    rng = np.random.default_rng(0)
    names = [names[i] for i in rng.permutation(len(names))[:n]]
    raw = ROOT.parent / "data/open/teeth3ds/raw"
    paths = []
    for nm in names:
        hit = sorted(raw.glob(f"data_part_*/*/*/{nm}.obj"))
        if hit:
            paths.append(hit[0])
    return paths


def summarise(rows: list[dict]) -> dict[str, float]:
    keys = ["teeth", "tooth_share", "score_mean", "score_min", "stab_iou", "stab_matched50", "stab_count_diff", "ms"]
    out = {k: float(np.median([r[k] for r in rows])) for k in keys}
    out["plausible"] = float(np.mean([r["plausible"] for r in rows]))
    out["stable"] = float(np.mean([r["stab_iou"] >= 0.8 and r["stab_count_diff"] <= 1 for r in rows]))
    out["n"] = len(rows)
    return out


def contact_sheet(results: list[tuple[Scan, SegmentationResult, dict]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 8
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    cmap = plt.get_cmap("tab20")
    for ax, (scan, r, row) in zip(np.ravel(axes), results):
        v = scan.positions.cpu().numpy().astype(np.float64)
        v -= v.mean(0)
        _, _, vt = np.linalg.svd(v[::20], full_matrices=False)
        p = v @ vt.T
        idx = np.random.default_rng(0).choice(len(p), min(len(p), 12000), replace=False)
        ins = r.instances[idx]
        col = np.where(ins[:, None] > 0, cmap((ins % 20) / 20)[:, :3], 0.85)
        order = np.argsort(p[idx, 2])
        ax.scatter(p[idx, 0][order], p[idx, 1][order], c=col[order], s=0.2)
        flag = "" if row["plausible"] and row["stab_iou"] >= 0.8 else "  ⚠"
        ax.set_title(f"{scan.name}\n{row['teeth']} teeth  stab {row['stab_iou']:.2f}{flag}", fontsize=6)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in np.ravel(axes)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=ROOT.parent / "data/ios/inhouse/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/zero_shot")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--teeth3ds", type=int, default=100, help="Teeth3DS testing scans for calibration")
    ap.add_argument("--names", type=Path, default=None, help="only these scans, one name per line (holdout.txt)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    seg = Segmenter(SegmentationConfig(checkpoint=a.ckpt, tta=a.tta))
    ev = Evaluation()
    ev.set_strategy(EvaluationType.INSTANCE_GPU if torch.cuda.is_available() else EvaluationType.INSTANCE_CPU)
    spec = seg.model.spec
    print(f"model {a.ckpt} | epoch {spec.epoch} | {spec.voxel_mm * 1000:.0f} um | curve {spec.curve.name} "
          f"| in_ch {spec.in_ch} | tta x{a.tta}", flush=True)

    # --- calibration: Teeth3DS testing, where the truth is known
    ap_meter, calib = InstanceAPMeter(), []
    for i, path in enumerate(teeth3ds_testing(a.teeth3ds)):
        scan = seg.load(path)
        r, row = label_free(seg, ev, scan, seed=10_000 + i)
        gt = np.asarray(json.loads(path.with_suffix(".json").read_text())["instances"], np.int64)
        out = EvaluationOutput()
        ev.execute(EvaluationInput(torch.from_numpy(r.instances), torch.from_numpy(gt),
                                   scores=torch.tensor(r.scores, dtype=torch.float64)), out)
        ap_meter.add(out.record)
        row.update(true_tsa=out.metrics["tsa"], true_iou=out.metrics["mean_iou"],
                   true_teeth=out.metrics["n_gt"])
        calib.append(row)
    calib_ap = ap_meter.result()

    # --- the unlabelled scans
    results, rows, failed = [], [], []
    paths = sorted(p for p in a.dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    only = set(a.names.read_text().split()) if a.names else None
    for i, path in enumerate(paths):             # i over every scan: a scan's seed is the same either way
        if only is not None and path.stem not in only:
            continue
        try:
            scan = seg.load(path)
            r, row = label_free(seg, ev, scan, seed=i)
        except Exception as e:                   # report it, keep going
            failed.append({"name": path.name, "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(row)
        results.append((scan, r, row))

    s_cal, s_in = summarise(calib), summarise(rows) if rows else {}
    corr = float(np.corrcoef([r["stab_iou"] for r in calib], [r["true_iou"] for r in calib])[0, 1]) if len(calib) > 2 else 0.0
    report = {"model": str(a.ckpt), "epoch": spec.epoch, "tta": a.tta,
              "teeth3ds": {"mAP": calib_ap["mAP"], "AP50": calib_ap["AP50"], **s_cal,
                           "true_tsa": float(np.mean([r["true_tsa"] for r in calib])),
                           "true_iou": float(np.median([r["true_iou"] for r in calib])),
                           "corr_stab_iou_vs_true_iou": corr},
              "inhouse": s_in, "failed": failed, "inhouse_scans": rows, "teeth3ds_scans": calib}
    (a.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    contact_sheet(results, a.out / "inhouse_sheet.png")

    print(f"\nTeeth3DS testing ({s_cal['n']}): mAP {calib_ap['mAP'] * 100:.1f}  AP50 {calib_ap['AP50'] * 100:.1f}  "
          f"TSA {report['teeth3ds']['true_tsa'] * 100:.1f}  | corr(stability, true IoU) {corr:.2f}")
    fmt = ("{:<10} teeth {teeth:5.1f}  share {tooth_share:.2f}  score {score_mean:.2f} (min {score_min:.2f})  "
           "stab IoU {stab_iou:.2f}  plausible {plausible:.0%}  stable {stable:.0%}  {ms:.0f} ms")
    print(fmt.format("Teeth3DS", **s_cal))
    if s_in:
        print(fmt.format("in-house", **s_in))
    print(f"failed {len(failed)}  -> {a.out}")


if __name__ == "__main__":
    main()
