"""Pseudo-labels for the unlabelled in-house scans: the instance model's own
answer, kept only where it is plausible and stable, cached as the npz
`preprocess.py` writes so `train.py --cache` mixes them with Teeth3DS.

The model finds teeth but does not number them, so every tooth point is
class 17 (a tooth of unknown position): the arch-position loss ignores it, the
offset head does not train on it, and crop / mirror / colour augmentation work
from the instance ids alone.

A scan is labelled when the model is plausible about it (1..16 teeth, tooth
share 0.20-0.75) and agrees with itself in a random SO(3) frame (matched IoU
>= 0.8, tooth count within 1) -- the signals of `tools/zero_shot_report.py`.
Byte-identical files are labelled once, and the held-out scans (whole
patients, `holdout.txt`) never are: they are the in-house test set.

    python training/pseudo_label.py --ckpt runs/instance/best.pt --out ../data/ios/inhouse/pseudo_r1
    python training/train.py --stage instance --init-from runs/instance/best.pt --out runs/instance_r1 \\
        --cache ../data/ios/teeth3ds/cache ../data/ios/inhouse/pseudo_r1 \\
        --names ../data/ios/inhouse/pseudo_r1/train_names.txt \\
        --val-names ../data/ios/inhouse/pseudo_r1/val_names.txt --epochs 60 --lr 5e-4 --warmup 2

The next round labels with `runs/instance_r1/best.pt` but trains again from
the Teeth3DS model: fine-tuning on its own labels round after round would
compound the mistakes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.augmentation.common.mirror import UNKNOWN_TOOTH  # noqa: E402
from algorithm.evaluation.evaluation import Evaluation, EvaluationType  # noqa: E402
from pipeline.config import SegmentationConfig  # noqa: E402
from pipeline.segmenter import Segmenter  # noqa: E402
from pipeline.types import Scan, SegmentationResult  # noqa: E402
from tools.zero_shot_report import EXTENSIONS, contact_sheet, label_free  # noqa: E402
from training.dataset import read_split  # noqa: E402
from training.train import split_names  # noqa: E402

DATA = ROOT.parent / "data"


def select(paths: list[Path], holdout: set[str]) -> tuple[list[Path], dict[str, str]]:
    """The scans to label, and why each other one is skipped.  Of byte-identical
    files one is labelled -- the copy whose name knows the jaw -- and a copy of
    a held-out scan is held out too, or the test set would leak into training."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(p)
    keep, skipped = [], {}
    for group in groups.values():
        if any(p.stem in holdout for p in group):
            skipped.update({p.stem: "holdout" for p in group})
            continue
        first = min(group, key=lambda p: ("unknown" in p.stem, p.name))
        keep.append(first)
        skipped.update({p.stem: f"duplicate of {first.stem}" for p in group if p != first})
    return sorted(keep), skipped


def write_label(scan: Scan, result: SegmentationResult, path: Path) -> None:
    """One labelled scan -> the cache format: every vertex, its normal as
    inference sees it, teeth as class 17 with ids 1..K."""
    ins = result.instances
    ids = np.unique(ins[ins > 0])
    remap = np.zeros(int(ins.max()) + 1, np.uint8)
    remap[ids] = np.arange(1, len(ids) + 1)
    ins = remap[ins]
    data = dict(xyz=scan.positions.cpu().numpy().astype(np.float32),
                nrm=scan.normals.cpu().numpy().astype(np.float16),
                cls=np.where(ins > 0, UNKNOWN_TOOTH, 0).astype(np.uint8), ins=ins,
                jaw=np.uint8(scan.jaw == "upper"))
    if scan.colours is not None:
        data["rgb"] = (scan.colours.cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    np.savez(path, **data)


def thin(scan: Scan, result: SegmentationResult, n: int = 12000) -> tuple[Scan, SegmentationResult]:
    """What the contact sheet draws, without keeping every full scan alive."""
    idx = np.random.default_rng(0).choice(result.n_vertices, min(result.n_vertices, n), replace=False)
    t = torch.from_numpy(idx)
    return (Scan(path=scan.path, name=scan.name, jaw=scan.jaw, positions=scan.positions.cpu()[t],
                 normals=scan.normals.cpu()[t]),
            SegmentationResult(path=result.path, name=result.name, n_vertices=len(idx),
                               instances=result.instances[idx]))


def write_names(path: Path, names: list[str]) -> None:
    path.write_text("".join(f"{n}\n" for n in names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="the instance checkpoint that labels")
    ap.add_argument("--out", type=Path, required=True, help="npz cache, name lists and report")
    ap.add_argument("--dir", type=Path, default=DATA / "ios/inhouse/raw")
    ap.add_argument("--holdout", type=Path, default=DATA / "ios/inhouse/splits/holdout.txt",
                    help="in-house test scans (whole patients), never labelled")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--min-stab-iou", type=float, default=0.8)
    ap.add_argument("--max-count-diff", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2,
                    help="times each labelled scan is listed in train_names.txt (in-house is ~15%% of the scans)")
    ap.add_argument("--split-dir", type=Path, default=DATA / "ios/teeth3ds/splits/Teeth3DS_train_test_split")
    ap.add_argument("--val-frac", type=float, default=0.1, help="as train.py: the same Teeth3DS validation scans")
    a = ap.parse_args()
    if not a.holdout.exists():
        raise SystemExit(f"{a.holdout} not found: hold out the in-house test patients first")
    a.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in a.dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    todo, skipped = select(paths, set(read_split(a.holdout)))
    seed = {p: i for i, p in enumerate(paths)}       # zero_shot_report's seeds, so the rows compare
    seg = Segmenter(SegmentationConfig(checkpoint=a.ckpt, tta=a.tta))
    ev = Evaluation()
    ev.set_strategy(EvaluationType.INSTANCE_GPU if torch.cuda.is_available() else EvaluationType.INSTANCE_CPU)
    n_held = sum(v == "holdout" for v in skipped.values())
    print(f"{len(paths)} scans: label {len(todo)} | held out {n_held} | duplicates {len(skipped) - n_held} "
          f"| {a.ckpt} tta x{a.tta}", flush=True)

    rows, failed, sheets = [], [], {True: [], False: []}
    for k, path in enumerate(todo, 1):
        try:
            scan = seg.load(path)
            r, row = label_free(seg, ev, scan, seed=seed[path])
        except Exception as e:                       # report it, keep going
            failed.append({"name": path.stem, "error": f"{type(e).__name__}: {e}"})
            continue
        row["accepted"] = bool(row["plausible"] and row["stab_iou"] >= a.min_stab_iou
                               and row["stab_count_diff"] <= a.max_count_diff)
        if row["accepted"]:
            write_label(scan, r, a.out / f"{path.stem}.npz")
        rows.append(row)
        sheets[row["accepted"]].append((*thin(scan, r), row))
        print(f"[{k}/{len(todo)}] {path.stem:<18} teeth {row['teeth']:2d}  share {row['tooth_share']:.2f}  "
              f"stab {row['stab_iou']:.2f} (count {int(row['stab_count_diff'])})  "
              f"{'label' if row['accepted'] else 'reject'}", flush=True)

    labelled = [r["name"] for r in rows if r["accepted"]]
    t3d_train, t3d_val = split_names(argparse.Namespace(names=None, split_dir=a.split_dir, exclude="",
                                                        val_names=None, val_frac=a.val_frac, limit=0))
    write_names(a.out / "inhouse_names.txt", labelled)
    write_names(a.out / "train_names.txt", t3d_train + labelled * a.repeat)
    write_names(a.out / "val_names.txt", t3d_val)
    report = {"ckpt": str(a.ckpt), "tta": a.tta, "min_stab_iou": a.min_stab_iou,
              "max_count_diff": a.max_count_diff, "labelled": len(labelled), "rejected": len(rows) - len(labelled),
              "skipped": skipped, "failed": failed, "scans": rows}
    (a.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for ok, file_name in ((True, "labelled.png"), (False, "rejected.png")):
        if sheets[ok]:
            contact_sheet(sheets[ok], a.out / file_name)
    print(f"\nlabelled {len(labelled)} / {len(todo)}  (rejected {len(rows) - len(labelled)}, failed {len(failed)})"
          f"  -> {a.out}\ntrain_names: {len(t3d_train)} Teeth3DS + {len(labelled)} in-house x{a.repeat}"
          f"  | val_names: {len(t3d_val)} Teeth3DS")


if __name__ == "__main__":
    main()
