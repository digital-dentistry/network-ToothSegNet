"""What the semantic model (gingiva + 16 FDI positions, `train.py --stage
semantic`) says about scans it never saw -- the in-house ones by default.

The instance pipeline (`Segmenter`, `zero_shot_report.py`) loads only
mask-transformer checkpoints.  This runs a semantic checkpoint through the
same steps: `LoadStage`, downsampling to the training resolution, the same
features and geometry over the same TTA poses, softmax averaged over the
poses, each vertex taking its voxel's class.

In --out:

* sheet_NN.png   occlusal views coloured and drawn as `preprocess_view.py`
                 draws Teeth3DS, every tooth numbered.  The in-house scanners
                 do not share Teeth3DS's frame, so each scan is first turned
                 into one: occlusal normal (least-variance axis, towards the
                 crowns) out of the page, predicted incisors up;
* <name>_sem.ply the mesh in the same colours, with --ply (MeshLab / CloudCompare);
* report.json    per scan: FDI teeth found, tooth share, confidence, and how
                 often a single TTA pose agrees with the fused class.

    python tools/semantic_view.py --ckpt runs/semantic/best.pt
    python tools/semantic_view.py --ckpt runs/semantic/best.pt --names ../data/ios/inhouse/splits/holdout.txt --ply
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

from algorithm.model_input.common.builder import CurveKey  # noqa: E402
from algorithm.model_input.common.features import standardise_colour, unit_normals  # noqa: E402
from algorithm.model_input.model_input import (GeometrySpec, ModelInput, ModelInputInput,  # noqa: E402
                                               ModelInputOutput, model_input_type)
from algorithm.result_encoder.result_encoder import (ResultEncoder, ResultEncoderInput,  # noqa: E402
                                                     ResultEncoderOutput, ResultEncoderType)
from algorithm.voxel_downsample.voxel_downsample import (VoxelDownsample, VoxelDownsampleInput,  # noqa: E402
                                                         VoxelDownsampleOutput, VoxelDownsampleType)
from layer.checkpoint import NUM_CLASSES, build_semantic_network  # noqa: E402
from pipeline.config import Placement, SegmentationConfig  # noqa: E402
from pipeline.rotations import tta_rotations  # noqa: E402
from pipeline.stages import LoadStage  # noqa: E402
from pipeline.types import Scan  # noqa: E402
from tools.preprocess_view import CLASS_RGB, View, cloud, fdi, frame, panel, save, tooth_labels  # noqa: E402
from tools.zero_shot_report import EXTENSIONS  # noqa: E402

MIN_FRAC = SegmentationConfig.min_frac     # a class under this share of the scan is not a tooth found
INCISORS = (1, 2, 9, 10)                   # central and lateral incisors of both quadrants
DRAW_POINTS = 40000                        # per panel
COLS, ROWS = 6, 6


class SemanticModel:
    """A `--stage semantic` checkpoint behind the inference half of `PrepareStage`."""

    def __init__(self, path: Path, tta: int, device: torch.device) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("stage") != "semantic":
            raise SystemExit(f"{path} is not a --stage semantic checkpoint (instance ones: zero_shot_report.py)")
        self.in_ch, self.voxel_mm, self.epoch = int(ck["in_ch"]), float(ck["args"]["res"]), ck["epoch"]
        self.device, gpu = device, device.type == "cuda"
        self.net = build_semantic_network(self.in_ch)
        self.net.load_state_dict(ck["model"])
        self.net.to(device).eval()
        self.down = VoxelDownsample()
        self.down.set_strategy(VoxelDownsampleType.GRID_GPU if gpu else VoxelDownsampleType.GRID_CPU)
        self.model_input = ModelInput()
        self.model_input.set_strategy(model_input_type(CurveKey[str(ck["curve"]).upper()], gpu))
        self.rotations = tta_rotations(max(1, tta))

    @torch.inference_mode()
    def predict(self, scan: Scan, tta_batch: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
        """-> per vertex (class, fused probability of it), and the share of
        (pose, point) pairs whose own argmax is the fused class."""
        pos = scan.positions.to(self.device)
        ds = VoxelDownsampleOutput()
        if not self.down.execute(VoxelDownsampleInput(pos, self.voxel_mm), ds):
            raise ValueError(f"{scan.name}: nothing to downsample")
        rep = ds.representative
        points = pos[rep] - pos[rep].double().mean(0).float()
        normals = unit_normals(scan.normals.to(self.device)[rep])
        colour = None
        if self.in_ch >= 10:
            rgb = scan.colours.to(self.device)[rep] if scan.colours is not None else torch.full_like(points, 0.5)
            colour = standardise_colour(rgb)
        probs = []
        for rot in torch.split(self.rotations, tta_batch):
            mi = ModelInputOutput()
            if not self.model_input.execute(ModelInputInput(points, normals, scan.jaw == "upper", colour, rot,
                                                            self.voxel_mm, GeometrySpec(n_query=0),
                                                            cells=ds.cells), mi):
                raise ValueError(f"{scan.name}: could not build the network input")
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
                sem = self.net(mi.features.to(self.device), mi.geometry.to(self.device))["sem"]
            probs.append(sem.float().softmax(-1).view(len(rot), points.shape[0], NUM_CLASSES))
        p = torch.cat(probs)                                     # [T,N,17]
        conf, cls = p.mean(0).max(1)
        agree = float((p.argmax(-1) == cls).float().mean())
        inv = ds.inverse.to(self.device)
        return cls[inv].cpu().numpy(), conf[inv].cpu().numpy(), agree


def occlusal_frame(p: np.ndarray, n: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """[3,3] rows (u, v, w): w the occlusal normal towards the crowns, v towards
    the predicted incisors, u = v x w -- a rotation, so the view is not mirrored."""
    c = p - p.mean(0)
    _, _, vt = np.linalg.svd(c[::max(1, len(c) // 20000)], full_matrices=False)
    tooth = cls > 0
    w = vt[2] if (n[tooth] if tooth.sum() > 100 else n).mean(0) @ vt[2] >= 0 else -vt[2]
    front = np.isin(cls, INCISORS)
    v = c[front].mean(0) - c[tooth].mean(0) if front.sum() > 50 else vt[0]
    v = v - (v @ w) * w
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else vt[0]
    return np.stack([np.cross(v, w), v, w])


def draw(ax, item: dict) -> None:
    p, n, cls, row = item["p"], item["n"], item["cls"], item["row"]
    rot = occlusal_frame(p, n, cls)
    p, n = (p - p.mean(0)) @ rot.T, n @ rot.T
    view = View(n, cls > 0 if (cls > 0).any() else np.ones(len(cls), bool))
    upper = row["jaw"] == "upper"
    frame(ax, cloud(ax, view, p, CLASS_RGB[cls], n, size=0.6))
    tooth_labels(ax, view, p, cls, np.where(np.isin(cls, row["classes"]), cls, 0), upper)
    jaw = f"{row['jaw']}?" if row["jaw_guessed"] else row["jaw"]
    panel(ax, row["name"], f"{jaw} · {len(row['classes'])} teeth · conf {row['conf']:.2f} · TTA {row['agree']:.2f}")


def sheets(items: list[dict], out: Path, title: str, sub: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from algorithm.loss.common.plot_style import INK, INK2, SURFACE
    per = COLS * ROWS
    pages = (len(items) + per - 1) // per
    for k in range(pages):
        chunk = items[k * per:(k + 1) * per]
        rows = (len(chunk) + COLS - 1) // COLS
        height = rows * 3.5 + 1.0
        fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.2, height), facecolor=SURFACE, squeeze=False)
        # the header sits a fixed distance from the top, in inches: as a share
        # of a tall page it would run into the first row's titles
        fig.text(0.008, 1 - 0.15 / height, f"{title}  ·  page {k + 1}/{pages}", ha="left", va="top", color=INK,
                 fontsize=13, fontweight="bold")
        fig.text(0.008, 1 - 0.45 / height, sub, ha="left", va="top", color=INK2, fontsize=8.5)
        for ax, item in zip(axes.flat, chunk):
            draw(ax, item)
        for ax in axes.flat[len(chunk):]:
            ax.set_axis_off()
        save(fig, out / f"sheet_{k + 1:02d}.png", top=1 - 0.85 / height)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/semantic/best.pt")
    ap.add_argument("--dir", type=Path, default=ROOT.parent / "data/ios/inhouse/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/semantic_view")
    ap.add_argument("--names", type=Path, default=None, help="only these scans, one name per line (holdout.txt)")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--ply", action="store_true", help="also write each scan as a coloured mesh")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticModel(a.ckpt, a.tta, dev)
    cfg = SegmentationConfig()
    load = LoadStage(cfg, Placement.resolve(cfg))
    encoder = ResultEncoder()
    encoder.set_strategy(ResultEncoderType.PLY_CPU)
    paths = sorted(p for p in a.dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if a.names:
        only = set(a.names.read_text().split())
        paths = [p for p in paths if p.stem in only]
    if a.limit:
        paths = paths[:a.limit]
    print(f"{a.ckpt} | epoch {model.epoch} | {model.voxel_mm * 1000:.0f} um | in_ch {model.in_ch} | tta x{a.tta} "
          f"| {len(paths)} scans -> {a.out}", flush=True)

    items, rows, failed = [], [], []
    for k, path in enumerate(paths, 1):
        try:
            t0 = time.perf_counter()
            scan = load.run(path)
            cls, conf, agree = model.predict(scan)
            ms = (time.perf_counter() - t0) * 1000
        except Exception as e:                       # report it, keep going
            failed.append({"name": path.stem, "error": f"{type(e).__name__}: {e}"})
            print(f"[{k}/{len(paths)}] {path.stem}: {failed[-1]['error']}", flush=True)
            continue
        share = np.bincount(cls, minlength=NUM_CLASSES) / len(cls)
        found = [c for c in range(1, NUM_CLASSES) if share[c] >= MIN_FRAC]
        upper = scan.jaw == "upper"
        row = {"name": scan.name, "jaw": scan.jaw, "jaw_guessed": "unknown" in scan.name, "vertices": len(cls),
               "colour": scan.colours is not None, "classes": found, "fdi": [fdi(c, upper) for c in found],
               "tooth_share": float(1 - share[0]), "conf": float(conf.mean()),
               "conf_teeth": float(conf[cls > 0].mean()) if (cls > 0).any() else 0.0, "agree": agree, "ms": ms}
        rows.append(row)
        idx = np.random.default_rng(0).choice(len(cls), min(len(cls), DRAW_POINTS), replace=False)
        items.append({"p": scan.positions.cpu().numpy()[idx].astype(np.float64),
                      "n": scan.normals.cpu().numpy()[idx].astype(np.float64), "cls": cls[idx], "row": row})
        if a.ply:
            encoder.execute(ResultEncoderInput(a.out / f"{scan.name}_sem.ply", scan.name, scan.jaw, scan.positions,
                                               torch.from_numpy(cls), scan.faces,
                                               colours=(CLASS_RGB[cls] * 255).round().astype(np.uint8)),
                            ResultEncoderOutput())
        print(f"[{k}/{len(paths)}] {scan.name:<18} {len(found):2d} teeth  share {row['tooth_share']:.2f}  "
              f"conf {row['conf']:.2f}  TTA {agree:.2f}  {ms:.0f} ms  {' '.join(row['fdi'])}", flush=True)

    (a.out / "report.json").write_text(json.dumps({"ckpt": str(a.ckpt), "epoch": model.epoch, "tta": a.tta,
                                                   "failed": failed, "scans": rows}, indent=1, ensure_ascii=False))
    if items:
        sheets(items, a.out, f"semantic model on {a.dir.name}",
               f"{a.ckpt.parent.name}/{a.ckpt.name} · epoch {model.epoch} · {model.voxel_mm * 1000:.0f} µm · "
               f"TTA x{a.tta}  ·  occlusal view from the crown side, predicted incisors up, never mirrored  ·  "
               f"grey gingiva, every tooth carries its FDI number  ·  jaw? = not in the file name, read as lower")
    if rows:
        print(f"\n{len(rows)} scans, {len(failed)} failed | median teeth "
              f"{np.median([len(r['classes']) for r in rows]):.0f}  conf {np.median([r['conf'] for r in rows]):.2f}  "
              f"TTA {np.median([r['agree'] for r in rows]):.2f} -> {a.out}")


if __name__ == "__main__":
    main()
