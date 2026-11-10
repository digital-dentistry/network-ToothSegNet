"""Segment every scan in a folder, pipelined across threads.

    python app.py --dir ../data/inhouse --ckpt toothinst_250um_msn.pt --tta 8

Four workflow threads -- load, prepare, infer, finish -- pass scans along
bounded buffers, so decoding the next file and building its geometry overlap
the current scan's forward.  Without --ckpt the weights are random: the
output shows shapes and timing, not teeth.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from queue import Empty
from typing import Callable

from layer.checkpoint import LoadedModel
from algorithm.model_input.common.builder import CurveKey
from algorithm.scan_decoder.scan_decoder import EXTENSIONS
from communication.communication_module import CommunicationModule
from interface.workflow import Workflow
from pipeline.config import Backend, Placement, SegmentationConfig, UpsampleMode
from pipeline.segmenter import load_model
from pipeline.stages import FinishStage, InferStage, LoadStage, PrepareStage
from pipeline.types import SegmentationResult
from workflow.finish_workflow import FinishWorkflow
from workflow.infer_workflow import InferWorkflow
from workflow.load_workflow import LoadWorkflow
from workflow.prepare_workflow import PrepareWorkflow


class Application:
    def __init__(self, config: SegmentationConfig, out_dir: Path | None,
                 model: LoadedModel | None = None, write_json: bool = True) -> None:
        self.model = model or load_model(config)
        placement = Placement.resolve(config)
        self._comm = CommunicationModule()                       # owner
        self._workflows: list[Workflow] = [                      # non-owning references
            LoadWorkflow(self._comm, LoadStage(config, placement)),
            PrepareWorkflow(self._comm, PrepareStage(config, placement, self.model.spec)),
            InferWorkflow(self._comm, InferStage(config, placement, self.model)),
            FinishWorkflow(self._comm, FinishStage(config, placement, out_dir, write_json)),
        ]

    def start(self) -> None:
        self._comm.clear()
        for wf in self._workflows:
            wf.start()

    def stop(self) -> None:
        self._comm.close()                                       # wake blocked threads
        for wf in reversed(self._workflows):
            wf.stop()

    def run(self, paths: list[Path],
            on_result: Callable[[SegmentationResult], None] | None = None) -> list[SegmentationResult]:
        """Segment `paths`; results arrive in completion order (the pipeline keeps file order)."""
        for p in paths:
            self._comm.scan_paths.push(p)
        results: list[SegmentationResult] = []
        while len(results) < len(paths):
            try:
                r = self._comm.results.pop(timeout=0.5)
            except Empty:
                if not all(wf.is_running() for wf in self._workflows):
                    raise RuntimeError("a workflow thread died; see the log") from None
                continue
            results.append(r)
            if on_result is not None:
                on_result(r)
        return results

    def __enter__(self) -> Application:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _print_result(r: SegmentationResult) -> None:
    if r.error:
        print(f"{r.name[:46]:<48} FAILED  {r.error}", flush=True)
        return
    t = r.timing
    seg = sum(v for k, v in t.items() if k not in ("load", "write"))
    print(f"{r.name[:46]:<48} {r.jaw:<5} {r.n_vertices:>7} verts | {r.n_teeth:>2} instances "
          f"| tooth {r.tooth_fraction * 100:4.1f}% | load {t.get('load', 0):>5.0f} + seg {seg:>6.0f} ms",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None, help="mask-transformer checkpoint (.pt)")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <dir>/segmented")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--backend", choices=[b.value for b in Backend], default="auto")
    ap.add_argument("--res", type=float, default=None, help="voxel mm; default: the checkpoint's")
    ap.add_argument("--jaw", choices=["upper", "lower"], default=None)
    ap.add_argument("--tta", type=int, default=4, help="poses averaged (x8 is the plateau)")
    ap.add_argument("--tta-batch", type=int, default=8, help="poses per forward")
    ap.add_argument("--upsample", choices=[m.value for m in UpsampleMode], default="knn")
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--erode", type=float, default=0.0, help="mm; 0 for the in-house fine-tuned models")
    ap.add_argument("--min-frac", type=float, default=0.005)
    ap.add_argument("--curve", choices=["legacy", "morton3d"], default=None,
                    help="default: the curve the checkpoint was trained on")
    ap.add_argument("--geometry", choices=["auto", "cpu", "gpu"], default="auto",
                    help="where to build the geometry; auto: GPU when --tta > 1")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(threadName)s %(levelname)s %(message)s")

    config = SegmentationConfig(
        checkpoint=a.ckpt, backend=Backend(a.backend), voxel_mm=a.res, tta=a.tta, tta_batch=a.tta_batch,
        upsample=UpsampleMode(a.upsample), knn=a.knn, min_frac=a.min_frac, erode_mm=a.erode,
        curve=CurveKey[a.curve.upper()] if a.curve else None, jaw=a.jaw,
        geometry_on_gpu={"auto": None, "cpu": False, "gpu": True}[a.geometry])
    out_dir = a.out or a.dir / "segmented"
    pattern = "**/*" if a.recursive else "*"
    paths = sorted(p for p in a.dir.glob(pattern)
                   if p.is_file() and p.suffix.lower() in EXTENSIONS and out_dir not in p.parents)

    app = Application(config, out_dir, write_json=not a.no_json)
    spec = app.model.spec
    print(f"{len(paths)} scans in {a.dir}.\nmodel {spec.source} | epoch {spec.epoch} | "
          f"{(config.voxel_mm or spec.voxel_mm) * 1000:.0f} um | {spec.n_query} queries | in_ch {spec.in_ch} "
          f"| tta x{config.tta} | {Placement.resolve(config)}\n", flush=True)
    if a.ckpt is None:
        print("!! no --ckpt: random weights -- instance counts below mean nothing\n", flush=True)

    t0 = time.perf_counter()
    with app:
        results = app.run(paths, _print_result)
    wall = time.perf_counter() - t0
    done = [r for r in results if not r.error]
    print(f"\n{len(done)}/{len(results)} segmented in {wall:.1f} s "
          f"({wall / max(len(results), 1) * 1000:.0f} ms per scan, pipelined) -> {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [dict(name=r.name, jaw=r.jaw, verts=r.n_vertices, teeth=r.n_teeth,
                    tooth_frac=r.tooth_fraction, scores=r.scores, ms=r.timing, error=r.error)
               for r in sorted(results, key=lambda r: r.name)]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
