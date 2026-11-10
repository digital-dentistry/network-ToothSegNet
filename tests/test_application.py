"""The threaded pipeline gives the synchronous one's answers, and survives bad files."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np

from tests.conftest import teeth3ds_scans


def test_threads_match_segmenter(tmp_path: Path) -> None:
    from layer.checkpoint import ModelSpec, random_model
    from app import Application
    from pipeline.config import Backend, SegmentationConfig
    from pipeline.segmenter import Segmenter

    scans = teeth3ds_scans("lower", 2) + teeth3ds_scans("upper", 1)
    paths = [Path(shutil.copy(p, tmp_path / p.name)) for p in scans]
    bad = tmp_path / "broken.ply"
    bad.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 10\n")   # no end_header
    paths.append(bad)

    # CPU: deterministic, so the two runs can be compared exactly
    config = SegmentationConfig(backend=Backend.CPU, tta=2, score_thresh=0.1, min_frac=0.0)
    model = random_model(ModelSpec())
    sync = Segmenter(config, model)
    want = {p.name: sync.segment_file(p) for p in paths[:-1]}

    before = threading.active_count()
    with Application(config, tmp_path / "out", model) as app:
        results = app.run(paths)
    assert threading.active_count() == before              # every workflow thread joined

    assert len(results) == len(paths)
    by_name = {r.path.name: r for r in results}
    assert by_name["broken.ply"].error is not None
    for name, w in want.items():
        got = by_name[name]
        assert got.error is None
        assert np.array_equal(got.instances, w.instances)
        assert np.allclose(got.scores, w.scores)
        assert all(f.exists() for f in got.outputs) and len(got.outputs) == 2
