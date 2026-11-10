from __future__ import annotations

from pathlib import Path

from layer.checkpoint import LoadedModel, ModelSpec, load_checkpoint, random_model

from .config import Placement, SegmentationConfig
from .stages import FinishStage, InferStage, LoadStage, PrepareStage
from .types import Scan, SegmentationResult


def load_model(config: SegmentationConfig) -> LoadedModel:
    if config.checkpoint is None:
        return random_model(ModelSpec())
    return load_checkpoint(config.checkpoint)


class Segmenter:
    """One scan at a time, in the calling thread -- for notebooks, tests and
    single files.  `app.Application` runs the same stages pipelined across
    threads for folders."""

    def __init__(self, config: SegmentationConfig, model: LoadedModel | None = None,
                 out_dir: Path | None = None, write_json: bool = True) -> None:
        self.config = config
        self.model = model or load_model(config)
        self.placement = Placement.resolve(config)
        self._load = LoadStage(config, self.placement)
        self._prepare = PrepareStage(config, self.placement, self.model.spec)
        self._infer = InferStage(config, self.placement, self.model)
        self._finish = FinishStage(config, self.placement, out_dir, write_json)

    def segment_file(self, path: Path) -> SegmentationResult:
        return self.segment(self._load.run(path))

    def segment(self, scan: Scan) -> SegmentationResult:
        return self._finish.run(self._infer.run(self._prepare.run(scan)))

    def load(self, path: Path) -> Scan:
        return self._load.run(path)
