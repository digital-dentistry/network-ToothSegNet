from __future__ import annotations

from pathlib import Path
from typing import Any

from communication.communication_module import CommunicationModule
from pipeline.stages import FinishStage
from pipeline.types import InferredScan, SegmentationResult

from .stage_workflow import StageWorkflow


class FinishWorkflow(StageWorkflow[InferredScan, SegmentationResult]):
    """InferredScan -> SegmentationResult: upsample, instances, output files."""

    def __init__(self, comm: CommunicationModule, stage: FinishStage) -> None:
        super().__init__(comm)
        self._stage = stage

    def _source(self) -> Any:
        return self._comm.inferred

    def _sink(self) -> Any:
        return self._comm.results

    def _process(self, item: InferredScan) -> SegmentationResult:
        return self._stage.run(item)

    def _path_of(self, item: InferredScan) -> Path:
        return item.prepared.scan.path
