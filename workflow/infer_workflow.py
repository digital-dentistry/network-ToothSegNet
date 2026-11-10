from __future__ import annotations

from pathlib import Path
from typing import Any

from communication.communication_module import CommunicationModule
from pipeline.stages import InferStage
from pipeline.types import InferredScan, PreparedScan

from .stage_workflow import StageWorkflow


class InferWorkflow(StageWorkflow[PreparedScan, InferredScan]):
    """PreparedScan -> InferredScan: the forward and the TTA fusion.

    Every thread here uses the default CUDA stream, so work one thread queued
    is finished before anything another thread queues after receiving it --
    tensors can be handed over without events.
    """

    def __init__(self, comm: CommunicationModule, stage: InferStage) -> None:
        super().__init__(comm)
        self._stage = stage

    def _source(self) -> Any:
        return self._comm.prepared

    def _sink(self) -> Any:
        return self._comm.inferred

    def _process(self, item: PreparedScan) -> InferredScan:
        return self._stage.run(item)

    def _path_of(self, item: PreparedScan) -> Path:
        return item.scan.path
