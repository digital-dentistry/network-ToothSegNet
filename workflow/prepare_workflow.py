from __future__ import annotations

from pathlib import Path
from typing import Any

from communication.communication_module import CommunicationModule
from pipeline.stages import PrepareStage
from pipeline.types import PreparedScan, Scan

from .stage_workflow import StageWorkflow


class PrepareWorkflow(StageWorkflow[Scan, PreparedScan]):
    """Scan -> PreparedScan.  With the geometry on the host, this overlaps the
    previous scan's forward on the GPU."""

    def __init__(self, comm: CommunicationModule, stage: PrepareStage) -> None:
        super().__init__(comm)
        self._stage = stage

    def _source(self) -> Any:
        return self._comm.loaded

    def _sink(self) -> Any:
        return self._comm.prepared

    def _process(self, item: Scan) -> PreparedScan:
        return self._stage.run(item)

    def _path_of(self, item: Scan) -> Path:
        return item.path
