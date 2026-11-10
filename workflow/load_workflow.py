from __future__ import annotations

from pathlib import Path
from typing import Any

from communication.communication_module import CommunicationModule
from pipeline.stages import LoadStage
from pipeline.types import Scan

from .stage_workflow import StageWorkflow


class LoadWorkflow(StageWorkflow[Path, Scan]):
    """File -> Scan.  Mostly file I/O and numpy parsing, which release the GIL."""

    def __init__(self, comm: CommunicationModule, stage: LoadStage) -> None:
        super().__init__(comm)
        self._stage = stage

    def _source(self) -> Any:
        return self._comm.scan_paths

    def _sink(self) -> Any:
        return self._comm.loaded

    def _process(self, item: Path) -> Scan:
        return self._stage.run(item)

    def _path_of(self, item: Path) -> Path:
        return item
