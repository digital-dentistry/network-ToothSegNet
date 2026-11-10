from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from queue import Empty
from typing import Any, Generic, TypeVar

from communication.communication_module import CommunicationModule
from interface.workflow import Workflow
from pipeline.types import SegmentationResult

logger = logging.getLogger(__name__)

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class StageWorkflow(Workflow, Generic[InT, OutT]):
    """Pop from one buffer, run one pipeline stage, push to the next.

    A scan that fails here is reported on `results` straight away, so every
    path still produces exactly one result and the owner's count adds up.
    """

    def __init__(self, comm: CommunicationModule) -> None:
        super().__init__(comm)

    @abstractmethod
    def _source(self) -> Any: ...

    @abstractmethod
    def _sink(self) -> Any: ...

    @abstractmethod
    def _process(self, item: InT) -> OutT: ...

    @abstractmethod
    def _path_of(self, item: InT) -> Path: ...

    def _step(self) -> None:
        try:
            item = self._source().pop(timeout=0.1)
        except Empty:
            return                              # re-check stop flag
        try:
            out = self._process(item)
        except Exception as exc:                # one bad scan must not stop the batch
            path = self._path_of(item)
            logger.exception("%s failed on %s", type(self).__name__, path.name)
            self._push(self._comm.results, SegmentationResult.failed(path, exc))
            return
        self._push(self._sink(), out)
