from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from queue import Full
from typing import TYPE_CHECKING, Any

from communication.buffers import BufferClosed

if TYPE_CHECKING:
    from communication.communication_module import CommunicationModule

logger = logging.getLogger(__name__)

_POLL_S = 0.1


class Workflow(ABC):
    def __init__(self, comm: CommunicationModule) -> None:
        self._comm = comm                       # non-owning reference
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name=type(self).__name__, daemon=False
            )
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._step()
        except BufferClosed:
            pass                                # normal shutdown path
        except Exception:
            logger.exception("%s stopped by an error", type(self).__name__)

    def _push(self, buffer: Any, value: Any) -> bool:
        """Push, re-checking the stop flag every 0.1 s instead of blocking on a
        full buffer forever.  False when stopped before the push went through."""
        while not self._stop_event.is_set():
            try:
                buffer.push(value, timeout=_POLL_S)
                return True
            except Full:
                continue
        return False

    @abstractmethod
    def _step(self) -> None:
        """One iteration of work. Must not block forever."""
