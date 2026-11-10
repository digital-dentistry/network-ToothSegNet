from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.types import InferredScan, PreparedScan, Scan, SegmentationResult

from .buffers import BlockingRingBuffer, Channel

# Scans in flight between two stages.  Each holds a few hundred MB at most
# (a million-vertex scan with its TTA geometry), so two per hop keeps every
# thread busy without letting a fast stage run far ahead of a slow one.
_HOP = 2


@dataclass
class CommunicationModule:
    scan_paths: Channel[Path] = field(default_factory=Channel)
    loaded: BlockingRingBuffer[Scan] = field(default_factory=lambda: BlockingRingBuffer(_HOP))
    prepared: BlockingRingBuffer[PreparedScan] = field(default_factory=lambda: BlockingRingBuffer(_HOP))
    inferred: BlockingRingBuffer[InferredScan] = field(default_factory=lambda: BlockingRingBuffer(_HOP))
    # every path ends here exactly once, segmented or failed
    results: Channel[SegmentationResult] = field(default_factory=Channel)

    def _buffers(self) -> tuple:
        return (self.scan_paths, self.loaded, self.prepared, self.inferred, self.results)

    def clear(self) -> None:
        """Call before the workflow threads start."""
        for buf in self._buffers():
            buf.clear()

    def close(self) -> None:
        """Call before the workflow threads stop."""
        for buf in self._buffers():
            buf.close()
