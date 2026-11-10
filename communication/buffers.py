from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Full
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class BufferClosed(Exception):
    """Push to a closed buffer, or pop from a closed and drained buffer."""


class BlockingRingBuffer(Generic[T]):
    """Fixed capacity. push() blocks while full, pop() blocks while empty."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            ok = self._not_full.wait_for(
                lambda: self._closed or len(self._items) < self._capacity, timeout
            )
            if not ok:
                raise Full
            if self._closed:
                raise BufferClosed
            self._items.append(value)
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            value = self._items.popleft()
            self._not_full.notify()
            return value

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False
            self._not_full.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class OverwriteRingBuffer(Generic[T]):
    """Fixed capacity. push() never blocks; the oldest item is dropped when full."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[T] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise BufferClosed
            self._items.append(value)           # deque(maxlen) drops the oldest
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            return self._items.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class Channel(Generic[T]):
    """Unbounded queue. pop() blocks until data arrives or the channel is closed."""

    def __init__(self) -> None:
        self._items: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise BufferClosed
            self._items.append(value)
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            return self._items.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False

    def __iter__(self) -> Iterator[T]:
        """Yield items until the channel is closed and drained."""
        while True:
            try:
                yield self.pop()
            except BufferClosed:
                return

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
