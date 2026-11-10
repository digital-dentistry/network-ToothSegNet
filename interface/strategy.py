from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class IStrategy(ABC, Generic[InT, OutT]):
    @abstractmethod
    def execute(self, inp: InT, out: OutT) -> bool:
        """Fill `out` from `inp`. Return True on success."""
