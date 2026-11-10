from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Generic, TypeVar

from interface.strategy import IStrategy

InT = TypeVar("InT")
OutT = TypeVar("OutT")
TypeT = TypeVar("TypeT", bound=Enum)


class IAlgorithm(ABC, Generic[InT, OutT, TypeT]):
    @abstractmethod
    def set_strategy(self, strategy_type: TypeT) -> None: ...

    @abstractmethod
    def execute(self, inp: InT, out: OutT) -> bool: ...


class StrategyAlgorithm(IAlgorithm[InT, OutT, TypeT]):
    """The factory / cache / dispatch part every algorithm module shares.

    A subclass only supplies the factories.  Each factory imports its strategy
    module when it is first selected, so a GPU strategy is never imported on a
    machine that does not ask for it.
    """

    def __init__(self, factories: dict[TypeT, Callable[[], IStrategy[InT, OutT]]]) -> None:
        self._factories = factories
        self._strategies: dict[TypeT, IStrategy[InT, OutT]] = {}
        self._current: IStrategy[InT, OutT] | None = None
        self._current_type: TypeT | None = None

    @property
    def strategy_type(self) -> TypeT | None:
        return self._current_type

    def set_strategy(self, strategy_type: TypeT) -> None:
        if strategy_type not in self._factories:
            raise ValueError(f"Unsupported strategy: {strategy_type}")
        if strategy_type not in self._strategies:
            self._strategies[strategy_type] = self._factories[strategy_type]()
        self._current = self._strategies[strategy_type]
        self._current_type = strategy_type

    def execute(self, inp: InT, out: OutT) -> bool:
        if self._current is None:
            raise RuntimeError("set_strategy() must be called before execute()")
        return self._current.execute(inp, out)
