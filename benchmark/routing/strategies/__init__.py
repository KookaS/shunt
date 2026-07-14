from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str: ...
