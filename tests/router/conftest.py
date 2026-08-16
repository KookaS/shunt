from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class FakeModel:
    name: str


class FakeModelPool:
    """Minimal ModelPoolProtocol fake: models given weakest -> strongest (rank order)."""

    def __init__(self, *names: str) -> None:
        self._ranked: list[FakeModel] = [FakeModel(n) for n in names]
        self._healthy: set[str] = set(names)

    def ranked_models(self) -> list[FakeModel]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        for i, m in enumerate(self._ranked):
            if m.name == name:
                return i
        return None

    def models_from_rank(self, i: int) -> list[FakeModel]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        return name in self._healthy


@pytest.fixture
def pool():
    return FakeModelPool
