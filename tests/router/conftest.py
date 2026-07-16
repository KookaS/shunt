from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class FakeModel:
    name: str


class FakeModelPool:
    def __init__(self, *names: str) -> None:
        self.models: dict[str, list[FakeModel]] = {
            "cheap": [],
            "mid": [],
            "frontier": [],
        }
        self._healthy: set[str] = set()
        for name in names:
            self.models["cheap"].append(FakeModel(name))
            self._healthy.add(name)

    def get_tier_models(self, tier: str) -> list[FakeModel]:
        return self.models.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return name in self._healthy


@pytest.fixture
def pool():
    return FakeModelPool
