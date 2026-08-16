"""A restored rank floor above the pool's top rank must clamp, not go silently dead.

`_apply_rank` never clamped the floor to the pool, so a floor restored into a smaller
pool made `models_from_rank(floor)` return `[]` forever and re-pick the cheapest model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from shunt.models.config import ModelConfig
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig


def _cfg(name: str) -> ModelConfig:
    return ModelConfig(name=name, provider="p", base_url="http://x", api_key_env_var="K")


class _Pool:
    """Ranked models, weakest -> strongest, all healthy."""

    def __init__(self, names: list[str]) -> None:
        self._models = {n: _cfg(n) for n in names}
        self._ranked = [self._models[n] for n in names]

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def ranked_models(self) -> list[ModelConfig]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        for i, m in enumerate(self._ranked):
            if m.name == name:
                return i
        return None

    def models_from_rank(self, i: int) -> list[ModelConfig]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        return name in self._models


@dataclass
class _Session:
    tool_identity: str


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity="toolA")


class _Index:
    def count_labeled(self) -> int:
        return 100

    def count_total_labeled(self) -> int:
        return 100

    def effective_labeled(self) -> float:
        return 100.0

    def effective_tier2(self) -> float:
        return 100.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: np.ndarray, k: int = 20) -> list[Any]:
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


def _engine(names: list[str]) -> RouterEngine:
    return RouterEngine(
        model_pool=_Pool(names),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="rank_only"),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine) -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key="t::a",
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _two_reds_then_decide(eng: RouterEngine, tag: str) -> str:
    eng.decide(f"a{tag}", "task")
    _fail(eng)
    eng.decide(f"b{tag}", "task")
    _fail(eng)
    model, _reason, _prov = eng.decide(f"c{tag}", "task")
    return model


def test_floor_restored_above_a_shrunk_pool_clamps_and_still_holds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wide = _engine(["m0", "m1", "m2"])
    assert _two_reds_then_decide(wide, "0") == "m1"  # one rung
    assert _two_reds_then_decide(wide, "1") == "m2"  # a second recurrence: the top rank
    state = wide.snapshot_escalation_state()
    assert state["rank_floor"], "the climbed floor is persisted"
    assert max(state["rank_floor"].values()) == 2

    shrunk = _engine(["m0", "m1"])  # the registry lost its top model across the restart
    with caplog.at_level(logging.INFO, logger="shunt.router.engine"):
        shrunk.restore_escalation_state(state)

    assert any("clamp" in r.getMessage().lower() for r in caplog.records), (
        "a floor that had to be clamped must say so — a silently dead floor is invisible"
    )

    # Every subsequent session holds the top of the SHRUNK pool. Pre-fix this returned
    # the cheapest model, and the ladder oscillated m0 -> m1 -> m0 forever.
    assert [shrunk.decide(f"s{i}", "task")[0] for i in range(3)] == ["m1", "m1", "m1"]
