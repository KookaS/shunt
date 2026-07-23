"""B6: a failure's staleness window measures decisions-SINCE-ROUTED, not since-capture."""

# The decision index is stamped onto the routing decision's provenance and carried back through
# capture, so interleaved sessions advancing the counter don't keep an old failure fresh.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.capture.coordinator import _decision_index_of
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig


@dataclass
class _M:
    name: str


class _TieredPool:
    def __init__(self) -> None:
        self._tiers = {"cheap": [_M("qwen")], "mid": [_M("glm")], "high": [], "frontier": []}

    def get_tier_models(self, tier: str) -> list[_M]:
        return self._tiers.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return True


class _SessionManager:
    def get_session(self, session_id: str) -> object:
        return object()


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

    def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


def _engine(*, stale_window: int) -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, stale_window=stale_window),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine, *, decision_index: int, key: str = "t::a") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=key,
        exit_code=1,
        blocking=True,
        confirmed=True,
        decision_index=decision_index,
    )


def test_stale_stamped_failure_ages_out_by_decisions_since_routed() -> None:
    eng = _engine(stale_window=3)
    for i in range(5):  # advance the per-task decision counter to 5
        eng.decide(f"s{i}", "task")
    # One failure was ROUTED long ago (index 0), one recently (index 4). Under decisions-since-
    # routed the old one is stale at the next decision (index 5), so only one live failure remains.
    _fail(eng, decision_index=0)
    _fail(eng, decision_index=4)
    m, r, _ = eng.decide("s5", "task")
    assert m == "qwen"  # only one in-window failure → below escalate_after_n
    assert r != "auto_escalation"


def test_two_recently_routed_failures_still_escalate() -> None:
    eng = _engine(stale_window=3)
    for i in range(5):
        eng.decide(f"s{i}", "task")
    _fail(eng, decision_index=3)  # both routed recently → both in window at index 5
    _fail(eng, decision_index=4)
    m, r, _ = eng.decide("s5", "task")
    assert m == "glm"  # two in-window same-key failures → escalate
    assert r == "auto_escalation"


def test_decision_index_extracted_from_provenance_row() -> None:
    import json

    row = {"decision_provenance": json.dumps({"decision_index": 7, "downshift": False})}
    assert _decision_index_of(row) == 7
    assert _decision_index_of({"decision_provenance": None}) is None
    assert _decision_index_of(None) is None
