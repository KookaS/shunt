"""Multi-rung tier climb, boundary-only application, and post-escalation retire.

Drives the REAL RouterEngine (decide + record_outcome), not a hand-built directive,
so the live tier ceiling math and retire semantics are exercised end to end. Fakes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig
from shunt.router.selection import NeighborResult


@dataclass
class _M:
    name: str


class _TieredPool:
    """cheap=qwen, mid=glm, high=opus, frontier empty — all healthy."""

    def __init__(self) -> None:
        self._tiers = {
            "cheap": [_M("qwen")],
            "mid": [_M("glm")],
            "high": [_M("opus")],
            "frontier": [],
        }

    def get_tier_models(self, tier: str) -> list[_M]:
        return self._tiers.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return True


@dataclass
class _Session:
    tool_identity: str


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity="toolA")


class _ClimbingIndex:
    """Neighborhood that returns whichever model the test currently pins as the base pick.

    Models the real feedback loop: an escalated, served model becomes the task's
    kNN-preferred base for the next decision, so the ladder can climb rung by rung.
    """

    def __init__(self) -> None:
        self.base = "qwen"

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

    def query(self, embedding: np.ndarray, k: int = 20) -> list[NeighborResult]:  # type: ignore[type-arg]
        return [
            NeighborResult(
                model=self.base,
                outcome=True,
                cost=1.0,
                verification_confidence=0.9,
                distance=0.1,
                session_id="t",
                truncation_rate=0.0,
            )
            for _ in range(5)
        ]


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


def _engine(index: _ClimbingIndex | None = None) -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=index if index is not None else _ClimbingIndex(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="tier_only"),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine) -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key="t::a",
        exit_code=1,
        blocking=True,
        confirmed=True,
    )


def _two_reds_then_decide(eng: RouterEngine, tag: str) -> tuple[str, str]:
    eng.decide(f"a{tag}", "task")
    _fail(eng)
    eng.decide(f"b{tag}", "task")
    _fail(eng)
    model, reason, _prov = eng.decide(f"c{tag}", "task")
    return model, reason


def test_ladder_climbs_one_rung_per_recurrence_then_holds_at_ceiling() -> None:
    index = _ClimbingIndex()
    eng = _engine(index)

    m1, r1 = _two_reds_then_decide(eng, "0")
    assert (m1, r1) == ("glm", "auto_escalation")  # cheap -> mid

    index.base = "glm"  # the escalated model is now the task's working base
    m2, r2 = _two_reds_then_decide(eng, "1")
    assert (m2, r2) == ("opus", "auto_escalation")  # mid -> high, one rung

    index.base = "opus"  # at the top tier; frontier is empty
    m3, r3 = _two_reds_then_decide(eng, "2")
    assert m3 == "opus"  # ceiling: no strictly-higher tier, model unchanged
    assert r3 != "auto_escalation"  # held, not escalated off the top


def test_record_outcome_does_not_mutate_the_served_decision_mid_turn() -> None:
    # Cache-safety spine: a verified failure only queues the NEXT boundary's directive; it
    # never switches the model or effort arm of an already-served turn.
    eng = _engine()
    served, _reason, _prov = eng.decide("s0", "task")
    assert served == "qwen"

    assert _fail(eng) is None  # record_outcome yields no decision
    assert _fail(eng) is None
    state = eng.snapshot_escalation_state()
    assert state["effort_arm"] == {}  # no mid-turn effort switch was applied
    assert len(state["failure_log"]["repoA"]) == 2  # the reds were only queued


def test_no_reescalation_off_already_consumed_reds() -> None:
    eng = _engine()
    _m, reason = _two_reds_then_decide(eng, "0")
    assert reason == "auto_escalation"  # the two reds fired one escalation

    # The immediately-following decision has no NEW verified failure — the consumed window
    # was retired, so it must HOLD rather than re-fire off the same evidence.
    _m2, reason2, _prov2 = eng.decide("d0", "task")
    assert reason2 != "auto_escalation"
