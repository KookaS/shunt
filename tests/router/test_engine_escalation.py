"""C: end-to-end auto-escalation through RouterEngine.decide() + record_outcome.

Proves the LIVE behaviour: after two verified same-check failures on a task, the next
decision for that task returns a strictly-higher-tier model. Uses fakes only (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig


@dataclass
class _M:
    name: str


class _TieredPool:
    """Models across real tiers: cheap=qwen, mid=glm, high=opus (all healthy)."""

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
    def __init__(self, tool_identity: str = "toolA") -> None:
        self._tool = tool_identity

    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity=self._tool)


class _Index:
    """Cold-start inactive, empty neighborhood → base selection is the cheapest (qwen)."""

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

    def query(self, embedding: np.ndarray, k: int = 20) -> list:
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


def _engine(*, enabled: bool) -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=enabled, escalate_after_n=2),
        # B3: the decide-side task key is the repo (resolved work_dir), matching the key the
        # record_outcome calls below use ("repoA") — not the client tool_identity.
        task_key_resolver=lambda _session: "repoA",
    )


def _fail(engine: RouterEngine, key: str = "t::a") -> None:
    # exit_code=1 (a real pytest red — NOT the hook contract's 2) with blocking+confirmed set,
    # proving the off-wire path escalates on the verified outcome, not the exit code (B1/B10).
    engine.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=key,
        exit_code=1,
        blocking=True,
        confirmed=True,
    )


def test_two_same_key_failures_escalate_the_next_decision() -> None:
    eng = _engine(enabled=True)
    m1, r1, _ = eng.decide("s1", "do the task")
    assert m1 == "qwen"  # base pick, cheapest
    _fail(eng)
    eng.decide("s2", "same task")  # holds — one failure only
    _fail(eng)
    m3, r3, prov = eng.decide("s3", "same task")
    assert m3 == "glm"  # escalated one tier: cheap → mid
    assert r3 == "auto_escalation"
    assert prov["tier_escalation_reason"] == "same_verified_failure_x2"


def test_distinct_failures_do_not_escalate() -> None:
    eng = _engine(enabled=True)
    eng.decide("s1", "task")
    _fail(eng, "t::a")
    eng.decide("s2", "task")
    _fail(eng, "t::b")  # different check
    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # distinct failures never aggregate


def test_success_retires_the_failure_log() -> None:
    eng = _engine(enabled=True)
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    # a verified pass on the task clears its stuck history
    eng.record_outcome(downshift=False, success=True, task_key="repoA", dedup_key=None, exit_code=0)
    _fail(eng)  # one fresh failure after the fix
    m3, _, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # only one live failure after the reset


def test_disabled_never_escalates() -> None:
    eng = _engine(enabled=False)
    eng.decide("s1", "task")
    _fail(eng)  # no-op when disabled (task_key path guarded)
    _fail(eng)
    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"
    assert r3 != "auto_escalation"
