"""Concurrent same-task escalation is serialized by the engine lock.

Interleaved record_outcome/decide on ONE task_key must neither tear the failure log
(no lost append) nor double-fire escalation (the retire is atomic under the lock).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationConfig


@dataclass
class _M:
    name: str


class _RankedPool:
    def __init__(self) -> None:
        self._ranked = [_M("qwen"), _M("glm"), _M("opus")]  # weakest -> strongest

    def ranked_models(self) -> list[_M]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        for i, m in enumerate(self._ranked):
            if m.name == name:
                return i
        return None

    def models_from_rank(self, i: int) -> list[_M]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        return True


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

    def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


def _engine(escalate_after_n: int) -> RouterEngine:
    return RouterEngine(
        model_pool=_RankedPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=escalate_after_n),
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


def test_concurrent_same_key_failures_all_accrue_no_lost_update() -> None:
    # 30 is under the log's growth cap, so a torn read-modify-write on the append is the
    # only way the count would fall short.
    n = 30
    eng = _engine(escalate_after_n=n + 1)  # not yet due, so nothing is retired mid-burst
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _i: _fail(eng), range(n)))

    log = eng.snapshot_escalation_state()["failure_log"][task_state_key("repoA")]
    assert len(log) == n  # every concurrent append landed exactly once


def test_concurrent_decides_escalate_at_most_once_for_the_same_evidence() -> None:
    eng = _engine(escalate_after_n=2)
    _fail(eng)
    _fail(eng)  # exactly two reds are now due — enough for ONE escalation

    n = 32
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda i: eng.decide(f"s{i}", "task"), range(n)))

    escalated = [reason for _m, reason, _prov in results if reason == "auto_escalation"]
    assert len(escalated) == 1  # the first decision retires the window; the rest see none
    log = eng.snapshot_escalation_state()["failure_log"]
    assert log.get(task_state_key("repoA"), []) == []
