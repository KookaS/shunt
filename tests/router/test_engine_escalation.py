"""C: end-to-end auto-escalation through RouterEngine.decide() + record_outcome.

Proves the LIVE behaviour: after two verified same-check failures on a task, the next
decision for that task returns a strictly-higher-tier model. Uses fakes only (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig

_AFTER_N = 2  # the threshold every engine test below configures; the reason string carries it


@dataclass
class _M:
    name: str


class _RankedPool:
    """Models across ranks: qwen < glm < opus (weakest -> strongest, all healthy)."""

    def __init__(self) -> None:
        self._ranked = [_M("qwen"), _M("glm"), _M("opus")]

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


class _ColdIndex(_Index):
    """Cold-start ACTIVE: too few effective verified outcomes for the router to be choosing."""

    def effective_labeled(self) -> float:
        return 1.0

    def effective_tier2(self) -> float:
        return 0.0


class _ColdStartPool(_RankedPool):
    """Models named so the cold-start default (deepseek-v4-flash) is a rankable member."""

    def __init__(self) -> None:
        self._ranked = [_M("deepseek-v4-flash"), _M("glm"), _M("opus")]


def _engine(*, enabled: bool, epsilon: float = 0.0) -> RouterEngine:
    return RouterEngine(
        model_pool=_RankedPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(
            enabled=enabled,
            escalate_after_n=2,
            exploration_epsilon=epsilon,
            exploration_seed=1234,
        ),
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
        is_infra_failure=False,
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
    assert prov["rank_escalation_reason"] == f"same_verified_failure_x{_AFTER_N}"


def test_cold_start_is_not_suppressed_by_the_collapse_alarm() -> None:
    # During cold start the router defaults every session to the cheap model — a degenerate
    # choice distribution BY DESIGN, so the routing-collapse alarm (entropy=0 on one model)
    # would fire and globally suppress auto-escalation exactly when verified failures first
    # accumulate. The collapse guard is a policy-ossification detector; it must abstain while
    # the router is not choosing at all (found by the live cold-start smoke).
    eng = RouterEngine(
        model_pool=_ColdStartPool(),  # deepseek-v4-flash is the cold-start default AND rankable
        session_manager=_SessionManager(),
        outcome_index=_ColdIndex(),  # cold-start ACTIVE
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        task_key_resolver=lambda _session: "repoA",
        loop_health_alarm=lambda: True,  # would suppress — but cold start disables the gate
    )
    _fail(eng)
    eng.decide("s2", "same task")
    _fail(eng)
    m3, r3, prov = eng.decide("s3", "same task")
    assert r3 == "auto_escalation"  # cold-start degenerate choices must NOT suppress escalation
    assert m3 != "deepseek-v4-flash"
    assert prov["rank_escalation_reason"] == f"same_verified_failure_x{_AFTER_N}"


def test_warm_degenerate_distribution_still_suppresses_escalation() -> None:
    # The gate relaxes ONLY during cold start: once the router has enough verified outcomes to
    # be making real choices, a degenerate choice distribution (here forced via the alarm
    # probe) must still suppress escalation.
    eng = RouterEngine(
        model_pool=_RankedPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),  # warm → cold-start INACTIVE
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        task_key_resolver=lambda _session: "repoA",
        loop_health_alarm=lambda: True,
    )
    _fail(eng)
    eng.decide("s2", "same task")
    _fail(eng)
    _m3, r3, _prov = eng.decide("s3", "same task")
    assert r3 != "auto_escalation"  # collapse suppression applies once the router is choosing


def test_infra_reds_do_not_escalate_via_the_shared_constructor() -> None:
    # An env/collection red (is_infra_failure=True) is not a capability failure: the shared
    # failure-event constructor derives blocking=False, so two of them never escalate.
    eng = _engine(enabled=True)
    for sid in ("s1", "s2"):
        eng.decide(sid, "same task")
        eng.record_outcome(
            downshift=False,
            success=False,
            task_key="repoA",
            dedup_key="t::a",
            exit_code=1,
            is_infra_failure=True,
            confirmed=True,
        )
    m3, r3, _ = eng.decide("s3", "same task")
    assert m3 == "qwen"  # infra reds never count → base pick held
    assert r3 != "auto_escalation"


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


def test_exploration_only_ever_withholds_an_escalation_never_invents_one() -> None:
    # Cache-safety under randomization: the explored arm is HOLD, so every decision is either
    # the escalation the deterministic policy would have made, or the base pick. There is no
    # third model, hence no switch the deterministic path would not also have made.
    # Stated as a rank invariant rather than a model literal: the ladder now CLIMBS across
    # sessions (the rank floor), so pinning the set to {qwen, glm} would only assert
    # that the cascade is still broken. One rung at a time, never backwards, is the property.
    ranks = _RankedPool()
    arms: set[str] = set()
    eng = _engine(enabled=True, epsilon=0.5)
    previous = 0
    for i in range(60):
        eng.decide(f"a{i}", "task")
        _fail(eng)
        eng.decide(f"b{i}", "task")
        _fail(eng)
        model, reason, provenance = eng.decide(f"c{i}", "task")
        rank = ranks.rank_of(model)
        assert rank is not None
        # No invented switch: at most ONE rung above where the task already stood, and never
        # below it. A third model the deterministic path would not have reached is excluded.
        assert previous <= rank <= previous + 1
        if rank == previous:  # explored HOLD -> the base decision is untouched
            assert reason != "auto_escalation"
        previous = rank
        record = provenance.get("escalation_exploration")
        if record is None:  # unflagged step — not a decision an estimator may weight
            continue
        if record["policy_action"] == "hold":
            # At the rank ceiling there is no rung to take, so there is no arm to randomize
            # over — exploration must not fabricate one, and must not claim a propensity.
            assert record["randomized"] is False
            assert record["propensity"] == 1.0
        else:
            assert record["randomized"] is True
            assert 0.0 < record["propensity"] < 1.0
            arms.add(str(record["action"]))
    assert arms == {"hold", "raise_rank"}, "both arms must actually be realized at epsilon=0.5"


class _TopUnhealthyPool(_RankedPool):
    """Headroom exists on paper, but every higher-rank model is circuit-broken."""

    def is_healthy(self, name: str) -> bool:
        return name == "qwen"


def test_an_escalation_that_cannot_be_delivered_is_not_logged_as_one() -> None:
    # The rung is unavailable (every higher model unhealthy), so the escalate arm did NOT
    # happen. Logging it as taken would corrupt exactly the arm the estimate is about.
    eng = RouterEngine(
        model_pool=_TopUnhealthyPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(
            enabled=True, escalate_after_n=2, exploration_epsilon=0.2, exploration_seed=7
        ),
        task_key_resolver=lambda _session: "repoA",
    )
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    _fail(eng)
    model, reason, provenance = eng.decide("s3", "task")
    assert model == "qwen"  # nothing was delivered
    assert reason != "auto_escalation"
    record = provenance["escalation_exploration"]
    assert record["action"] == "hold"
    assert record["randomized"] is False  # carries no counterfactual information
    assert record["propensity"] == 1.0
