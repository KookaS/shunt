"""Regression tests for the Tranche-0 auto-escalation bug fixes (B1-B4, B7, B8, B10)."""

# Each test targets one confirmed bug. The load-bearing one is
# ``test_realistic_verifier_failure_escalates_end_to_end`` — the decorrelated coordinator →
# record_outcome → decide_escalation drive that a two-agent audit said would have caught B1
# (no unit test hardcoding exit_code=2 could). Fakes only; no I/O, no live keys.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from shunt.capture.coordinator import CaptureCoordinator, WorkDirResolver
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig
from shunt.session import Session
from shunt.verifiers.base import VerifierResult
from shunt.verifiers.rerun import RerunConfirmingVerifier
from shunt.verifiers.tier2 import _failing_check_id


@dataclass
class _M:
    name: str


class _TieredPool:
    """cheap=qwen, mid=glm, high=opus — all healthy."""

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


def _engine(
    *,
    resolver_key: str = "repoA",
    alarm: bool = False,
    session_manager: _SessionManager | None = None,
) -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=session_manager or _SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        task_key_resolver=lambda _s: resolver_key,
        loop_health_alarm=(lambda: alarm),
    )


# ── B1 (CRITICAL): the mandatory decorrelated real-verifier drive ──────────────
class _ScriptedVerifier:
    """Returns a scripted result each call — stands in for AutoDetectVerifier (never runs)."""

    def __init__(self, result: VerifierResult) -> None:
        self._result = result

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        return self._result


def _store_stub() -> object:
    """Minimal OutcomeStore surface CaptureCoordinator._append_tier2 touches (fresh insert)."""

    class _Row(dict):  # type: ignore[type-arg]
        pass

    class _Store:
        def get_session(self, session_id: str) -> dict | None:  # type: ignore[type-arg]
            return _Row(model_fingerprint=None, decision_provenance=None)

        def append_outcome_event(self, event: object) -> bool:
            return True  # a fresh insert → the record_outcome callback fires

        def persist_index(self) -> None:
            return None

    return _Store()


def _closed_session(sid: str = "s1", tool_identity: str = "toolA") -> Session:
    now = datetime.now(UTC)
    sess = Session(session_id=sid, tool_identity=tool_identity, start_time=now)
    sess.end_time = now
    return sess


def test_realistic_verifier_failure_escalates_end_to_end() -> None:
    """B1: a real VerifierResult(outcome=failure, exit_code=1) drives escalation after 2 reds."""
    # Exit code 1 is a genuine pytest red, NOT the hook contract's 2 — under the old
    # ``exit_code == 2`` gate this never counted and the feature was dead. Wrapped in a real
    # RerunConfirmingVerifier so ``confirmed`` is set structurally (B10), keyed on the repo
    # work_dir on both the decide and capture sides (B3).
    eng = _engine(resolver_key="/repo")
    # A real failure the off-wire verifier would emit: subprocess exit 1, a pytest node id.
    fail = VerifierResult(
        outcome="failure",
        confidence=0.7,
        failing_check_id="tests/test_x.py::test_y",
        exit_code=1,
    )
    verifier = RerunConfirmingVerifier(_ScriptedVerifier(fail), reruns=2)
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=verifier,
        store=_store_stub(),  # type: ignore[arg-type]
        record_outcome_callback=eng.record_outcome,
    )

    m1, _, _ = eng.decide("s1", "do the task")
    assert m1 == "qwen"  # base pick
    coord.capture(_closed_session("s1"))  # 1st verified failure
    eng.decide("s2", "same task")  # still holds after one
    coord.capture(_closed_session("s2"))  # 2nd verified same-key failure

    m3, r3, _ = eng.decide("s3", "same task")
    assert m3 == "glm"  # escalated cheap → mid on the SUBPROCESS-exit-1 failures
    assert r3 == "auto_escalation"


# ── B2: a live routing-collapse alarm suppresses escalation ────────────────────
def _fail(eng: RouterEngine, key: str = "t::a", task_key: str = "repoA") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key=task_key,
        dedup_key=key,
        exit_code=1,
        blocking=True,
        confirmed=True,
    )


def test_collapse_alarm_suppresses_escalation_live() -> None:
    eng = _engine(alarm=True)
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    _fail(eng)
    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # suppressed: escalating into a routing collapse voids the cost gate
    assert r3 != "auto_escalation"


def test_no_alarm_still_escalates() -> None:
    eng = _engine(alarm=False)  # control: same drive without the alarm DOES escalate
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    _fail(eng)
    m3, _, _ = eng.decide("s3", "task")
    assert m3 == "glm"


# ── B3: two different repos with the same test node id do NOT aggregate ─────────
def test_cross_repo_same_node_id_does_not_aggregate() -> None:
    eng = RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        # decide-side key follows the session — here always /repoA (one client, but the point
        # is the CAPTURE side keys by repo, so /repoB's failures never bleed into /repoA).
        task_key_resolver=lambda _s: "/repoA",
    )
    eng.decide("s1", "task")
    _fail(eng, "tests/test_x.py::test_y", task_key="/repoA")  # repoA, once
    eng.decide("s2", "task")
    _fail(eng, "tests/test_x.py::test_y", task_key="/repoB")  # SAME node id, different repo
    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # repoA has only ONE failure — the repoB red did not aggregate in
    assert r3 != "auto_escalation"


# ── B4: escalation state survives a snapshot/restore round-trip ────────────────
def test_escalation_state_round_trips() -> None:
    eng = _engine()
    eng.decide("s1", "task")
    _fail(eng)  # one accrued failure for repoA
    state = eng.snapshot_escalation_state()
    assert state["failure_log"]["repoA"]  # the log serialized

    restored = _engine()
    restored.restore_escalation_state(state)
    # A restart that restored the log needs only ONE more same-key red to escalate — the first
    # was not wiped.
    restored.decide("s2", "task")
    _fail(restored)
    m3, r3, _ = restored.decide("s3", "task")
    assert m3 == "glm"
    assert r3 == "auto_escalation"


def test_snapshot_empty_when_disabled() -> None:
    eng = RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=False),
        task_key_resolver=lambda _s: "repoA",
    )
    assert eng.snapshot_escalation_state() == {}


# ── B7: an escalated turn's provenance is marked non-policy ─────────────────────
def test_escalated_provenance_is_non_policy() -> None:
    eng = _engine()
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    _fail(eng)
    _m, _r, prov = eng.decide("s3", "task")
    assert prov["auto_escalated"] is True
    assert prov["new_label_window"] is True
    # The base model's propensity/scores are neutralized so the learner never trains the
    # escalated model as a free policy choice.
    assert prov["router_propensity"] is None
    assert prov["candidate_model_scores"] == {}


# ── B8: a recurring go/rust failure with only timing/address noise hashes stably ─
def test_dedup_key_stable_across_timing_noise() -> None:
    run1 = (
        "--- FAIL: TestWidget (0.42s)\n"
        "    widget_test.go:31: got 0x7ffee3b2a1c0, want match\n"
        "FAIL\texample/pkg\t0.512s\n"
    )
    run2 = (
        "--- FAIL: TestWidget (1.07s)\n"
        "    widget_test.go:31: got 0x55a9f0e12340, want match\n"
        "FAIL\texample/pkg\t1.998s\n"
    )
    assert _failing_check_id(run1) == _failing_check_id(run2)


def test_dedup_key_distinguishes_genuinely_different_failures() -> None:
    a = "panic: index out of range [5] with length 3\n"
    b = "panic: nil pointer dereference\n"
    assert _failing_check_id(a) != _failing_check_id(b)


# ── B10: a bare (non-confirming) verifier failure is NOT trusted as confirmed ───
def test_unconfirmed_failure_does_not_escalate() -> None:
    """A single-run red (confirmed=False) never counts — the flake guard is structural."""
    eng = _engine()
    for i in (1, 2):
        eng.decide(f"s{i}", "task")
        eng.record_outcome(
            downshift=False,
            success=False,
            task_key="repoA",
            dedup_key="t::a",
            exit_code=1,
            blocking=True,
            confirmed=False,  # a bare AutoDetectVerifier never sets this
        )
    m3, r3, _ = eng.decide("s3", "task")
    assert m3 == "qwen"  # unconfirmed reds are abstained → no escalation
    assert r3 != "auto_escalation"
