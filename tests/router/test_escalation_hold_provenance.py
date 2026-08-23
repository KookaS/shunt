"""Every held escalation records WHY, with a token from the documented hold vocabulary."""

# A HOLD leaves the served model untouched, so without `escalation_hold_reason` a held boundary
# is indistinguishable from one where escalation never ran — the breakdown is underivable, not
# merely empty. Fakes only; no I/O.

from __future__ import annotations

from typing import Any

from shunt.models.config import ReasoningConfig
from shunt.router.engine import RouterEngine
from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    ExplorationStream,
    FailureEvent,
    decide_escalation,
)
from tests.router.escalation_fakes import (
    Embedder,
    Index,
    RankedReasoningPool,
    SessionManager,
    reasoning_ladder,
)

# The published hold vocabulary. Written out here on purpose: `decide_escalation` builds these
# as literals, so comparing the engine's provenance to its own directive would let a rename pass
# silently. Membership is cross-checked against the live function below.
HOLD_REASONS = frozenset(
    {
        "disabled",
        "collapse_suppressed",
        "no_recurring_failure",
        "escalation_ceiling",
        "exploration_hold",
    }
)


def _ladders(top_has_arms: bool = False) -> dict[str, ReasoningConfig | None]:
    top = reasoning_ladder("nothink", "think") if top_has_arms else None
    return {"qwen": reasoning_ladder("low", "high"), "glm": top}


def _engine(
    config: EscalationConfig,
    *,
    ladders: dict[str, ReasoningConfig | None] | None = None,
    alarm: bool = False,
) -> RouterEngine:
    return RouterEngine(
        model_pool=RankedReasoningPool(ladders if ladders is not None else _ladders()),
        session_manager=SessionManager(),
        outcome_index=Index(),  # warm → cold start inactive, so the alarm is not suppressed
        embedder=Embedder(),
        escalation=config,
        task_key_resolver=lambda _session: "repoA",
        loop_health_alarm=(lambda: True) if alarm else None,
    )


def _fail(eng: RouterEngine, dedup_key: str = "t::a") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=dedup_key,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _held(eng: RouterEngine, session_id: str) -> dict[str, Any]:
    """Route once and return the provenance, asserting the decision really was a hold."""
    _model, reason, provenance = eng.decide(session_id, "same task")
    assert reason != "auto_escalation"
    return provenance


def _assert_hold_reason(provenance: dict[str, Any], expected: str) -> None:
    assert expected in HOLD_REASONS
    assert provenance["escalation_hold_reason"] == expected


def test_disabled_config_never_reaches_the_hold_branch_at_all() -> None:
    # `disabled` is unreachable through the ENGINE: a disabled config makes `_task_key` return
    # None, so `_maybe_escalate` — and therefore `decide_escalation` — never runs and there is no
    # hold to explain. The token is emitted only on the direct/offline path, covered by the
    # vocabulary test below. Pinning the absence here stops a future refactor from routing a
    # disabled router through the branch and flooding the breakdown with a non-decision.
    prov = _held(_engine(EscalationConfig(enabled=False, escalate_after_n=2)), "s0")
    assert "escalation_hold_reason" not in prov


def test_no_recurring_failure_records_the_hold_reason() -> None:
    eng = _engine(EscalationConfig(enabled=True, escalate_after_n=2))
    _fail(eng)  # one failure only — below the threshold
    _assert_hold_reason(_held(eng, "s1"), "no_recurring_failure")


def test_collapse_suppressed_records_the_hold_reason() -> None:
    # Two same-key verified reds WOULD escalate; the routing-collapse alarm suppresses first.
    eng = _engine(EscalationConfig(enabled=True, escalate_after_n=2), alarm=True)
    _fail(eng)
    eng.decide("s1", "same task")
    _fail(eng)
    _assert_hold_reason(_held(eng, "s2"), "collapse_suppressed")


def test_escalation_ceiling_records_the_hold_reason() -> None:
    # A single-model pool with no reasoning arms: recurrence flags the checkpoint, but neither
    # the effort nor the rank axis has headroom, so the ladder holds instead of thrashing.
    eng = _engine(
        EscalationConfig(enabled=True, escalate_after_n=2),
        ladders={"qwen": None},
    )
    _fail(eng)
    eng.decide("s1", "same task")
    _fail(eng)
    _assert_hold_reason(_held(eng, "s2"), "escalation_ceiling")


def test_exploration_hold_records_the_hold_reason() -> None:
    # epsilon just under 1.0 with a fixed seed: the flagged checkpoint is randomized onto the
    # HOLD arm, which is a hold the estimator must be able to tell apart from the others.
    eng = _engine(
        EscalationConfig(
            enabled=True,
            escalate_after_n=2,
            exploration_epsilon=0.99,
            exploration_seed=7,
        )
    )
    _fail(eng)
    eng.decide("s1", "same task")
    _fail(eng)
    prov = _held(eng, "s2")
    _assert_hold_reason(prov, "exploration_hold")
    # The explored hold still carries its propensity — the two records are complementary.
    assert prov["escalation_exploration"]["action"] == "hold"


def test_every_hold_token_the_decider_emits_is_in_the_published_vocabulary() -> None:
    # Binds the literals above to the live decision function: a renamed token fails here rather
    # than silently passing a test that compares the engine to its own directive.
    capped = EscalationContext(
        current_rank_index=0, max_rank_index=0, current_effort_index=0, max_effort_index=0
    )
    headroom = EscalationContext(
        current_rank_index=0, max_rank_index=1, current_effort_index=0, max_effort_index=1
    )
    alarmed = EscalationContext(
        current_rank_index=0,
        max_rank_index=1,
        current_effort_index=0,
        max_effort_index=1,
        loop_health_alarm=True,
    )
    on = EscalationConfig(enabled=True, escalate_after_n=2)
    explore = EscalationConfig(
        enabled=True, escalate_after_n=2, exploration_epsilon=0.99, exploration_seed=7
    )
    events = [
        FailureEvent(
            decision_index=i,
            dedup_key="t::a",
            exit_code=1,
            success=False,
            confirmed=True,
            blocking=True,
        )
        for i in range(2)
    ]
    directives = [
        decide_escalation([], 0, capped, EscalationConfig(enabled=False)),
        decide_escalation([], 0, alarmed, on),
        decide_escalation([], 0, headroom, on),
        decide_escalation(events, 2, capped, on),
        decide_escalation(events, 2, headroom, explore, ExplorationStream.from_seed(7)),
    ]
    assert all(d.action is EscalationAction.HOLD for d in directives)
    assert {d.reason for d in directives} == HOLD_REASONS


def test_the_fire_path_is_untouched() -> None:
    # The escalation FIRE path keeps writing `rank_escalation_reason` and grows no hold key.
    eng = _engine(EscalationConfig(enabled=True, escalate_after_n=2))
    _fail(eng)
    eng.decide("s1", "same task")
    _fail(eng)
    model, reason, prov = eng.decide("s2", "same task")
    assert reason == "auto_escalation"
    assert model == "qwen"  # effort rung first (cache-safe), same model
    assert prov["rank_escalation_reason"] == "same_verified_failure_x2"
    assert prov["auto_escalated"] is True
    assert "escalation_hold_reason" not in prov
