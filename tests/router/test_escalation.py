"""Unit tests for the auto-escalation decision (pure logic — no I/O, no live keys)."""

from __future__ import annotations

from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    FailureEvent,
    counts_as_failure,
    decide_escalation,
)


def _fail(
    idx: int,
    key: str = "test::foo",
    *,
    exit_code: int = 1,
    confirmed: bool = True,
    blocking: bool = True,
) -> FailureEvent:
    # exit_code defaults to 1 (a real pytest/jest red) to prove it no longer gates escalation;
    # `blocking` is the gate the CaptureCoordinator sets from the verified outcome.
    return FailureEvent(
        decision_index=idx,
        dedup_key=key,
        exit_code=exit_code,
        success=False,
        confirmed=confirmed,
        blocking=blocking,
    )


def _ctx(
    *, tier: int = 0, max_tier: int = 3, effort: int = 2, max_effort: int = 2, alarm: bool = False
) -> EscalationContext:
    return EscalationContext(
        current_tier_index=tier,
        max_tier_index=max_tier,
        current_effort_index=effort,
        max_effort_index=max_effort,
        loop_health_alarm=alarm,
    )


_ON = EscalationConfig(enabled=True)


# ── AC-A1: two same-key verified failures → escalate, reason recorded ──────────
def test_same_key_twice_escalates() -> None:
    events = [_fail(0), _fail(1)]
    # effort already at ceiling → the ladder steps tier
    d = decide_escalation(events, current_index=2, ctx=_ctx(effort=2, max_effort=2), config=_ON)
    assert d.action is EscalationAction.RAISE_TIER
    assert d.reason == "same_verified_failure_x2"


# ── AC-A2: a single failure, or two DIFFERENT keys, does NOT escalate ──────────
def test_single_failure_holds() -> None:
    d = decide_escalation([_fail(0)], current_index=1, ctx=_ctx(), config=_ON)
    assert d.action is EscalationAction.HOLD
    assert d.reason == "no_recurring_failure"


def test_two_different_keys_hold() -> None:
    events = [_fail(0, "test::a"), _fail(1, "test::b")]
    d = decide_escalation(events, current_index=2, ctx=_ctx(), config=_ON)
    assert d.action is EscalationAction.HOLD


# ── AC-A3 / non-blocking (lint / infra) not counted ─────────
def test_non_blocking_failures_not_counted() -> None:
    events = [_fail(0, blocking=False), _fail(1, blocking=False)]
    d = decide_escalation(events, current_index=2, ctx=_ctx(), config=_ON)
    assert d.action is EscalationAction.HOLD
    assert not counts_as_failure(events[0], _ON)


# ── B1 regression: a real subprocess exit code (1, 101 …) STILL counts when blocking ──
def test_blocking_failure_counts_regardless_of_exit_code() -> None:
    # The old code gated on exit_code == 2; a real pytest red is exit 1 → was dropped. Prove a
    # blocking, confirmed failure counts at exit 1 (pytest/jest) AND 101 (cargo).
    for code in (1, 101):
        events = [_fail(0, exit_code=code), _fail(1, exit_code=code)]
        d = decide_escalation(events, current_index=2, ctx=_ctx(effort=2, max_effort=2), config=_ON)
        assert d.action is EscalationAction.RAISE_TIER, f"exit {code} should escalate"
        assert counts_as_failure(events[0], _ON)


# ── AC-A4 / flake guard: unconfirmed failure not counted ──────────────────────
def test_unconfirmed_failures_not_counted() -> None:
    events = [_fail(0, confirmed=False), _fail(1, confirmed=False)]
    d = decide_escalation(events, current_index=2, ctx=_ctx(), config=_ON)
    assert d.action is EscalationAction.HOLD


def test_success_never_counts() -> None:
    ok = FailureEvent(
        decision_index=0, dedup_key="test::foo", exit_code=0, success=True, confirmed=True
    )
    assert not counts_as_failure(ok, _ON)


# ── AC-A5: escalation opens a new, non-policy label window ─────────────────────
def test_escalation_opens_new_label_window() -> None:
    d = decide_escalation(
        [_fail(0), _fail(1)], current_index=2, ctx=_ctx(effort=2, max_effort=2), config=_ON
    )
    assert d.new_label_window is True


# ── effort-then-tier ladder: effort first (cache-safe), then tier ─────────
def test_ladder_raises_effort_before_tier() -> None:
    # effort below ceiling → the cache-safe effort rung is taken first
    d = decide_escalation(
        [_fail(0), _fail(1)], current_index=2, ctx=_ctx(effort=0, max_effort=2), config=_ON
    )
    assert d.action is EscalationAction.RAISE_EFFORT


def test_tier_only_ladder_skips_effort() -> None:
    cfg = EscalationConfig(enabled=True, ladder="tier_only")
    d = decide_escalation(
        [_fail(0), _fail(1)], current_index=2, ctx=_ctx(effort=0, max_effort=2), config=cfg
    )
    assert d.action is EscalationAction.RAISE_TIER


# ── at the ceiling → hold, don't thrash ────────────────────────────────
def test_at_ceiling_holds() -> None:
    ctx = _ctx(tier=3, max_tier=3, effort=2, max_effort=2)
    d = decide_escalation([_fail(0), _fail(1)], current_index=2, ctx=ctx, config=_ON)
    assert d.action is EscalationAction.HOLD
    assert d.reason == "escalation_ceiling"


# ── routing-collapse alarm suppresses escalation ───────────────────────
def test_collapse_alarm_suppresses() -> None:
    ctx = _ctx(alarm=True)
    d = decide_escalation([_fail(0), _fail(1)], current_index=2, ctx=ctx, config=_ON)
    assert d.action is EscalationAction.HOLD
    assert d.reason == "collapse_suppressed"


# ── staleness window (the "+10 calls" idea): a non-recurring failure retires ───
def test_stale_failure_retired() -> None:
    # two same-key failures but the first is older than stale_window from current_index
    events = [_fail(0), _fail(15)]
    cfg = EscalationConfig(enabled=True, stale_window=10)
    d = decide_escalation(events, current_index=16, ctx=_ctx(), config=cfg)
    assert d.action is EscalationAction.HOLD  # only one live failure → no recurrence


def test_recurrence_within_window_escalates() -> None:
    events = [_fail(7), _fail(15)]  # gaps 9 and 1 from index 16, both < stale_window=10
    cfg = EscalationConfig(enabled=True, stale_window=10)
    d = decide_escalation(events, current_index=16, ctx=_ctx(effort=2, max_effort=2), config=cfg)
    assert d.action is EscalationAction.RAISE_TIER


# ── AC-A "default OFF": disabled config always holds (kill-gate discipline) ────
def test_disabled_config_always_holds() -> None:
    default = EscalationConfig()  # enabled defaults False
    assert default.enabled is False
    d = decide_escalation([_fail(0), _fail(1)], current_index=2, ctx=_ctx(), config=default)
    assert d.action is EscalationAction.HOLD
    assert d.reason == "disabled"


# ── no failure events at all (no tests / polite retry) → hold ─────────
def test_no_events_holds() -> None:
    d = decide_escalation([], current_index=5, ctx=_ctx(), config=_ON)
    assert d.action is EscalationAction.HOLD
    assert d.reason == "no_recurring_failure"


# Progress guard note: a verified suite pass retires failures at the ENGINE, which pops the
# whole per-task failure log (suite-level capture ⇒ all keys resolved) — see
# ``test_success_retires_the_failure_log`` in test_engine_escalation.py. The escalation logic
# therefore never sees a success event mixed into a key's failures, so there is no per-key
# success-retirement path here to test.


# ── the config-file schema and the pure-logic config stay in lockstep ──────────
def test_policy_to_config_bridge() -> None:
    from shunt.router.policy import EscalationPolicy

    cfg = EscalationPolicy(enabled=True, escalate_after_n=3, ladder="tier_only").to_config()
    assert cfg == EscalationConfig(enabled=True, escalate_after_n=3, ladder="tier_only")
