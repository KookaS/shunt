"""Per-lane scheduler: pure admission windows, 429 quarantine, and the pull-loop re-drain.

Hand-written cases pin each window and each transition; a Hypothesis state machine drives
interleaved admit/complete/429/recover/refill sequences and asserts the invariants the
campaign depends on. Zero model calls, zero I/O outside the persistence round-trip.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from benchmark.runner import lane_scheduler as ls

_LANES = ("a", "b")
# a: 2 cells/min, 100 calls/day -> one 60-call reservation fits, the second does not.
# b: 3 cells/min, 300 calls/day -> three 80-call reservations fit.
_LIMITS: Final[dict[str, ls.LaneLimits]] = {
    "a": ls.LaneLimits(rpm=2, rpd=100),
    "b": ls.LaneLimits(rpm=3, rpd=300),
}
_RESERVES: Final[dict[str, int]] = {"a": 60, "b": 80}


def _scheduler(**limits_overrides: ls.LaneLimits) -> ls.LaneScheduler:
    limits = dict(_LIMITS)
    limits.update(limits_overrides)
    return ls.LaneScheduler(limits=limits, reserves=dict(_RESERVES), stall_timeout_s=900.0)


# ── UNKNOWN limits and backoff ────────────────────────────────────────────────


def test_unknown_limits_resolve_to_the_declared_defaults() -> None:
    limits = ls.LaneLimits()
    assert limits.rpm is None and limits.rpd is None
    assert limits.effective_rpm == ls.UNKNOWN_RPM == 10
    assert limits.effective_rpd == ls.UNKNOWN_RPD == 100


def test_p90_calls_is_nearest_rank_and_never_zero() -> None:
    assert ls.p90_calls([]) == 1
    assert ls.p90_calls([5]) == 5
    assert ls.p90_calls([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert ls.p90_calls([0, 0]) == 1


def test_backoff_is_monotone_and_capped() -> None:
    values = [ls.backoff_seconds(n) for n in range(0, 20)]
    assert values == sorted(values)
    assert values[-1] == ls.BACKOFF_CAP_S == 3600.0
    assert ls.backoff_seconds(0) == 30.0


def test_quarantine_honours_retry_after_over_jitter() -> None:
    # rng draw 0.0 makes the jittered bound 0, so only Retry-After can move the deadline.
    until = ls.quarantine_until(100.0, 0, retry_after=120.0, rng=random.Random(0))
    assert until == 220.0


def test_quarantine_full_jitter_never_exceeds_the_capped_base() -> None:
    now = 1_000.0
    for attempt in range(10):
        until = ls.quarantine_until(now, attempt, rng=random.Random(attempt))
        assert now <= until <= now + ls.backoff_seconds(attempt)


# ── admission windows ─────────────────────────────────────────────────────────


def test_rpm_window_blocks_then_refills() -> None:
    sched = _scheduler(a=ls.LaneLimits(rpm=2, rpd=1000))
    sched.reserves["a"] = 1
    assert sched.admit(("c1", "a", "default"), now=0.0)
    assert sched.admit(("c2", "a", "default"), now=0.0)
    assert not sched.admit(("c3", "a", "default"), now=0.0)  # 2 == rpm
    assert sched.admit(("c3", "a", "default"), now=61.0)  # the first minute aged out


def test_rpd_window_charges_the_reservation_not_one() -> None:
    sched = _scheduler()
    assert sched.admit(("c1", "a", "default"), now=0.0)
    assert not sched.admit(("c2", "a", "default"), now=0.0)  # 60 + 60 > 100
    assert sched.lane_state("a").day.total(0.0, ls.DAY_S) == 60


def test_complete_reconciles_the_reservation_to_the_rows_calls() -> None:
    sched = _scheduler()
    cell = ("c1", "a", "default")
    assert sched.admit(cell, now=0.0)
    assert sched.lane_state("a").day.total(0.0, ls.DAY_S) == 60
    sched.complete(cell, now=5.0, calls=7)
    assert sched.lane_state("a").day.total(5.0, ls.DAY_S) == 7


def test_complete_reflects_actual_calls_above_the_reservation() -> None:
    # The day counter is ACTUAL consumption: an over-issuing cell is charged what it issued,
    # not silently capped at the reservation.
    sched = _scheduler()
    cell = ("c1", "a", "default")
    assert sched.admit(cell, now=0.0)
    sched.complete(cell, now=1.0, calls=150)  # reserved 60, actual 150 > rpd 100
    assert sched.lane_state("a").day.total(1.0, ls.DAY_S) == 150


def test_bucket_reconcile_returns_the_replaced_reservation() -> None:
    bucket = ls.Bucket()
    bucket.reserve("k", 0.0, 60)
    assert bucket.reconcile("k", 7) == 60
    assert bucket.reconcile("absent", 1) is None


def test_overshoot_blocks_further_same_day_admissions_until_the_day_rolls() -> None:
    # rpd=100, reserve=10: admit two cells; truing the first up to 90 puts the day total at
    # 100 >= rpd, so the lane is blocked by a NAMED state for the rest of the day — even after
    # the second cell reconciles back DOWN below rpd (arithmetic alone would re-admit it).
    sched = _scheduler(a=ls.LaneLimits(rpm=100, rpd=100))
    sched.reserves["a"] = 10
    first, second = ("c1", "a", "default"), ("c2", "a", "default")
    assert sched.admit(first, now=0.0)
    assert sched.admit(second, now=0.0)
    sched.complete(first, now=1.0, calls=90)  # 90 + 10 == rpd -> block
    state = sched.lane_state("a")
    assert state.day_blocked_reason is not None
    assert "RPD" in state.day_blocked_reason
    assert not sched.can_admit("a", now=2.0)
    sched.complete(second, now=3.0, calls=0)  # day total drops to 90 < rpd...
    assert sched.lane_state("a").day.total(3.0, ls.DAY_S) == 90
    assert not sched.can_admit("a", now=3.0)  # ...but the named block still holds
    # The block self-clears only once every charge has aged out of the trailing day.
    assert sched.can_admit("a", now=3.0 + ls.DAY_S)


def test_daily_token_budget_blocks_an_over_budget_cell() -> None:
    sched = _scheduler(c=ls.LaneLimits(rpm=10, rpd=1000, daily_token_budget=500))
    sched.reserves["c"] = 1
    assert not sched.admit(("c1", "c", "default"), now=0.0, tokens=501)
    assert sched.admit(("c1", "c", "default"), now=0.0, tokens=400)


# ── the limitation registry: one rule, every provider ─────────────────────────


def test_tpm_token_bucket_sums_actual_tokens_in_the_trailing_minute() -> None:
    limits = ls.LaneLimits(rpm=100, rpd=1_000, tpm=10_000, max_request_tokens=4_000)
    sched = _scheduler(a=limits)
    sched.reserves["a"] = 1
    first = ("c1", "a", "default")
    assert sched.admit(first, now=0.0)
    sched.complete(first, now=1.0, calls=1, tokens=6_001)
    # 6,001 actual + the 4,000-token single request > 10,000 TPM in the same minute.
    assert not sched.can_admit("a", now=2.0)
    assert "TPM" in (sched.refusal_reason("a", now=2.0) or "")
    # The whole bucket ages out of the trailing minute.
    assert sched.can_admit("a", now=1.0 + ls.MINUTE_S)


def test_daily_token_budget_sums_actual_tokens_and_rolls_after_a_day() -> None:
    limits = ls.LaneLimits(rpm=100, rpd=1_000, daily_token_budget=1_000, max_request_tokens=100)
    sched = _scheduler(a=limits)
    sched.reserves["a"] = 1
    cell = ("c1", "a", "default")
    assert sched.admit(cell, now=0.0)
    sched.complete(cell, now=1.0, calls=1, tokens=950)
    assert not sched.can_admit("a", now=2.0)  # 950 actual + 100 required > 1,000
    assert sched.can_admit("a", now=1.0 + ls.DAY_S)


def test_max_request_gate_refuses_a_lane_that_can_never_fit_tpm() -> None:
    # Groq's measured free shape: an ~8026-token SWE-bench turn against an 8000 TPM cap.
    limits = ls.LaneLimits(rpm=30, rpd=1_000, tpm=8_000, max_request_tokens=8_026)
    reason = ls.structural_refusal(limits)
    assert reason is not None and ls.LANE_TPM_TOO_SMALL in reason
    sched = _scheduler(a=limits)
    sched.reserves["a"] = 1
    assert not sched.can_admit("a", now=0.0)
    assert sched.structural_refusals() == {"a": reason}
    # A request that fits the minute is not structurally refused.
    assert ls.structural_refusal(ls.LaneLimits(tpm=8_000, max_request_tokens=8_000)) is None


def test_is_available_agrees_with_can_admit_on_the_structural_max_request() -> None:
    # A lane whose single request can NEVER fit its TPM is structurally unusable, so the health
    # view must say unavailable — not report healthy while admission always refuses.
    limits = ls.LaneLimits(rpm=30, rpd=1_000, tpm=8_000, max_request_tokens=8_026)
    sched = ls.LaneScheduler(limits={"a": limits}, reserves={"a": 1})
    assert not sched.is_available("a", now=0.0)
    assert not sched.can_admit("a", now=0.0)
    # A lane whose request does fit stays available and admissible.
    ok = ls.LaneScheduler(
        limits={"b": ls.LaneLimits(rpm=30, rpd=1_000, tpm=8_000, max_request_tokens=8_000)},
        reserves={"b": 1},
    )
    assert ok.is_available("b", now=0.0)
    assert ok.can_admit("b", now=0.0)


def test_no_free_access_lane_is_structurally_refused_with_its_named_reason() -> None:
    # OpenCode Zen's shape: a `-free` id that the API gates behind the app/session.
    limits = ls.LaneLimits(
        rpm=10, rpd=100, free_access=False, access_note="app/session-gated (MissingSessionID)"
    )
    reason = ls.structural_refusal(limits)
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason
    assert "MissingSessionID" in reason
    sched = _scheduler(a=limits)
    sched.reserves["a"] = 1
    assert not sched.can_admit("a", now=0.0)
    assert not sched.is_available("a", now=0.0)
    assert sched.structural_refusals() == {"a": reason}
    # A lane without the marker is admitted normally.
    assert ls.structural_refusal(ls.LaneLimits(rpm=10, rpd=100)) is None


def test_structurally_refused_lane_is_skipped_without_a_stall(monkeypatch) -> None:
    from benchmark.runner import run_matrix

    executed: list[ls.Cell] = []
    monkeypatch.setattr(run_matrix, "_run_one_cell", lambda cell, ctx: executed.append(cell))
    scheduler = ls.LaneScheduler(
        limits={"groq-lane": ls.LaneLimits(rpm=30, rpd=1_000, tpm=8_000, max_request_tokens=8_026)},
        stall_timeout_s=0.01,
    )
    rows, spent, stopped = run_matrix._run_scheduled_batch(
        [("c1", "groq-lane", "default")],
        run_matrix._LiveContext.__new__(run_matrix._LiveContext),
        scheduler,
        run_matrix._FailureTracker(None, None),
        None,
        0.0,
        None,
        None,
        "",
    )
    assert rows == [] and spent == 0.0 and stopped is False
    assert executed == []


def _state(**fields: object) -> ls.LaneState:
    return ls.LaneState(**fields)  # type: ignore[arg-type]


# TEMPLATE: one context per registered limitation that MUST trip it. Adding a limitation to
# ``LIMITATIONS`` without adding its name here fails ``test_registry_names_and_cases_agree``
# (set mismatch) and ``test_each_registered_limitation_blocks`` (KeyError) — the test is the
# enforcement. To add a rule: append it to ``LIMITATIONS``, then add a case below.
_LIMITATION_BLOCK_CASES: Final[dict[str, tuple[ls.LaneLimits, ls.LaneState, float, int]]] = {
    "disabled": (ls.LaneLimits(rpm=10, rpd=100), _state(disabled_reason="operator"), 0.0, 0),
    "expired": (
        ls.LaneLimits(expires_at="1970-01-01T00:00:00+00:00"),
        ls.LaneState(),
        0.0,
        0,
    ),
    "day_blocked": (
        ls.LaneLimits(rpm=10, rpd=100),
        _state(day_blocked_reason="rpd", day_blocked_at=0.0),
        0.0,
        0,
    ),
    "quarantined": (ls.LaneLimits(rpm=10, rpd=100), _state(quarantined_until=100.0), 0.0, 0),
    "free_access": (
        ls.LaneLimits(free_access=False, access_note="app/session-gated"),
        ls.LaneState(),
        0.0,
        0,
    ),
    "max_request_tokens": (
        ls.LaneLimits(tpm=8_000, max_request_tokens=8_026),
        ls.LaneState(),
        0.0,
        0,
    ),
    "rpm": (ls.LaneLimits(rpm=0, rpd=100), ls.LaneState(), 0.0, 0),
    "rpd": (ls.LaneLimits(rpm=100, rpd=0), ls.LaneState(), 0.0, 0),
    "tpm": (ls.LaneLimits(tpm=1), ls.LaneState(), 0.0, 2),
    "daily_token_budget": (ls.LaneLimits(daily_token_budget=1), ls.LaneState(), 0.0, 2),
}


def test_registry_names_and_cases_agree() -> None:
    assert {limitation.name for limitation in ls.LIMITATIONS} == set(_LIMITATION_BLOCK_CASES)


@pytest.mark.parametrize("limitation", ls.LIMITATIONS, ids=lambda item: item.name)
def test_each_registered_limitation_blocks(limitation: ls.Limitation) -> None:
    limits, state, now, tokens = _LIMITATION_BLOCK_CASES[limitation.name]
    assert limitation.evaluate(ls.LimitContext(limits, state, now, tokens, reserve=1)) is not None


@given(
    used=st.integers(min_value=0, max_value=5),
    rpm=st.integers(min_value=0, max_value=5),
)
def test_rpm_blocks_exactly_at_the_limit(used: int, rpm: int) -> None:
    sched = ls.LaneScheduler(limits={"a": ls.LaneLimits(rpm=rpm, rpd=10**9)})
    for index in range(used):
        sched.lane_state("a").minute.reserve(f"c{index}", 0.0, 1)
    assert sched.can_admit("a", now=1.0) == (used + 1 <= rpm)


@given(
    reserved=st.integers(min_value=0, max_value=200),
    reserve=st.integers(min_value=1, max_value=100),
    rpd=st.integers(min_value=0, max_value=500),
)
def test_rpd_blocks_exactly_when_the_reservation_does_not_fit(
    reserved: int, reserve: int, rpd: int
) -> None:
    sched = ls.LaneScheduler(
        limits={"a": ls.LaneLimits(rpm=1000, rpd=rpd)}, reserves={"a": reserve}
    )
    sched.lane_state("a").day.reserve("c0", 0.0, reserved)
    assert sched.can_admit("a", now=1.0) == (reserved + reserve <= rpd)


@given(
    charges=st.lists(st.integers(min_value=0, max_value=5_000), max_size=8),
    charge=st.integers(min_value=0, max_value=5_000),
    tpm=st.integers(min_value=0, max_value=20_000),
)
def test_tpm_blocks_exactly_when_the_trailing_token_sum_exceeds_tpm(
    charges: list[int], charge: int, tpm: int
) -> None:
    limits = ls.LaneLimits(rpm=1_000, rpd=10**9, tpm=tpm, max_request_tokens=0)
    sched = ls.LaneScheduler(limits={"a": limits})
    for index, value in enumerate(charges):
        sched.lane_state("a").token_minute.reserve(f"c{index}", 0.0, value)
    assert sched.can_admit("a", now=1.0, tokens=charge) == (sum(charges) + charge <= tpm)


@given(
    charges=st.lists(st.integers(min_value=0, max_value=5_000), max_size=8),
    charge=st.integers(min_value=0, max_value=5_000),
    budget=st.integers(min_value=0, max_value=20_000),
)
def test_daily_token_budget_blocks_exactly_when_the_day_sum_exceeds_it(
    charges: list[int], charge: int, budget: int
) -> None:
    limits = ls.LaneLimits(rpm=1_000, rpd=10**9, daily_token_budget=budget)
    sched = ls.LaneScheduler(limits={"a": limits})
    for index, value in enumerate(charges):
        sched.lane_state("a").token_day.reserve(f"c{index}", 0.0, value)
    assert sched.can_admit("a", now=1.0, tokens=charge) == (sum(charges) + charge <= budget)


@given(
    tpm=st.integers(min_value=0, max_value=100_000),
    request=st.integers(min_value=0, max_value=100_000),
)
def test_max_request_gate_is_exactly_request_exceeding_tpm(tpm: int, request: int) -> None:
    reason = ls.structural_refusal(ls.LaneLimits(tpm=tpm, max_request_tokens=request))
    assert (reason is not None) == (request > tpm)
    if reason is not None:
        assert ls.LANE_TPM_TOO_SMALL in reason


def test_expired_lane_is_never_admitted() -> None:
    sched = _scheduler(a=ls.LaneLimits(rpm=10, rpd=1000, expires_at="1970-01-01T00:00:00+00:00"))
    assert not sched.can_admit("a", now=0.0)


# ── quarantine ────────────────────────────────────────────────────────────────


def test_a_quarantined_lane_is_never_admitted_and_recovers_after_expiry() -> None:
    sched = _scheduler()
    sched.record_rate_limit("a", now=0.0, rng=random.Random(0))
    assert not sched.admit(("c1", "a", "default"), now=0.0)
    assert not sched.is_available("a", now=0.0)
    until = sched.lane_state("a").quarantined_until
    assert until is not None
    sched.recover("a", now=until)
    assert sched.is_available("a", now=until)
    assert sched.admit(("c1", "a", "default"), now=until)


def test_consecutive_429s_back_off_monotonically() -> None:
    sched = _scheduler()
    delays = []
    now = 0.0
    for _ in range(6):
        sched.record_rate_limit("a", now=now, rng=random.Random(0))
        until = sched.lane_state("a").quarantined_until
        assert until is not None
        delays.append(until - now)
        now = until  # advance to expiry, then rate-limit again with no success
    assert delays == sorted(delays)
    assert max(delays) <= ls.BACKOFF_CAP_S


def test_a_completion_resets_the_429_streak() -> None:
    sched = _scheduler()
    cell = ("c1", "a", "default")
    sched.record_rate_limit("a", now=0.0, rng=random.Random(0))
    assert sched.lane_state("a").consecutive_429 == 1
    assert sched.admit(cell, now=10_000.0)
    sched.complete(cell, now=10_001.0, calls=1)
    assert sched.lane_state("a").consecutive_429 == 0


# ── the pull loop ─────────────────────────────────────────────────────────────


def test_next_ready_stalls_when_every_lane_is_blocked() -> None:
    sched = _scheduler(a=ls.LaneLimits(rpm=1, rpd=1), b=ls.LaneLimits(rpm=1, rpd=1))
    sched.reserves["a"] = 1
    sched.reserves["b"] = 1
    pending = [("c1", "a", "default"), ("c2", "b", "default")]
    assert sched.admit(("c1", "a", "default"), now=0.0)
    assert sched.admit(("c2", "b", "default"), now=0.0)
    assert isinstance(sched.next_ready(pending, now=0.0), ls.Stalled)


def test_next_ready_is_fair_round_robin_across_lanes() -> None:
    sched = _scheduler(a=ls.LaneLimits(rpm=100, rpd=10_000), b=ls.LaneLimits(rpm=100, rpd=10_000))
    sched.reserves["a"] = 1
    sched.reserves["b"] = 1
    pending = [("c1", "a", "default"), ("c2", "b", "default")]
    picks = []
    for index in range(6):
        cell = sched.next_ready(pending, now=float(index))
        assert not isinstance(cell, ls.Stalled)
        assert sched.admit(cell, now=float(index))
        picks.append(cell[1])
    assert picks == ["a", "b", "a", "b", "a", "b"]


# ── persistence ───────────────────────────────────────────────────────────────


def test_lane_state_round_trips_through_corpus_lock(tmp_path) -> None:
    sched = _scheduler()
    sched.admit(("c1", "a", "default"), now=0.0)
    sched.complete(("c1", "a", "default"), now=1.0, calls=3)
    sched.record_rate_limit("b", now=2.0, rng=random.Random(0))
    path = tmp_path / "free-tier" / "lane_state.json"
    ls.save_lane_state(dict(sched.state), path)
    loaded = ls.load_lane_state(path)
    assert loaded == sched.state
    assert loaded["a"].day.total(1.0, ls.DAY_S) == 3
    assert loaded["b"].consecutive_429 == 1


def test_missing_state_file_loads_empty(tmp_path) -> None:
    assert ls.load_lane_state(tmp_path / "absent.json") == {}


# ── the state machine ─────────────────────────────────────────────────────────


class LaneSchedulerMachine(RuleBasedStateMachine):
    """Interleaves admissions, completions, 429s, recovery and the passage of time."""

    def __init__(self) -> None:
        super().__init__()
        self.sched = _scheduler()
        self.now = 1_000.0
        self.pending: list[ls.Cell] = []
        self.inflight: dict[str, tuple[ls.Cell, int]] = {}
        # lane -> key -> (admission time, charge). Holds the reservation until the cell
        # completes, then the row's ACTUAL calls — the oracle for the day-counter invariant.
        self.charges: dict[str, dict[str, tuple[float, int]]] = {}
        self._next = 0

    def _new_cell(self, lane: str) -> ls.Cell:
        self._next += 1
        return (f"c{self._next}", lane, "default")

    @rule(lane=st.sampled_from(_LANES))
    def enqueue(self, lane: str) -> None:
        self.pending.append(self._new_cell(lane))

    @rule()
    def admit(self) -> None:
        cell = self.sched.next_ready(self.pending, self.now)
        if isinstance(cell, ls.Stalled):
            return
        assert self.sched.admit(cell, self.now)
        self.pending.remove(cell)
        reserved = self.sched.reserve_for(cell[1])
        self.inflight[ls.cell_key(cell)] = (cell, reserved)
        self.charges.setdefault(cell[1], {})[ls.cell_key(cell)] = (self.now, reserved)

    @rule(calls=st.integers(min_value=0, max_value=200))
    def complete(self, calls: int) -> None:
        if not self.inflight:
            return
        key = sorted(self.inflight)[0]
        cell, _reserved = self.inflight.pop(key)
        self.sched.complete(cell, self.now, calls=calls)
        at = self.charges[cell[1]][key][0]
        self.charges[cell[1]][key] = (at, max(0, calls))

    @rule(lane=st.sampled_from(_LANES), seed=st.integers(min_value=0, max_value=10_000))
    def rate_limit(self, lane: str, seed: int) -> None:
        self.sched.record_rate_limit(lane, self.now, rng=random.Random(seed))

    @rule(lane=st.sampled_from(_LANES))
    def recover(self, lane: str) -> None:
        self.sched.recover(lane, self.now)

    @rule(delta=st.floats(min_value=0.0, max_value=4_000.0, allow_nan=False))
    def refill(self, delta: float) -> None:
        self.now += delta

    @invariant()
    def rpm_is_never_exceeded(self) -> None:
        for lane in _LANES:
            state = self.sched.lane_state(lane)
            assert (
                state.minute.total(self.now, ls.MINUTE_S)
                <= self.sched.limits_for(lane).effective_rpm
            )

    @invariant()
    def day_counter_equals_sum_of_actuals(self) -> None:
        # The day bucket holds reservations while in flight and ACTUAL calls once complete; it
        # is never re-capped. This is the invariant the old `min(calls, reserved)` cap hid.
        for lane in _LANES:
            state = self.sched.lane_state(lane)
            expected = sum(
                charge
                for at, charge in self.charges.get(lane, {}).values()
                if self.now - at < ls.DAY_S
            )
            assert state.day.total(self.now, ls.DAY_S) == expected

    @invariant()
    def admission_respects_the_day_budget(self) -> None:
        # No admission is granted when day_total + the cell's reservation would exceed rpd.
        for lane in _LANES:
            if not self.sched.can_admit(lane, self.now):
                continue
            state = self.sched.lane_state(lane)
            assert self.sched.is_available(lane, self.now)
            assert (
                state.day.total(self.now, ls.DAY_S) + self.sched.reserve_for(lane)
                <= self.sched.limits_for(lane).effective_rpd
            )

    @invariant()
    def a_day_block_blocks_same_day_admission(self) -> None:
        for lane in _LANES:
            state = self.sched.lane_state(lane)
            if state.day_blocked_reason is not None and self.sched._day_block_active(
                state, self.now
            ):
                assert not self.sched.can_admit(lane, self.now)
                assert "RPD" in state.day_blocked_reason

    @invariant()
    def a_quarantine_blocks_admission(self) -> None:
        for lane in _LANES:
            state = self.sched.lane_state(lane)
            if state.quarantined_until is not None and self.now < state.quarantined_until:
                assert not self.sched.can_admit(lane, self.now)

    @invariant()
    def backoff_stays_monotone_and_capped(self) -> None:
        values = [ls.backoff_seconds(n) for n in range(12)]
        assert values == sorted(values) and values[-1] <= ls.BACKOFF_CAP_S

    @invariant()
    def a_ready_pending_cell_is_never_starved(self) -> None:
        if self.pending and any(self.sched.can_admit(c[1], self.now) for c in self.pending):
            assert not isinstance(self.sched.next_ready(self.pending, self.now), ls.Stalled)


TestLaneScheduler = LaneSchedulerMachine.TestCase
TestLaneScheduler.settings = settings(
    max_examples=60, stateful_step_count=30, deadline=timedelta(seconds=2)
)


@pytest.mark.parametrize("lane", _LANES)
def test_limits_for_unknown_lane_is_fully_declared(lane: str) -> None:
    sched = ls.LaneScheduler()
    assert sched.limits_for(lane).rpm is None
    assert sched.reserve_for(lane) == 1


def test_adapt_probe_limits_uses_the_canonical_type() -> None:
    from benchmark.runner.free_lane_probe import LaneLimits as ProbeLimits

    adapted = ls.adapt_probe_limits(ProbeLimits(rpm=30, rpd=1000, known=True))
    assert adapted == ls.LaneLimits(rpm=30, rpd=1000)
    assert isinstance(adapted, ls.LaneLimits)


# ── wiring: the runner must actually construct and persist a scheduler ─────────


def test_run_full_builds_passes_and_persists_a_lane_scheduler(monkeypatch) -> None:
    """WIRING: with a `lanes:` block, `_run_full` constructs one scheduler, passes it to the
    live path, and loads/saves the day buckets.

    This is the regression guard for the defect where ``_run_full`` never passed ``lanes=``,
    so every lane-admission branch was dead. The run path is monkeypatched — no model calls.
    """
    import argparse
    from pathlib import Path

    from benchmark import config
    from benchmark.routing import integrity
    from benchmark.runner import run_matrix, swebench_specs

    cell = ("c1", "lane-a", "default")
    captured: dict[str, object] = {}
    loaded: list[bool] = []
    saved: list[dict] = []

    def _record_load(*_a: object, **_k: object) -> dict:
        loaded.append(True)
        return {}

    def _record_save(state: dict, *_a: object, **_k: object) -> None:
        saved.append(state)

    monkeypatch.setattr(
        config,
        "lanes_config",
        lambda: {
            "unknown_limits": {"rpm": 10, "rpd": 100},
            "stall_timeout_s": 900,
            "limits": {"lane-a": {"rpm": 10, "rpd": 100}},
        },
    )
    monkeypatch.setattr(ls, "load_lane_state", _record_load)
    monkeypatch.setattr(ls, "save_lane_state", _record_save)

    monkeypatch.setattr(config, "challenges_path", lambda: Path("challenges.json"))
    monkeypatch.setattr(config, "load_matrix", lambda *a, **k: {})
    monkeypatch.setattr(config, "enabled_models", lambda: ["lane-a"])
    monkeypatch.setattr(config, "register_collection_models", lambda names: None)
    monkeypatch.setattr(config, "load_results", lambda: {})
    monkeypatch.setattr(config, "sample_tasks", lambda tasks, seed=42: ["c1"])
    monkeypatch.setattr(config, "concordance_subset_models", lambda: set())
    monkeypatch.setattr(config, "models_missing_cache", lambda models: [])
    monkeypatch.setattr(config, "results_csv_path", lambda: Path("results_free.csv"))
    monkeypatch.setattr(swebench_specs, "manifest_source", lambda: "swebench_verified")
    monkeypatch.setattr(swebench_specs, "spec_module_for", lambda source: object())
    monkeypatch.setattr(integrity, "all_hashes", lambda source: {"c1": "h"})
    monkeypatch.setattr(integrity, "model_versions", lambda: {"lane-a": "v"})
    monkeypatch.setattr(integrity, "scaffold_prompt_hash", lambda: "")
    monkeypatch.setattr(integrity, "sampling_hash_map", lambda models: {})
    monkeypatch.setattr(run_matrix, "_apply_multimodal_gate", lambda source, models: (models, {}))
    monkeypatch.setattr(run_matrix, "_arm_context", lambda tasks, models: ({}, {}))
    monkeypatch.setattr(
        run_matrix,
        "classify_cells",
        lambda *a, **k: run_matrix.CellStatus(missing=[cell]),
    )
    monkeypatch.setattr(run_matrix, "_print_status", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "_has_keys", lambda: True)
    monkeypatch.setattr(run_matrix, "preflight_refuses", lambda *a, **k: False)
    monkeypatch.setattr(
        run_matrix,
        "_run_and_merge",
        lambda *a, **k: (captured.update(lanes=k.get("lanes")), 1)[1],
    )
    monkeypatch.setattr(run_matrix, "_report_coverage", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "refresh_summary", lambda *a, **k: None)
    monkeypatch.setattr(run_matrix, "regenerate_plots", lambda: None)

    args = argparse.Namespace(
        live=True,
        extra_models=None,
        check_images=False,
        step_limit=10,
        cells=None,
        timeout=1,
        workers=1,
        max_cost=None,
        max_cost_overshoot=0.0,
        max_start_failures=5,
        max_consecutive_failures=5,
        require_zero_cost=False,
        no_summary=True,
        no_plots=True,
    )
    assert run_matrix._run_full(args) == 0
    scheduler = captured.get("lanes")
    assert isinstance(scheduler, ls.LaneScheduler)
    assert scheduler.limits_for("lane-a").effective_rpd == 100
    assert loaded == [True]
    assert len(saved) == 1 and isinstance(saved[0], dict)


def test_build_lane_scheduler_refuses_a_paid_registry_model(monkeypatch) -> None:
    """LIVE GUARD: a model whose provider is not a declared free lane is structurally refused.

    The provider is resolved from the pricing view (the overlay has no row for a shipped paid
    model), so a paid extra cannot slip past the scheduler and spend.
    """
    from benchmark import config
    from benchmark.runner import run_matrix

    monkeypatch.setattr(
        config,
        "lanes_config",
        lambda: {"unknown_limits": {"rpm": 10, "rpd": 100}, "limits": {}},
    )
    monkeypatch.setattr(config, "lane_unknown_limits", lambda: {"rpm": 10, "rpd": 100})
    monkeypatch.setattr(config, "lane_limits_config", lambda: {})
    monkeypatch.setattr(config, "lane_stall_timeout_s", lambda: 900.0)
    monkeypatch.setattr(config, "free_registry", lambda: {})
    monkeypatch.setattr(config, "load_pricing", lambda: {"paid-model": {"provider": "deepseek"}})
    monkeypatch.setattr(ls, "load_lane_state", lambda: {})

    scheduler = run_matrix._build_lane_scheduler(["paid-model"], {})
    assert scheduler is not None
    assert not scheduler.can_admit("paid-model", now=0.0)
    assert not scheduler.is_available("paid-model", now=0.0)
    reason = scheduler.refusal_reason("paid-model", now=0.0)
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason
