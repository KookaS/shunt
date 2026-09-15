"""Per-lane free-tier rate limiting: pure admission predicates plus persisted lane state.

A cell is one (challenge, model, arm) run that issues up to ``step_limit`` provider requests,
so a lane is charged a conservative ``p90(calls)`` reservation at admission and trued up at
completion from the row's ``calls`` — the reservation idea the collector ladder already ships
as its cold-start cost. Persisted under ``corpus_lock`` so a restart cannot burn a day's RPD
in a minute. Pure and I/O-free apart from the two serialization helpers at the end; the
re-drain is a pull loop (:meth:`LaneScheduler.next_ready`), never a timer or a queue object.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from benchmark import corpus_lock

# A cell is the same triple run_matrix passes everywhere: (challenge, model, arm).
type Cell = tuple[str, str, str]

UNKNOWN_RPM: Final[int] = 10
UNKNOWN_RPD: Final[int] = 100
MINUTE_S: Final[float] = 60.0
DAY_S: Final[float] = 86_400.0
BACKOFF_BASE_S: Final[float] = 30.0
BACKOFF_CAP_S: Final[float] = 3_600.0
DEFAULT_STALL_TIMEOUT_S: Final[float] = 900.0
# Named reason for a lane whose single request can never fit its TPM: a request larger than
# the whole minute's token budget is rejected however it is spaced, so the lane is refused up
# front instead of thrashing 429s.
LANE_TPM_TOO_SMALL: Final[str] = "LANE_TPM_TOO_SMALL"
# Named reason for a lane whose provider serves no free tier through the API (the free ids are
# app/session-gated or merely promotional), so it is refused in the free campaign up front.
LANE_NO_FREE_ACCESS: Final[str] = "LANE_NO_FREE_ACCESS"
STATE_PATH: Final[Path] = Path("benchmark/runner/artifacts/free-tier/lane_state.json")


@dataclass(frozen=True)
class LaneLimits:
    """Published per-lane limits. ``None`` is UNKNOWN — never replaced by an invented number.

    ``tpm`` is a true trailing-minute token budget, fed by the ACTUAL tokens each completed
    cell reports (see :meth:`LaneScheduler.complete`). ``max_request_tokens`` is the largest
    single request the lane must send; when it exceeds ``tpm`` the lane can never serve a
    cell and :func:`structural_refusal` names it ``LANE_TPM_TOO_SMALL``. ``free_access`` is
    the registry's access marker (``False`` on a provider that serves no free tier through
    the API); ``access_note`` carries the evidence the named refusal quotes.
    """

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    max_request_tokens: int | None = None
    daily_token_budget: int | None = None
    free_access: bool = True
    access_note: str | None = None
    expires_at: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> LaneLimits:
        """Build from a config/published limit mapping; absent or null means UNKNOWN."""
        free_access = raw.get("free_access")
        access_note = raw.get("access_note")
        return cls(
            rpm=_optional_int(raw.get("rpm")),
            rpd=_optional_int(raw.get("rpd")),
            tpm=_optional_int(raw.get("tpm")),
            max_request_tokens=_optional_int(raw.get("max_request_tokens")),
            daily_token_budget=_optional_int(raw.get("daily_token_budget")),
            free_access=free_access if isinstance(free_access, bool) else True,
            access_note=str(access_note) if access_note else None,
            expires_at=str(raw["expires_at"]) if raw.get("expires_at") else None,
        )

    @property
    def effective_rpm(self) -> int:
        """The RPM to enforce: the published limit, else the conservative unknown default."""
        return UNKNOWN_RPM if self.rpm is None else self.rpm

    @property
    def effective_rpd(self) -> int:
        """The RPD to enforce: the published limit, else the conservative unknown default."""
        return UNKNOWN_RPD if self.rpd is None else self.rpd

    @property
    def required_request_tokens(self) -> int:
        """The single-request token charge to reserve; ``0`` when the request size is UNKNOWN."""
        return 0 if self.max_request_tokens is None else self.max_request_tokens


def _optional_int(value: object) -> int | None:
    """An int limit from a config value, or UNKNOWN (``None``) — never an invented number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def adapt_probe_limits(probe: object) -> LaneLimits:
    """Adapt ``free_lane_probe.LaneLimits`` (rpm/rpd/known) to the scheduler's canonical type.

    The probe's dataclass is a static-admission record; the scheduler owns the runtime limits.
    One canonical class, two views — an adapter, never a duplicated class.
    """
    return LaneLimits(
        rpm=_optional_int(getattr(probe, "rpm", None)),
        rpd=_optional_int(getattr(probe, "rpd", None)),
    )


@dataclass
class _Reservation:
    """One admitted cell's charge in a window; ``calls`` is reserved then trued up."""

    key: str
    at: float
    calls: int


@dataclass
class Bucket:
    """A sliding window of cell charges: reservations at admission, trued up at completion."""

    entries: list[_Reservation] = field(default_factory=list)

    def total(self, now: float, window: float) -> int:
        """Charged calls still inside ``window`` ending at ``now``."""
        return sum(entry.calls for entry in self.entries if now - entry.at < window)

    def reserve(self, key: str, now: float, calls: int) -> None:
        """Charge ``calls`` for ``key`` as of ``now``."""
        self.entries.append(_Reservation(key, now, calls))

    def reconcile(self, key: str, actual: int) -> int | None:
        """Replace the reservation named ``key`` with the cell's actual ``calls``.

        Returns the reservation that was replaced (``None`` if the key is not present), so a
        caller can record whether the cell over-issued its reservation.
        """
        for entry in self.entries:
            if entry.key == key:
                previous = entry.calls
                entry.calls = actual
                return previous
        return None

    def prune(self, now: float, window: float) -> None:
        """Drop entries that have aged out of ``window``."""
        self.entries = [entry for entry in self.entries if now - entry.at < window]


@dataclass
class LaneState:
    """A lane's mutable window counters, quarantine, and failure streaks.

    ``minute``/``day`` meter REQUESTS (cell admissions and their actual calls); the two
    ``token_*`` buckets meter ACTUAL tokens over the same windows, so a token budget is
    summed rather than merely reserved per cell.
    """

    minute: Bucket = field(default_factory=Bucket)
    day: Bucket = field(default_factory=Bucket)
    token_minute: Bucket = field(default_factory=Bucket)
    token_day: Bucket = field(default_factory=Bucket)
    quarantined_until: float | None = None
    consecutive_429: int = 0
    consecutive_fail: int = 0
    disabled_reason: str | None = None
    # Named block set when a cell's ACTUAL calls true-up the day total to/over effective_rpd.
    # Distinct from ``disabled_reason`` (operator intent, cleared only by :meth:`enable`): this
    # self-clears once every charge has aged out of the trailing-day window. Set with the
    # timestamp so a restart keeps the lane blocked for the rest of the day, not forever.
    day_blocked_reason: str | None = None
    day_blocked_at: float | None = None


class Stalled:
    """Sentinel: no pending cell's lane is currently admissible."""

    def __repr__(self) -> str:
        return "STALLED"


STALLED: Final[Stalled] = Stalled()


def cell_key(cell: Cell) -> str:
    """The stable identity of a cell within the scheduler's reservation bookkeeping."""
    return f"{cell[0]}:{cell[1]}:{cell[2]}"


def p90_calls(history: Sequence[int], *, default: int = 1) -> int:
    """Nearest-rank 90th percentile of a lane's per-cell call counts (``default`` if empty)."""
    if not history:
        return default
    ordered = sorted(int(value) for value in history)
    rank = max(1, math.ceil(0.9 * len(ordered)))
    return max(1, ordered[rank - 1])


def backoff_seconds(consecutive_429: int) -> float:
    """The full-jitter UPPER bound for the Nth consecutive 429: ``min(30 * 2**N, 3600)``."""
    return min(BACKOFF_BASE_S * (2.0 ** max(0, consecutive_429)), BACKOFF_CAP_S)


def quarantine_until(
    now: float,
    consecutive_429: int,
    *,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """``now`` + full jitter of the capped backoff, never earlier than a server Retry-After."""
    draw = rng.random() if rng is not None else random.random()
    jittered = backoff_seconds(consecutive_429) * draw
    floor = retry_after if retry_after is not None and retry_after > 0 else 0.0
    return now + max(jittered, floor)


def _expires_epoch(limits: LaneLimits) -> float | None:
    """The lane's expiry as an epoch second, or None when absent/unparseable (never invented)."""
    if not limits.expires_at:
        return None
    try:
        return datetime.fromisoformat(limits.expires_at).timestamp()
    except ValueError:
        return None


def _day_block_active(state: LaneState, now: float) -> bool:
    """True while the lane's named day-RPD block is still inside its trailing day.

    A missing timestamp is treated as still-blocked (fail-closed): an unknown age never
    silently reopens a lane that over-issued.
    """
    if state.day_blocked_reason is None:
        return False
    return state.day_blocked_at is None or now - state.day_blocked_at < DAY_S


# ── the limitation registry: every admission rule, expressed once ─────────────────────────
#
# Each limitation is one named predicate over an immutable LimitContext. The scheduler walks
# the same tuple for every provider, so support for a new rule is a new entry plus its test —
# never a provider-specific branch. A rule marked ``structural`` depends only on the lane's
# published limits, so :func:`structural_refusal` can name an unusable lane before any request.


@dataclass(frozen=True)
class LimitContext:
    """One admission decision's inputs, shared by every limitation."""

    limits: LaneLimits
    state: LaneState
    now: float
    tokens: int
    reserve: int


LimitationCheck = Callable[[LimitContext], str | None]


@dataclass(frozen=True)
class Limitation:
    """A named admission constraint: a non-empty reason blocks, ``None`` admits."""

    name: str
    evaluate: LimitationCheck
    structural: bool = False


def _disabled_reason(ctx: LimitContext) -> str | None:
    return ctx.state.disabled_reason


def _expired_reason(ctx: LimitContext) -> str | None:
    expiry = _expires_epoch(ctx.limits)
    if expiry is not None and ctx.now >= expiry:
        return f"lane expired at {ctx.limits.expires_at}"
    return None


def _day_blocked_reason(ctx: LimitContext) -> str | None:
    return ctx.state.day_blocked_reason if _day_block_active(ctx.state, ctx.now) else None


def _quarantined_reason(ctx: LimitContext) -> str | None:
    until = ctx.state.quarantined_until
    if until is not None and ctx.now < until:
        return f"quarantined for another {until - ctx.now:.0f}s"
    return None


def _free_access_reason(ctx: LimitContext) -> str | None:
    if ctx.limits.free_access:
        return None
    note = ctx.limits.access_note or "this lane serves no free tier through the API"
    return f"{LANE_NO_FREE_ACCESS}: {note}"


def _max_request_reason(ctx: LimitContext) -> str | None:
    limits = ctx.limits
    if limits.tpm is None or limits.max_request_tokens is None:
        return None
    if limits.max_request_tokens > limits.tpm:
        return (
            f"{LANE_TPM_TOO_SMALL}: a single {limits.max_request_tokens}-token request "
            f"exceeds the {limits.tpm} TPM budget; spacing cannot admit it"
        )
    return None


def _rpm_reason(ctx: LimitContext) -> str | None:
    limit = ctx.limits.effective_rpm
    used = ctx.state.minute.total(ctx.now, MINUTE_S)
    if used + 1 > limit:
        return f"RPM {limit}: {used} admission(s) in the trailing minute"
    return None


def _rpd_reason(ctx: LimitContext) -> str | None:
    limit = ctx.limits.effective_rpd
    used = ctx.state.day.total(ctx.now, DAY_S)
    if used + ctx.reserve > limit:
        return f"RPD {limit}: {used}+{ctx.reserve} calls in the trailing day"
    return None


def _tpm_reason(ctx: LimitContext) -> str | None:
    if ctx.limits.tpm is None:
        return None
    used = ctx.state.token_minute.total(ctx.now, MINUTE_S)
    if used + ctx.tokens > ctx.limits.tpm:
        return f"TPM {ctx.limits.tpm}: {used}+{ctx.tokens} tokens in the trailing minute"
    return None


def _daily_token_budget_reason(ctx: LimitContext) -> str | None:
    budget = ctx.limits.daily_token_budget
    if budget is None:
        return None
    used = ctx.state.token_day.total(ctx.now, DAY_S)
    if used + ctx.tokens > budget:
        return f"daily token budget {budget}: {used}+{ctx.tokens} tokens in the trailing day"
    return None


_DISABLED = Limitation("disabled", _disabled_reason)
_EXPIRED = Limitation("expired", _expired_reason)
_DAY_BLOCKED = Limitation("day_blocked", _day_blocked_reason)
_QUARANTINED = Limitation("quarantined", _quarantined_reason)
_FREE_ACCESS = Limitation("free_access", _free_access_reason, structural=True)
_MAX_REQUEST = Limitation("max_request_tokens", _max_request_reason, structural=True)
_RPM = Limitation("rpm", _rpm_reason)
_RPD = Limitation("rpd", _rpd_reason)
_TPM = Limitation("tpm", _tpm_reason)
_DAILY_TOKEN_BUDGET = Limitation("daily_token_budget", _daily_token_budget_reason)

LIMITATIONS: Final[tuple[Limitation, ...]] = (
    _DISABLED,
    _EXPIRED,
    _DAY_BLOCKED,
    _QUARANTINED,
    _FREE_ACCESS,
    _MAX_REQUEST,
    _RPM,
    _RPD,
    _TPM,
    _DAILY_TOKEN_BUDGET,
)

# The health-only subset: a lane is "available" when none of these blocks. Capacity limits
# (rpm/rpd/tpm/budget) are deliberately excluded — they gate a specific admission, not the
# lane's health, and a caller may peek at availability without consuming a slot. The two
# STRUCTURAL limits ARE included: a lane with no free access, or whose single request can never
# fit its own TPM, can never serve a cell, so `is_available` must agree with `can_admit` rather
# than report a lane healthy that admission will always refuse.
_AVAILABILITY_LIMITATIONS: Final[tuple[Limitation, ...]] = (
    _DISABLED,
    _DAY_BLOCKED,
    _QUARANTINED,
    _FREE_ACCESS,
    _MAX_REQUEST,
)


def structural_refusal(limits: LaneLimits) -> str | None:
    """The named reason these limits can never serve a cell, or ``None`` (limits-only)."""
    ctx = LimitContext(limits, LaneState(), 0.0, 0, 1)
    for limitation in LIMITATIONS:
        if limitation.structural:
            reason = limitation.evaluate(ctx)
            if reason is not None:
                return reason
    return None


@dataclass
class LaneScheduler:
    """Admission control over named lanes: reserve at admission, reconcile at completion."""

    limits: dict[str, LaneLimits] = field(default_factory=dict)
    state: dict[str, LaneState] = field(default_factory=dict)
    reserves: dict[str, int] = field(default_factory=dict)
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    served: dict[str, int] = field(default_factory=dict)

    # ── reads ─────────────────────────────────────────────────────────────────
    def lane_state(self, lane: str) -> LaneState:
        """The lane's state, created empty on first touch."""
        return self.state.setdefault(lane, LaneState())

    def limits_for(self, lane: str) -> LaneLimits:
        """The lane's limits, or a fully-UNKNOWN set (which collapses to the defaults)."""
        return self.limits.get(lane, LaneLimits())

    def reserve_for(self, lane: str) -> int:
        """The lane's per-cell reservation (``p90(calls)``), defaulting to one call."""
        return max(1, self.reserves.get(lane, 1))

    def _day_block_active(self, state: LaneState, now: float) -> bool:
        """True while the lane's named day-RPD block is still inside its trailing day."""
        return _day_block_active(state, now)

    def is_available(self, lane: str, now: float) -> bool:
        """True unless a STRUCTURAL or health limit blocks the lane.

        Disabled, day-blocked, quarantined, no free access, or a single request that can never
        fit its TPM all make the lane unavailable. Capacity limits (RPM/RPD/TPM/budget) are
        excluded: this is lane HEALTH, so a caller may peek without consuming a slot. The same
        named predicates back full admission, so the two views cannot disagree.
        """
        ctx = LimitContext(
            self.limits_for(lane), self.lane_state(lane), now, 0, self.reserve_for(lane)
        )
        return all(check.evaluate(ctx) is None for check in _AVAILABILITY_LIMITATIONS)

    def refusal_reason(self, lane: str, now: float, *, tokens: int | None = None) -> str | None:
        """The named reason ``lane`` cannot take one more admission at ``now``, or ``None``.

        ``tokens=None`` charges the lane's own single-request size (``max_request_tokens``),
        so a caller that does not know the next cell's token count still reserves the largest
        request it must send; an explicit value overrides it.
        """
        limits = self.limits_for(lane)
        charge = limits.required_request_tokens if tokens is None else tokens
        ctx = LimitContext(limits, self.lane_state(lane), now, charge, self.reserve_for(lane))
        for limitation in LIMITATIONS:
            reason = limitation.evaluate(ctx)
            if reason is not None:
                return reason
        return None

    def can_admit(self, lane: str, now: float, *, tokens: int | None = None) -> bool:
        """True iff one more admission on ``lane`` fits every published window at ``now``."""
        return self.refusal_reason(lane, now, tokens=tokens) is None

    def structural_refusals(self) -> dict[str, str]:
        """Named reasons for every lane whose published limits can never serve a cell."""
        return {
            lane: reason
            for lane, limits in self.limits.items()
            if (reason := structural_refusal(limits)) is not None
        }

    # ── transitions ───────────────────────────────────────────────────────────
    def admit(self, cell: Cell, now: float, *, tokens: int | None = None) -> bool:
        """Reserve ``p90(calls)`` on the cell's lane; False (no charge) when inadmissible."""
        lane = cell[1]
        if not self.can_admit(lane, now, tokens=tokens):
            return False
        key = cell_key(cell)
        state = self.lane_state(lane)
        # The minute bucket meters cell ADMISSIONS (one slot each); the provider's per-request
        # RPM is ultimately enforced by the 429/quarantine backstop. Charging the full p90
        # reservation to the minute window would make any lane with rpm < p90 permanently
        # inadmissible — unknown lanes are 10 RPM against an ~85-call cell — which is not the
        # intent; the p90 reservation is a DAY-budget guard.
        state.minute.prune(now, MINUTE_S)  # bucket hygiene: aged-out entries never count again
        state.day.prune(now, DAY_S)
        state.token_minute.prune(now, MINUTE_S)
        state.token_day.prune(now, DAY_S)
        state.minute.reserve(key, now, 1)
        state.day.reserve(key, now, self.reserve_for(lane))
        self.served[lane] = self.served.get(lane, 0) + 1
        return True

    def complete(self, cell: Cell, now: float, *, calls: int, tokens: int = 0) -> None:
        """True up the reservation to the row's ACTUAL ``calls`` and tokens; reset streaks.

        The day counter always reflects actual consumption. If the true-up pushes the
        trailing-day total to or past ``effective_rpd`` — the over-issue case the p90
        reservation exists to bound — the lane is blocked for the rest of the day through the
        named :attr:`LaneState.day_blocked_reason`, never silently over-counted.

        ``tokens`` is the cell's ACTUAL prompt + completion total, recorded in the trailing
        minute and day token buckets; a cell's tokens are attributed to its completion instant,
        so between completions the token windows are conservative rather than exact.
        """
        lane = cell[1]
        limits = self.limits_for(lane)
        state = self.lane_state(lane)
        reserved = state.day.reconcile(cell_key(cell), max(0, calls))
        actual_tokens = max(0, tokens)
        state.token_minute.reserve(cell_key(cell), now, actual_tokens)
        state.token_day.reserve(cell_key(cell), now, actual_tokens)
        state.consecutive_429 = 0
        state.consecutive_fail = 0
        total = state.day.total(now, DAY_S)
        if total >= limits.effective_rpd:
            if state.day_blocked_reason is None:
                state.day_blocked_reason = (
                    f"day RPD reached: {total}/{limits.effective_rpd} after a cell trued up a "
                    f"reservation of {reserved if reserved is not None else 'unknown'} to "
                    f"{max(0, calls)} calls"
                )
            state.day_blocked_at = now

    def record_rate_limit(
        self,
        lane: str,
        now: float,
        *,
        retry_after: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Back a lane off after a 429: exponential full jitter, honouring a Retry-After floor."""
        state = self.lane_state(lane)
        exponent = state.consecutive_429
        state.consecutive_429 = exponent + 1
        state.consecutive_fail += 1
        state.quarantined_until = quarantine_until(now, exponent, retry_after=retry_after, rng=rng)

    def record_fail(self, lane: str) -> None:
        """Count a non-429 failure; the caller decides whether to disable the lane."""
        self.lane_state(lane).consecutive_fail += 1

    def recover(self, lane: str, now: float) -> None:
        """Clear an expired quarantine (admission already treats it as expired)."""
        state = self.lane_state(lane)
        if state.quarantined_until is not None and now >= state.quarantined_until:
            state.quarantined_until = None

    def disable(self, lane: str, reason: str) -> None:
        """Take a lane out of rotation with a named reason (never silently)."""
        self.lane_state(lane).disabled_reason = reason

    def enable(self, lane: str) -> None:
        """Return a disabled lane to rotation."""
        self.lane_state(lane).disabled_reason = None

    def next_ready(self, pending: Sequence[Cell], now: float) -> Cell | Stalled:
        """The least-recently-served admissible cell, or ``STALLED`` — a pull, not a push.

        Read-only: recovery from an expired quarantine is handled by the expiry check in
        :meth:`is_available`, so a caller may peek freely and only :meth:`admit` mutates.
        """
        candidates = [
            (self.served.get(cell[1], 0), index, cell)
            for index, cell in enumerate(pending)
            if self.can_admit(cell[1], now)
        ]
        if not candidates:
            return STALLED
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]


# ── persistence (the only I/O in this module) ─────────────────────────────────


def _state_to_dict(state: LaneState) -> dict:
    """Serialize one lane state to JSON-ready primitives."""
    return {
        "minute": [[e.key, e.at, e.calls] for e in state.minute.entries],
        "day": [[e.key, e.at, e.calls] for e in state.day.entries],
        "token_minute": [[e.key, e.at, e.calls] for e in state.token_minute.entries],
        "token_day": [[e.key, e.at, e.calls] for e in state.token_day.entries],
        "quarantined_until": state.quarantined_until,
        "consecutive_429": state.consecutive_429,
        "consecutive_fail": state.consecutive_fail,
        "disabled_reason": state.disabled_reason,
        "day_blocked_reason": state.day_blocked_reason,
        "day_blocked_at": state.day_blocked_at,
    }


def _bucket_from(raw: list) -> Bucket:
    """Rebuild a bucket from its serialized ``[key, at, calls]`` rows."""
    return Bucket(entries=[_Reservation(str(k), float(at), int(c)) for k, at, c in raw])


def _state_from_dict(raw: dict) -> LaneState:
    """Rebuild one lane state from its serialized form.

    A state file written before token windows existed carries only ``tokens_today``; the key
    is ignored, so upgrading mid-campaign loses no REQUEST accounting and no lane that declares
    a token budget can have relied on the old scalar.
    """
    return LaneState(
        minute=_bucket_from(raw.get("minute", [])),
        day=_bucket_from(raw.get("day", [])),
        token_minute=_bucket_from(raw.get("token_minute", [])),
        token_day=_bucket_from(raw.get("token_day", [])),
        quarantined_until=raw.get("quarantined_until"),
        consecutive_429=int(raw.get("consecutive_429", 0)),
        consecutive_fail=int(raw.get("consecutive_fail", 0)),
        disabled_reason=raw.get("disabled_reason"),
        day_blocked_reason=raw.get("day_blocked_reason"),
        day_blocked_at=raw.get("day_blocked_at"),
    )


def dump_lane_state(states: dict[str, LaneState]) -> str:
    """Serialize lane states to stable, sorted JSON text."""
    return json.dumps(
        {lane: _state_to_dict(state) for lane, state in states.items()}, indent=2, sort_keys=True
    )


def parse_lane_state(text: str) -> dict[str, LaneState]:
    """Parse persisted lane-state JSON back into ``LaneState`` objects."""
    return {lane: _state_from_dict(raw) for lane, raw in json.loads(text).items()}


def load_lane_state(path: Path = STATE_PATH) -> dict[str, LaneState]:
    """Read persisted lane state under ``corpus_lock``; ``{}`` when the file is absent."""
    if not path.exists():
        return {}
    with corpus_lock.corpus_lock(path.parent):
        text = path.read_text(encoding="utf-8")
    return parse_lane_state(text)


def save_lane_state(states: dict[str, LaneState], path: Path = STATE_PATH) -> None:
    """Atomically persist lane state under ``corpus_lock`` so a restart keeps the day's RPD."""
    with corpus_lock.corpus_lock(path.parent):
        corpus_lock.atomic_write_text(path, dump_lane_state(states))
