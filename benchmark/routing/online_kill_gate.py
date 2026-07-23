"""Peeking-safe online kill-gate adjudication over the verified-outcome stream.

Paired non-inferiority (not-worse quality within a margin AND cheaper on cache-aware cost)
via the anytime-valid confidence sequence in ``frontier_estimate``; see the design doc.
"""

# Decision logic only — this module is not yet wired to a live/served endpoint (it is
# imported by the benchmark, never by ``src/shunt``); the "served" wiring is future work.

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Protocol

from benchmark.routing.frontier_estimate import (
    McNemarResult,
    SeqState,
    mcnemar_noninferiority,
    update_confidence_sequence,
)

_NORMAL = NormalDist()

# Verdict labels (behaviour-only; no internal identifiers by design).
PASS = "PASS"
FAIL = "FAIL"
CONTINUE = "CONTINUE"
UNDERPOWERED = "UNDERPOWERED"

DEFAULT_MARGIN = 0.05  # non-inferiority margin on paired pass-rate difference
DEFAULT_ALPHA = 0.05  # anytime-valid gate level (false-PASS budget across ALL looks)


def _sign(x: float) -> int:
    """Sign of a real value as an integer in {-1, 0, 1}."""
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


@dataclass(frozen=True)
class PairedOutcome:
    """One paired production observation: same task, router arm vs fixed-frontier arm.

    ``*_pass`` are verified pass/fail in {0,1}; ``*_cost`` are realized cache-aware costs;
    ``cost_known=False`` excludes the pair from the cost ratio (never a real 0.0).
    """

    router_pass: int
    baseline_pass: int
    router_cost: float = 0.0
    baseline_cost: float = 0.0
    cost_known: bool = True


@dataclass(frozen=True)
class OnlineGateState:
    """Accumulated online-gate state: two anytime-valid sequences + a cost ratio."""

    quality: SeqState | None = None
    cost_sign: SeqState | None = None
    router_cost_sum: float = 0.0
    baseline_cost_sum: float = 0.0
    n: int = 0
    n_cost: int = 0

    @property
    def cache_aware_ratio(self) -> float:
        """Router cache-aware cost / baseline cache-aware cost; +inf when no cost data."""
        # A zero/absent baseline cost yields +inf, which the verdict treats as 'not
        # cheaper' (conservative) — mirrors the offline gate's _cache_aware_ratio.
        if self.baseline_cost_sum <= 0.0:
            return float("inf")
        return self.router_cost_sum / self.baseline_cost_sum


def _fold_sign_sequence(prev: SeqState | None, c: int, alpha: float) -> SeqState:
    """Fold a paired cost-saving sign c in {-1,0,1} into a margin-0 confidence sequence."""
    # Reuse update_confidence_sequence verbatim by encoding the sign as a paired pair:
    # +1 -> (1,0), -1 -> (0,1), 0 -> (0,0), so its X = router-baseline equals c. margin=0
    # tests H0: median saving = 0 -> 'router_wins' means credibly cheaper per session.
    router_i, baseline_i = (1, 0) if c > 0 else (0, 1) if c < 0 else (0, 0)
    return update_confidence_sequence(prev, router_i, baseline_i, margin=0.0, alpha=alpha)


def update_online_gate(
    state: OnlineGateState,
    obs: PairedOutcome,
    *,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> OnlineGateState:
    """Fold one paired production observation into the online-gate state (pure, O(1))."""
    quality = update_confidence_sequence(
        state.quality, obs.router_pass, obs.baseline_pass, margin=margin, alpha=alpha
    )
    cost_sign = state.cost_sign
    router_sum = state.router_cost_sum
    baseline_sum = state.baseline_cost_sum
    n_cost = state.n_cost
    if obs.cost_known:
        saving_sign = _sign(obs.baseline_cost - obs.router_cost)
        cost_sign = _fold_sign_sequence(cost_sign, saving_sign, alpha)
        router_sum += obs.router_cost
        baseline_sum += obs.baseline_cost
        n_cost += 1
    return OnlineGateState(
        quality=quality,
        cost_sign=cost_sign,
        router_cost_sum=router_sum,
        baseline_cost_sum=baseline_sum,
        n=state.n + 1,
        n_cost=n_cost,
    )


def run_online_gate(
    stream: list[PairedOutcome],
    *,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> OnlineGateState:
    """Fold a whole paired stream into a final state (convenience over update_online_gate)."""
    state = OnlineGateState()
    for obs in stream:
        state = update_online_gate(state, obs, margin=margin, alpha=alpha)
    return state


def decide_online_verdict(
    state: OnlineGateState,
    *,
    available_n: int | None = None,
    reachable: bool = True,
) -> tuple[str, str]:
    """Emit PASS / FAIL / CONTINUE / UNDERPOWERED from the accumulated state.

    Priority: quality-worse FAIL > cost-not-cheaper FAIL > PASS/cost-FAIL on a decided
    win > UNDERPOWERED (window exhausted and unreachable) > CONTINUE.
    """
    q = state.quality
    cs = state.cost_sign
    ratio = state.cache_aware_ratio

    if q is not None and q.decided and q.direction == "no_win":
        return (FAIL, "quality credibly worse than the fixed-frontier baseline")

    # The authoritative cost criterion is the AGGREGATE cache-aware ratio (Sum d_i < 0,
    # i.e. mean per-pair cost difference below 0) — NOT the per-session saving sign, which
    # is the MEDIAN and can contradict the aggregate. The peeking-safe sign sequence is
    # therefore only allowed to trigger an early FAIL when it AGREES with the aggregate
    # (ratio >= 1); it can never FAIL a router that is aggregate-cheaper (ratio < 1).
    if cs is not None and cs.decided and cs.direction == "no_win" and ratio >= 1.0:
        return (FAIL, "router credibly not cheaper on cache-aware cost")

    if q is not None and q.decided and q.direction == "router_wins":
        if state.n_cost == 0:
            return (CONTINUE, "quality decided; awaiting cache-aware cost evidence")
        if ratio < 1.0:
            return (PASS, f"non-inferior quality and cheaper (cache-aware ratio {ratio:.4f})")
        return (
            FAIL,
            f"quality non-inferior but not cheaper (cache-aware ratio {ratio:.4f} >= 1.0)",
        )

    if available_n is not None and state.n >= available_n and not reachable:
        return (UNDERPOWERED, "effect not separable within the window at this label volume")
    return (CONTINUE, "insufficient evidence; keep collecting paired outcomes")


def confirm_fixed_n(
    stream: list[PairedOutcome], *, margin: float = DEFAULT_MARGIN, alpha: float = DEFAULT_ALPHA
) -> McNemarResult:
    """Decorrelated day-30 confirmation: the fixed-N Tango-McNemar non-inferiority test.

    A different estimator family from the sequential monitor, so a peeking bug in the
    online path cannot also corrupt this cross-check. A genuine PASS survives both.
    """
    router_pass = {f"t{i}": o.router_pass for i, o in enumerate(stream)}
    baseline_pass = {f"t{i}": o.baseline_pass for i, o in enumerate(stream)}
    return mcnemar_noninferiority(router_pass, baseline_pass, margin=margin, alpha=alpha)


def final_verdict(
    stream: list[PairedOutcome],
    *,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    available_n: int | None = None,
    reachable: bool = True,
    confirm: bool = True,
) -> tuple[str, str]:
    """End-to-end verdict: run the online monitor, then require the decorrelated cross-check.

    A reported PASS must survive BOTH the sequential monitor and the fixed-N Tango-McNemar
    test; a monitor PASS the fixed-N test does not confirm is downgraded to CONTINUE.
    """
    state = run_online_gate(stream, margin=margin, alpha=alpha)
    verdict, reason = decide_online_verdict(state, available_n=available_n, reachable=reachable)
    if verdict == PASS and confirm:
        confirmation = confirm_fixed_n(stream, margin=margin, alpha=alpha)
        if confirmation.decision != "non_inferior":
            return (CONTINUE, "monitor PASS not confirmed by the fixed-N cross-check; hold")
    return (verdict, reason)


# ---------------------------------------------------------------------------
# Reachability (pre-registration): can the effect be separated within the window?
# ---------------------------------------------------------------------------

_UNREACHABLE_N = 10**12  # sentinel required-N when the effect is not detectable at all


@dataclass(frozen=True)
class Reachability:
    """Whether the anytime-valid procedure can separate the effect within the window."""

    required_n: int
    available_n: int
    reachable: bool
    gap: float
    sigma2: float
    n_fixed: float


def reachability(
    *,
    effect: float = 0.054,
    margin: float = DEFAULT_MARGIN,
    discordance_rate: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = 0.80,
    daily_pairs: float = 10.0,
    window_days: float = 30.0,
    anytime_penalty: float = 2.0,
) -> Reachability:
    """Required-N vs available-N for the online gate; ``reachable`` gates UNDERPOWERED.

    Fixed-N paired non-inferiority sample size inflated by ``anytime_penalty`` for
    continuous monitoring — a planning estimate; power is verified empirically in tests.
    """
    gap = effect + margin  # distance from the true mean to the H0 boundary -margin
    sigma2 = max(discordance_rate - effect * effect, 0.0)
    z_alpha = _NORMAL.inv_cdf(1.0 - alpha)
    z_power = _NORMAL.inv_cdf(power)
    n_fixed = math.inf if gap <= 0.0 else (z_alpha + z_power) ** 2 * sigma2 / (gap * gap)
    required_n = math.ceil(n_fixed * anytime_penalty) if math.isfinite(n_fixed) else _UNREACHABLE_N
    available_n = int(daily_pairs * window_days)
    reachable = required_n <= available_n
    return Reachability(
        required_n=required_n,
        available_n=available_n,
        reachable=reachable,
        gap=gap,
        sigma2=sigma2,
        n_fixed=n_fixed,
    )


def format_reachability(r: Reachability) -> str:
    """One-line human summary: required-N vs available-N and the verdict."""
    req = "unreachable" if r.required_n >= _UNREACHABLE_N else str(r.required_n)
    status = "REACHABLE" if r.reachable else "UNDERPOWERED"
    return (
        f"{status}: required_n={req} vs available_n={r.available_n} "
        f"(gap={r.gap:.4f}, per-obs var={r.sigma2:.4f})"
    )


# ---------------------------------------------------------------------------
# Live-store adapter: materialize the paired stream from the OutcomeStore.
# ---------------------------------------------------------------------------


class _StoreLike(Protocol):
    """The slice of OutcomeStore the adapter reads (duck-typed for testability)."""

    def labeled_outcome_rows(self, *, tier2_only: bool = ...) -> list[dict[str, Any]]: ...
    def get_session(self, session_id: str) -> dict[str, Any] | None: ...


def _pass_from_outcome(outcome: str | None) -> int:
    """Verified Tier-2 outcome -> pass in {0,1}. Only 'success' counts as a pass."""
    # Intentionally STRICTER than the capture path's _SUCCESS_LABELS (which also credits
    # 'weak_success'): the make-or-break gate raises the router's bar, so 'weak_success'
    # scores as a fail here. This is the safe direction — do NOT relax it to match capture
    # without a deliberate decision, as that makes the gate easier to PASS.
    return 1 if outcome == "success" else 0


def read_paired_outcomes(store: _StoreLike) -> list[PairedOutcome]:
    """Build the paired stream from verified Tier-2 store rows, arm-tagged in provenance.

    Each session's ``decision_provenance`` JSON carries ``arm`` in {'router','baseline'}
    and a shared ``pair_key``; sessions without a complete pair are skipped (empty is fine).
    """
    pairs: dict[str, dict[str, PairedOutcome | None]] = {}
    order: list[str] = []
    for row in store.labeled_outcome_rows(tier2_only=True):
        session_id = row.get("session_id")
        if session_id is None:
            continue
        session = store.get_session(session_id)
        if session is None:
            continue
        arm, pair_key = _arm_and_key(session)
        if arm not in ("router", "baseline") or pair_key is None:
            continue
        leg = PairedOutcome(
            router_pass=_pass_from_outcome(row.get("tier2_outcome")),
            baseline_pass=_pass_from_outcome(row.get("tier2_outcome")),
            router_cost=float(session.get("cost", 0.0)),
            baseline_cost=float(session.get("cost", 0.0)),
            cost_known=bool(session.get("cost_known", 1)),
        )
        slot = pairs.setdefault(pair_key, {"router": None, "baseline": None})
        if pair_key not in order:
            order.append(pair_key)
        slot[arm] = leg
    return [_merge_legs(pairs[k]) for k in order if _complete(pairs[k])]


def _arm_and_key(session: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (arm, pair_key) from a session's decision_provenance JSON, tolerant."""
    raw = session.get("decision_provenance")
    if not raw:
        return (None, None)
    try:
        prov = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return (None, None)
    if not isinstance(prov, dict):
        return (None, None)
    return (prov.get("arm"), prov.get("pair_key"))


def _complete(slot: dict[str, PairedOutcome | None]) -> bool:
    return slot["router"] is not None and slot["baseline"] is not None


def _merge_legs(slot: dict[str, PairedOutcome | None]) -> PairedOutcome:
    """Merge the router leg and baseline leg of one dual-run task into one PairedOutcome."""
    router = slot["router"]
    baseline = slot["baseline"]
    assert router is not None and baseline is not None  # guarded by _complete
    return PairedOutcome(
        router_pass=router.router_pass,
        baseline_pass=baseline.baseline_pass,
        router_cost=router.router_cost,
        baseline_cost=baseline.baseline_cost,
        cost_known=router.cost_known and baseline.cost_known,
    )
