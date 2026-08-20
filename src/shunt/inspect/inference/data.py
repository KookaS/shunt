"""Pure transforms behind the inference figures — public store reads in, dataclasses out."""

# No matplotlib here on purpose: the arithmetic that decides what a figure claims is testable
# without a rendering backend, and the draw functions in `figures.py` do no arithmetic beyond
# axis limits. Everything reads public `OutcomeStore` methods only.
#
# Emptiness is a first-class result. Every dataclass below can be empty, carries the count that
# explains WHY it is empty, and never substitutes a seeded number for a live one.

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from shunt.analysis.ope import (
    _DEFAULT_WEIGHT_CLIP,
    _MIN_CLUSTERS,
    _MIN_PER_ARM,
    _MIN_PROPENSITY,
    ExplorationLogRow,
    PolicyValueEstimate,
    _usable,
    _weights,
    always_escalate,
    ess_fraction,
    estimate_policy_value,
    ips_estimate,
    never_escalate,
    rows_from_records,
    snips_estimate,
)
from shunt.db.loop_health import LiveCostAggregates, LoopHealthThresholds, StratumCensus
from shunt.inspect.inference.estimators import (
    ESCALATION,
    ESTIMATORS,
    ROUTING,
    CertifiedEstimates,
    routing_agreement_rows,
)
from shunt.inspect.inference.specs import HOLD_TOKENS, RUNG_TOKENS, SEED_PREFIX
from shunt.router.escalation import EscalationAction
from shunt.router.inspection import top_capability_cluster

if TYPE_CHECKING:
    from shunt.db.store import OutcomeStore
    from shunt.models.config import ModelPool

SEEDED: Final[str] = "seeded"
LIVE: Final[str] = "live"
AMBIGUOUS: Final[str] = "ambiguous"

_SEED_RULE: Final[str] = "benchmark_seed"
_PAGE: Final[int] = 2000
_DEFAULT_K: Final[int] = 10
_RELIABILITY_BINS: Final[int] = 5


@dataclass(frozen=True)
class SessionRow:
    """One session, with its origin already adjudicated and its provenance already parsed."""

    session_id: str
    timestamp: datetime | None
    model_chosen: str
    cost: float
    cost_known: bool
    stratum: str
    selection_rule_used: str | None
    selection_propensity: float | None
    hold_reason: str | None
    rung: str | None
    undeliverable: bool
    tier2_success: bool | None


def _parse_time(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _provenance(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify_stratum(session_id: str, rule: str | None, outcome_source: str | None) -> str:
    """Adjudicate origin from the prefix and the decision rule; the label source corroborates."""
    # Prefix and rule are the two symmetric witnesses: both are written at DECISION time and
    # neither is ever rewritten, so a row where they dissent is a real seeding or migration
    # defect and is surfaced rather than folded into a stratum.
    #
    # The label source is deliberately ONE-DIRECTIONAL. `store._SOURCE_PRIORITY` gives
    # `benchmark_seed` priority zero, so the moment a seeded task is genuinely verified the
    # winning event's source becomes `auto_tier2` or `human` — the normal path, and the whole
    # point of the rig. Treating that as a dissenting vote would flag every verified seeded row
    # as ambiguous and quietly shrink the seeded count this family publishes. So the source can
    # only ever ADD evidence of seeding (only the seeder writes `benchmark_seed`); its absence
    # proves nothing.
    prefixed = session_id.startswith(SEED_PREFIX)
    seeded_rule = rule == _SEED_RULE
    sourced = outcome_source == _SEED_RULE  # only the seeder ever writes this source
    if prefixed and seeded_rule:
        return SEEDED
    if not prefixed and not seeded_rule:
        # Both decision-time witnesses say live. A `benchmark_seed` label source contradicts
        # them outright, which is a real conflict and is surfaced rather than resolved.
        return AMBIGUOUS if sourced else LIVE
    return SEEDED if sourced else AMBIGUOUS


def _rung_of(prov: dict[str, Any]) -> str | None:
    rule = prov.get("selection_rule_used")
    if rule == "escalation_floor":
        return "escalation_floor"
    if rule != "auto_escalation":
        return None
    return "raise_effort" if "escalated_reasoning_arm" in prov else "raise_rank"


def _undeliverable(prov: dict[str, Any]) -> bool:
    """A raise directive that could not be delivered: a hold the engine never tokenised."""
    # `engine._void_exploration` rewrites the stamped record to action=HOLD at propensity 1.0
    # with randomized=False, then returns BEFORE the branch that writes `escalation_hold_reason`.
    # All four conditions together are the void's exact signature and the only trace the
    # undeliverable case leaves. A genuinely EXPLORED hold also serialises action="hold", but it
    # carries a token (`exploration_hold`) and randomized=True, so the first and last checks
    # exclude it twice over. `HOLD` is a StrEnum, so the persisted value is its `.value`.
    if prov.get("escalation_hold_reason") is not None:
        return False
    record = prov.get("escalation_exploration")
    if not isinstance(record, dict):
        return False
    return (
        record.get("action") == EscalationAction.HOLD.value
        and record.get("randomized") is False
        and record.get("propensity") == 1.0
    )


def read_sessions(store: OutcomeStore) -> list[SessionRow]:
    """Every session in the store, origin-adjudicated, oldest first."""
    sources = {
        str(row["session_id"]): row.get("outcome_source") for row in store.labeled_outcome_rows()
    }
    successes = {
        str(row["session_id"]): row.get("tier2_outcome") == "success"
        for row in store.labeled_outcome_rows(tier2_only=True)
    }
    raw: list[dict[str, Any]] = []
    while True:
        page = store.get_sessions(limit=_PAGE, offset=len(raw))
        raw.extend(page)
        if len(page) < _PAGE:
            break
    rows = [_session_row(item, sources, successes) for item in raw]
    return sorted(rows, key=lambda r: (r.timestamp is None, r.timestamp, r.session_id))


def _session_row(
    item: dict[str, Any],
    sources: dict[str, str | None],
    successes: dict[str, bool],
) -> SessionRow:
    prov = _provenance(item.get("decision_provenance"))
    session_id = str(item["session_id"])
    rule = prov.get("selection_rule_used")
    return SessionRow(
        session_id=session_id,
        timestamp=_parse_time(str(item.get("timestamp") or "")),
        model_chosen=str(item.get("model_chosen") or "(unknown)"),
        cost=float(item.get("cost") or 0.0),
        cost_known=bool(item.get("cost_known", 1)),
        stratum=classify_stratum(session_id, rule, sources.get(session_id)),
        selection_rule_used=str(rule) if isinstance(rule, str) else None,
        selection_propensity=_opt_float(item.get("selection_propensity")),
        hold_reason=_opt_str(prov.get("escalation_hold_reason")),
        rung=_rung_of(prov),
        undeliverable=_undeliverable(prov),
        tier2_success=successes.get(session_id),
    )


def _opt_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


# ------------------------------------------------------------------- F1 strata


@dataclass(frozen=True)
class StrataData:
    """The two-stratum census, arrival times and per-model label counts."""

    census: StratumCensus
    ambiguous: list[str]
    times: list[tuple[str, datetime]]
    per_model: list[tuple[str, int, int]]  # model, seeded labeled, live labeled
    # Adjudicated by `classify_stratum`, NOT by the census. The census partitions on the id
    # prefix alone, so it assigns every row to one of two strata and counts an ambiguous row
    # inside one of them — summing its two `stored` figures with `ambiguous` double-counts.
    n_sessions: int
    n_seeded: int
    n_live: int
    # Rows the prefix census files under the OTHER stratum than the adjudication reached. Panel
    # A is the one panel that cannot be rebuilt from `rows` (embedded/indexed live in the
    # index, not in a session row), so the disagreement is disclosed rather than hidden.
    n_prefix_disagree: int
    # The census stages are NOT a nested chain. `embedded` is a property of the session row,
    # while `labeled` (a non-tombstoned row in `outcome_events`) and `tier2` (a materialized
    # `outcomes.tier2_outcome`) are counted off two DIFFERENT tables, so a later stage can and
    # does exceed an earlier one. The v5 backfill removed the legacy population where `tier2`
    # exceeded `labeled`, but no schema constraint forbids a future inversion, so any pair that
    # actually inverts is still named here and surfaced on the canvas.
    nesting_breaks: tuple[str, ...]


def strata(store: OutcomeStore, rows: list[SessionRow]) -> StrataData:
    """F1's inputs: the store's own stage census, plus the arrival and per-model views."""
    labeled = Counter(
        (row.model_chosen, row.stratum) for row in rows if row.tier2_success is not None
    )
    models = sorted({model for model, _ in labeled})
    strata_of = Counter(row.stratum for row in rows)
    census = store.stratum_census()
    return StrataData(
        census=census,
        ambiguous=[row.session_id for row in rows if row.stratum == AMBIGUOUS],
        times=[(row.stratum, row.timestamp) for row in rows if row.timestamp is not None],
        per_model=[(model, labeled[(model, SEEDED)], labeled[(model, LIVE)]) for model in models],
        n_sessions=len(rows),
        n_seeded=strata_of[SEEDED],
        n_live=strata_of[LIVE],
        n_prefix_disagree=sum(1 for row in rows if _prefix_disagrees(row)),
        nesting_breaks=_nesting_breaks(census),
    )


CENSUS_STAGES: Final[tuple[str, ...]] = ("stored", "embedded", "labeled", "tier2", "indexed")


def _nesting_breaks(census: StratumCensus) -> tuple[str, ...]:
    """Adjacent census stages where the later count exceeds the earlier one it is drawn under."""
    return tuple(
        f"{funnel.stratum}: {later} ({getattr(funnel, later)}) > {earlier} "
        f"({getattr(funnel, earlier)})"
        for funnel in (census.seeded, census.live)
        for earlier, later in zip(CENSUS_STAGES, CENSUS_STAGES[1:], strict=False)
        if getattr(funnel, later) > getattr(funnel, earlier)
    )


def _prefix_disagrees(row: SessionRow) -> bool:
    """True when the id prefix files this row under the other stratum than the adjudication."""
    if row.stratum == AMBIGUOUS:
        return False  # already disclosed by its own count; not a second, separate defect
    return (SEEDED if row.session_id.startswith(SEED_PREFIX) else LIVE) != row.stratum


# --------------------------------------------------------------------- F2 cost


@dataclass(frozen=True)
class CostData:
    """Live spend by window, its coverage, and its cumulative curve. Seeded rows excluded."""

    windows: list[tuple[str, LiveCostAggregates]]
    cumulative: list[tuple[datetime, float]]
    n_seeded_excluded: int
    n_live: int


def cost(rows: list[SessionRow], windows: tuple[int | None, ...] = (7, 30, None)) -> CostData:
    """F2's inputs. Every number here is live-only; the excluded seeded count is carried."""
    # Aggregated from `rows` rather than from `store.live_cost_aggregates`, so ONE definition
    # of live — the adjudicated `stratum` — drives every panel and the subtitle. The store's
    # clause partitions on the `bench:` id prefix alone, so a seeded row that lost its prefix
    # was billed as live spend by the same subtitle asserting it had been excluded.
    running = 0.0
    curve: list[tuple[datetime, float]] = []
    for row in rows:
        if row.stratum != LIVE or row.timestamp is None or not row.cost_known:
            continue
        running += row.cost
        curve.append((row.timestamp, running))
    return CostData(
        windows=[(_window_label(w), _live_cost(rows, w)) for w in windows],
        cumulative=curve,
        n_seeded_excluded=sum(1 for row in rows if row.stratum == SEEDED),
        n_live=sum(1 for row in rows if row.stratum == LIVE),
    )


def _live_cost(rows: list[SessionRow], days: int | None) -> LiveCostAggregates:
    """One window of live spend, in the shape the store used to hand back."""
    live = [row for row in rows if row.stratum == LIVE and _in_window(row, days)]
    known = [row for row in live if row.cost_known]
    counts = Counter(row.model_chosen for row in known)
    totals: dict[str, float] = dict.fromkeys(counts, 0.0)
    for row in known:
        totals[row.model_chosen] += row.cost
    return LiveCostAggregates(
        by_model=[(model, counts[model], totals[model]) for model in sorted(counts)],
        total=sum(row.cost for row in known),
        n_cost_known=len(known),
        n_cost_unknown=len(live) - len(known),
        window_days=days,
    )


def _in_window(row: SessionRow, days: int | None) -> bool:
    """The ONE window this family means by `7d`: the last N days of wall clock, ending now."""
    # Every windowed panel in the family shares this predicate. The rejected alternative anchored
    # the window on the NEWEST ROW instead, which relabels a year-old burst as "the last 7 days"
    # and made F2 and F6 print different `7d` denominators over one corpus. The whole delta is
    # compared, not `.days`, which floors and would let a `7d` bar quietly cover eight.
    if days is None:
        return True
    if row.timestamp is None:
        return False  # an unstamped row cannot be placed in a window; it is not "recent"
    return row.timestamp >= datetime.now(row.timestamp.tzinfo) - timedelta(days=days)


def _window_label(days: int | None) -> str:
    return "all" if days is None else f"{days}d"


# ----------------------------------------------------------- F3 unit economics


@dataclass(frozen=True)
class ModelEconomics:
    """One model's verified-success rate and cost per success within one stratum."""

    model: str
    n_labeled: int
    n_success: int
    rate: float
    lo: float
    hi: float
    cost_known_total: float
    cost_per_success: float | None


@dataclass(frozen=True)
class UnitEconomicsData:
    seeded: list[ModelEconomics]
    live: list[ModelEconomics]


def unit_economics(rows: list[SessionRow]) -> UnitEconomicsData:
    """F3's inputs: the same computation run once per stratum, never pooled across them."""
    return UnitEconomicsData(
        seeded=_economics(row for row in rows if row.stratum == SEEDED),
        live=_economics(row for row in rows if row.stratum == LIVE),
    )


def _economics(rows: Iterable[SessionRow]) -> list[ModelEconomics]:
    # Imported here, not at module scope: `plot_style` pulls matplotlib, and this module is
    # the half that must stay importable (and testable) without a rendering backend. Reusing
    # the shipped interval still beats re-deriving one that could drift from it.
    from shunt.inspect.plot_style import wilson_interval

    grouped: dict[str, list[SessionRow]] = {}
    for row in rows:
        grouped.setdefault(row.model_chosen, []).append(row)
    out: list[ModelEconomics] = []
    for model in sorted(grouped):
        members = grouped[model]
        labeled = [r for r in members if r.tier2_success is not None]
        n_success = sum(1 for r in labeled if r.tier2_success)
        spend = sum(r.cost for r in members if r.cost_known)
        lo, hi = wilson_interval(n_success, len(labeled)) if labeled else (0.0, 0.0)
        out.append(
            ModelEconomics(
                model=model,
                n_labeled=len(labeled),
                n_success=n_success,
                rate=n_success / len(labeled) if labeled else 0.0,
                lo=lo,
                hi=hi,
                cost_known_total=spend,
                # Undefined, not zero and not infinite: a model with no verified success has no
                # cost-per-success, and drawing one would invent a comparison.
                cost_per_success=spend / n_success if n_success else None,
            )
        )
    return out


# ------------------------------------------------------------- F4 neighbourhood


@dataclass(frozen=True)
class ReliabilityBin:
    centre: float
    predicted: float
    observed: float
    n: int


@dataclass(frozen=True)
class NeighbourhoodData:
    bins: list[ReliabilityBin]
    distances: list[float]
    origin_mix: list[float]
    n_probed: int
    n_live_decisions: int
    k: int


def neighbourhood(
    store: OutcomeStore, rows: list[SessionRow], k: int = _DEFAULT_K
) -> NeighbourhoodData:
    """F4's inputs: leave-one-out over the indexed population, plus the live origin mix."""
    success = {row.session_id: row.tier2_success for row in rows}
    stratum = {row.session_id: row.stratum for row in rows}
    probes = _loo_probes(store, success, k)
    live_ids = {row.session_id for row in rows if row.stratum == LIVE}
    return NeighbourhoodData(
        bins=_reliability_bins(probes),
        distances=[d for _p, _o, dists in probes for d in dists],
        origin_mix=[
            # Field 1 is the seeded share; field 3 is mean distance. Binding `share` to the
            # LAST field plotted distances under the seeded-share axis for as long as the
            # committed corpus had no live rows to expose it.
            share
            for sid, share, _k_found, _dist in _origin_probes(store, stratum, k)
            if sid in live_ids
        ],
        n_probed=len(probes),
        n_live_decisions=len(live_ids),
        k=k,
    )


def _neighbours(store: OutcomeStore, sid: str, blob: bytes, k: int) -> list[tuple[str, float]]:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    hits = store.query_index(vector, k=k + 1)
    return [(nid, dist) for nid, dist in hits if nid != sid][:k]


def _loo_probes(
    store: OutcomeStore, success: dict[str, bool | None], k: int
) -> list[tuple[float, bool, list[float]]]:
    """(neighbour success rate, own outcome, distances) for every indexed labeled session."""
    out: list[tuple[float, bool, list[float]]] = []
    for sid, blob in store.get_labeled_embeddings():
        own = success.get(sid)
        if own is None:
            continue
        hits = _neighbours(store, sid, blob, k)
        outcomes = [success[nid] for nid, _d in hits if success.get(nid) is not None]
        if not outcomes:
            continue
        rate = sum(1 for o in outcomes if o) / len(outcomes)
        out.append((rate, own, [dist for _n, dist in hits]))
    return out


def _origin_probes(
    store: OutcomeStore, stratum: dict[str, str], k: int
) -> list[tuple[str, float, int, float]]:
    """(session, seeded share of its top-k, k found, mean distance) for every embedded session."""
    out: list[tuple[str, float, int, float]] = []
    for sid, blob in store.get_all_embeddings():
        hits = _neighbours(store, sid, blob, k)
        if not hits:
            continue
        seeded = sum(1 for nid, _d in hits if stratum.get(nid) == SEEDED)
        mean_d = sum(dist for _n, dist in hits) / len(hits)
        out.append((sid, seeded / len(hits), len(hits), mean_d))
    return out


def _reliability_bins(probes: list[tuple[float, bool, list[float]]]) -> list[ReliabilityBin]:
    buckets: dict[int, list[tuple[float, bool]]] = {}
    for rate, own, _dists in probes:
        index = min(_RELIABILITY_BINS - 1, int(rate * _RELIABILITY_BINS))
        buckets.setdefault(index, []).append((rate, own))
    out: list[ReliabilityBin] = []
    for index in sorted(buckets):
        members = buckets[index]
        out.append(
            ReliabilityBin(
                centre=(index + 0.5) / _RELIABILITY_BINS,
                predicted=sum(r for r, _o in members) / len(members),
                observed=sum(1 for _r, o in members if o) / len(members),
                n=len(members),
            )
        )
    return out


# ------------------------------------------------------------------- F5 policy


@dataclass(frozen=True)
class PolicyData:
    """Live share over time, the collapse alarms, and per-model propensity support."""

    live_series: list[tuple[datetime, dict[str, float]]]
    seed_mix: list[tuple[str, int]]
    entropy: list[tuple[int, float]]
    frontier_share: list[tuple[int, float]]
    propensities: list[tuple[str, int, float, float]]
    window: int
    n_live: int
    thresholds: LoopHealthThresholds
    # None when no model registry was supplied. Normalized entropy divides by the number of
    # arms the router COULD have chosen, which the store does not know; without it the alarm
    # is undrawable, and panel B says so rather than dividing by the arms it happens to see.
    candidate_models: int | None = None


def policy(rows: list[SessionRow], model_pool: ModelPool | None = None) -> PolicyData:
    """F5's inputs. The seed mix is carried separately and never enters the live series."""
    # `model_pool` supplies the two things the store cannot know: which models are the expensive
    # tail, and how many arms the router could have picked. Both come from the SHIPPED
    # definitions (`top_capability_cluster`, `ModelPool.model_names`) so the alarm this figure
    # draws is the alarm `/admin/loop-health` raises, not a lookalike that drifts from it.
    thresholds = LoopHealthThresholds()
    live = [row for row in rows if row.stratum == LIVE and row.timestamp is not None]
    frontier = top_capability_cluster(model_pool) if model_pool is not None else set()
    candidates = len(model_pool.model_names()) if model_pool is not None else None
    return PolicyData(
        live_series=_share_series(live, thresholds.recent_window),
        seed_mix=sorted(Counter(row.model_chosen for row in rows if row.stratum == SEEDED).items()),
        entropy=(
            _rolling(live, thresholds.recent_window, lambda n: normalized_entropy(n, candidates))
            if candidates is not None
            else []
        ),
        frontier_share=(
            _rolling(live, thresholds.recent_window, lambda names: _share_of(names, frontier))
            if model_pool is not None
            else []
        ),
        propensities=_propensities(rows),
        window=thresholds.recent_window,
        n_live=len(live),
        thresholds=thresholds,
        candidate_models=candidates,
    )


def _propensities(rows: list[SessionRow]) -> list[tuple[str, int, float, float]]:
    """Panel C's support floor input: logged propensity per model over LIVE rows only."""
    # Shaped exactly like `LoopHealthSnapshot.model_propensities`, but adjudicated by
    # `classify_stratum` rather than by the id prefix the snapshot filters on — so panel C and
    # the rest of this family agree on which rows are live even where the two rules dissent.
    by_model: dict[str, list[float]] = {}
    for row in rows:
        if row.stratum != LIVE or row.selection_propensity is None:
            continue
        by_model.setdefault(row.model_chosen, []).append(row.selection_propensity)
    return [
        (model, len(seen), statistics.fmean(seen), min(seen))
        for model, seen in sorted(by_model.items())
    ]


def _share_series(live: list[SessionRow], window: int) -> list[tuple[datetime, dict[str, float]]]:
    """Model share within a trailing window — NOT cumulative share."""
    # A running total plotted as "share over time" flattens: a router that went single-arm over
    # its last fifty sessions still shows the mix of everything before it, so the collapse this
    # panel exists to reveal is diluted by history that stopped being true.
    out: list[tuple[datetime, dict[str, float]]] = []
    for index, row in enumerate(live):
        if row.timestamp is None:
            continue
        recent = Counter(r.model_chosen for r in live[max(0, index - window + 1) : index + 1])
        total = sum(recent.values()) or 1
        out.append((row.timestamp, {model: n / total for model, n in recent.items()}))
    return out


def _rolling(
    live: list[SessionRow], window: int, metric: Callable[[list[str]], float]
) -> list[tuple[int, float]]:
    names = [row.model_chosen for row in live]
    return [
        (index, float(metric(names[max(0, index - window + 1) : index + 1])))
        for index in range(len(names))
    ]


def normalized_entropy(names: list[str], candidate_models: int | None) -> float:
    """Choice entropy over the arms the router COULD have picked, as loop-health defines it."""
    # Mirrors `loop_health._routing_collapse`: log2 bits over log2(candidate arms), clamped at
    # 1.0, and 1.0 (no alarm possible) below two arms. Dividing by the arms OBSERVED instead —
    # the shortcut this replaced — reports a router pinned to two of six models as perfectly
    # diverse, i.e. all-clear on exactly the collapse the panel exists to catch. A test pins
    # this against the shipped computation rather than against a number written here.
    if candidate_models is None or candidate_models <= 1:
        return 1.0
    counts = Counter(names)
    total = sum(counts.values())
    if total == 0:
        return 1.0
    bits = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return min(bits / math.log2(candidate_models), 1.0)


def _share_of(names: list[str], subset: set[str]) -> float:
    return sum(1 for n in names if n in subset) / len(names) if names else 0.0


# --------------------------------------------------------------- F6 escalation


@dataclass(frozen=True)
class EscalationData:
    """Escalation rate, rungs, holds — with the untokenised hold recovered as its own bar."""

    rates: list[tuple[str, int, int]]  # window label, escalated, live sessions
    rungs: list[tuple[str, int]]
    holds: list[tuple[str, int]]
    unknown_holds: list[tuple[str, int]]
    n_undeliverable: int
    outcomes: list[tuple[str, int, int]]  # cohort, successes, labeled
    n_live: int
    n_records: int = field(default=0)


def escalation(
    rows: list[SessionRow], windows: tuple[int | None, ...] = (7, 30, None)
) -> EscalationData:
    """F6's inputs. Holds split into the five tokens, anything else, and the derived case."""
    live = [row for row in rows if row.stratum == LIVE]
    holds = Counter(row.hold_reason for row in live if row.hold_reason is not None)
    escalated = [row for row in live if row.rung is not None]
    rungs = Counter(row.rung for row in escalated)
    plain = [row for row in live if row.rung is None]
    return EscalationData(
        rates=[_rate(live, w) for w in windows],
        rungs=[(token, rungs.get(token, 0)) for token in RUNG_TOKENS],
        holds=[(token, holds.get(token, 0)) for token in HOLD_TOKENS],
        unknown_holds=sorted((t, n) for t, n in holds.items() if t not in HOLD_TOKENS),
        n_undeliverable=sum(1 for row in live if row.undeliverable),
        outcomes=[
            ("escalated", *_verified(escalated)),
            ("not escalated", *_verified(plain)),
        ],
        n_live=len(live),
        n_records=sum(1 for row in live if row.hold_reason is not None or row.rung is not None),
    )


def _rate(live: list[SessionRow], days: int | None) -> tuple[str, int, int]:
    members = [row for row in live if _in_window(row, days)]
    return (_window_label(days), sum(1 for row in members if row.rung is not None), len(members))


def _verified(rows: list[SessionRow]) -> tuple[int, int]:
    labeled = [row for row in rows if row.tier2_success is not None]
    return (sum(1 for row in labeled if row.tier2_success), len(labeled))


# ---------------------------------------------------------------------- F7 OPE


# `_usable`, `_weights` and `_DEFAULT_WEIGHT_CLIP` are `shunt.analysis.ope`'s own internals,
# imported for the same reason `estimators.py` imports them: the diagnostics panel has to show
# the WEIGHTS THE SHIPPED ESTIMATOR USED, and a local re-derivation of the clip or the target
# probability would let the panel and the number it explains drift apart — which is exactly the
# failure the family exists to catch. The floors are imported rather than restated for the same
# reason: a figure that hardcoded 5 would keep drawing a green line after the floor moved.
FLOOR_PER_ARM: Final[int] = _MIN_PER_ARM
FLOOR_CLUSTERS: Final[int] = _MIN_CLUSTERS
FLOOR_PROPENSITY: Final[float] = _MIN_PROPENSITY

WEIGHT_CLIP: Final[float] = _DEFAULT_WEIGHT_CLIP

ROUTING_POLICY: Final[str] = "top-scored candidate"
ALWAYS_POLICY: Final[str] = "always_escalate"
NEVER_POLICY: Final[str] = "never_escalate"


@dataclass(frozen=True)
class LegEstimates:
    """One (leg, target policy) pair: the three estimators, and the refusal when there is one."""

    leg: str
    policy: str
    estimate: PolicyValueEstimate  # DR, its interval, the contrast and the counts
    ips: float | None
    snips: float | None
    quotable: tuple[str, ...]  # estimator names whose instrument certificate cleared
    on_policy_mean: float | None  # what the LOGGED policy paid, over the same rows
    n_rewarded: int

    @property
    def identified(self) -> bool:
        return self.estimate.identified

    def value(self, estimator: str) -> float | None:
        """This leg's point estimate under one estimator, or None when it has none."""
        return {"ips": self.ips, "snips": self.snips, "dr": self.estimate.dr_estimate}[estimator]


@dataclass(frozen=True)
class LegDiagnostics:
    """Why a leg was or was not identified: the weights, and every floor against its measurement."""

    leg: str
    weights: list[float]
    ess_fraction: float
    n_clipped: int
    min_propensity: float | None
    n_logged: int
    n_usable: int
    n_escalated: int
    n_held: int
    n_clusters: int

    @property
    def floors(self) -> list[tuple[str, float, float]]:
        """Each floor as (label, measured, floor) — the ledger panel D divides and draws."""
        return [
            ("target arm rows", float(self.n_escalated), float(FLOOR_PER_ARM)),
            ("other arm rows", float(self.n_held), float(FLOOR_PER_ARM)),
            ("independent sessions", float(self.n_clusters), float(FLOOR_CLUSTERS)),
            ("min propensity", self.min_propensity or 0.0, FLOOR_PROPENSITY),
        ]


@dataclass(frozen=True)
class OpeData:
    """F7's inputs: one entry per target policy, the diagnostics, and the instrument verdict."""

    headline: str
    admissible: bool
    admissibility_reason: str
    routing: LegEstimates
    escalation: tuple[LegEstimates, ...]  # always_escalate, then never_escalate
    diagnostics: tuple[LegDiagnostics, ...]  # routing, then escalation


def ope(store: OutcomeStore, estimates: CertifiedEstimates) -> OpeData:
    """F7's inputs. A NOT_IDENTIFIED leg is a result, so it is carried, never dropped."""
    routing_rows = routing_agreement_rows(rows_from_records(store.routing_ope_rows()))
    escalation_rows = rows_from_records(store.escalation_exploration_rows())
    return OpeData(
        headline=estimates.headline,
        admissible=estimates.admissibility.admissible,
        admissibility_reason=estimates.admissibility.reason,
        routing=_leg(routing_rows, ROUTING, ROUTING_POLICY, estimates, estimates.routing),
        escalation=(
            _leg(escalation_rows, ESCALATION, ALWAYS_POLICY, estimates, estimates.escalation),
            _leg(escalation_rows, ESCALATION, NEVER_POLICY, estimates, _held(escalation_rows)),
        ),
        diagnostics=(
            _diagnostics(routing_rows, ROUTING),
            _diagnostics(escalation_rows, ESCALATION),
        ),
    )


def _held(rows: list[ExplorationLogRow]) -> PolicyValueEstimate:
    """The hold-everything comparator, on the same rows and the same shipped orchestrator."""
    return estimate_policy_value(rows, never_escalate)


def _leg(
    rows: list[ExplorationLogRow],
    leg: str,
    policy: str,
    estimates: CertifiedEstimates,
    estimate: PolicyValueEstimate,
) -> LegEstimates:
    """One target policy's three estimators, gated on the certificate each one earned."""
    target = never_escalate if policy == NEVER_POLICY else always_escalate
    rewarded = [row.reward for row in rows if row.reward is not None]
    return LegEstimates(
        leg=leg,
        policy=policy,
        estimate=estimate,
        ips=ips_estimate(rows, target),
        snips=snips_estimate(rows, target),
        quotable=tuple(name for name in ESTIMATORS if estimates.quotable(leg, name)),
        on_policy_mean=statistics.fmean(rewarded) if rewarded else None,
        n_rewarded=len(rewarded),
    )


def _diagnostics(rows: list[ExplorationLogRow], leg: str) -> LegDiagnostics:
    """The overlap evidence, computed on the rows the estimator may weight — its own population."""
    usable = _usable(rows)
    weights, n_clipped = _weights(usable, always_escalate, _DEFAULT_WEIGHT_CLIP)
    escalated = sum(1 for row in usable if row.escalated)
    return LegDiagnostics(
        leg=leg,
        weights=sorted(weights),
        ess_fraction=ess_fraction(weights),
        n_clipped=n_clipped,
        min_propensity=min((row.propensity for row in usable), default=None),
        n_logged=len(rows),
        n_usable=len(usable),
        n_escalated=escalated,
        n_held=len(usable) - escalated,
        n_clusters=len({row.session_id or row.checkpoint_id for row in usable}),
    )
