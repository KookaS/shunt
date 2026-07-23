"""Loop-health telemetry: is the learning loop healthy, independent of reward?"""

# The load-bearing signal is the routing-collapse alarm, which keys on the model-choice
# distribution ALONE. A degenerate feedback loop keeps reward looking fine while the policy
# ossifies onto one arm, so a reward-based detector is blind to it. Every function feeding the
# collapse alarm takes only choices (never outcomes) — that is what makes the alarm
# reward-independent.

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopHealthThresholds:
    """Alarm knobs. Defaults chosen for the router's 'cheap-first, frontier-tail' thesis."""

    # An arm whose mean selection propensity falls below this has effectively stopped
    # being tried (MNAR censoring) — the router no longer explores it.
    propensity_epsilon: float = 0.01
    # The product routes the routine ~70-80% away from frontier; a recent frontier share
    # at or above half the window is a collapse toward the expensive tail.
    frontier_share_alarm: float = 0.5
    # Normalized (0..1) choice entropy at or below this means the distribution has
    # concentrated onto a few arms.
    entropy_collapse: float = 0.5
    recent_window: int = 100


@dataclass(frozen=True)
class LoopHealthSnapshot:
    """Raw aggregates read from the store, before any metric is derived."""

    total_sessions: int
    eligible_sessions: int
    verified_labeled: int
    any_labeled: int
    model_propensities: list[tuple[str, int, float, float]]  # model, n, mean_p, min_p
    recent_choices: list[str]  # reward-independent: model_chosen only, no outcome join
    cost_by_model: list[tuple[str, int, float]]  # model, n, total_cost (cost_known only)


@dataclass(frozen=True)
class LabelCoverage:
    total_sessions: int
    eligible_sessions: int
    verified_labeled: int
    any_labeled: int
    verified_coverage: float
    labeled_coverage: float


@dataclass(frozen=True)
class ModelPropensity:
    model: str
    n: int
    mean_propensity: float
    min_propensity: float
    support_deficient: bool


@dataclass(frozen=True)
class RoutingCollapse:
    window_size: int
    frontier_share: float
    choice_entropy: float  # normalized 0..1
    entropy_bits: float
    distinct_models: int
    frontier_share_alarm: bool
    entropy_collapse_alarm: bool
    alarm: bool


@dataclass(frozen=True)
class CostObservability:
    router_cost: float
    frontier_mean_cost: float | None
    frontier_baseline_cost: float | None
    cost_ratio: float | None
    n_frontier: int
    n_total: int


@dataclass(frozen=True)
class LoopHealth:
    label_coverage: LabelCoverage
    propensity_support: list[ModelPropensity]
    support_deficient_models: list[str]
    routing_collapse: RoutingCollapse
    cost: CostObservability


def _label_coverage(snapshot: LoopHealthSnapshot) -> LabelCoverage:
    eligible = snapshot.eligible_sessions
    verified = snapshot.verified_labeled / eligible if eligible else 0.0
    labeled = snapshot.any_labeled / eligible if eligible else 0.0
    return LabelCoverage(
        total_sessions=snapshot.total_sessions,
        eligible_sessions=eligible,
        verified_labeled=snapshot.verified_labeled,
        any_labeled=snapshot.any_labeled,
        verified_coverage=verified,
        labeled_coverage=labeled,
    )


def _propensity_support(
    rows: list[tuple[str, int, float, float]],
    candidate_models: set[str],
    thresholds: LoopHealthThresholds,
) -> list[ModelPropensity]:
    # Seed every candidate arm so an arm the router stopped selecting ENTIRELY (zero rows)
    # still surfaces as support-deficient — that fully-censored arm is the strongest MNAR
    # case, and a GROUP BY over observed sessions alone would render it invisible.
    observed = {model: (n, mean_p, min_p) for model, n, mean_p, min_p in rows}
    out: list[ModelPropensity] = []
    for model in sorted(candidate_models | set(observed)):
        n, mean_p, min_p = observed.get(model, (0, 0.0, 0.0))
        out.append(
            ModelPropensity(
                model=model,
                n=n,
                mean_propensity=mean_p,
                min_propensity=min_p,
                support_deficient=mean_p < thresholds.propensity_epsilon,
            )
        )
    return out


def _entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    bits = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            bits -= p * math.log2(p)
    return bits


def _routing_collapse(
    choices: list[str],
    frontier_models: set[str],
    num_candidate_models: int,
    thresholds: LoopHealthThresholds,
) -> RoutingCollapse:
    # Reward-independent by construction: this reads only the choice sequence. The alarm
    # cannot be quieted by good outcomes because no outcome ever reaches it.
    window = len(choices)
    if window == 0:
        return RoutingCollapse(0, 0.0, 1.0, 0.0, 0, False, False, False)
    counts = Counter(choices)
    frontier_n = sum(c for model, c in counts.items() if model in frontier_models)
    frontier_share = frontier_n / window
    entropy_bits = _entropy_bits(list(counts.values()))
    # With 0 or 1 candidate arm there is no distribution to collapse — neither alarm applies.
    collapse_possible = num_candidate_models > 1
    normalized = (
        min(entropy_bits / math.log2(num_candidate_models), 1.0) if collapse_possible else 1.0
    )
    fs_alarm = collapse_possible and frontier_share >= thresholds.frontier_share_alarm
    ent_alarm = collapse_possible and normalized <= thresholds.entropy_collapse
    return RoutingCollapse(
        window_size=window,
        frontier_share=frontier_share,
        choice_entropy=normalized,
        entropy_bits=entropy_bits,
        distinct_models=len(counts),
        frontier_share_alarm=fs_alarm,
        entropy_collapse_alarm=ent_alarm,
        alarm=fs_alarm or ent_alarm,
    )


def _cost_observability(
    rows: list[tuple[str, int, float]], frontier_models: set[str]
) -> CostObservability:
    # Observability only — NOT the kill-gate. The fixed-frontier baseline is a crude
    # counterfactual: the mean realized cost of frontier sessions extrapolated across
    # every costed session. The peeking-safe adjudication lives elsewhere.
    router_cost = sum(total for _, _, total in rows)
    n_total = sum(n for _, n, _ in rows)
    frontier_total = sum(total for model, _, total in rows if model in frontier_models)
    frontier_n = sum(n for model, n, _ in rows if model in frontier_models)
    frontier_mean: float | None = None
    baseline: float | None = None
    ratio: float | None = None
    if frontier_n > 0:
        frontier_mean = frontier_total / frontier_n
        baseline = frontier_mean * n_total
        ratio = router_cost / baseline if baseline else None
    return CostObservability(
        router_cost=router_cost,
        frontier_mean_cost=frontier_mean,
        frontier_baseline_cost=baseline,
        cost_ratio=ratio,
        n_frontier=frontier_n,
        n_total=n_total,
    )


def compute_loop_health(
    snapshot: LoopHealthSnapshot,
    *,
    frontier_models: set[str],
    candidate_models: set[str],
    thresholds: LoopHealthThresholds | None = None,
) -> LoopHealth:
    """Derive every loop-health metric from a raw snapshot. Pure — no I/O."""
    thresholds = thresholds or LoopHealthThresholds()
    support = _propensity_support(snapshot.model_propensities, candidate_models, thresholds)
    return LoopHealth(
        label_coverage=_label_coverage(snapshot),
        propensity_support=support,
        support_deficient_models=[p.model for p in support if p.support_deficient],
        routing_collapse=_routing_collapse(
            snapshot.recent_choices, frontier_models, len(candidate_models), thresholds
        ),
        cost=_cost_observability(snapshot.cost_by_model, frontier_models),
    )
