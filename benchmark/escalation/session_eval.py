"""Session-cadence escalation value: what the PRODUCT's per-session mechanism is worth."""

# The per-step sweep (`policy_eval`) measures a policy production does not run: live, a verified
# outcome is produced once per CLOSED SESSION (the off-wire verifier runs at the session boundary),
# so the escalation rule counts failures across sessions, never within one. The committed corpus has
# no multi-session trajectories, which is why the per-step eval exists at all — but it does hold
# several (model, arm) sessions per INSTANCE, i.e. repeated independent attempts at the same task.
# Ordered by price and read as a session sequence, those attempts measure the session-cadence
# question directly:
#
#   given a CHEAP session failed on this task, does escalating the NEXT session to a FRONTIER
#   model resolve it more often than a same-cost retry would?
#
# At session cadence the detector is trivially satisfied — the first session failed with the task's
# target failing-check id, which IS the recurrence key — so the question is pure VALUE, not
# detection. The numbers below are observational, not causal: the arms ran in parallel, and which
# instances got frontier coverage was adaptive. The overlap subset (instances with >=2 cheap AND a
# frontier session) is reported separately so the two rates are read on the SAME instances, which
# removes the worst of the coverage-selection bias. On the current corpus that subset still shows a
# ~2.5x resolution lift for escalating.
#
# This module reports no skill verdict and gates nothing — it is context for the per-step sweep, and
# its samples are small. Its numbers answer "is the escalation ladder pointed the right way",
# not "is the trigger well-tuned".

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark.escalation import features, metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.schema import Trajectory

# How many of the most-expensive models present in the corpus count as "frontier" for the
# escalation ladder. Two, not one: the ladder steps one rank at a time, so the top two prices are
# both legitimate escalation targets on this corpus.
_FRONTIER_TOP: Final[int] = 2
# The boundary below which a model counts as "cheap" for the session sequence: the cheapest price
# present. On this corpus that is deepseek-v4-flash; a retry at that same price is the
# no-escalation counterfactual.
_CHEAP_BOTTOM: Final[int] = 1


@dataclass(frozen=True)
class SessionCadenceReport:
    """The session-cadence escalation contrast, measured on the overlap subset."""

    n_overlap_instances: int
    n_escalated: int
    n_escalated_resolved: int
    escalate_rate: float
    escalate_ci: tuple[float, float]
    n_retried: int
    n_retried_resolved: int
    retry_rate: float
    retry_ci: tuple[float, float]
    # The full-corpus context: frontier resolve rate wherever a cheaper session failed, and the
    # cheap base rate — so a reader can see the subset's selection without trusting it.
    n_frontier_after_fail: int
    n_frontier_after_fail_resolved: int
    frontier_after_fail_rate: float
    cheap_base_rate: float

    @property
    def lift(self) -> float | None:
        """escalate_rate / retry_rate — resolution escalation buys over a same-cost retry."""
        if self.retry_rate <= 0.0:
            return None
        return self.escalate_rate / self.retry_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "n_overlap_instances": self.n_overlap_instances,
            "escalate": {
                "n": self.n_escalated,
                "resolved": self.n_escalated_resolved,
                "rate": round(self.escalate_rate, 4),
                "ci95": [round(v, 4) for v in self.escalate_ci],
            },
            "cheap_retry": {
                "n": self.n_retried,
                "resolved": self.n_retried_resolved,
                "rate": round(self.retry_rate, 4),
                "ci95": [round(v, 4) for v in self.retry_ci],
            },
            "lift": _rounded(self.lift),
            "context": {
                "n_frontier_after_fail": self.n_frontier_after_fail,
                "n_frontier_after_fail_resolved": self.n_frontier_after_fail_resolved,
                "frontier_after_fail_rate": round(self.frontier_after_fail_rate, 4),
                "cheap_base_rate": round(self.cheap_base_rate, 4),
            },
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _price_ranks(trajectories: Sequence[Trajectory]) -> dict[str, int]:
    """Model -> ascending price rank among the models PRESENT in this corpus (0 = cheapest)."""
    # Read off the registry's own price ordering, never a literal list: `ModelPool.ranked_models`
    # sorts by total list price, which is the same order the router's capability rank uses.
    from shunt.models.config import ModelPool  # noqa: PLC0415

    pool = ModelPool()
    prices: dict[str, float] = {}
    for model in pool.ranked_models():
        if model.pricing is not None:
            prices[model.name] = model.pricing.input_cost_per_1m + model.pricing.output_cost_per_1m
    present = {features.model_of(t) for t in trajectories}
    ordered = sorted((m for m in present if m in prices), key=lambda m: prices[m])
    return {m: i for i, m in enumerate(ordered)}


def _bucket_of(rank: int, n_models: int) -> str:
    if rank < _CHEAP_BOTTOM:
        return "cheap"
    if rank >= n_models - _FRONTIER_TOP:
        return "frontier"
    return "mid"


def session_cadence(trajectories: Sequence[Trajectory]) -> SessionCadenceReport | None:
    """The escalation-vs-cheap-retry contrast at the session cadence, on the overlap subset.

    None when the corpus cannot support the estimate — no task has both >=2 cheap sessions AND
    a frontier session — so the caller can skip the figure rather than draw NaN bars.
    """
    ranks = _price_ranks(trajectories)
    n_models = len(ranks)
    by_instance: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        by_instance[features.group_of(traj)].append(traj)

    def bucket(traj: Trajectory) -> str:
        # A model absent from the registry is neither a cheap base pick nor a frontier target;
        # it is "mid" and takes no part in either arm of the contrast.
        rank = ranks.get(features.model_of(traj))
        if rank is None:
            return "mid"
        return _bucket_of(rank, n_models)

    cheap_base = metrics.prevalence(
        [t.header.terminal_resolved for t in trajectories if bucket(t) == "cheap"]
    )
    # Full-corpus context: frontier resolve wherever a cheaper session failed.
    frontier_after_fail: list[bool] = []
    for inst in by_instance.values():
        sessions = sorted(inst, key=lambda t: ranks.get(features.model_of(t), n_models))
        if any(not s.header.terminal_resolved for s in sessions if bucket(s) == "cheap"):
            frontier_after_fail.extend(
                s.header.terminal_resolved for s in sessions if bucket(s) == "frontier"
            )

    # The controlled subset: instances with BOTH >=2 cheap sessions AND a frontier session, so
    # the escalate and retry arms are read on the same instances (adaptive-coverage selection
    # removed as far as this corpus allows).
    escalate_outcomes: list[bool] = []
    retry_outcomes: list[bool] = []
    n_overlap = 0
    for inst in by_instance.values():
        cheap = [s for s in inst if bucket(s) == "cheap"]
        frontier = [s for s in inst if bucket(s) == "frontier"]
        if len(cheap) < 2 or not frontier:
            continue
        n_overlap += 1
        if any(not s.header.terminal_resolved for s in cheap):
            escalate_outcomes.extend(s.header.terminal_resolved for s in frontier)
        cheap_by_length = sorted(cheap, key=lambda s: s.header.n_steps)
        for i in range(1, len(cheap_by_length)):
            if not cheap_by_length[i - 1].header.terminal_resolved:
                retry_outcomes.append(cheap_by_length[i].header.terminal_resolved)

    if not n_overlap:
        # No task carries both a retry arm and a frontier arm: neither rate is estimable and the
        # contrast is meaningless, so report "not supported" rather than NaN bars.
        return None

    return SessionCadenceReport(
        n_overlap_instances=n_overlap,
        n_escalated=len(escalate_outcomes),
        n_escalated_resolved=sum(escalate_outcomes),
        escalate_rate=metrics.prevalence(escalate_outcomes),
        escalate_ci=(
            metrics.wilson_interval(sum(escalate_outcomes), len(escalate_outcomes))
            if escalate_outcomes
            else (float("nan"), float("nan"))
        ),
        n_retried=len(retry_outcomes),
        n_retried_resolved=sum(retry_outcomes),
        retry_rate=metrics.prevalence(retry_outcomes),
        retry_ci=(
            metrics.wilson_interval(sum(retry_outcomes), len(retry_outcomes))
            if retry_outcomes
            else (float("nan"), float("nan"))
        ),
        n_frontier_after_fail=len(frontier_after_fail),
        n_frontier_after_fail_resolved=sum(frontier_after_fail),
        frontier_after_fail_rate=metrics.prevalence(frontier_after_fail),
        cheap_base_rate=cheap_base,
    )


__all__ = ["SessionCadenceReport", "session_cadence"]
