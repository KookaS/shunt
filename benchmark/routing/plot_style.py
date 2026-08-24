"""Corpus-coupled plotting helpers, plus a re-export of the shipped shared half."""

# The statistical, palette and layout helpers moved into `shunt.inspect.plot_style` so the
# shipped figure code can use them without importing `benchmark` (SH006 forbids that
# direction). What stays here is everything typed on `RawResults` — the benchmark's raw
# challenge x model x arm cache — which has no place in the wheel. Every existing
# `benchmark.routing.plot_style.<name>` import keeps working.

from __future__ import annotations

from dataclasses import dataclass

from shunt.inspect.plot_style import (
    _ARM_BASE_SIZE,  # noqa: F401 (re-export: tests pin the arm size ramp)
    _ARM_SIZE_STEP,  # noqa: F401 (re-export: tests pin the arm size ramp)
    ARM_SWEEP_PENDING_NOTE,
    MIN_N_PROVISIONAL,
    OKABE_ITO,
    TRISTATE_FAIL,
    TRISTATE_PASS,
    TRISTATE_UNSAMPLED,
    UNEVEN_COVERAGE_NOTE,
    LabelCluster,
    LabelPoint,
    Point,
    area_under_frontier,
    arm_marker_size,
    arm_size_legend_values,
    ci_footer,
    ci_yerr,
    fit_end_labels,
    free_region,
    is_provisional,
    label_clusters,
    label_extent,
    model_color_map,
    pareto_prune,
    place_labels,
    stack_labels,
    upper_hull,
    usd,
    wilson_interval,
)

__all__ = [
    "ARM_SWEEP_PENDING_NOTE",
    "MIN_N_PROVISIONAL",
    "OKABE_ITO",
    "TRISTATE_FAIL",
    "TRISTATE_PASS",
    "TRISTATE_UNSAMPLED",
    "UNEVEN_COVERAGE_NOTE",
    "ArmStats",
    "LabelCluster",
    "LabelPoint",
    "Point",
    "RawResults",
    "area_under_frontier",
    "arm_columns",
    "arm_marker_size",
    "arm_size_legend_values",
    "arm_stats",
    "ci_footer",
    "ci_yerr",
    "fit_end_labels",
    "free_region",
    "is_provisional",
    "label_extent",
    "is_single_arm",
    "model_color_map",
    "pareto_prune",
    "label_clusters",
    "place_labels",
    "stack_labels",
    "row_real_cost",
    "upper_hull",
    "usd",
    "wilson_interval",
]

# challenge_id -> model -> arm_id -> outcome row (config.load_results() shape).
RawResults = dict[str, dict[str, dict[str, dict]]]


# ---------------------------------------------------------------------------
# (model, arm) coverage data helpers — the uneven-sampling-by-design axis.
# ---------------------------------------------------------------------------


def arm_columns(raw: RawResults) -> list[tuple[str, str]]:
    """Every (model, arm) pair observed anywhere in the raw challenge x model x
    arm cache, sorted for a deterministic column order."""
    seen: set[tuple[str, str]] = set()
    for per_model in raw.values():
        for model, per_arm in per_model.items():
            seen.update((model, arm) for arm in per_arm)
    return sorted(seen)


@dataclass(frozen=True)
class ArmStats:
    """Sampled-n, pass-count, and cost for one (model, arm) column across all tasks."""

    model: str
    arm: str
    n: int
    passes: int
    total_cost: float

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.n if self.n else 0.0

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_interval(self.passes, self.n)

    @property
    def provisional(self) -> bool:
        return is_provisional(self.n)


def row_real_cost(row: dict) -> float:
    """The provider's own billed cost for one row — `real_cost`, never `estimated_cost`."""
    # Falls back to `cost` only when `real_cost` is absent (older rows, hand-built test
    # fixtures). `cost` and `real_cost` happen to agree on every row of the current
    # results.csv, so reading `cost` is right by luck; `estimated_cost` is ~4.8x larger
    # across the same file, so a row written with a divergent `cost` would silently
    # inflate every cost axis. Naming the column makes the invariant a construction.
    value = row.get("real_cost")
    if value is None or value == "":
        value = row.get("cost", 0.0)
    return float(value)


def arm_stats(raw: RawResults, model: str, arm: str) -> ArmStats:
    """Aggregate one (model, arm) column's sampled n / passes / real cost across all tasks."""
    n = 0
    passes = 0
    cost = 0.0
    for per_model in raw.values():
        row = per_model.get(model, {}).get(arm)
        if row is not None:
            n += 1
            if row.get("pass"):
                passes += 1
            cost += row_real_cost(row)
    return ArmStats(model=model, arm=arm, n=n, passes=passes, total_cost=cost)


def is_single_arm(raw: RawResults) -> bool:
    """True iff every model in the cache has exactly one observed arm — the
    current committed-data reality (live per-arm execution isn't wired up yet):
    degrade gracefully rather than imply fake arm variation.
    """
    per_model_arms: dict[str, set[str]] = {}
    for per_model in raw.values():
        for model, per_arm in per_model.items():
            per_model_arms.setdefault(model, set()).update(per_arm)
    return all(len(arms) <= 1 for arms in per_model_arms.values())
