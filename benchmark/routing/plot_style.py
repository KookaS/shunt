"""Shared plotting infra for the arm-aware benchmark report."""

# Built once, reused by every K1-K4 / N1-N6 plot function in report.py: Wilson
# score CIs on pass-rate estimates, the fixed Okabe-Ito model palette with an
# arm-rank size ramp nested inside each hue, the cost-quality Pareto frontier
# (non-decreasing convex hull) + an AIQ-style area scalar, and the tri-state
# pass/fail/not-sampled encoding.
#
# Color note: the model palette is genuine Okabe-Ito (matches the existing
# CB_PALETTE convention in plot_strategies.py and the task's
# design brief), validated with the dataviz skill's six-check validator — all
# 6 slots PASS in both light (surface #fcfcfb) and dark (#1a1a19) modes on the
# adjacent pairlist; the all-pairs pairlist (scatter/bubble/small-multiples,
# where any two models can be neighbors) lands CVD in the 6-8 floor band, so
# every scatter-form plot using this palette MUST ship secondary encoding
# (direct labels) — never color alone. The same hex values serve both themes:
# Okabe-Ito is an externally fixed standard (not re-stepped per surface);
# contrast, CVD, and the normal-vision floor all clear on both surfaces, only
# the design system's own "lightness band" cosmetic guideline (tuned for its
# own ramps) does not apply.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Point = tuple[float, float]
# challenge_id -> model -> arm_id -> outcome row (config.load_results() shape).
RawResults = dict[str, dict[str, dict[str, dict]]]

# ---------------------------------------------------------------------------
# Wilson score interval — valid near 0/1 and at small n, unlike a normal
# approximation. The #1 fix this design brief calls for: p(arm|model) sampling
# makes per-arm n uneven, so every pass-rate estimate must carry a CI.
# ---------------------------------------------------------------------------

MIN_N_PROVISIONAL: Final[int] = 10

# Shared caveat text for the degrade-gracefully case (only the default arm has
# data yet — the live executor doesn't issue divergent per-arm requests yet).
ARM_SWEEP_PENDING_NOTE: Final[str] = "single-arm data — arm sweep pending"
UNEVEN_COVERAGE_NOTE: Final[str] = "uneven n/N per column is BY DESIGN (p(arm|model) sampling)"


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI (default z=1.96 -> 95%) for a binomial pass rate."""
    if n <= 0:
        return (0.0, 0.0)
    phat = passes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def ci_yerr(rate: float, lo: float, hi: float) -> tuple[float, float]:
    """(lower, upper) error-bar offsets from ``rate`` for matplotlib's ``yerr=``."""
    return (max(0.0, rate - lo), max(0.0, hi - rate))


def is_provisional(n: int, min_n: int = MIN_N_PROVISIONAL) -> bool:
    """True when a sample is too small to trust — render hollow/greyed, not solid."""
    return n < min_n


def ci_footer(method: str = "Wilson", level: float = 0.95) -> str:
    """One-line caption stating the CI method + level (state it, never imply it)."""
    return f"error bars = {level:.0%} {method} CI (binomial)"


# ---------------------------------------------------------------------------
# Okabe-Ito categorical palette — model = hue, fixed order, never cycled.
# Arm-rank is a channel NESTED INSIDE the hue (marker size), never a shared
# axis across models: arms are ordinal WITHIN one model only.
# ---------------------------------------------------------------------------

OKABE_ITO: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#56B4E9",  # sky blue     -- contrast WARN vs white surface: always direct-label
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange       -- contrast WARN vs white surface: always direct-label
    "#F0E442",  # yellow
    "#000000",  # black (reserved; not assigned to a series)
)


def model_color_map(models_in_order: Sequence[str]) -> dict[str, str]:
    """Assign each model its fixed Okabe-Ito hue by POSITION in ``models_in_order``."""
    # Callers must compute this once from the full/stable model list (e.g.
    # tier-then-price order) and pass the same map to every figure — never
    # re-derive it from a filtered subset, or a model's color would repaint
    # when the subset changes (the recolor-on-filter anti-pattern).
    return {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(models_in_order)}


# ---------------------------------------------------------------------------
# Arm-rank encoding: marker size grows with within-model reasoning-effort rank.
# ---------------------------------------------------------------------------

_ARM_BASE_SIZE: Final[float] = 45.0
_ARM_SIZE_STEP: Final[float] = 75.0


def arm_marker_size(rank: int, max_rank: int) -> float:
    """Marker area (pts^2) for a scatter point, growing with within-model arm rank."""
    # max_rank is the model's OWN highest rank — never compared across models
    # (a rank-1 arm on a 2-arm model is not the same effort as rank-1 on a
    # 4-arm model; the size only orders arms within one model's facet/hue).
    if max_rank <= 0:
        return _ARM_BASE_SIZE
    frac = max(0.0, min(1.0, rank / max_rank))
    return _ARM_BASE_SIZE + _ARM_SIZE_STEP * frac


def arm_size_legend_values(max_rank: int) -> list[tuple[int, float]]:
    """[(rank, marker_size), ...] spanning 0..max_rank, for a size-legend swatch."""
    return [(r, arm_marker_size(r, max_rank)) for r in range(max_rank + 1)]


# ---------------------------------------------------------------------------
# Tri-state outcome encoding — grey is NEVER a fail, it means "not sampled".
# ---------------------------------------------------------------------------

TRISTATE_PASS: Final[str] = "#2E7D32"
TRISTATE_FAIL: Final[str] = "#C62828"
TRISTATE_UNSAMPLED: Final[str] = "#BDBDBD"


def label_points_with_leaders(
    ax: Axes,
    points: Sequence[tuple[float, float, str]],
    fontsize: float = 8,
    color: str = "#333333",
    margin_x: float = 1.02,
) -> None:
    """Direct-label points via a leader-line column in the right margin."""
    # Robust to clustering (tightly-packed or identical points never stack raw
    # text on top of itself, the anti-pattern this replaces): labels are
    # ordered top-to-bottom by y descending and spread evenly down the margin,
    # each connected to its point by a thin leader line.
    if not points:
        return
    ordered = sorted(points, key=lambda p: -p[1])
    n = len(ordered)
    for i, (x, y, name) in enumerate(ordered):
        frac_y = 1.0 - (i + 0.5) / n
        ax.annotate(
            name,
            xy=(x, y),
            xycoords="data",
            xytext=(margin_x, frac_y),
            textcoords=("axes fraction", "axes fraction"),
            fontsize=fontsize,
            va="center",
            ha="left",
            color=color,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=2, shrinkB=2),
        )


# ---------------------------------------------------------------------------
# Cost-quality Pareto frontier: NON-DECREASING CONVEX HULL, not keep-max
# staircase — a mixture router reaches interpolated points on the hull edges.
# ---------------------------------------------------------------------------


def pareto_prune(points: Sequence[Point]) -> list[Point]:
    """Keep only non-dominated (cost, pass_rate) points (lower cost AND higher
    rate beats; ties on both do not dominate)."""
    pts = list(points)
    keep: list[Point] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return keep


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def upper_hull(points: Sequence[Point]) -> list[Point]:
    """Upper convex hull of (cost, pass_rate) points, cost ascending."""
    # This is the achievable region a mixture router reaches by
    # probabilistically interpolating between two strategies' (cost, pass)
    # points — the honest cost-quality frontier, not a keep-max staircase that
    # ignores mixtures. Duplicate costs keep only the highest rate.
    by_cost: dict[float, float] = {}
    for cost, rate in points:
        if cost not in by_cost or rate > by_cost[cost]:
            by_cost[cost] = rate
    pts = sorted(by_cost.items())
    if len(pts) <= 2:
        return pts
    hull: list[Point] = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def area_under_frontier(hull: Sequence[Point]) -> float:
    """AIQ-style scalar in [0,1]: area under the (cost, pass%) frontier curve,
    normalized by the bounding rectangle (max cost x 100%). Extends flat from
    x=0 at the cheapest hull point's rate when that point's cost is > 0.
    """
    if not hull:
        return 0.0
    pts = list(hull)
    if pts[0][0] > 0:
        pts = [(0.0, pts[0][1]), *pts]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if xs[-1] <= 0:
        return 0.0
    area = float(np.trapezoid(ys, xs))
    return area / (xs[-1] * 100.0)


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


def arm_stats(raw: RawResults, model: str, arm: str) -> ArmStats:
    """Aggregate one (model, arm) column's sampled n / passes / cost across all tasks."""
    n = 0
    passes = 0
    cost = 0.0
    for per_model in raw.values():
        row = per_model.get(model, {}).get(arm)
        if row is not None:
            n += 1
            if row.get("pass"):
                passes += 1
            cost += float(row.get("cost", 0.0))
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
