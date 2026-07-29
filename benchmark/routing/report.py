#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import zlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import impute, plot_style, summary
from benchmark.routing.impute import ImputedMatrix
from benchmark.routing.metrics import _reward
from benchmark.routing.plot_style import RawResults, row_real_cost, usd
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.oracle import OracleRewardAware

# Strategies that read realised outcomes. They are upper bounds, never routers: a
# live router cannot see the answer, so a hindsight row must never anchor a hull,
# win a headline, or be compared as a peer. Drawn as grey reference markers only.
HINDSIGHT_STRATEGIES: Final[frozenset[str]] = frozenset({"Oracle", "Oracle-reward", "Arm-oracle"})

# A (model, arm) cell below this many tasks is provisional HERE — stricter than
# plot_style's shared floor of 10, because at 10 a single cell was silently
# defining the arm frontier and moving AIQ by 0.03.
MIN_N_RELIABLE: Final[int] = 30

_HINDSIGHT_LABEL: Final[str] = "hindsight — not achievable"


# ---------------------------------------------------------------------------
# Progress + memory diagnostics. This report holds the kNN family's ONNX embedders
# and their per-task index in RSS at once and has peaked near 5 GB; an OOM kill is
# SIGKILL, so nothing it was about to print survives. Every step therefore reports
# as it completes, on a line-buffered stdout, with the peak RSS so far — the last
# line printed names the step that was running when the kernel stepped in.
# ---------------------------------------------------------------------------

_OOM_HEADROOM_MB: Final[float] = 6000.0


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _available_mb() -> float | None:
    """MemAvailable from /proc, or None where the kernel does not publish it."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _memory_preflight() -> None:
    """Warn BEFORE the embedders load when this host cannot hold the report."""
    available = _available_mb()
    if available is None:
        return
    print(f"Memory: {available:,.0f} MB available; this report has peaked near 5,000 MB.")
    if available < _OOM_HEADROOM_MB:
        print(
            f"WARNING: under {_OOM_HEADROOM_MB:,.0f} MB free — the kernel may OOM-kill this "
            "run (exit 137) mid-figure. Free memory or run it alone.",
            file=sys.stderr,
        )


def _step(label: str, value: object) -> None:
    """One completed step, with the peak RSS reached by the time it finished."""
    print(f"  {label:13}: {value}   [peak RSS {_peak_rss_mb():,.0f} MB]")


def _is_hindsight(name: str) -> bool:
    """True for a strategy that peeks at realised outcomes (never deployable)."""
    return name in HINDSIGHT_STRATEGIES


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p for ``b``/``c`` discordant pairs (1.0 when none)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def _const_factory(strategy: Strategy) -> Callable[[], Strategy]:
    """Wrap an already-built strategy instance as a 0-arg factory — re-use, not
    rebuild, so a kNN strategy's embeddings/index aren't recomputed per plot.
    """
    return lambda: strategy


def _build_strategy_factories(gamma: float) -> dict[str, Callable[[], Strategy]]:
    """One source of truth for the regret plot's strategy set."""
    # The SAME config-enabled list run_eval.get_strategies reads for every
    # other plot (a hardcoded set here previously added Oracle-reward+Random
    # but omitted an enabled headline strategy — silently absent from the
    # regret plot while still shown everywhere else).
    # Oracle-reward is added unconditionally: the regret plot's internal
    # reference baseline every other strategy's regret is measured against,
    # independent of whether it is itself config-enabled for display.
    from benchmark.routing import run_eval

    factories: dict[str, Callable[[], Strategy]] = {
        s.name: _const_factory(s) for s in run_eval.get_strategies()
    }
    factories.setdefault("Oracle-reward", lambda: OracleRewardAware(gamma=gamma))
    return factories


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_matrix(path: Path) -> dict | None:
    try:
        return config.load_matrix(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _validate_rows(results: list[dict]) -> str | None:
    """Reason the row set cannot be plotted, or ``None`` if it is usable.

    Plotting reads TotalCost/AvgPerf% unguarded; a thin or malformed matrix must
    fail here with a diagnosis rather than half-write artifacts and KeyError later.
    """
    for row in results:
        for field in ("TotalCost", "AvgPerf%"):
            try:
                float(row[field])
            except (KeyError, TypeError, ValueError):
                return f"strategy {row.get('strategy', '?')!r} has no usable {field}"
    if not any(int(float(r.get("n_tasks", 0) or 0)) > 0 for r in results):
        return (
            "no strategy has a single scorable task — every chosen cell is a "
            "coverage gap. Run the matrix before reporting."
        )
    return None


def print_summary(results: list[dict[str, str]]) -> None:
    """Print the strategy ranking to stdout. Rows are derived in-memory from the
    results.csv source of truth (no committed derived CSV).
    """
    rows = sorted(results, key=lambda r: float(r.get("Reward", 0)), reverse=True)
    cols = ["strategy", "n_pass", "AvgPerf%", "TotalCost", "AvgCost", "Reward", "Pareto"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


# ---------------------------------------------------------------------------
# Model palette + arm ranks — computed once, shared by every arm-aware plot.
# ---------------------------------------------------------------------------


def _model_colors() -> dict[str, str]:
    """Okabe-Ito hue per model, fixed by price order (cheapest -> priciest)."""
    try:
        order = config.enabled_models()
    except Exception:  # noqa: BLE001 (registry optional at plot time)
        order = []
    return plot_style.model_color_map(order)


def _arm_ranks() -> dict[tuple[str, str], int]:
    """(model, arm_id) -> within-model rank, from the registry's reasoning bracket."""
    ranks: dict[tuple[str, str], int] = {}
    try:
        resolved = config.resolved_models()
    except Exception:  # noqa: BLE001 (registry optional at plot time)
        return ranks
    for model, mc in resolved.items():
        if mc.reasoning is None:
            continue
        for arm in mc.reasoning.arms:
            ranks[(model, arm.id)] = arm.rank
    return ranks


def _arm_cloud_points(
    raw: RawResults,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
) -> list[dict]:
    """One dict per (model, arm) column: cost/pass_rate/CI/size/color, on the
    per-task cost scale. ``provisional`` uses MIN_N_RELIABLE, not plot_style's
    looser floor — see the constant.
    """
    cols = plot_style.arm_columns(raw)
    max_rank_by_model: dict[str, int] = {}
    for model, arm in cols:
        r = arm_ranks.get((model, arm), 0)
        max_rank_by_model[model] = max(max_rank_by_model.get(model, 0), r)
    points: list[dict] = []
    for model, arm in cols:
        stats = plot_style.arm_stats(raw, model, arm)
        if stats.n == 0:
            continue
        rank = arm_ranks.get((model, arm), 0)
        points.append(
            {
                "model": model,
                "arm": arm,
                "cost": stats.avg_cost,
                "pass_rate": stats.pass_rate * 100,
                "n": stats.n,
                "wilson": stats.wilson,
                "color": model_colors.get(model, "#9E9E9E"),
                "size": plot_style.arm_marker_size(rank, max_rank_by_model.get(model, 0)),
                "provisional": plot_style.is_provisional(stats.n, min_n=MIN_N_RELIABLE),
            }
        )
    return points


def _deployable_pareto(
    names: Sequence[str], costs: Sequence[float], perfs: Sequence[float], ns: Sequence[int]
) -> set[str]:
    """Names that no OTHER DEPLOYABLE strategy beats on both cost and pass rate."""
    # Recomputed here rather than read off the summary's `Pareto` column: that column
    # ranks hindsight oracles alongside routers, so it paints a real router "dominated"
    # when the only thing dominating it cannot be deployed. It is the SINGLE source of
    # Pareto membership on the cost-quality plane — marker colour and hull both read it,
    # so the legend's key and the drawn frontier cannot disagree.
    live = [
        (n, c, p)
        for n, c, p, k in zip(names, costs, perfs, ns, strict=True)
        if not _is_hindsight(n) and k > 0
    ]
    keep: set[str] = set()
    for name, cost, perf in live:
        if not any(
            oc <= cost and op >= perf and (oc < cost or op > perf)
            for on, oc, op in live
            if on != name
        ):
            keep.add(name)
    return keep


def _hull_pareto_indices(names: list[str], pareto_map: dict[str, bool]) -> list[int]:
    """Indices of the DEPLOYABLE Pareto-flagged rows entering the hull/AIQ computation.

    A hindsight oracle is excluded: it anchored the hull's upper vertex, so the shaded
    'achievable region' reached a height no live router can buy (AIQ 0.86 vs 0.78).
    """
    return [
        i for i, name in enumerate(names) if pareto_map.get(name, False) and not _is_hindsight(name)
    ]


# ---------------------------------------------------------------------------
# Footer content. Static construction facts live in the module-level FigureSpec
# constants; anything that depends on the DATA (counts, coverage, imputation,
# whether arm variation exists at all) is computed per run and merged in as
# Annotations, so a caveat can never go stale when the matrix grows.
# ---------------------------------------------------------------------------

_CI_NOTE = "Error bars and bands are 95% Wilson binomial CIs on the pass rate."


def _merge_annotations(*parts: Annotations) -> Annotations:
    """Concatenate several Annotations (plot_frame drops duplicates on merge)."""
    return Annotations(
        definitions=tuple(d for p in parts for d in p.definitions),
        notes=tuple(n for p in parts for n in p.notes),
        limitations=tuple(lim for p in parts for lim in p.limitations),
    )


def _row_coverage_annotations(ns: list[int]) -> Annotations:
    """Whether the plotted strategy rows really share one task denominator."""
    scored = [n for n in ns if n > 0]
    empty = sum(1 for n in ns if n <= 0)
    notes: list[str] = []
    limits: list[str] = []
    if scored and len(set(scored)) == 1:
        # Only the SCORED rows were checked — saying "every plotted strategy" while
        # zero-n rows sit on the canvas contradicts the no-evidence limitation below.
        subject = (
            f"Every scored strategy ({len(scored)} of {len(ns)} plotted)"
            if empty
            else "Every plotted strategy"
        )
        notes.append(f"{subject} is scored on the same {scored[0]} task(s).")
    elif scored:
        limits.append(
            f"Strategies are scored on uneven task counts (n={min(scored)}-{max(scored)}), "
            "so their totals are not strictly comparable."
        )
    if empty:
        limits.append(
            f"{empty} strategy row(s) have no scorable task and sit at \\$0 / 0% — "
            "that is absence of evidence, not efficiency."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _single_arm_limits(raw: RawResults | None, synthesized: bool = False) -> tuple[str, ...]:
    """The 'only the default arm has data' caveat, when that is true of this cache.

    Silent on a SYNTHESIZED cache: collapsing every model to one "default" column is
    an artifact of the missing per-arm cache, not a fact about the sampled arms.
    """
    if synthesized or raw is None or not raw or not plot_style.is_single_arm(raw):
        return ()
    return (
        "Every model has exactly one sampled arm on this data (arm sweep pending), "
        "so nothing here shows reasoning-effort variation.",
    )


def _banner_annotations(banner: str | None) -> Annotations:
    """The equal-coverage/imputation disclosure, as a footer note instead of a caption."""
    return Annotations(notes=(banner,) if banner else ())


def _provisional_limits(points: list[dict], what: str) -> tuple[str, ...]:
    """How many plotted (model, arm) cells rest on fewer than MIN_N_PROVISIONAL tasks."""
    if not points:
        return ()
    thin = sum(1 for p in points if p["provisional"])
    if not thin:
        return ()
    return (
        f"{thin} of {len(points)} {what} rest on fewer than "
        f"{MIN_N_RELIABLE} tasks (drawn hollow) — their pass rate is provisional.",
    )


_HINDSIGHT_NOTE: Final[str] = (
    "Hindsight strategies (Oracle / Oracle-reward / Arm-oracle) read the realised outcome. "
    "They are marked 'hindsight — not achievable' (see READ for this figure's encoding) and "
    "are excluded from every frontier, ranking and headline here."
)

_PARETO_SPEC = FigureSpec(
    reading=(
        "x is the total dollars a routing strategy spent over every task it scored; y is its "
        "average pass rate in percent, with a 95% Wilson error bar. One big marker per strategy: "
        "orange diamond = Always-Frontier, green = on the deployable Pareto set, grey circle = "
        "beaten by another DEPLOYABLE strategy on both axes (or parked at zero with no scorable "
        "task), grey open star = a hindsight oracle. The green dashed line is the convex hull "
        "through the deployable Pareto strategies; the shaded area under it is what a router "
        "mixing two of THOSE can reach."
    ),
    goal=(
        "Aim top-left: the same pass rate for fewer dollars. A strategy well to the LEFT of the "
        "orange diamond at a similar height is the win this benchmark is looking for. Ignore the "
        "stars when judging that — they are upper bounds, not options."
    ),
    definitions=(
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
        ("Pareto-optimal", "no other DEPLOYABLE strategy is both cheaper and better"),
        ("AIQ", "area under the frontier over the plotted-cost x 100% rectangle, 0-1"),
        ("hindsight", "the strategy reads the realised outcome; no live router can"),
    ),
    notes=(
        _CI_NOTE,
        _HINDSIGHT_NOTE,
        "The frontier is a convex hull (mixtures of two strategies), not a best-so-far staircase.",
    ),
    limitations=(
        "AIQ is normalised by the widest cost actually plotted, so it moves when the plotted "
        "set changes — read it within one figure, never across figures or runs.",
        "The per-(model, arm) cell cloud that used to sit behind these markers was removed: its "
        "dots rested on 10-198 differently-selected tasks, which is not comparable with the "
        "strategy denominators. It lives on its own axes in arm_cost_quality_cloud.png.",
    ),
)


def _pareto_annotations(
    ns: list[int], banner: str | None, aiq: float, hindsight: list[str]
) -> Annotations:
    """Runtime footer content for K1: coverage, hull height, imputation, hindsight rows."""
    notes: list[str] = []
    if aiq > 0:
        notes.append(
            f"AIQ={aiq:.2f} of the cost-quality rectangle lies under the DEPLOYABLE frontier."
        )
    if hindsight:
        notes.append(
            f"{len(hindsight)} hindsight row(s) drawn but excluded from the hull and AIQ: "
            + ", ".join(hindsight)
            + "."
        )
    return _merge_annotations(
        _banner_annotations(banner),
        _row_coverage_annotations(ns),
        Annotations(notes=tuple(notes)),
    )


def plot_pareto(
    results: list[dict[str, str]],
    out_dir: Path,
    raw_results: RawResults | None = None,
    n_tasks: int = 0,
    banner: str | None = None,
) -> Path:
    """K1 — cost-quality Pareto plane over DEPLOYABLE strategies. Wilson CI per
    strategy and a convex-hull frontier (the region a mixture router can reach)
    with AIQ in the title; hindsight oracles are drawn as reference stars only.
    """
    _ = (raw_results, n_tasks)  # the (model, arm) underlay moved to its own figure
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]
    # ONE Pareto definition for this plane. The summary's own `Pareto` column ranks the
    # hindsight oracles alongside the routers, so reading it here painted kNN-cascade
    # "dominated" while the legend's colour key called it Pareto-optimal.
    deployable_pareto = _deployable_pareto(names, costs.tolist(), perfs.tolist(), ns)
    pareto_map = {n: n in deployable_pareto for n in names}
    hindsight = [n for n in names if _is_hindsight(n)]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    label_points: list[tuple[float, float, str]] = []
    for i, name in enumerate(names):
        is_frontier = name == "Always-Frontier"
        if _is_hindsight(name):
            color, size, marker = "#9E9E9E", 150, "*"
        elif is_frontier:
            color, size, marker = "#D55E00", 150, "D"
        elif name in deployable_pareto:
            color, size, marker = "#009E73", 110, "o"
        else:
            color, size, marker = "#9E9E9E", 75, "o"
        if ns[i] > 0:
            lo, hi = plot_style.wilson_interval(n_pass[i], ns[i])
            down, up = plot_style.ci_yerr(perfs[i] / 100.0, lo, hi)
            ax.errorbar(
                costs[i],
                perfs[i],
                yerr=[[down * 100], [up * 100]],
                fmt="none",
                ecolor="#555555",
                elinewidth=1,
                capsize=3,
                zorder=4,
            )
        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=size,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
        suffix = f" ({_HINDSIGHT_LABEL})" if _is_hindsight(name) else ""
        label_points.append((float(costs[i]), float(perfs[i]), name + suffix))

    plot_style.label_points_with_leaders(ax, label_points)

    pareto_idx = _hull_pareto_indices(names, pareto_map)
    aiq = 0.0
    if pareto_idx:
        pts = [(float(costs[i]), float(perfs[i])) for i in pareto_idx]
        hull = plot_style.upper_hull(pts)
        aiq = plot_style.area_under_frontier(hull)
        fx = [p[0] for p in hull]
        fy = [p[1] for p in hull]
        if fx and fx[0] > 0:
            fx = [0.0, *fx]
            fy = [fy[0], *fy]
        ax.plot(
            fx,
            fy,
            color="#009E73",
            linewidth=2,
            linestyle="--",
            label=f"Deployable Pareto frontier (convex hull, AIQ={aiq:.2f})",
        )
        ax.fill_between(
            fx,
            0,
            fy,
            alpha=0.06,
            color="#009E73",
            label="achievable region (mixtures of DEPLOYABLE strategies)",
        )

    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Average pass rate (%)")
    ax.set_title(
        f"Cost vs quality — the DEPLOYABLE frontier spans AIQ={aiq:.2f} of the "
        "cost-quality rectangle\n(hindsight oracles drawn as grey stars, excluded from it)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "pareto_scatter.png",
        _PARETO_SPEC,
        extra=_pareto_annotations(ns, banner, aiq, hindsight),
    )


def _evaluate_strategies(
    factory_map: dict[str, Callable[[], Strategy]],
    matrix: dict,
    tasks: list[str],
) -> dict[str, tuple[list[tuple[str, str, bool, float]], set[str]]]:
    """Per strategy: ``(decisions, unscorable)`` from ``summary.evaluate``."""
    # Delegates rather than re-implementing. The private copy that used to live here
    # dropped a cascade's failed-probe cost and skipped the censoring check, so the
    # regret plot charged kNN-cascade $3.09 less than strategy_summary.csv did and
    # printed a headline regret the CSV contradicted.
    return {
        name: summary.evaluate(factory(), matrix, tasks) for name, factory in factory_map.items()
    }


def _compute_per_task_regret(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    gamma: float = 0.1,
    excluded: set[str] | None = None,
) -> np.ndarray:
    """Cumulative regret over scorable tasks only; a task in ``excluded`` (a
    coverage gap) is DROPPED, never scored fail@$0. Filtering both series on the
    same task id keeps strategy and oracle position-aligned."""
    excluded = excluded or set()
    regrets: list[float] = []
    for sd, od in zip(strategy_decisions, oracle_decisions, strict=True):
        if sd[0] in excluded:
            continue
        sr = _reward(sd[2], sd[3], gamma)
        or_ = _reward(od[2], od[3], gamma)
        regrets.append(or_ - sr)
    return np.cumsum(regrets)


def _arm_oracle_decisions(
    raw: RawResults, tasks: list[str], gamma: float
) -> list[tuple[str, str, bool, float]]:
    """Best REALIZED (model, arm) per task among whichever arms were actually
    sampled for it — the ceiling given the sparse p(arm|model) coverage,
    not a full-bracket oracle.
    """
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        per_model = raw.get(tid, {})
        best_key, best_reward, best_outcome = "", -math.inf, (False, 0.0)
        for model, per_arm in per_model.items():
            for arm, row in per_arm.items():
                passed, cost = bool(row.get("pass")), row_real_cost(row)
                r = _reward(passed, cost, gamma)
                if r > best_reward:
                    best_reward, best_key, best_outcome = r, f"{model}:{arm}", (passed, cost)
        decisions.append((tid, best_key, best_outcome[0], best_outcome[1]))
    return decisions


def _arm_bandit_score(
    opt: tuple[str, str, dict], history: dict[tuple[str, str], list[float]]
) -> float:
    hist = history.get((opt[0], opt[1]))
    return (sum(hist) / len(hist)) if hist else math.inf


def _arm_bandit_decisions(
    raw: RawResults, tasks: list[str], gamma: float
) -> list[tuple[str, str, bool, float]]:
    """Illustrative optimistic-greedy bandit over sampled (model, arm) options.
    Single consumer (this plot) — not a production Strategy (the arm sampler
    itself is likewise kept inline for the same one-consumer reason).
    """
    history: dict[tuple[str, str], list[float]] = {}
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        per_model = raw.get(tid, {})
        options = [(m, a, row) for m, per_arm in per_model.items() for a, row in per_arm.items()]
        if not options:
            decisions.append((tid, "", False, 0.0))
            continue
        model, arm, row = max(options, key=lambda opt: _arm_bandit_score(opt, history))
        passed, cost = bool(row.get("pass")), row_real_cost(row)
        history.setdefault((model, arm), []).append(_reward(passed, cost, gamma))
        decisions.append((tid, f"{model}:{arm}", passed, cost))
    return decisions


_REGRET_TERMS = (
    ("regret", "reward the oracle collected that this strategy did not"),
    ("reward", "1 for a pass, 0 for a fail, minus gamma x cost in dollars"),
    ("oracle", "picks the best-reward model per task with hindsight"),
)

_REGRET_SPEC = FigureSpec(
    reading=(
        "x is the task index in evaluation order — an ordinal position, not time and not "
        "difficulty. y is cumulative regret: per task the oracle's reward minus this strategy's "
        "reward, summed left to right. A curve hugging the green 0 line routes about as well as "
        "the oracle; a rising curve is losing reward either by failing a task the oracle solved "
        "(quality regret) or by paying for a bigger model than the task needed (cost regret). "
        "Each curve is indexed over ITS OWN scorable tasks — a coverage gap is dropped, never "
        "scored — so the same x position is not the same task on two curves. The bracket at each "
        "curve's right-hand end is that strategy's bootstrap 95% interval on the TOTAL, taken "
        "from strategy_summary.csv."
    ),
    goal=(
        "Look for the lowest, flattest curve — then check the end brackets before believing the "
        "ordering. Where the top two brackets overlap, the ranking is a leading hypothesis, not "
        "a result. The title ranks routers by TOTAL regret, which is only comparable where two "
        "series span the same task count — when the counts below differ, prefer the SLOPE."
    ),
    definitions=_REGRET_TERMS,
    notes=(
        "The Oracle-reward baseline is flat at 0 by construction: it is the reference every "
        "other curve is measured against, not a competitor.",
        "Totals are computed through summary.evaluate — the same accounting strategy_summary.csv "
        "uses, cascade probe costs included — so the plotted total and the CSV agree.",
    ),
    limitations=(
        "The end bracket is an interval on the TOTAL only. It is not a band around the "
        "trajectory: nothing here bounds the curve at an intermediate task index.",
    ),
)

_REGRET_AGGREGATE_SPEC = FigureSpec(
    reading=(
        "Fallback view, drawn when no task matrix is available for the per-task curves. x is the "
        "strategy name (categorical, no order); y is that strategy's total cumulative regret in "
        "reward units, read straight from its summary row and printed above the bar. A shorter "
        "bar gave up less reward than the oracle would have collected."
    ),
    goal="Look for the shortest bar among the deployable routers — closest to optimal routing.",
    definitions=_REGRET_TERMS,
    notes=(
        "Bar colours are a fixed semantic map (green oracle, red Always-Frontier, blue "
        "Always-Cheap, orange Random, grey everything else); they do not encode model identity.",
    ),
    limitations=(
        "Aggregate totals only: no per-task trajectory, no confidence interval, and no "
        "disclosure of how many tasks each strategy could actually be scored on.",
    ),
)


def _regret_annotations(
    gamma: float,
    tasks: list[str],
    excluded: set[str],
    lengths: list[int],
    raw: RawResults | None,
    im: ImputedMatrix | None,
) -> Annotations:
    """Runtime footer content for K2: gamma, imputation share, dropped tasks, arm reality."""
    notes = [f"gamma = {gamma} per dollar on this run; reward = pass(1/0) minus gamma x cost."]
    limits: list[str] = []
    if raw:
        # Only asserted when the arm series are actually on the canvas.
        limits.append(
            "Arm-oracle is hindsight-only — it picks the best REALISED arm per task, so it can "
            "dip below 0 and no live router can match it."
        )
        limits.append(
            "Arm-bandit is an illustrative inline learner drawn for this figure only, not a "
            "shipped routing strategy."
        )
    covered = (im.n_real + im.n_imputed) if im is not None else 0
    if im is not None and covered:
        notes.append(
            f"Scored on the coverage-completed matrix: {im.n_imputed} of {covered} cells "
            f"({im.n_imputed / covered * 100:.0f}%) are monotone-imputed, "
            f"{im.n_unknown} left UNKNOWN and excluded from scoring."
        )
    if excluded:
        limits.append(
            f"{len(excluded)} of {len(tasks)} sampled task(s) are dropped from at least one "
            "series because the chosen model was never measured — never scored fail@\\$0."
        )
    if len(set(lengths)) > 1:
        limits.append(
            f"Series span different task counts ({min(lengths)}-{max(lengths)}), so the "
            "endpoints are not directly comparable."
        )
    limits.extend(_single_arm_limits(raw))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _regret_cis(results: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """strategy -> (CumReg_ci_lower, CumReg_ci_upper) for rows that carry both."""
    cis: dict[str, tuple[float, float]] = {}
    for row in results:
        try:
            cis[row["strategy"]] = (
                float(row["CumReg_ci_lower"]),
                float(row["CumReg_ci_upper"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return cis


def _regret_title(
    finals: dict[str, float], cis: dict[str, tuple[float, float]], n_tasks: int
) -> str:
    """Headline the best DEPLOYABLE router, softened to a tie when the top two overlap."""
    ranked = sorted(
        ((n, v) for n, v in finals.items() if not _is_hindsight(n)), key=lambda kv: kv[1]
    )
    if not ranked:
        return "Cumulative Regret vs Oracle (Per-Task)"
    best_name, best_val = ranked[0]
    if len(ranked) >= 2:
        second_name, _second_val = ranked[1]
        a, b = cis.get(best_name), cis.get(second_name)
        if a and b and a[0] <= b[1] and b[0] <= a[1]:
            return (
                f"{best_name} has the lowest regret ({best_val:.2f} over {n_tasks} tasks) "
                f"but its interval OVERLAPS {second_name}'s — the ordering is unresolved"
            )
    return (
        f"{best_name} tracks the oracle closest among routers "
        f"(regret={best_val:.2f} over {n_tasks} tasks)"
    )


def plot_cumulative_regret(
    results: list[dict[str, str]],
    out_dir: Path,
    matrix_path: Path | None = None,
    gamma: float = 0.1,
    strategy_factories: dict[str, Callable[[], Strategy]] | None = None,
    raw_results: RawResults | None = None,
) -> Path:
    """K2 — per-task cumulative regret vs Oracle-reward. Adds an Arm-oracle
    (best realized arm per task among whatever was sampled) and an
    illustrative Arm-bandit when arm-level data is available.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if matrix_path is not None and strategy_factories is not None:
        matrix = load_matrix(matrix_path)
        if matrix is not None:
            # Score regret on the same equal-coverage matrix the summary rows use.
            matrix, im = summary.complete_scored_matrix(matrix)
            # Same sampled denominator the summary rows use (derive_tasks), not the
            # full results.csv key set — otherwise this plot scores a different task
            # set than run_eval/plot_strategies.
            tasks = derive_tasks(matrix, config.benchmark_params().get("seed", 42))
            if tasks:
                all_decisions = _evaluate_strategies(strategy_factories, matrix, tasks)
                oracle_pair = all_decisions.get("Oracle-reward")
                if oracle_pair:
                    oracle_decisions, oracle_unscorable = oracle_pair
                    finals: dict[str, float] = {}
                    lengths: list[int] = []
                    # Tasks dropped from ≥1 series as a coverage gap (chosen model
                    # unmeasured) — disclosed on canvas, never imputed fail@$0.
                    excluded_union: set[str] = set(oracle_unscorable)
                    cis = _regret_cis(results)
                    for name in [r["strategy"] for r in results]:
                        pair = all_decisions.get(name)
                        if pair and name != "Oracle-reward":
                            decisions, strat_unscorable = pair
                            excluded = strat_unscorable | oracle_unscorable
                            excluded_union |= strat_unscorable
                            cumreg = _compute_per_task_regret(
                                decisions, oracle_decisions, gamma, excluded
                            )
                            if not len(cumreg):
                                continue
                            label = name + f" (total={cumreg[-1]:.2f})"
                            (line,) = ax.plot(
                                range(1, len(cumreg) + 1), cumreg, label=label, lw=1.5
                            )
                            finals[name] = float(cumreg[-1])
                            lengths.append(len(cumreg))
                            # The interval belongs on the TOTAL, so it is drawn as an end
                            # bracket, not as a band along a trajectory it does not bound.
                            ci = cis.get(name)
                            if ci:
                                lo_err = max(0.0, float(cumreg[-1]) - ci[0])
                                hi_err = max(0.0, ci[1] - float(cumreg[-1]))
                                ax.errorbar(
                                    len(cumreg),
                                    cumreg[-1],
                                    yerr=[[lo_err], [hi_err]],
                                    fmt="none",
                                    ecolor=line.get_color(),
                                    elinewidth=1.4,
                                    capsize=4,
                                    zorder=6,
                                )

                    if raw_results:
                        raw_sampled = {cid: raw_results[cid] for cid in tasks if cid in raw_results}
                        for extra_name, fn, style in (
                            ("Arm-oracle", _arm_oracle_decisions, "--"),
                            ("Arm-bandit", _arm_bandit_decisions, ":"),
                        ):
                            extra_decisions = fn(raw_sampled, tasks, gamma)
                            cumreg = _compute_per_task_regret(
                                extra_decisions, oracle_decisions, gamma, oracle_unscorable
                            )
                            if not len(cumreg):
                                continue
                            # Arm-oracle peeks at realized outcomes, so its curve can
                            # dip BELOW 0 — say so on the legend, don't let a reader
                            # read "better than optimal" off a reference line.
                            suffix = " — hindsight only" if extra_name == "Arm-oracle" else ""
                            label = f"{extra_name} (total={cumreg[-1]:.2f}){suffix}"
                            ax.plot(
                                range(1, len(cumreg) + 1),
                                cumreg,
                                label=label,
                                lw=1.5,
                                linestyle=style,
                            )
                            finals[extra_name] = float(cumreg[-1])
                            lengths.append(len(cumreg))

                    # The baseline itself, drawn so the reader sees what "0" means.
                    ax.axhline(
                        0.0,
                        color="#4CAF50",
                        lw=1.2,
                        label="Oracle-reward (baseline — regret 0 by definition)",
                    )
                    ax.set_xlabel("Task (evaluation order) — regret accumulates left to right")
                    ax.set_ylabel(
                        f"Cumulative regret vs Oracle-reward (γ={gamma}, reward units)\n"
                        "lower = closer to the best possible routing"
                    )
                    # Headline the best deployable ROUTER, never an oracle — and say
                    # "unresolved" when the top two intervals overlap rather than
                    # printing a ranking the data does not support.
                    ax.set_title(_regret_title(finals, cis, len(tasks)))
                    # Outside the axes: with ~10 series the "best" in-axes slot
                    # always lands on top of the early, steepest part of a curve.
                    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
                    ax.grid(True, alpha=0.3)

                    return plot_frame.save(
                        fig,
                        out_dir / "cumulative_regret.png",
                        _REGRET_SPEC,
                        extra=_regret_annotations(
                            gamma, tasks, excluded_union, lengths, raw_results, im
                        ),
                    )

    # Fallback: bar chart from aggregate CumReg
    names = [r["strategy"] for r in results]
    cumregs = [float(r["CumReg"]) for r in results]

    color_map = {
        "Oracle": "#4CAF50",
        "Oracle-reward": "#4CAF50",
        "Always-Frontier": "#F44336",
        "Random": "#FF9800",
        "Always-Cheap": "#2196F3",
    }
    colors = [color_map.get(n, "#9E9E9E") for n in names]

    bars = ax.bar(names, cumregs, color=colors, edgecolor="white")
    for bar, val in zip(bars, cumregs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Strategy")
    ax.set_ylabel(
        f"Cumulative regret vs oracle (γ={gamma}, reward units)\n"
        "lower = closer to the best possible routing"
    )
    ax.set_title("Cumulative Regret vs Oracle (Aggregate) — total reward lost vs perfect routing")
    ax.grid(True, axis="y", alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "cumulative_regret.png",
        _REGRET_AGGREGATE_SPEC,
        extra=Annotations(
            notes=(
                f"gamma = {gamma} per dollar on this run; reward = pass(1/0) minus gamma x cost.",
            )
        ),
    )


_COST_SAVINGS_SPEC = FigureSpec(
    reading=(
        "One bar per routing strategy, sorted cheapest first. y is the total dollars that "
        "strategy's SELECTIONS cost over every task it scored. The text above each bar repeats "
        "the cost, then the strategy's pass rate with its 95% Wilson interval in brackets. The "
        "red dotted line is Always-Frontier's total — the kill-gate baseline a router has to "
        "undercut. A HATCHED bar is a hindsight oracle: it read the realised outcome, so its "
        "height is a lower bound on spend that no live router can buy."
    ),
    goal=(
        "Look for a short SOLID bar whose bracketed pass rate still overlaps Always-Frontier's: "
        "cheap is only interesting at matched quality, and a hatched bar is not an option."
    ),
    definitions=(
        ("Always-Frontier", "always route to the strongest enabled model"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
        ("hindsight", "the strategy reads the realised outcome; no live router can"),
    ),
    notes=(
        "Bars are what each ROUTER SELECTS, not per-model data — a single bar can mix many models.",
        _HINDSIGHT_NOTE,
        _CI_NOTE,
        "Superseded by cost_quality_equal.png, which carries the same strategies on both axes "
        "plus the measured-only panel. Keep this one for the cost breakdown, not as the headline.",
    ),
    limitations=(
        "The y axis carries NO uncertainty: imputed cell costs enter as a per-model median point "
        "estimate with no interval, while the pass rates above the bars do have one.",
        "Bar HEIGHT alone answers no question this benchmark asks — the cheapest bar is usually "
        "the worst router. Height is only meaningful next to the bracketed pass rate.",
    ),
)


def _equal_quality_names(
    names: Sequence[str], ns: Sequence[int], n_pass: Sequence[int], frontier: str
) -> set[str]:
    """Deployable strategies whose Wilson interval overlaps the frontier's own."""
    idx = {n: i for i, n in enumerate(names)}
    f = idx.get(frontier)
    if f is None or ns[f] <= 0:
        return set()
    flo, fhi = plot_style.wilson_interval(n_pass[f], ns[f])
    out: set[str] = set()
    for i, name in enumerate(names):
        if ns[i] <= 0 or name == frontier or _is_hindsight(name):
            continue
        lo, hi = plot_style.wilson_interval(n_pass[i], ns[i])
        if hi >= flo and lo <= fhi:
            out.add(name)
    return out


def _cost_savings_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    names: list[str],
    ns: list[int],
    n_pass: list[int],
    costs: list[float],
    ref: float | None,
    banner: str | None,
) -> Annotations:
    """Runtime footer content for the cost bars: coverage, imputation, headroom vs baseline."""
    notes: list[str] = []
    limits: list[str] = []
    # The figure's GOAL is equal-quality cost. Naming the cheapest bar outright
    # violated it: Always-Cheap is 19 points below the baseline it was compared to.
    equal = _equal_quality_names(names, ns, n_pass, "Always-Frontier")
    matched = [(c, n) for n, c, k in zip(names, costs, ns, strict=True) if n in equal and k > 0]
    if matched and ref:
        cheapest_cost, cheapest_name = min(matched)
        cut = (1 - cheapest_cost / ref) * 100 if ref else 0.0
        notes.append(
            f"Cheapest DEPLOYABLE strategy whose quality CI overlaps Always-Frontier: "
            f"{cheapest_name} at {usd(cheapest_cost, 4)} against the {usd(ref, 4)} baseline "
            f"({cut:.0f}% less)."
        )
    elif ref:
        notes.append(
            f"No deployable strategy's quality CI overlaps the {usd(ref, 4)} Always-Frontier "
            "baseline, so there is no equal-quality cost claim to make here."
        )
    if any("cascade" in n.lower() for n in names):
        limits.append(
            "A cascade strategy's bar is the CASCADE TOTAL — every failed cheaper probe plus the "
            "model that finally passed — so it is not one model's price."
        )
    return _merge_annotations(
        _banner_annotations(banner),
        _row_coverage_annotations(ns),
        Annotations(notes=tuple(notes), limitations=tuple(limits)),
    )


def plot_cost_savings(
    results: list[dict[str, str]], out_dir: Path, banner: str | None = None
) -> Path:
    # Sorted cheapest-first: `results` order is Reward-descending, which made the
    # reader decode every label to find the short bar this figure exists to show.
    order = sorted(range(len(results)), key=lambda i: float(results[i]["TotalCost"]))
    rows = [results[i] for i in order]
    names = [r["strategy"] for r in rows]
    costs = [float(r["TotalCost"]) for r in rows]
    perfs = [float(r["AvgPerf%"]) for r in rows]
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in rows]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {"Always-Frontier": "#F44336", "Always-Cheap": "#2196F3"}
    colors = ["#9E9E9E" if _is_hindsight(n) else color_map.get(n, "#9E9E9E") for n in names]
    hatches = ["///" if _is_hindsight(n) else "" for n in names]

    bars = ax.bar(names, costs, color=colors, edgecolor="white", hatch=hatches)
    for bar, name, cost, perf, n, p in zip(bars, names, costs, perfs, ns, n_pass, strict=True):
        ci_str = ""
        if n > 0:
            lo, hi = plot_style.wilson_interval(p, n)
            ci_str = f"[{lo * 100:.0f}-{hi * 100:.0f}]"
        tag = f"\n{_HINDSIGHT_LABEL}" if _is_hindsight(name) else ""
        # Two lines, not one: bars with an identical cost (a real tie in this
        # data) sit adjacent, and a single wide line would bleed into the
        # neighbor's label — stacking keeps each bar's own text narrow.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"{usd(cost, 4)}\n{perf:.0f}% {ci_str}{tag}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xlabel("Strategy (routing rule — bars are what each router SELECTS, not per-model data)")
    ax.set_ylabel("Total cost over all tasks (USD)")
    ax.set_ylim(0, max(costs) * 1.14 or 1.0)  # headroom for the multi-line bar labels
    ax.set_title(
        "Total cost by routing strategy, cheapest first — equal-quality framing\n"
        "(pass rate bracketed above each bar; hatched = hindsight, not achievable)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)

    ref_cost: float | None = None
    if len(names) >= 2:
        # Under equal coverage the frontier is measured on every task, so its total
        # cost is a valid reference line (no sparse-frontier guard needed — S6).
        ref_cost = next(
            (float(r["TotalCost"]) for r in rows if r["strategy"] == "Always-Frontier"), None
        )
        if ref_cost is not None:
            ax.axhline(
                y=ref_cost,
                color="#F44336",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Always-Frontier cost = {usd(ref_cost, 4)}",
            )
            ax.legend()

    return plot_frame.save(
        fig,
        out_dir / "cost_savings.png",
        _COST_SAVINGS_SPEC,
        extra=_cost_savings_annotations(names, ns, n_pass, costs, ref_cost, banner),
    )


# ---------------------------------------------------------------------------
# Measured vs imputed accounting. 409 of 1062 analytical cells are monotone-imputed
# and EVERY imputed cell is filled pass=True, so "how much of this number is
# evidence" is a first-class question these figures must be able to answer.
# ---------------------------------------------------------------------------

# tid -> (passed, cost, imputed-anywhere-on-the-billed-path)
StrategyCells = dict[str, tuple[bool, float, bool]]


class _PathRecorder:
    """Wraps a strategy so ``summary.evaluate`` also yields each task's billed path."""

    # A cascade's reported cost sums every model it tried, but only the model it RETURNED
    # is visible in the decision tuple. Reading the imputed flag off that final cell alone
    # let projected dollars enter a panel labelled "no imputed cell on either side".
    # Recording the path during the one evaluate pass keeps that single-producer property
    # (no second, re-embedding evaluation) while making the flag cover what was billed.

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.paths: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return str(self._inner.name)  # type: ignore[attr-defined]

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        model: str = self._inner.select(task_id, task_meta, matrix)  # type: ignore[attr-defined]
        tried = getattr(self._inner, "cascade_tried_models", None)
        self.paths[task_id] = list(tried) if tried else [model]
        return model

    def __getattr__(self, item: str) -> object:
        # Forwards `cascade_total_cost` (set per select) to evaluate; absent on
        # single-shot strategies, where evaluate's getattr default takes over.
        return getattr(self._inner, item)


def strategy_cells(
    matrix: dict, tasks: list[str], strategies: Iterable[object]
) -> dict[str, tuple[StrategyCells, set[str]]]:
    """Per strategy: the chosen cell's (pass, cost, imputed) per task, plus its unscorable set."""
    # Cost is read from `real_cost` for a single-shot pick; a cascade keeps
    # summary.evaluate's cascade total so these numbers reconcile with
    # strategy_summary.csv rather than quietly forming a second accounting path.
    # `imputed` is PATH-AWARE: true when ANY cell the decision billed was projected.
    out: dict[str, tuple[StrategyCells, set[str]]] = {}
    for strategy in strategies:
        recorder = _PathRecorder(strategy)
        decisions, unscorable = summary.evaluate(recorder, matrix, tasks)
        is_cascade = getattr(strategy, "cascade_total_cost", None) is not None
        cells: StrategyCells = {}
        for tid, model, passed, cost in decisions:
            per_task = matrix.get("results", {}).get(tid, {})
            spend = float(cost) if is_cascade else row_real_cost(per_task.get(model, {}))
            path = recorder.paths.get(tid) or [model]
            imputed = any(bool(per_task.get(m, {}).get("imputed", False)) for m in path)
            cells[tid] = (bool(passed), spend, imputed)
        out[strategy.name] = (cells, unscorable)  # type: ignore[attr-defined]
    return out


def _split_measured(cells: StrategyCells, unscorable: set[str]) -> dict[str, float]:
    """Measured vs imputed dollars and passes for one strategy's scored selections."""
    scored = [(p, c, i) for tid, (p, c, i) in cells.items() if tid not in unscorable]
    return {
        "measured_cost": sum(c for _p, c, i in scored if not i),
        "imputed_cost": sum(c for _p, c, i in scored if i),
        "measured_cells": float(sum(1 for _p, _c, i in scored if not i)),
        "imputed_cells": float(sum(1 for _p, _c, i in scored if i)),
        "measured_pass": float(sum(1 for p, _c, i in scored if p and not i)),
        "imputed_pass": float(sum(1 for p, _c, i in scored if p and i)),
    }


def paired_measured(
    router: str, baseline: str, by_strategy: dict[str, tuple[StrategyCells, set[str]]]
) -> dict[str, float] | None:
    """Router vs baseline on the tasks where NEITHER side billed a projected cell.

    This is the kill gate with the projection taken out. ``None`` when either side
    is absent or the overlap is empty.
    """
    # The imputed flag is path-aware (see strategy_cells), so a cascade that probed an
    # imputed cell on its way to a measured final pick is excluded here — the panel's
    # "no imputed cell on either side" title is then literally true of every dollar shown.
    if router not in by_strategy or baseline not in by_strategy:
        return None
    r_cells, r_un = by_strategy[router]
    b_cells, b_un = by_strategy[baseline]
    shared = [
        tid
        for tid in r_cells
        if tid in b_cells
        and tid not in r_un
        and tid not in b_un
        and not r_cells[tid][2]
        and not b_cells[tid][2]
    ]
    if not shared:
        return None
    r_pass = sum(1 for t in shared if r_cells[t][0])
    b_pass = sum(1 for t in shared if b_cells[t][0])
    only_b = sum(1 for t in shared if b_cells[t][0] and not r_cells[t][0])
    only_r = sum(1 for t in shared if r_cells[t][0] and not b_cells[t][0])
    return {
        "n": float(len(shared)),
        "router_cost": sum(r_cells[t][1] for t in shared),
        "baseline_cost": sum(b_cells[t][1] for t in shared),
        "router_pass": float(r_pass),
        "baseline_pass": float(b_pass),
        "baseline_only": float(only_b),
        "router_only": float(only_r),
        "mcnemar_p": _mcnemar_exact_p(only_b, only_r),
    }


_COST_QUALITY_SPEC = FigureSpec(
    reading=(
        "x is the total dollars a strategy spent; y is its pass rate in percent with a 95% "
        "Wilson error bar. The orange diamond is Always-Frontier and the orange horizontal band "
        "is ITS OWN interval — the 'equal quality' zone. A point is blue when its own interval "
        "overlaps that band and grey when it does not. A blue point far LEFT of the diamond is a "
        "strategy buying indistinguishable quality for less money."
    ),
    goal=(
        "Track the orange band leftwards: the leftmost BLUE point is the cheapest DEPLOYABLE "
        "strategy this data cannot tell apart from Always-Frontier. Then read the RIGHT panel "
        "before believing the saving — the left panel's dollars are roughly half projection."
    ),
    definitions=(
        ("Always-Frontier", "always route to the strongest enabled model"),
        ("equal quality", "the two pass-rate intervals overlap; not proof they are equal"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
        ("imputed cell", "a (task, model) outcome projected from the monotone axiom, not run"),
        ("hindsight", "the strategy reads the realised outcome; no live router can"),
    ),
    notes=(
        "This is the kill-gate figure: the router has to reach the orange band at materially "
        "lower cost, or the project stops.",
        _HINDSIGHT_NOTE,
        _CI_NOTE,
    ),
    limitations=(
        "Overlapping intervals at small n are WEAK evidence — the band widens as n shrinks and "
        "can swallow every point.",
        "The LEFT panel is an UNPAIRED comparison over partly-imputed cells. Every imputed cell "
        "is filled pass=True at a median price, so both axes of the baseline are part projection. "
        "The RIGHT panel is the paired, measured-only contrast — that is the real gate.",
        "Imputation is NOT neutral with respect to the conclusion, and the direction is not "
        "assumed here: the right panel measures it — see its own footer line for how much of "
        "the left panel's saving survives once every projected cell is removed.",
    ),
)


def _paired_annotations(paired: dict[str, float] | None) -> Annotations:
    """The measured-only panel's own numbers, stated plainly including a null result."""
    if paired is None:
        return Annotations()
    n = int(paired["n"])
    delta = paired["router_cost"] - paired["baseline_cost"]
    verdict = (
        f"NULL RESULT: on the {n} co-measured task(s) the router saves nothing — it costs "
        f"{usd(abs(delta), 2)} MORE than the baseline"
        if delta >= 0
        else f"On the {n} co-measured task(s) the router costs {usd(abs(delta), 2)} less"
    )
    return Annotations(
        notes=(
            f"{verdict}, at {paired['router_pass'] / n * 100:.1f}% vs "
            f"{paired['baseline_pass'] / n * 100:.1f}% pass "
            f"(McNemar exact p={paired['mcnemar_p']:.3f} — no quality difference resolved).",
        ),
        limitations=(
            f"The measured overlap is only {n} task(s) and was never DESIGNED to answer the "
            "kill gate — it is what survived after removing every projected cell, so it is an "
            "opportunistic subset, not a pre-registered one. "
            + (
                "The saving here is not a passed gate; read it as directional evidence."
                if delta < 0
                else "Read the gate as UNTESTED on measured data, not as passed or failed."
            ),
        ),
    )


def _cost_quality_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    ns: list[int],
    frontier_idx: int | None,
    band: tuple[float, float] | None,
    overlaps: int,
    banner: str | None,
    paired: dict[str, float] | None = None,
) -> Annotations:
    """Runtime footer content for N1: how wide the equal-quality band actually is."""
    notes: list[str] = []
    limits: list[str] = []
    if band is None or frontier_idx is None:
        limits.append(
            "Always-Frontier has no scorable row here, so there is no equal-quality band and "
            "no kill-gate comparison to read."
        )
    else:
        width = band[1] - band[0]
        notes.append(
            f"The band is Always-Frontier's own interval on n={ns[frontier_idx]} task(s): "
            f"{band[0]:.0f}-{band[1]:.0f}%. {overlaps} other strategy point(s) overlap it."
        )
        if width >= 20.0:
            limits.append(
                f"That band is {width:.0f} percentage points wide, so 'overlaps the frontier' "
                "is a very low bar on this sample size."
            )
    return _merge_annotations(
        _banner_annotations(banner),
        _row_coverage_annotations(ns),
        Annotations(notes=tuple(notes), limitations=tuple(limits)),
        _paired_annotations(paired),
    )


def _draw_measured_panel(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    router: str,
    paired: dict[str, float],
) -> None:
    """The paired MEASURED-ONLY kill gate: baseline vs router on co-measured cells."""
    n = int(paired["n"])
    series = (
        ("Always-Frontier", paired["baseline_cost"], paired["baseline_pass"], "#D55E00", "D"),
        (router, paired["router_cost"], paired["router_pass"], "#0072B2", "o"),
    )
    for label, cost, passes, color, marker in series:
        rate = passes / n * 100
        lo, hi = plot_style.wilson_interval(int(passes), n)
        down, up = plot_style.ci_yerr(rate, lo * 100, hi * 100)
        ax.errorbar(
            cost, rate, yerr=[[down], [up]], fmt="none", ecolor="#555555", elinewidth=1, capsize=3
        )
        ax.scatter(cost, rate, c=color, s=140, marker=marker, zorder=5, edgecolors="white")
        ax.annotate(
            f"{label}\n{usd(cost, 2)} · {rate:.1f}%",
            (cost, rate),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    delta = paired["router_cost"] - paired["baseline_cost"]
    direction = "MORE expensive" if delta > 0 else "cheaper"
    ax.set_title(
        f"MEASURED ONLY — no imputed cell on either side (n={n})\n"
        f"{router} is {usd(abs(delta), 2)} {direction}; McNemar "
        f"{int(paired['baseline_only'])} vs {int(paired['router_only'])} discordant, "
        f"p={paired['mcnemar_p']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("Total cost on the co-measured tasks (USD)")
    ax.set_ylim(0, 100)
    ax.margins(x=0.35)
    # The two costs differ by cents on a ~$43 base, so the default locator packs six
    # 6-digit tick labels into one another.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")
        lbl.set_fontsize(7)
    ax.grid(True, alpha=0.3)


def plot_cost_quality_equal(
    results: list[dict[str, str]],
    out_dir: Path,
    banner: str | None = None,
    by_strategy: dict[str, tuple[StrategyCells, set[str]]] | None = None,
) -> Path:
    """N1 — the kill-gate plot: cost vs pass rate, plus a measured-only companion."""
    # LEFT: pass% (Wilson CI) vs cost over the coverage-completed matrix, with
    # Always-Frontier's CI as a horizontal band, headlining the cheapest DEPLOYABLE
    # strategy inside it. RIGHT: the same contrast restricted to the tasks where both
    # sides landed on a genuinely measured cell — the gate with the projection removed.
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]

    if by_strategy:
        fig, (ax, ax_m) = plt.subplots(1, 2, figsize=(14, 6.5), width_ratios=[1.6, 1.0])
    else:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax_m = None

    frontier_idx = names.index("Always-Frontier") if "Always-Frontier" in names else None
    band: tuple[float, float] | None = None
    if frontier_idx is not None and ns[frontier_idx] > 0:
        lo, hi = plot_style.wilson_interval(n_pass[frontier_idx], ns[frontier_idx])
        band = (lo * 100, hi * 100)
        ax.axhspan(
            band[0],
            band[1],
            color="#D55E00",
            alpha=0.12,
            zorder=0,
            label=f"Always-Frontier 95% Wilson CI [{band[0]:.0f}%, {band[1]:.0f}%]",
        )

    best_cut: tuple[str, float, float, float] | None = None
    label_points: list[tuple[float, float, str]] = []
    overlap_count = 0
    for i, name in enumerate(names):
        overlaps = False
        if ns[i] > 0:
            lo, hi = plot_style.wilson_interval(n_pass[i], ns[i])
            lo, hi = lo * 100, hi * 100
            down, up = plot_style.ci_yerr(perfs[i], lo, hi)
            ax.errorbar(
                costs[i],
                perfs[i],
                yerr=[[down], [up]],
                fmt="none",
                ecolor="#555555",
                elinewidth=1,
                capsize=3,
                zorder=4,
            )
            overlaps = band is not None and hi >= band[0] and lo <= band[1]
        is_frontier = name == "Always-Frontier"
        hindsight = _is_hindsight(name)
        overlap_count += int(overlaps and not is_frontier and not hindsight)
        if hindsight:
            color, marker, size = "#9E9E9E", "*", 190
        elif is_frontier:
            color, marker, size = "#D55E00", "D", 140
        else:
            color, marker, size = ("#0072B2" if overlaps else "#9E9E9E"), "o", 110
        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=size,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.6,
        )
        suffix = f" ({_HINDSIGHT_LABEL})" if hindsight else ""
        label_points.append((float(costs[i]), float(perfs[i]), name + suffix))
        # Hindsight rows are excluded from the headline: the shipped figure crowned
        # Oracle at "84% less cost", a strategy that reads the answers.
        if (
            overlaps
            and frontier_idx is not None
            and not is_frontier
            and not hindsight
            and costs[frontier_idx] > 0
        ):
            cut = (1 - costs[i] / costs[frontier_idx]) * 100
            if best_cut is None or cut > best_cut[1]:
                best_cut = (name, cut, float(perfs[i]), float(perfs[frontier_idx]))
    plot_style.label_points_with_leaders(ax, label_points)

    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Pass rate (%)")
    # Full range: the shipped figure cropped to 60-100%, magnifying a 0.6-point
    # difference on the one figure the project has pre-committed to being killed by.
    ax.set_ylim(0, 100)

    if best_cut is not None:
        name, cut, perf, fperf = best_cut
        title = (
            f"Best DEPLOYABLE router: {name} matches Always-Frontier quality "
            f"({perf:.0f}% vs {fperf:.0f}%, CIs overlap) at {max(cut, 0.0):.0f}% less cost"
        )
    else:
        title = (
            "No DEPLOYABLE strategy's quality CI overlaps Always-Frontier's yet — "
            "no cost-equal-quality win to report"
        )
    ax.set_title(title + "\n(coverage-completed matrix — part measured, part imputed)", fontsize=9)
    if band is not None:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    paired: dict[str, float] | None = None
    if ax_m is not None and by_strategy is not None:
        router = best_cut[0] if best_cut else None
        paired = paired_measured(router, "Always-Frontier", by_strategy) if router else None
        if paired and router:
            _draw_measured_panel(ax_m, router, paired)
        else:
            ax_m.axis("off")
            ax_m.text(
                0.5,
                0.5,
                "No task where the baseline and a deployable\nrouter both landed on a MEASURED "
                "cell —\nthe kill gate is untested on measured data.",
                ha="center",
                va="center",
                fontsize=9,
                color=plot_frame.LIMIT_RED,
            )
    fig.tight_layout()

    return plot_frame.save(
        fig,
        out_dir / "cost_quality_equal.png",
        _COST_QUALITY_SPEC,
        extra=_cost_quality_annotations(ns, frontier_idx, band, overlap_count, banner, paired),
    )


_MEASURED_SPLIT_SPEC = FigureSpec(
    reading=(
        "One bar per strategy, sorted by total spend. Each bar splits that strategy's total into "
        "the dollars a provider actually billed on a run we performed (solid) and the dollars "
        "PROJECTED for cells we never ran (hatched), filled from a model's median measured price "
        "under the capability-ordering axiom. The text above each bar gives the projected share "
        "of the dollars and, on the second line, how many of the strategy's passes are projected "
        "rather than observed."
    ),
    goal=(
        "Read the hatched fraction as the benchmark telling you how much of its own headline is "
        "inference. Where a bar is roughly half hatched, the correct reading of any saving "
        "computed against it is 'what we expect IF the ordering holds', never 'what we measured'."
    ),
    definitions=(
        ("imputed cell", "a (task, model) outcome projected from the monotone axiom, not run"),
        ("projected pass", "a pass the axiom granted; every imputed cell is filled pass=True"),
    ),
    notes=(
        "Imputation is NOT neutral with respect to the conclusion: the always-frontier baseline "
        "is charged its full price on tasks a cheaper model demonstrably solved, which is exactly "
        "where a router's apparent saving comes from.",
        "Measured dollars are read from the real_cost column; imputed dollars are a per-model "
        "median of MEASURED real_cost, never a fabricated proxy.",
    ),
    limitations=(
        "A cascade strategy's bar is its cascade TOTAL, bucketed as PROJECTED whenever ANY cell "
        "on the billed path was imputed — including a failed cheaper probe ahead of a measured "
        "final pick. That over-attributes rather than under-attributes: the bar's measured "
        "segment is a floor on billed dollars, never an overstatement of them.",
        "The projected segment carries no uncertainty at all: it is a point estimate with no "
        "interval, stacked on top of billed dollars that are exact.",
    ),
)


def _measured_split_annotations(splits: dict[str, dict[str, float]]) -> Annotations:
    """State the projected share for every plotted strategy — the number this figure exists for."""
    notes: list[str] = []
    cells_i = sum(s["imputed_cells"] for s in splits.values())
    cells_m = sum(s["measured_cells"] for s in splits.values())
    if cells_i + cells_m:
        notes.append(
            f"Across the plotted selections {int(cells_i)} of {int(cells_i + cells_m)} chosen "
            f"cells ({cells_i / (cells_i + cells_m) * 100:.1f}%) are projected, and EVERY "
            "projected cell is filled pass=True."
        )
    for name, s in splits.items():
        total = s["measured_cost"] + s["imputed_cost"]
        passes = s["measured_pass"] + s["imputed_pass"]
        if total <= 0:
            continue
        notes.append(
            f"{name}: {usd(s['imputed_cost'], 2)} of {usd(total, 2)} "
            f"({s['imputed_cost'] / total * 100:.1f}%) projected; "
            f"{int(s['imputed_pass'])} of {int(passes)} passes projected."
        )
    return Annotations(notes=tuple(notes))


def plot_measured_vs_imputed(
    by_strategy: dict[str, tuple[StrategyCells, set[str]]], out_dir: Path
) -> Path | None:
    """Measured vs projected dollars per strategy — the evidentiary state of every
    cost number the other figures quote."""
    splits = {name: _split_measured(cells, un) for name, (cells, un) in by_strategy.items()}
    splits = {n: s for n, s in splits.items() if (s["measured_cost"] + s["imputed_cost"]) > 0}
    if not splits:
        return None
    order = sorted(splits, key=lambda n: splits[n]["measured_cost"] + splits[n]["imputed_cost"])
    measured = [splits[n]["measured_cost"] for n in order]
    imputed = [splits[n]["imputed_cost"] for n in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(order, measured, color="#0072B2", edgecolor="white", label="measured (really billed)")
    ax.bar(
        order,
        imputed,
        bottom=measured,
        color="#E69F00",
        edgecolor="white",
        hatch="///",
        label="projected (cell never run)",
    )
    totals = [m + i for m, i in zip(measured, imputed, strict=True)]
    for x, (name, total) in enumerate(zip(order, totals, strict=True)):
        s = splits[name]
        passes = int(s["measured_pass"] + s["imputed_pass"])
        ax.text(
            x,
            total + max(totals) * 0.01,
            f"{usd(total, 2)}\n{s['imputed_cost'] / total * 100:.0f}% projected\n"
            f"{int(s['imputed_pass'])}/{passes} passes proj.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_ylabel("Total cost over all scored tasks (USD)")
    ax.set_xlabel("Strategy")
    worst = max(
        splits,
        key=lambda n: (
            splits[n]["imputed_cost"] / (splits[n]["measured_cost"] + splits[n]["imputed_cost"])
        ),
    )
    worst_share = (
        splits[worst]["imputed_cost"]
        / (splits[worst]["measured_cost"] + splits[worst]["imputed_cost"])
        * 100
    )
    ax.set_title(
        "How much of each strategy's cost was measured and how much was projected\n"
        f"(most projected: {worst} at {worst_share:.0f}% of its dollars)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    return plot_frame.save(
        fig,
        out_dir / "measured_vs_imputed_cost.png",
        _MEASURED_SPLIT_SPEC,
        extra=_measured_split_annotations({n: splits[n] for n in order}),
    )


def _cap_names(names: list[str], k: int = 6) -> str:
    """Join names, capping the list so titles can't explode at 500 tasks."""
    if not names:
        return "none"
    if len(names) <= k:
        return ", ".join(names)
    return ", ".join(names[:k]) + f" (+{len(names) - k} more)"


def _model_total_price(models: dict, name: str) -> float:
    info = models.get(name, {})
    return float(info.get("input_price", 0)) + float(info.get("output_price", 0))


def _synthesize_raw(results_2level: dict) -> RawResults:
    """Wrap a flattened challenge x model view as challenge x model x "default"
    for the arm-aware heatmap when no true per-arm cache is supplied — keeps the
    2-level test fixtures (and the old callers) working unchanged.
    """
    return {
        cid: {model: {"default": row} for model, row in per_model.items()}
        for cid, per_model in results_2level.items()
    }


def _sorted_arm_columns(
    columns: list[tuple[str, str]],
    models_meta: dict,
    arm_ranks: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    """Column order: price ascending, then within-model arm rank ascending."""
    return sorted(
        columns,
        key=lambda c: (_model_total_price(models_meta, c[0]), c[0], arm_ranks.get(c, 0), c[1]),
    )


_HEATMAP_SPEC = FigureSpec(
    reading=(
        "One column per measured (model, arm) combination, ordered cheapest model first then by "
        "within-model arm rank; the tick under each column reads passes/sampled. One row per "
        "task. Green = that combination passed the task, red = it failed, GREY = it was never "
        "sampled. Grey is missing data, never a failure — that is the single most misread thing "
        "on this figure. Small matrices also carry a tick / cross / 'n/a' glyph per cell."
    ),
    goal=(
        "Scan across a row for complementarity: a task green in one column and red in another is "
        "a task routing can win. A column green where cheaper columns are red earns its price."
    ),
    definitions=(
        ("arm", "one reasoning-effort setting of a single model"),
        ("n/N under a column", "tasks that column passed over tasks it was sampled on"),
    ),
    notes=("Every cell here is a REAL measured outcome — nothing on this figure is imputed.",),
    limitations=(
        "Coverage is uneven BY DESIGN (p(arm|model) sampling), so two columns are only fairly "
        "compared on the rows where both are non-grey.",
        "The 'columns disagree on K task(s)' count in the title is computed over sampled cells "
        "only, so it understates real disagreement.",
        "This grid covers EVERY task in the matrix, not the sampled subset the strategy "
        "figures are scored on, so its per-column n/N uses a different denominator than the "
        "Pareto and cost figures.",
    ),
)


_MIN_ROW_LABEL_PT: Final[float] = 4.0
_MAX_ROW_LABEL_PT: Final[float] = 7.0


def _row_label_step(n_tasks: int, fig_h: float) -> tuple[int, float]:
    """``(every-nth-row-to-label, fontsize)`` sized to the canvas the grid was given."""
    # A label needs roughly its own font size in vertical points to stay legible, so the
    # row pitch — not a hardcoded row budget — decides how many rows can carry a name.
    axes_pt = max(1.0, (fig_h - 2.0) * 72.0)
    row_pt = axes_pt / max(1, n_tasks)
    if row_pt >= _MIN_ROW_LABEL_PT:
        return 1, min(_MAX_ROW_LABEL_PT, row_pt * 0.75)
    return max(1, int(np.ceil(_MIN_ROW_LABEL_PT / row_pt))), _MIN_ROW_LABEL_PT


def _heatmap_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    grid: np.ndarray,
    coverage: np.ndarray,
    glyphs: bool,
    raw: RawResults,
    synthesized: bool,
    ystep: int = 1,
) -> Annotations:
    """Runtime footer content for K4/N3: how much of the grid was actually sampled."""
    n_tasks, n_cols = grid.shape
    unsampled = int(np.isnan(grid).sum())
    total = n_tasks * n_cols
    notes = [
        f"{n_tasks} task rows x {n_cols} (model, arm) columns; {unsampled} of {total} cells "
        f"({unsampled / total * 100:.0f}%) were never sampled and are grey.",
        f"Per-column coverage runs from {int(coverage.min())} to {int(coverage.max())} "
        f"of {n_tasks} tasks.",
    ]
    limits: list[str] = []
    if not glyphs:
        limits.append(
            "At this size per-cell glyphs are off, so a cell's outcome is read from its colour "
            "against the legend rather than from a tick or cross."
        )
    if ystep > 1:
        limits.append(
            f"More rows than the canvas can label: only every {ystep}th task carries a name, "
            f"so {n_tasks - len(range(0, n_tasks, ystep))} of {n_tasks} rows are unlabelled."
        )
    if synthesized:
        limits.append(
            "No per-arm cache was available, so every column is a synthesized "
            "(model, 'default') column and no arm structure is shown."
        )
    limits.extend(_single_arm_limits(raw, synthesized))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_heatmap(matrix_path: Path, out_dir: Path, raw_results: RawResults | None = None) -> Path:
    """K4 / N3 — task x (model, arm) tri-state heatmap, columns per-column n/N —
    doubles as the coverage audit proving sampling is uneven-by-design.
    Degrades to (model, "default") columns with no true per-arm cache given.
    """
    matrix = load_matrix(matrix_path)
    if matrix is None:
        raise FileNotFoundError(f"Cannot load matrix from {matrix_path}")

    models_meta = matrix["models"]
    results_data = matrix.get("results", {})
    tasks = sorted(results_data.keys())
    raw = raw_results if raw_results is not None else _synthesize_raw(results_data)
    columns = _sorted_arm_columns(plot_style.arm_columns(raw), models_meta, _arm_ranks())
    if not columns or not tasks:
        raise ValueError("No (model, arm) columns or tasks found in matrix")

    # 1 = pass, 0 = fail, NaN = not evaluated (distinct from a real failure).
    grid = np.full((len(tasks), len(columns)), np.nan, dtype=float)
    for i, tid in enumerate(tasks):
        per_model = raw.get(tid, {})
        for j, (model, arm) in enumerate(columns):
            row = per_model.get(model, {}).get(arm)
            if row is not None:
                grid[i, j] = 1.0 if row.get("pass") else 0.0

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n_tasks, n_cols = len(tasks), len(columns)
    # Dynamic canvas: grows with the matrix but capped so it stays a usable image
    # at 500+ tasks / many (model,arm) columns rather than a fixed size that
    # crushes everything.
    fig_w = min(26.0, max(8.0, 1.15 * n_cols + 3.0))
    fig_h = min(32.0, max(5.0, 0.32 * n_tasks + 2.0))

    cmap = ListedColormap([plot_style.TRISTATE_FAIL, plot_style.TRISTATE_PASS]).with_extremes(
        bad=plot_style.TRISTATE_UNSAMPLED
    )
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    coverage = np.sum(~np.isnan(grid), axis=0)
    passes = np.nansum(grid, axis=0)
    col_labels = [
        f"{m}\n{a}  {int(passes[j])}/{int(coverage[j])}" for j, (m, a) in enumerate(columns)
    ]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=35, ha="right")

    # Label every row the canvas can physically fit, and only thin when it cannot.
    # The previous fixed "~40 evenly spaced" rule left 160 of 200 rows anonymous on a
    # 32-inch canvas whose row pitch (10.8 pt) had room for all of them.
    ystep, ylabel_fs = _row_label_step(n_tasks, fig_h)
    yticks = list(range(0, n_tasks, ystep))
    ax.set_yticks(yticks)
    ax.set_yticklabels([tasks[i].split("__")[-1] for i in yticks], fontsize=ylabel_fs)
    ax.tick_params(axis="y", length=2, pad=1.5)

    # Per-cell glyphs only when cells are big enough to read; else colour-only + legend.
    glyphs = n_tasks <= 40 and n_cols <= 20
    if glyphs:
        for i in range(n_tasks):
            for j in range(n_cols):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#616161")
                else:
                    mark = "✓" if grid[i, j] == 1.0 else "✗"
                    ax.text(j, i, mark, ha="center", va="center", fontsize=10, color="white")
    else:
        ax.legend(
            handles=[
                Patch(color=plot_style.TRISTATE_PASS, label="pass"),
                Patch(color=plot_style.TRISTATE_FAIL, label="fail"),
                Patch(color=plot_style.TRISTATE_UNSAMPLED, label="not sampled"),
            ],
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
            frameon=False,
        )

    # Data-driven subtitle so it stays correct on every regen: which tasks split the
    # field (columns disagree), which none solve, and the frontier column's coverage.
    split, none_solve = [], []
    for i, tid in enumerate(tasks):
        seen = grid[i, ~np.isnan(grid[i])]
        if seen.size and seen.min() != seen.max():
            split.append(tid.split("__")[-1])
        elif seen.size and seen.max() == 0.0:
            none_solve.append(tid.split("__")[-1])
    frontier_model, frontier_arm = columns[-1]  # priciest model, its highest-ranked sampled arm
    frontier_n = int(coverage[-1])
    glyph_note = "" if glyphs else "  (colour-only at scale — see legend)"
    ax.set_title(
        f"Task × (model, arm) outcomes — ✓ pass · ✗ fail · gray = not sampled{glyph_note}\n"
        f"columns disagree on {len(split)} task(s): {_cap_names(split)} · "
        f"solved by none: {_cap_names(none_solve)} · "
        f"{frontier_model}/{frontier_arm} sampled on {frontier_n}/{n_tasks}",
        fontsize=9,
    )
    fig.tight_layout()

    return plot_frame.save(
        fig,
        out_dir / "model_complementarity_heatmap.png",
        _HEATMAP_SPEC,
        extra=_heatmap_annotations(grid, coverage, glyphs, raw, raw_results is None, ystep),
    )


_ARM_MONOTONICITY_SPEC = FigureSpec(
    reading=(
        "One row per within-model arm pair (lower effort -> higher effort), restricted to the "
        "tasks where BOTH arms actually ran. LEFT panel: the paired difference in pass rate in "
        "percentage points, with a 95% interval for correlated proportions; the dot is the "
        "estimate, the vertical line is zero. RIGHT panel: the paired difference in average cost "
        "per task in dollars on the same tasks. The row label carries the co-measured n, and the "
        "discordant counts (how many tasks the higher arm lost / won) with an exact McNemar p."
    ),
    goal=(
        "Look for rows whose LEFT interval clears zero. A row straddling zero means this data "
        "cannot tell the two effort settings apart, which is the answer for every row here — "
        "read the dataset-wide line in the footer, not the individual dots."
    ),
    definitions=(
        ("arm", "one reasoning-effort setting of a single model"),
        ("arm rank", "ordinal effort order WITHIN one model, from the registry"),
        ("co-measured", "only tasks on which BOTH arms of the pair were actually run"),
        ("violation", "the higher-effort arm FAILS a task the lower-effort arm passed"),
    ),
    notes=(
        "Paired by construction. The earlier version of this figure compared each arm's MARGINAL "
        "pass rate over whatever tasks that arm happened to run, and those subsets differ "
        "systematically in difficulty — 3 of 5 adjacent pairs flip sign once paired.",
        "Zero cells are imputed here: an arm pair enters only where both arms have a real "
        "measured outcome on the same task.",
    ),
    limitations=(
        "A pair's n is the co-measured overlap, not the marginal n, so it can be far smaller "
        "than the per-arm sample sizes quoted elsewhere.",
        "The dataset-wide total pools every rank pair of every model and counts a task once per "
        "pair, so a 3-arm model contributes a task up to three times — read it as a pooled "
        "effect over arm pairs, not over tasks.",
    ),
)


def arm_pair_contrasts(raw: RawResults, arm_ranks: dict[tuple[str, str], int]) -> list[dict]:
    """Every within-model (lower-rank, higher-rank) arm pair, on CO-MEASURED tasks only.

    The metamorphic check this replaces an unpaired marginal with: a within-model
    effect estimate must not change sign under restriction to the co-measured subset.
    """
    by_model: dict[str, list[str]] = {}
    for model, arm in plot_style.arm_columns(raw):
        by_model.setdefault(model, []).append(arm)
    out: list[dict] = []
    for model in sorted(by_model):
        arms = sorted(by_model[model], key=lambda a: arm_ranks.get((model, a), 0))
        for i, low in enumerate(arms):
            for high in arms[i + 1 :]:
                out.append(_arm_pair(raw, model, low, high))
    return [p for p in out if p["n"] > 0]


def _arm_pair(raw: RawResults, model: str, low: str, high: str) -> dict:
    """One paired arm contrast over the tasks where both arms ran."""
    shared = [
        (per_model[model][low], per_model[model][high])
        for per_model in raw.values()
        if low in per_model.get(model, {}) and high in per_model.get(model, {})
    ]
    n = len(shared)
    lost = sum(1 for lo, hi in shared if lo.get("pass") and not hi.get("pass"))
    won = sum(1 for lo, hi in shared if hi.get("pass") and not lo.get("pass"))
    low_pass = sum(1 for lo, _hi in shared if lo.get("pass"))
    high_pass = sum(1 for _lo, hi in shared if hi.get("pass"))
    delta = (high_pass - low_pass) / n * 100 if n else 0.0
    # Wald interval for the difference of CORRELATED proportions (the paired form —
    # an independent-samples interval would be far too narrow here).
    var = (lost + won - (won - lost) ** 2 / n) if n else 0.0
    half = 1.96 * math.sqrt(max(0.0, var)) / n * 100 if n else 0.0
    cost_delta = sum(row_real_cost(hi) - row_real_cost(lo) for lo, hi in shared) / n if n else 0.0
    return {
        "model": model,
        "low": low,
        "high": high,
        "n": n,
        "low_rate": low_pass / n * 100 if n else 0.0,
        "high_rate": high_pass / n * 100 if n else 0.0,
        "delta_pp": delta,
        "ci": (delta - half, delta + half),
        "violations": lost,
        "gains": won,
        "p": _mcnemar_exact_p(lost, won),
        "cost_delta": cost_delta,
    }


def arm_pair_totals(pairs: list[dict]) -> dict[str, float]:
    """Pooled paired effect over every arm pair: net pp, violation rate, exact McNemar p."""
    n = sum(p["n"] for p in pairs)
    lost = sum(p["violations"] for p in pairs)
    won = sum(p["gains"] for p in pairs)
    return {
        "n": float(n),
        "violations": float(lost),
        "gains": float(won),
        "net_pp": (won - lost) / n * 100 if n else 0.0,
        "violation_rate": lost / n if n else 0.0,
        "p": _mcnemar_exact_p(lost, won),
    }


def _monotonicity_annotations(pairs: list[dict]) -> Annotations:
    """The dataset-wide paired result — the honest headline this figure exists to state."""
    t = arm_pair_totals(pairs)
    n = int(t["n"])
    if not n:
        return Annotations(
            limitations=("No arm pair has a single co-measured task, so nothing here is paired.",)
        )
    verdict = (
        "indistinguishable from zero"
        if t["p"] > 0.05
        else ("positive" if t["net_pp"] > 0 else "negative")
    )
    notes = [
        f"DATASET-WIDE PAIRED RESULT: over {n} co-measured within-model arm pair-observations, "
        f"the net effect of more reasoning effort is {t['net_pp']:+.1f}pp "
        f"({int(t['gains'])} gains vs {int(t['violations'])} losses), exact McNemar two-sided "
        f"p={t['p']:.3f} — {verdict}.",
        f"Monotonicity is VIOLATED on {t['violation_rate'] * 100:.1f}% of co-measured pairs "
        f"({int(t['violations'])} of {n}): the higher-effort arm failed a task the lower-effort "
        "arm passed.",
        f"{len(pairs)} arm pair(s) plotted; co-measured n runs "
        f"{min(p['n'] for p in pairs)}-{max(p['n'] for p in pairs)}.",
    ]
    straddling = sum(1 for p in pairs if p["ci"][0] <= 0.0 <= p["ci"][1])
    limits: list[str] = []
    if straddling:
        limits.append(
            f"{straddling} of {len(pairs)} pair interval(s) straddle zero — on this data the "
            "arm dimension is not something to route on."
        )
    thin = [p for p in pairs if p["n"] < MIN_N_RELIABLE]
    if thin:
        limits.append(
            f"{len(thin)} of {len(pairs)} pair(s) rest on fewer than {MIN_N_RELIABLE} "
            "co-measured tasks (marked provisional) — their interval is wide enough to contain "
            "almost anything."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _draw_pair_panel(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    pairs: list[dict],
    ys: list[int],
    model_colors: dict[str, str],
) -> None:
    """LEFT panel: paired delta in pass rate with its correlated-proportion interval."""
    for pair, y in zip(pairs, ys, strict=True):
        color = model_colors.get(pair["model"], "#9E9E9E")
        lo, hi = pair["ci"]
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, alpha=0.8, zorder=3)
        ax.scatter(
            pair["delta_pp"],
            y,
            s=90,
            facecolors=color if pair["n"] >= MIN_N_RELIABLE else "none",
            edgecolors=color,
            linewidth=1.4,
            zorder=4,
        )
        ax.annotate(
            f"{pair['gains']}↑/{pair['violations']}↓  p={pair['p']:.2f}",
            (max(hi, pair["delta_pp"]), y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
            color="#555555",
        )
    ax.axvline(0.0, color="#333333", lw=1.2, zorder=2)
    ax.set_xlabel("paired Δ pass rate, higher arm − lower arm (percentage points)", fontsize=8)


def _draw_pair_cost_panel(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    pairs: list[dict],
    ys: list[int],
    model_colors: dict[str, str],
) -> None:
    """RIGHT panel: paired delta in average cost per task, on the same co-measured tasks."""
    for pair, y in zip(pairs, ys, strict=True):
        color = model_colors.get(pair["model"], "#9E9E9E")
        ax.barh(y, pair["cost_delta"], height=0.55, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="#333333", lw=1.2, zorder=2)
    ax.set_xlabel("paired Δ avg cost per task, higher − lower (USD)", fontsize=8)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
        lbl.set_fontsize(7)


def plot_arm_monotonicity(
    raw_results: RawResults,
    out_dir: Path,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
) -> Path | None:
    """N2 — paired arm-monotonicity forest: does more reasoning effort buy quality?

    One row per within-model arm pair on CO-MEASURED tasks only, so the estimate
    cannot be a coverage artifact of which arm ran on which tasks.
    """
    pairs = arm_pair_contrasts(raw_results, arm_ranks)
    if not pairs:
        return None
    ys = list(range(len(pairs) - 1, -1, -1))
    height = max(4.0, 0.62 * len(pairs) + 2.2)
    fig, (ax, ax_cost) = plt.subplots(
        1, 2, figsize=(12.5, height), sharey=True, width_ratios=[1.7, 1.0]
    )
    _draw_pair_panel(ax, pairs, ys, model_colors)
    _draw_pair_cost_panel(ax_cost, pairs, ys, model_colors)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{p['model']}\n{p['low']} → {p['high']}  (n={p['n']})" for p in pairs], fontsize=7
    )
    ax.set_ylim(-0.8, len(pairs) - 0.2)
    for axis in (ax, ax_cost):
        axis.grid(True, axis="x", alpha=0.3)

    totals = arm_pair_totals(pairs)
    fig.suptitle(
        "Does more reasoning effort buy quality? Paired, co-measured tasks only\n"
        f"pooled over {int(totals['n'])} arm pair-observations: {totals['net_pp']:+.1f}pp, "
        f"exact McNemar p={totals['p']:.3f}, "
        f"{totals['violation_rate'] * 100:.1f}% of pairs violate monotonicity",
        fontsize=10,
    )
    fig.tight_layout()
    return plot_frame.save(
        fig,
        out_dir / "arm_monotonicity.png",
        _ARM_MONOTONICITY_SPEC,
        extra=_monotonicity_annotations(pairs),
    )


def _arm_size_legend_handles(max_rank: int) -> list[Line2D]:
    """Grey marker-size proxies explaining the size=arm-rank channel."""
    # Endpoints only (min/max effort, plus a midpoint when the bracket has
    # one) so the legend stays small regardless of how many ranks a model's
    # bracket has. Ranks are within-model only (see
    # plot_style.arm_marker_size), so this reads as an ordinal "less -> more
    # effort" scale, never a cross-model rank comparison.
    values = dict(plot_style.arm_size_legend_values(max_rank))
    picks = {0, max_rank} if max_rank > 0 else {0}
    if max_rank > 1:
        picks.add(max_rank // 2)
    handles = []
    for rank in sorted(picks):
        size = values[rank]
        if rank == 0:
            label = "arm rank 0 (less effort)"
        elif rank == max_rank:
            label = f"arm rank {rank} (more effort)"
        else:
            label = f"arm rank {rank}"
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="#666666",
                markersize=math.sqrt(size) / 2,
                label=label,
            )
        )
    return handles


_ARM_CLOUD_SPEC = FigureSpec(
    reading=(
        "One marker per measured (model, arm) cell. x is that cell's average cost PER TASK in "
        "dollars, not a total; y is its pass rate in percent with a 95% Wilson error bar. Hue is "
        "the model in registry price order, cheapest first. Marker SIZE orders arms WITHIN one "
        "hue only — a large marker in one hue is not more effort than a large marker in another. "
        "A HOLLOW marker rests on fewer tasks than the reliability floor and is excluded from the "
        "frontier. The dark dashed line is the convex hull through the RELIABLE non-dominated "
        "cells; the faint grey line is the hull you would get if the hollow cells were trusted."
    ),
    goal=(
        "Aim top-left: most pass rate per dollar. The RELIABLE cells sitting on the dashed hull "
        "are the ones a router should be choosing between. A cell under the hull is not "
        "necessarily dominated — a convex hull cuts corners, so a Pareto-optimal cell can sit "
        "just beneath it."
    ),
    definitions=(
        ("arm", "one reasoning-effort setting of a single model"),
        ("arm rank", "ordinal effort order WITHIN one model, from the registry"),
        ("AIQ", "area under the frontier over the plotted-cost x 100% rectangle, 0-1"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
        ("reliable", f"the cell ran on at least {MIN_N_RELIABLE} tasks"),
    ),
    notes=(
        _CI_NOTE,
        f"The frontier takes only cells with n >= {MIN_N_RELIABLE}. At the old floor of 10 a "
        "single 9-of-10 cell anchored the hull's apex, pushed the frontier model off the Pareto "
        "set entirely, and no marker on this data ever rendered hollow.",
    ),
    limitations=(
        "AIQ is normalised by the widest cost actually plotted, so it moves when the plotted "
        "set changes — read it within one figure, never across figures or runs.",
        "x is an average over tasks, so it hides per-task cost variance entirely.",
        "Each cell's pass rate is a MARGINAL over the tasks that cell happened to run on, and "
        "those subsets differ in difficulty — two cells are only fairly compared where both ran.",
    ),
)


def _arm_cloud_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    points: list[dict],
    raw: RawResults,
    aiq: float,
    plotted_max_rank: int,
    hull: list[tuple[float, float]],
    loo: tuple[float, float] | None,
) -> Annotations:
    """Runtime footer content for N4: cell count, sample thinness, how stable the hull is."""
    ns = [p["n"] for p in points]
    notes = [
        f"{len(points)} measured (model, arm) cell(s) from "
        f"{len({p['model'] for p in points})} model(s); sample sizes span n={min(ns)}-{max(ns)}.",
        f"AIQ={aiq:.2f} of the cost-quality rectangle lies under the reliable hull.",
    ]
    on_hull = {(round(x, 6), round(y, 6)) for x, y in hull}
    members = [
        f"{p['model']}·{p['arm']} (n={p['n']}, "
        f"CI {p['wilson'][0] * 100:.0f}-{p['wilson'][1] * 100:.0f}%)"
        for p in points
        if (round(p["cost"], 6), round(p["pass_rate"], 6)) in on_hull
    ]
    if members:
        notes.append("Hull members: " + "; ".join(members) + ".")
    limits = list(_provisional_limits(points, "plotted cells"))
    if loo is not None:
        limits.append(
            f"Leave-one-cell-out AIQ spans {loo[0]:.3f}-{loo[1]:.3f} against the plotted "
            f"{aiq:.3f} — a single cell moves this frontier, so read its shape as provisional."
        )
    if plotted_max_rank == 0:
        limits.append(
            "Every plotted cell is at arm rank 0, so the marker-size channel carries no "
            "information on this data — all markers are the base size."
        )
    limits.extend(_single_arm_limits(raw))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _cell_hull(points: list[dict]) -> list[tuple[float, float]]:
    """Upper convex hull through a set of (model, arm) cells."""
    if not points:
        return []
    return plot_style.upper_hull(
        plot_style.pareto_prune([(p["cost"], p["pass_rate"]) for p in points])
    )


def _draw_hull(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    hull: list[tuple[float, float]],
    color: str,
    style: str,
    label: str,
) -> None:
    if not hull:
        return
    fx = [p[0] for p in hull]
    fy = [p[1] for p in hull]
    if fx[0] > 0:
        fx = [0.0, *fx]
        fy = [fy[0], *fy]
    ax.plot(fx, fy, color=color, lw=1.6, linestyle=style, label=label)


def _loo_aiq_range(points: list[dict]) -> tuple[float, float] | None:
    """(min, max) AIQ over leaving each cell out — how much one cell moves the frontier."""
    if len(points) < 3:
        return None
    vals = [
        plot_style.area_under_frontier(_cell_hull([q for q in points if q is not p]))
        for p in points
    ]
    return (min(vals), max(vals))


def plot_arm_cloud(
    raw_results: RawResults,
    out_dir: Path,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
) -> Path | None:
    """N4 — single-panel (model, arm) cost-quality cloud: hue=model, size=arm
    rank, Wilson CIs, convex-hull frontier + AIQ.
    """
    points = _arm_cloud_points(raw_results, model_colors, arm_ranks)
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    label_points: list[tuple[float, float, str]] = []
    for pt in points:
        lo, hi = pt["wilson"]
        down, up = plot_style.ci_yerr(pt["pass_rate"] / 100.0, lo, hi)
        face = pt["color"] if not pt["provisional"] else "none"
        ax.errorbar(
            pt["cost"],
            pt["pass_rate"],
            yerr=[[down * 100], [up * 100]],
            fmt="none",
            ecolor=pt["color"],
            alpha=0.5,
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
        ax.scatter(
            pt["cost"],
            pt["pass_rate"],
            s=pt["size"],
            facecolors=face,
            edgecolors=pt["color"],
            linewidth=1.1,
            alpha=0.9,
            zorder=4,
        )
        label_points.append((pt["cost"], pt["pass_rate"], f"{pt['model']}·{pt['arm']}"))
    plot_style.label_points_with_leaders(ax, label_points, fontsize=6.5)

    # The frontier takes only cells above the reliability floor: at n=10 one 9/10
    # cell was the hull's apex and pushed the frontier model off the Pareto set.
    reliable = [p for p in points if not p["provisional"]]
    hull = _cell_hull(reliable)
    aiq = plot_style.area_under_frontier(hull)
    _draw_hull(
        ax, hull, "#333333", "--", f"frontier over n>={MIN_N_RELIABLE} cells (AIQ={aiq:.2f})"
    )
    if len(reliable) < len(points):
        loose = _cell_hull(points)
        _draw_hull(
            ax,
            loose,
            "#BDBDBD",
            ":",
            f"hull if provisional cells counted (AIQ={plot_style.area_under_frontier(loose):.2f})",
        )
    loo = _loo_aiq_range(reliable)

    ax.set_xlabel("avg cost per task (USD)")
    ax.set_ylabel("pass rate (%)")
    ax.set_title(
        f"(model, arm) cost-quality cloud — hue=model, size=arm rank (AIQ={aiq:.2f})\n"
        f"frontier over cells with n>={MIN_N_RELIABLE}; hollow markers are provisional "
        "and excluded from it",
        fontsize=9,
    )
    seen_models = {p["model"] for p in points}
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in model_colors.items()
        if m in seen_models
    ]
    plotted_max_rank = max((arm_ranks.get((p["model"], p["arm"]), 0) for p in points), default=0)
    size_handles = _arm_size_legend_handles(plotted_max_rank)
    if handles:
        # Opaque frame at lower-right, drawn on top: the right margin is taken by
        # the per-point leader labels, so the model key stays inside the axes and
        # its solid box masks the leader lines that would otherwise cross it.
        model_legend = ax.legend(
            handles=handles, fontsize=7, loc="lower right", title="model (hue)", framealpha=1.0
        )
        model_legend.set_zorder(20)
        ax.add_artist(model_legend)
    if size_handles:
        size_legend = ax.legend(
            handles=size_handles,
            fontsize=6.5,
            loc="upper left",
            title="size = arm rank",
            framealpha=1.0,
        )
        size_legend.set_zorder(20)
    ax.grid(True, alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "arm_cost_quality_cloud.png",
        _ARM_CLOUD_SPEC,
        extra=_arm_cloud_annotations(points, raw_results, aiq, plotted_max_rank, hull, loo),
    )


_CHOSEN_ARM_SPEC = FigureSpec(
    reading=(
        "One point per task, showing what the kNN-cascade router SELECTED for it. x is "
        "solve-breadth — how many of the sampled (model, arm) combinations passed that task, so "
        "further right means easier — with a small jitter so tied tasks do not overplot. y is the "
        "dollar cost of the single cell the router chose. Hue is the chosen model, and the legend "
        "title names models it never selected. Marker size follows the chosen arm's within-model "
        "rank, but while only default arms are selected it is effectively constant — do not read "
        "it. A router adapting to difficulty would push cost down as you move right."
    ),
    goal=(
        "Look for a downward slope left-to-right: expensive models on the hard tasks, cheap ones "
        "on the easy tasks. A flat cloud means the router is not adapting to difficulty."
    ),
    definitions=(
        ("solve-breadth", "how many sampled model-arm combos passed that task"),
        ("kNN-cascade", "router trying nearest-neighbour-predicted models cheapest-first"),
        ("arm", "one reasoning-effort setting of a single model"),
    ),
    notes=(
        "Strategy-conditioned: one routed cell per task, not per-model data.",
        "On the runs to date this cloud is flat — the embedding difficulty signal measured near "
        "chance on agentic coding. Treat any slope you do see as a finding to verify, not as "
        "confirmation.",
    ),
    limitations=(
        "solve-breadth is confounded by uneven coverage — a task with fewer sampled cells "
        "mechanically scores lower, so 'hard' and 'under-sampled' are not separated.",
        "y is the chosen cell's own cost, not a cascade total: the failed cheaper probes the "
        "cascade paid for before it are not counted.",
        "Only the default arm is ever chosen; live per-arm routing is not wired up yet.",
    ),
)


def _routing_share_lines(plotted: dict[str, int], dropped: dict[str, int]) -> list[str]:
    """The routed-model distribution as PLOTTED vs as actually ROUTED."""
    # Missingness here is perfectly confounded with the y axis — every dropped task on
    # this data is a frontier pick — so a bare "44 tasks are missing" reads as incidental
    # when it is a systematic understatement of what the router spends.
    if not dropped:
        return []
    true_counts = {m: plotted.get(m, 0) + dropped.get(m, 0) for m in set(plotted) | set(dropped)}
    n_plot, n_true = sum(plotted.values()), sum(true_counts.values())
    top = max(true_counts, key=lambda m: true_counts[m])
    composition = ", ".join(f"{m} {c}" for m, c in sorted(dropped.items(), key=lambda kv: -kv[1]))
    lines = [
        f"The {sum(dropped.values())} dropped task(s) are NOT a random sample of the routing: "
        f"by chosen model they are {composition}."
    ]
    if n_plot and n_true:
        lines.append(
            f"So this cloud understates what the router actually buys: {top} is "
            f"{plotted.get(top, 0) / n_plot * 100:.0f}% of the plotted points but "
            f"{true_counts[top] / n_true * 100:.0f}% of the true routing."
        )
    return lines


def _chosen_arm_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    plotted: dict[str, int],
    tasks: int,
    dropped: dict[str, int],
    unsampled: int,
    omitted: list[str],
) -> Annotations:
    """Runtime footer content for N5: how many tasks silently fell out, and which."""
    n_plotted = sum(plotted.values())
    unmeasured = sum(dropped.values())
    notes = [
        f"{n_plotted} of {tasks} sampled task(s) are plotted; the router picked "
        f"{len(plotted)} distinct model(s)."
    ]
    limits: list[str] = []
    if unmeasured or unsampled:
        limits.append(
            f"{unmeasured + unsampled} task(s) are missing from this cloud: {unmeasured} where "
            f"the chosen (model, default-arm) cell was never measured and {unsampled} with no "
            "sampled cell at all."
        )
    limits.extend(_routing_share_lines(plotted, dropped))
    if omitted:
        limits.append(
            f"{len(omitted)} model(s) in the palette were never selected and carry no point here."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_chosen_arm_vs_difficulty(
    raw_results: RawResults,
    matrix: dict,
    tasks: list[str],
    out_dir: Path,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
) -> Path | None:
    """N5 — chosen (model, arm) cost vs solve-breadth (# sampled combos that
    passed; higher = easier). Arm choice is always the default arm until the
    live executor wires per-arm routing — stated in the title, not hidden.
    """
    try:
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy
    except ImportError:
        return None
    strat_cfg = config.strategies()
    cascade_p = dict(strat_cfg.get("knn", {}))
    cascade_p.update(strat_cfg.get("knn_cascade", {}))
    try:
        strategy = kNNCascadeStrategy(**cascade_p)
    except (TypeError, ValueError):
        return None

    xs, ys, colors, sizes = [], [], [], []
    plotted_by_model: dict[str, int] = {}
    dropped_by_model: dict[str, int] = {}
    n_unsampled = 0
    for tid in tasks:
        per_model = raw_results.get(tid, {})
        sampled = [(m, a) for m, per_arm in per_model.items() for a in per_arm]
        if not sampled:
            n_unsampled += 1
            continue
        solved = sum(1 for m, a in sampled if per_model[m][a].get("pass"))
        # Deterministic jitter (crc32 of the task id) so identical solve-counts
        # don't silently overplot into a single dot — the same input always
        # renders the same jitter, so the PNG stays byte-stable on a re-run.
        jitter = (zlib.crc32(tid.encode()) % 1000) / 1000 * 0.5 - 0.25
        model = strategy.select(tid, matrix.get("tasks", {}).get(tid, {}), matrix)
        default_arm = config.default_arm_ids([model]).get(model, "default")
        row = per_model.get(model, {}).get(default_arm)
        if row is None:
            # Recorded BY MODEL: on this data every dropped task is a frontier pick,
            # so deleting them silently understates the router's spend.
            dropped_by_model[model] = dropped_by_model.get(model, 0) + 1
            continue
        xs.append(solved + jitter)
        ys.append(row_real_cost(row))
        colors.append(model_colors.get(model, "#9E9E9E"))
        plotted_by_model[model] = plotted_by_model.get(model, 0) + 1
        max_rank = max(
            (arm_ranks.get((model, a), 0) for m2, a in sampled if m2 == model), default=0
        )
        sizes.append(plot_style.arm_marker_size(arm_ranks.get((model, default_arm), 0), max_rank))
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(
        "solve-breadth: # sampled (model, arm) combos that passed\n"
        "(higher = easier; jittered to reveal ties; coverage is uneven by design,\n"
        "p(arm|model) sampling)",
        fontsize=9,
    )
    ax.set_ylabel("chosen (model, default-arm) cost (USD/task)")
    dropped_note = ""
    if dropped_by_model:
        composition = ", ".join(
            f"{m} {c}" for m, c in sorted(dropped_by_model.items(), key=lambda kv: -kv[1])
        )
        dropped_note = (
            f"\n{sum(dropped_by_model.values())} task(s) NOT drawn (chosen cell never measured), "
            f"all of them: {composition}"
        )
    ax.set_title(
        "What kNN-cascade SELECTS: chosen model cost vs task solve-breadth\n"
        "(strategy-conditioned — one routed model per task, not per-model data; "
        "arm is always the default — live per-arm routing isn't wired up yet)" + dropped_note,
        fontsize=9,
    )
    # Legend covers only the models this strategy actually routed to; models in the
    # palette it never selected are omitted (noted) rather than shown as dead keys.
    chosen_models = set(plotted_by_model)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in model_colors.items()
        if m in chosen_models
    ]
    omitted = [m for m in model_colors if m not in chosen_models and m not in dropped_by_model]
    title = "model routed to (hue)"
    if omitted:
        title += f"\nnever selected: {_cap_names(omitted, 3)}"
    ax.legend(handles=handles, fontsize=7, loc="upper right", title=title)
    ax.grid(True, alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "chosen_arm_vs_difficulty.png",
        _CHOSEN_ARM_SPEC,
        extra=_chosen_arm_annotations(
            plotted_by_model, len(tasks), dropped_by_model, n_unsampled, omitted
        ),
    )


_EMBEDDING_MAP_SPEC = FigureSpec(
    reading=(
        "Each point is one task, placed by the first two principal components of its REAL jina "
        "prompt embedding — the same encoder the router ships. The axis labels give how much of "
        "the embedding variance each component explains. Colour is the task's measured p_solve on "
        "a fixed 0-1 scale: dark = few models solved it, bright = most did. If difficulty were "
        "separable in embedding space, dark and bright points would form distinct regions."
    ),
    goal=(
        "Look for separated dark and bright regions. On the runs to date none appear, which is "
        "the finding: prompt embeddings alone did not predict task difficulty here. Treat any "
        "structure you do see as something to verify, not as confirmation."
    ),
    definitions=(
        ("p_solve", "fraction of the models measured on a task that passed it"),
        ("PC1 / PC2", "the two directions of largest variance in the embedding cloud"),
    ),
    notes=(
        "Embeddings come from the shipped jina-embeddings-v2-base-code encoder, never a TF-IDF "
        "or hash stand-in.",
        "p_solve counts real measured cells only; nothing on this figure is imputed.",
    ),
    limitations=(
        "PCA is linear and unsupervised: structure that is nonlinear, or that lives in later "
        "components, is invisible here — so absence of clusters is suggestive, not conclusive.",
    ),
)


def _embedding_map_annotations(
    task_ids: list[str], dropped: int, p_solve: list[float], denominators: list[int]
) -> Annotations:
    """Runtime footer content for N6: variance kept, coverage behind each colour."""
    notes = [
        f"{len(task_ids)} task(s) plotted; p_solve runs {min(p_solve):.2f}-{max(p_solve):.2f} "
        f"(mean {float(np.mean(p_solve)):.2f})."
    ]
    limits: list[str] = []
    if denominators and min(denominators) != max(denominators):
        limits.append(
            f"Each colour rests on a different denominator ({min(denominators)}-"
            f"{max(denominators)} measured models per task), so two identically-coloured points "
            "are not equally well evidenced."
        )
    if dropped:
        limits.append(f"{dropped} task(s) have no description and are dropped from this map.")
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _variance_limit(explained: np.ndarray) -> tuple[str, ...]:
    """Flag the projection when PC1+PC2 keep only a small share of the variance."""
    kept = float(explained[0] + explained[1]) * 100
    if kept >= 60.0:
        return ()
    return (
        f"PC1 and PC2 together explain only {kept:.1f}% of the embedding variance, so most of "
        "the space this plane claims to summarise is not shown.",
    )


def plot_embedding_routing_map(
    matrix: dict, out_dir: Path, model_colors: dict[str, str], k: int = 10
) -> Path | None:
    """N6 — PCA of the REAL jina prompt embeddings, coloured by each task's measured
    p_solve: does task difficulty cluster in the shipped embedding space? (~no here).
    """
    _ = (model_colors, k)  # coloured by continuous p_solve now, not per-model arms
    try:
        from sklearn.decomposition import PCA

        from benchmark.routing.strategies.knn import _embed_texts
    except ImportError:
        return None

    results = matrix.get("results", {})
    tasks = matrix.get("tasks", {})
    task_ids = [t for t in sorted(results.keys()) if t in tasks and tasks[t].get("description")]
    if len(task_ids) < 3:
        return None

    # p_solve = fraction of MEASURED models that passed (real outcomes only, no imputation).
    p_solve = []
    denominators = []
    for tid in task_ids:
        cells = [c for c in results[tid].values() if isinstance(c, dict)]
        passes = [1.0 if c.get("pass") else 0.0 for c in cells]
        p_solve.append(float(np.mean(passes)) if passes else 0.0)
        denominators.append(len(passes))

    embeddings = np.asarray(_embed_texts([tasks[tid]["description"] for tid in task_ids]))
    # random_state pins the randomized SVD solver ('auto' picks it at 768-d input), so the
    # committed figure is byte-reproducible run-to-run instead of re-churning git on every
    # regeneration. Same fix, same reason, as viz_knn's PCA scatter.
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=p_solve,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=48,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax, label="own p_solve  (fraction of models that passed)")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(
        "Does task difficulty cluster in the REAL jina embedding space?\n"
        "PCA of the shipped jina-embeddings-v2-base-code vectors, coloured by measured p_solve\n"
        "— hard (dark) and easy (bright) tasks intermix ⇒ difficulty is ~not embedding-separable",
        fontsize=9,
    )
    fig.tight_layout()

    extra = _embedding_map_annotations(
        task_ids, len(results) - len(task_ids), p_solve, denominators
    )
    return plot_frame.save(
        fig,
        out_dir / "embedding_routing_map.png",
        _EMBEDDING_MAP_SPEC,
        extra=Annotations(
            notes=extra.notes, limitations=extra.limitations + _variance_limit(explained)
        ),
    )


# ---------------------------------------------------------------------------
# Equal-coverage imputation reporting — capability distribution,
# coverage audit, violation rate, cascade overhead-leak, per-stratum win-rates,
# and the disclosure banner that replaces the removed sparse-frontier (phantom) machinery.
# ---------------------------------------------------------------------------


# Report-time capability BANDS — an ordinal, metadata-described grouping of the measured
# capability rank (band 1 = weakest ... band N = strongest), NOT stored and NOT the routing
# unit. Bands carry NO semantic names: each is described only by its member models, price
# range, marginal-pass-rate range (+CI), n_models, and the share of tasks it is the weakest
# to solve. The band COUNT is data-driven — adjacent models whose capability CIs overlap
# merge into one band — never a fixed cap.


def capability_bands(rank: config.CapabilityRank) -> dict[str, int]:
    """Ordinal band index per model (1 = weakest ... N = strongest), merged on PAIRWISE
    CI overlap: every member of a band overlaps every other member."""
    # A set of intervals overlaps pairwise iff max(lo) <= min(hi), so the band carries a
    # running envelope and cuts when a candidate falls outside it. The previous rule
    # compared each model only with its predecessor, which chains: it put deepseek
    # [0.613, 0.745] and qwen [0.337, 0.547] — disjoint — in one band while the legend
    # told the reader a band means overlapping CIs.
    bands: dict[str, int] = {}
    band = 1
    envelope: tuple[float, float] | None = None
    for rm in rank.ordered:
        ev = rank.evidence.get(rm.model)
        if ev is not None:
            if envelope is None:
                envelope = (ev.ci_lo, ev.ci_hi)
            else:
                lo, hi = max(envelope[0], ev.ci_lo), min(envelope[1], ev.ci_hi)
                if lo > hi:
                    band += 1
                    lo, hi = ev.ci_lo, ev.ci_hi
                envelope = (lo, hi)
        bands[rm.model] = band
    return bands


def _band_order(bands: dict[str, int]) -> list[int]:
    """Unique band indices in weakest -> strongest order."""
    return sorted(set(bands.values()))


def band_metadata(
    rank: config.CapabilityRank,
    bands: dict[str, int],
    im: ImputedMatrix | None = None,
) -> dict[int, dict]:
    """Per-band metadata: member models (rank order), n_models, price_range [min,max],
    capability_range (marginal pass-rate min/max + CI envelope), and pct_tasks_min_solved
    (share of tasks whose crossover model τ falls in the band, when a matrix is given)."""
    members: dict[int, list[str]] = {}
    for rm in rank.ordered:
        b = bands.get(rm.model)
        if b is not None:
            members.setdefault(b, []).append(rm.model)
    tau_total = len(im.tau) if im is not None else 0
    tau_counts: dict[int, int] = {}
    if im is not None:
        for model in im.tau.values():
            b = bands.get(model) if model is not None else None
            if b is not None:
                tau_counts[b] = tau_counts.get(b, 0) + 1
    meta: dict[int, dict] = {}
    for b in _band_order(bands):
        models = members.get(b, [])
        evs = [rank.evidence[m] for m in models if m in rank.evidence]
        prices = [e.price for e in evs]
        rates = [e.pass_rate for e in evs]
        meta[b] = {
            "band": b,
            "models": models,
            "n_models": len(models),
            "price_range": [min(prices), max(prices)] if prices else None,
            "capability_range": {
                "pass_rate_min": round(min(rates), 4),
                "pass_rate_max": round(max(rates), 4),
                "ci_lo": round(min(e.ci_lo for e in evs), 4),
                "ci_hi": round(max(e.ci_hi for e in evs), 4),
            }
            if evs
            else None,
            "pct_tasks_min_solved": (
                round(tau_counts.get(b, 0) / tau_total, 4) if tau_total else None
            ),
        }
    return meta


def _band_label(b: int, meta: dict[int, dict]) -> str:
    """Compact axis label: the ordinal index plus the band's defining metadata (member
    models, price range, marginal pass-rate range) — never a semantic name."""
    m = meta.get(b, {})
    who = "\n".join(m.get("models") or ["?"])  # one per line: names collided at fontsize 7
    pr = m.get("price_range")
    # Escaped: an unescaped pair of `$` in one tick label is parsed as mathtext and
    # both currency markers are deleted, so the shipped axis read "2-4" not "$2-$4".
    price = f"{usd(pr[0], 2)}-{usd(pr[1], 2)}/Mtok" if pr else "price ?"
    cr = m.get("capability_range")
    rate = (
        f"marginal pass {cr['pass_rate_min'] * 100:.0f}-{cr['pass_rate_max'] * 100:.0f}%"
        if cr
        else "?"
    )
    return f"band {b}\n{who}\n{price}\n{rate}"


def _rank_bands() -> tuple[config.CapabilityRank, dict[str, int], list[int]]:
    """The live capability rank, its ordinal display bands, and their weak->strong order."""
    rank = config.capability_rank()
    bands = capability_bands(rank)
    return rank, bands, _band_order(bands)


def coverage_rows(im: ImputedMatrix, bands: dict[str, int], band_order: list[int]) -> list[dict]:
    """Per-band real / imputed / UNKNOWN cell counts — the audit that coverage is
    now equal and how much rests on imputation."""
    n_tasks = len(im.matrix)
    models_per_band: dict[int, int] = {}
    for b in bands.values():
        models_per_band[b] = models_per_band.get(b, 0) + 1
    real: dict[int, int] = {}
    imputed: dict[int, int] = {}
    for cells in im.matrix.values():
        for model, cell in cells.items():
            mb = bands.get(model)
            if mb is None:
                continue
            bucket = imputed if cell.get("imputed") else real
            bucket[mb] = bucket.get(mb, 0) + 1
    rows: list[dict] = []
    for b in band_order:
        if b not in models_per_band:
            continue
        r, i = real.get(b, 0), imputed.get(b, 0)
        expected = n_tasks * models_per_band[b]
        rows.append({"band": b, "real": r, "imputed": i, "unknown": max(0, expected - r - i)})
    return rows


def write_coverage_table(im: ImputedMatrix, out_dir: Path) -> Path:
    """Write the per-band coverage audit CSV (regenerable, gitignored)."""
    _rank, bands, order = _rank_bands()
    rows = coverage_rows(im, bands, order)
    path = out_dir / "coverage_table.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["band", "real", "imputed", "unknown"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def write_capability_evidence(out_dir: Path, im: ImputedMatrix | None = None) -> Path:
    """Write the per-model capability-rank evidence + ordinal-band grouping artifact (rank,
    price, pass-rate, CI, source, band metadata; regenerable, gitignored). Prints a loud
    finding if the derived strongest model is not the configured control."""
    rank, bands, _order = _rank_bands()
    knobs = config.capability_rank_config()
    control = config.frontier_model()
    if rank.ordered and rank.strongest() != control:
        print(
            f"  ⚠ FINDING: derived strongest model {rank.strongest()!r} != control_model "
            f"{control!r} — the kill-gate baseline may be mis-chosen (investigate, do not ignore)."
        )
    csv_path = config.results_csv_path()
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else "absent"
    models = [
        {
            "model": rm.model,
            "n": rank.evidence[rm.model].n,
            "pass_rate": round(rank.evidence[rm.model].pass_rate, 4),
            "ci_lo": round(rank.evidence[rm.model].ci_lo, 4),
            "ci_hi": round(rank.evidence[rm.model].ci_hi, 4),
            "rank": rm.rank,
            "source": rm.source,
            "price": rank.evidence[rm.model].price,
            "band": bands.get(rm.model),
        }
        for rm in rank.ordered
    ]
    payload = {
        "generated_from": f"results.csv@{digest}",
        "knobs": {k: knobs[k] for k in ("K", "W", "min_pairs")},
        "control_model": control,
        "bands": list(band_metadata(rank, bands, im).values()),
        "models": models,
    }
    path = out_dir / "capability_evidence.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _violation_line(im: ImputedMatrix) -> str:
    """One-line monotonicity violation rate with a Wilson CI — 'measured' only when it was."""
    # violation_ci returns (0, 0, 0) at n=0; rendering that as v̂=0 [0,0] sells a
    # vacuous denominator as a perfect measurement.
    if im.n_multi_observed <= 0:
        return (
            "Monotonicity violation rate UNMEASURED — no task has two or more observed "
            "ranked models, so the axiom is assumed here, not verified"
        )
    v, lo, hi = impute.violation_ci(len(im.violations), im.n_multi_observed)
    return (
        f"Monotonicity violation rate v̂={v:.3f} 95% CI [{lo:.3f}, {hi:.3f}] "
        f"({len(im.violations)} of {im.n_multi_observed} multi-observed tasks) — "
        "measured, not assumed"
    )


def _disclosure_banner(im: ImputedMatrix | None, rows: list[dict] | None) -> str | None:
    """Coverage disclosure for the Pareto / cost=quality planes.

    Downgrades 'equal-coverage' to 'coverage-completed' with the residual UNKNOWN
    frontier count whenever per-strategy n_tasks are not actually equal (FIX B).
    """
    if im is None:
        return None
    _rank, bands, order = _rank_bands()
    strongest = order[-1] if order else None
    fr = next(
        (r for r in coverage_rows(im, bands, order) if r["band"] == strongest),
        None,
    )
    covered = (fr["real"] + fr["imputed"]) if fr else 0
    unknown = fr["unknown"] if fr else 0
    frac = (fr["imputed"] / covered * 100) if fr and covered else 0.0
    ns = [int(r.get("n_tasks", 0) or 0) for r in rows] if rows else []
    equal = bool(ns) and len(set(ns)) == 1
    if equal:
        head = (
            f"equal-coverage via monotone imputation — {frac:.0f}% of frontier cells "
            f"imputed (every strategy scored on n={ns[0]})."
        )
    else:
        rng = f"{min(ns)}–{max(ns)}" if ns else "?"
        head = (
            f"coverage-completed via monotone imputation — {frac:.0f}% of frontier cells "
            f"imputed, {unknown} still unmeasured (per-strategy n={rng}); true equal-coverage "
            "requires ladder collection."
        )
    if im.n_multi_observed > 0:
        v, _lo, _hi = impute.violation_ci(len(im.violations), im.n_multi_observed)
        axiom = (
            f" Monotonicity holds on {(1 - v) * 100:.0f}% of "
            f"{im.n_multi_observed} multi-observed task(s) (measured, not assumed)."
        )
    else:
        axiom = (
            " Monotonicity is UNVERIFIED here — no task has two or more observed ranked "
            "models, so the axiom is assumed."
        )
    # No directional claim here. The shipped banner asserted "imputation is
    # conservative, so a broken axiom only widens the router's lead"; that is not
    # demonstrated — the router takes free imputed passes too, and on the
    # measured-only overlap its lead is zero. See cost_quality_equal's right panel.
    return (
        head + axiom + " EVERY imputed cell is filled pass=True at a median measured price, "
        "for the router as well as for the baseline — see measured_vs_imputed_cost.png for how "
        "much of each strategy's number that is, and cost_quality_equal.png's measured-only "
        "panel for what survives when the projection is removed."
    )


_BAND_TERMS = (
    ("tau (crossover)", "the weakest capability band that solves a given task"),
    (
        "capability band",
        "consecutive ranked models whose capability CIs ALL overlap each other pairwise",
    ),
)

_CAPABILITY_SPEC = FigureSpec(
    reading=(
        "One bar per capability band, ordered weakest-first from the left by CAPABILITY rank, "
        "with 'unsolvable' last. Each x tick carries the band index, its member models, its "
        "price range and its members' marginal pass-rate range. y is how many tasks that band is "
        "the WEAKEST to solve, labelled with the count and the share of the suite."
    ),
    goal=(
        "Read where the mass sits: mass in the weaker (left-hand) bands is capability headroom — "
        "most tasks do not need the strongest model. This axis is a CAPABILITY ordering, NOT a "
        "price ordering, so it does not by itself say those tasks route CHEAPLY: check the "
        "footer for where the cheapest model actually sits. Mass in 'unsolvable' is headroom no "
        "router has."
    ),
    definitions=(
        *_BAND_TERMS,
        ("Copeland score", "pairwise win count over tasks both models were measured on"),
        ("marginal pass rate", "a model's pass rate over ITS OWN sampled tasks, not a shared set"),
    ),
    notes=(
        "Model rank is a Copeland score over CO-MEASURED tasks, deliberately not raw pass rate — "
        "the strongest models only ran on the hard subset. The pass rates printed on the ticks "
        "are the MARGINAL rates, each over a different task subset, so they can run backwards "
        "against the band order; they describe the members, they do not define the ordering.",
        "Band boundaries are data-driven: a band holds consecutive models whose capability "
        "intervals all overlap pairwise, so the band count moves with the data and is not capped.",
        "The bars themselves are 0% imputed — tau is read from real observed cells only.",
    ),
    limitations=(
        "'unsolvable' conflates two different things: nothing solved the task, and nothing strong "
        "enough was ever run on it.",
        "tau is capped by what was sampled — it can never be stronger than the strongest model "
        "that actually ran on the task.",
        "Band position is capability, not price. A left-hand bar does NOT mean 'routes to the "
        "cheapest model' — read the cheapest-model line in the footer before drawing a cost "
        "conclusion from this figure.",
    ),
)


def _cheapest_model_line(rank: config.CapabilityRank, bands: dict[str, int]) -> str | None:
    """Where the CHEAPEST model actually sits, so 'left = cheap' cannot be inferred."""
    # The GOAL used to say mass in the left-hand bands means tasks fall to the cheapest
    # models. On this data band 1 holds gpt-5-mini and kimi-k2.5 while the cheapest model
    # in the registry sits a band to the RIGHT of them.
    priced = [
        (rank.evidence[rm.model].price, rm.model)
        for rm in rank.ordered
        if rm.model in rank.evidence
    ]
    if not priced:
        return None
    price, cheapest = min(priced)
    band = bands.get(cheapest)
    leftmost = min(bands.values()) if bands else None
    where = (
        "the leftmost band, so left-hand mass does read as cheap here"
        if band == leftmost
        else f"band {band}, NOT the leftmost band — left-hand mass is not the same as cheap"
    )
    return (
        f"The cheapest model in the registry is {cheapest} at {usd(price, 2)}/Mtok and it sits "
        f"in {where}."
    )


def _capability_annotations(  # noqa: PLR0913 (one footer per plotted channel)
    im: ImputedMatrix,
    rank: config.CapabilityRank,
    bands: dict[str, int],
    n_bands: int,
    total: int,
    unsolvable: int,
    excluded_incomplete: int = 0,
) -> Annotations:
    """Runtime footer content for the capability histogram: imputation and rank provenance."""
    covered = im.n_real + im.n_imputed
    denom = total or 1  # safe division; the PRINTED count stays the true one
    notes = [
        f"{total} task(s) across {n_bands} capability band(s); {unsolvable} "
        f"({unsolvable / denom * 100:.0f}%) have no band that solves them.",
        f"{_violation_line(im)}.",
    ]
    cheapest = _cheapest_model_line(rank, bands)
    if cheapest:
        notes.append(cheapest)
    if n_bands <= 1:
        notes.append(
            "The band axis has COLLAPSED to a single capability band on this data — the models "
            "never separated, so there is no left-to-right weak-to-strong gradient to read."
        )
    if covered:
        notes.append(
            f"{im.n_imputed} of {covered} cells ({im.n_imputed / covered * 100:.0f}%) of the "
            "MATRIX are monotone-imputed, but these bars are not: tau is read from real observed "
            f"cells only ({im.n_unknown} cells left UNKNOWN and excluded from tau)."
        )
    limits: list[str] = []
    if excluded_incomplete:
        notes.append(
            f"{excluded_incomplete} task(s) whose crossover band was never established are "
            "EXCLUDED from these bars, per impute.py's own contract."
        )
    prior = [rm.model for rm in rank.ordered if rm.source != "measured"]
    if prior:
        limits.append(
            f"{len(prior)} of {len(rank.ordered)} models are placed by PRICE PRIOR rather than "
            "measured evidence, so their band position is an assumption, not a finding."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_capability_distribution(im: ImputedMatrix, out_dir: Path) -> Path:
    """Capability histogram: the crossover model τ per task grouped into ordinal capability
    bands plus 'unsolvable' — what fraction of the suite each band is the weakest to solve."""
    rank, bands, order = _rank_bands()
    meta = band_metadata(rank, bands, im)
    buckets: list[int | str] = [*order, "unsolvable"]
    counts: dict[int | str, int] = {b: 0 for b in buckets}
    # COMPLETE tasks only. impute.py's contract is that a task with an open UNKNOWN
    # band "is excluded from analysis entirely", but this histogram counted every tau
    # it found — 13 tasks whose crossover was never established were in the bars.
    excluded_incomplete = 0
    for tid, model in im.tau.items():
        if tid not in im.complete:
            excluded_incomplete += 1
            continue
        key = bands.get(model, "unsolvable") if model is not None else "unsolvable"
        counts[key] += 1
    total = sum(counts.values())
    denom = total or 1

    fig, ax = plt.subplots(figsize=(11, 6.0))
    heights = [counts[b] for b in buckets]
    labels = [_band_label(b, meta) if isinstance(b, int) else "unsolvable" for b in buckets]
    bars = ax.bar(range(len(buckets)), heights, color="#0072B2", edgecolor="white")
    for bar, h in zip(bars, heights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(heights) * 0.01,
            f"{h}\n{h / denom * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    # Headroom for the two-line count/percent label above the tallest bar, which the
    # axes edge otherwise clips.
    ax.set_ylim(0, max(heights) * 1.16 or 1.0)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_xlabel(
        "crossover capability band τ (weakest band that solves the task)\n"
        "— a CAPABILITY ordering, not a price ordering"
    )
    ax.set_ylabel("number of tasks")
    ax.set_title(
        f"Capability distribution — where the suite's difficulty lives ({total} tasks)\n"
        "τ = weakest capability band that solves each task ('unsolvable' = none does); "
        "bands group models whose capability CIs all overlap pairwise",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return plot_frame.save(
        fig,
        out_dir / "capability_distribution.png",
        _CAPABILITY_SPEC,
        extra=_capability_annotations(
            im, rank, bands, len(order), total, counts["unsolvable"], excluded_incomplete
        ),
    )


def _stratum_winner(rows: list[dict]) -> tuple[str, float] | None:
    """Best DEPLOYABLE strategy in a stratum by Reward (oracles excluded — upper bounds)."""
    deployable = [
        r
        for r in rows
        if r["strategy"] not in ("Oracle", "Oracle-reward") and int(r.get("n_tasks", 0) or 0) > 0
    ]
    if not deployable:
        return None
    best = max(deployable, key=lambda r: float(r.get("Reward", 0)))
    return best["strategy"], float(best.get("Reward", 0))


_PER_STRATUM_SPEC = FigureSpec(
    reading=(
        "One GROUP of bars per task stratum — the tasks sharing a crossover capability band — "
        "ordered weakest to strongest with 'unsolvable' last. Inside a group there is one bar "
        "per DEPLOYABLE strategy, in a fixed colour, and its height is that strategy's MEAN "
        "reward per task in the stratum with a 95% interval on the mean. Every strategy in a "
        "group is scored on exactly the same tasks, so bars inside a group are directly "
        "comparable; the tick under each group carries the band metadata and that shared n."
    ),
    goal=(
        "Compare bar heights WITHIN a group, and read overlapping intervals as unresolved. If "
        "one strategy is tallest in every group, per-task routing buys nothing here; if the "
        "tallest bar changes across groups, that difference is the routing headroom — but only "
        "where the intervals actually separate."
    ),
    definitions=(
        *_BAND_TERMS,
        ("Reward", "1 for a pass, 0 for a fail, minus gamma x cost — per task, then averaged"),
        ("deployable", "a strategy a live router could run — oracles excluded"),
    ),
    notes=(
        "Oracles are excluded: they peek at realised outcomes and would win every stratum by "
        "construction.",
        "Mean, not sum. The earlier version drew one bar per stratum at the WINNER's summed "
        "reward, so bar height tracked stratum size — the one variable the reader was told to "
        "ignore — and an exact three-way tie was printed as a single confident winner.",
        "'unsolvable' strata normally go negative — nothing passes there, so only cost accrues.",
    ),
    limitations=(
        "Scored on the partly-imputed matrix, so a bar inherits whatever the monotone axiom "
        "granted the cells it selected.",
        "The interval is a normal-approximation interval on a mean of bounded per-task rewards; "
        "at a handful of tasks it is indicative, not exact.",
        "gamma fixes the whole dollars-per-solve exchange rate and no sensitivity view exists: "
        "every ordering here is conditional on that one number.",
    ),
)

# stratum -> strategy -> {n, mean, lo, hi}
StratumStats = dict[object, dict[str, dict[str, float]]]


def stratum_reward_stats(
    matrix: dict,
    im: ImputedMatrix,
    strategies: list[object],
    tasks: list[str],
    gamma: float,
) -> StratumStats:
    """Per (capability-band stratum, deployable strategy): mean per-task reward + 95% CI.

    Within a stratum every strategy is scored on the INTERSECTION of the tasks all of
    them could score, so the bars are paired and a tie shows up as a tie.
    """
    _rank, bands, _order = _rank_bands()
    by_stratum: dict[object, list[str]] = {}
    for tid in tasks:
        if tid not in im.complete:
            continue
        tau = im.tau.get(tid)
        stratum: object = "unsolvable" if tau is None else bands.get(tau, "unsolvable")
        by_stratum.setdefault(stratum, []).append(tid)
    deployable = [s for s in strategies if not _is_hindsight(getattr(s, "name", ""))]
    out: StratumStats = {}
    for stratum, stratum_tasks in by_stratum.items():
        per_strategy = {
            s.name: summary.evaluate(s, matrix, stratum_tasks)  # type: ignore[attr-defined]
            for s in deployable
        }
        shared = [
            t for t in stratum_tasks if not any(t in un for _dec, un in per_strategy.values())
        ]
        if not shared:
            continue
        out[stratum] = {
            name: _reward_stats([_reward(p, c, gamma) for t, _m, p, c in dec if t in shared])
            for name, (dec, _un) in per_strategy.items()
        }
    return out


def _reward_stats(rewards: list[float]) -> dict[str, float]:
    """Mean per-task reward with a 95% normal-approximation interval on the mean."""
    n = len(rewards)
    if not n:
        return {"n": 0.0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    mean = sum(rewards) / n
    if n < 2:
        return {"n": float(n), "mean": mean, "lo": mean, "hi": mean}
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return {"n": float(n), "mean": mean, "lo": mean - half, "hi": mean + half}


def _stratum_ties(stats: dict[str, dict[str, float]], tol: float = 1e-9) -> list[str]:
    """Strategies tied (to ``tol``) with the stratum's best mean reward."""
    if not stats:
        return []
    best = max(s["mean"] for s in stats.values())
    return sorted(n for n, s in stats.items() if abs(s["mean"] - best) <= tol)


def _per_stratum_annotations(stats_by_stratum: StratumStats, strata: list) -> Annotations:
    """Runtime footer: shared task counts, exact ties, and whether any win is resolved."""
    sizes = {
        s: int(next(iter(stats_by_stratum[s].values()))["n"]) for s in strata if stats_by_stratum[s]
    }
    notes = [
        "Per-stratum shared task counts: " + ", ".join(f"{s}={n}" for s, n in sizes.items()) + "."
    ]
    limits: list[str] = []
    for s in strata:
        tied = _stratum_ties(stats_by_stratum.get(s, {}))
        if len(tied) > 1:
            notes.append(
                f"Stratum {s} is an EXACT TIE between {', '.join(tied)} — identical mean reward, "
                "so naming a single winner there would be an artifact of list order."
            )
    unresolved = []
    for s in strata:
        st = stats_by_stratum.get(s, {})
        ordered = sorted(st.items(), key=lambda kv: -kv[1]["mean"])
        if len(ordered) >= 2 and ordered[0][1]["lo"] <= ordered[1][1]["hi"]:
            unresolved.append(str(s))
    if unresolved:
        limits.append(
            f"In stratum/strata {', '.join(unresolved)} the top two intervals OVERLAP — the "
            "leader there is not resolved by this data."
        )
    if sizes and min(sizes.values()) < 10:
        limits.append(
            f"The smallest stratum holds only {min(sizes.values())} shared task(s), so its bars "
            "rest on almost no evidence."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_per_stratum_winrate(stats_by_stratum: StratumStats, out_dir: Path) -> Path:
    """Per-τ-stratum MEAN reward for every deployable strategy, with intervals — where
    routing helps and where it cannot, without a tie-break deciding the conclusion."""
    rank, bands, order = _rank_bands()
    meta = band_metadata(rank, bands)
    strata = [s for s in [*order, "unsolvable"] if s in stats_by_stratum]
    names = sorted({n for s in strata for n in stats_by_stratum[s]})
    if not strata or not names:
        strata = strata or ["unsolvable"]
        names = names or ["none"]
    colors = plot_style.model_color_map(names)

    fig, ax = plt.subplots(figsize=(max(9.0, 2.4 * len(strata) + 3.0), 6.0))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        xs = [i + (j - (len(names) - 1) / 2) * width for i in range(len(strata))]
        means = [stats_by_stratum[s].get(name, {}).get("mean", 0.0) for s in strata]
        errs = [
            [
                stats_by_stratum[s].get(name, {}).get("mean", 0.0)
                - stats_by_stratum[s].get(name, {}).get("lo", 0.0)
                for s in strata
            ],
            [
                stats_by_stratum[s].get(name, {}).get("hi", 0.0)
                - stats_by_stratum[s].get(name, {}).get("mean", 0.0)
                for s in strata
            ],
        ]
        ax.bar(
            xs,
            means,
            width=width * 0.92,
            label=name,
            color=colors[name],
            edgecolor="white",
            yerr=errs,
            ecolor="#333333",
            capsize=2,
            error_kw={"elinewidth": 0.9},
        )

    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(
        [_band_label(s, meta) if isinstance(s, int) else "unsolvable" for s in strata],
        fontsize=6.5,
    )
    ax.set_xlabel("task stratum (crossover capability band τ)")
    ax.set_ylabel("mean Reward per task (pass − γ·cost)")
    ties = [str(s) for s in strata if len(_stratum_ties(stats_by_stratum.get(s, {}))) > 1]
    tie_note = f"  ·  exact tie in stratum {', '.join(ties)}" if ties else ""
    ax.set_title(
        "Per-stratum mean reward by strategy — where routing helps and where it cannot\n"
        "(oracles excluded; every bar in a group scored on the SAME tasks)" + tie_note,
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=min(len(names), 3), loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return plot_frame.save(
        fig,
        out_dir / "per_stratum_winrate.png",
        _PER_STRATUM_SPEC,
        extra=_per_stratum_annotations(stats_by_stratum, strata),
    )


def cascade_overhead(
    matrix: dict, im: ImputedMatrix, tasks: list[str], strategy: Strategy
) -> float:
    """Mean ``cascade_total_cost − single_shot_cost(τ)`` over solvable tasks — dollars a
    cascade burns on failed cheaper probes before the passing model."""
    results = matrix.get("results", {})
    overheads: list[float] = []
    for tid in tasks:
        tau = im.tau.get(tid)  # the crossover MODEL (weakest that solves the task), or None
        if tau is None:
            continue
        strategy.select(tid, matrix.get("tasks", {}).get(tid, {}), matrix)
        cascade_cost = getattr(strategy, "cascade_total_cost", None)
        if cascade_cost is None:
            continue
        single = row_real_cost(results.get(tid, {}).get(tau, {}))
        overheads.append(float(cascade_cost) - single)
    return sum(overheads) / len(overheads) if overheads else 0.0


def derive_tasks(matrix: dict, seed: int) -> list[str]:
    """Covered tasks (present in results.csv) sampled by ``sample_size`` — matches run_eval."""
    # Evaluate over the MEASURED denominator, not the full suite: an uncovered task is
    # unmeasured, NOT a failure — scoring the 151 unrun-of-200 as fail@$0 diluted every
    # plotted pass-rate ~4x and made report.py disagree with run_eval/plot_strategies.
    return config.sample_tasks(sorted(matrix.get("results", {}).keys()), seed=seed)


def derive_rows(
    matrix: dict, tasks: list[str], strategies: list[object] | None = None
) -> list[dict]:
    """Per-strategy summary rows derived in-memory from the results.csv cache."""
    # The single source of truth. Mirrors run_matrix.refresh_summary exactly (same
    # strategies, gamma, bootstrap, seed, and task set — see derive_tasks). Pass
    # `strategies` to reuse an already-built set: each kNN-family strategy embeds every
    # task description and builds its own HNSW index on first select, so building a
    # second set costs GBs of RSS for no new information.
    from benchmark.routing import run_eval

    bm = config.benchmark_params()
    strategies = strategies if strategies is not None else run_eval.get_strategies()
    return summary.compute_strategy_rows(
        matrix,
        tasks,
        strategies,
        gamma=config.gamma(),
        bootstrap=bm.get("bootstrap_iterations", 1000),
        seed=bm.get("seed", 42),
    )


def _load_raw_results() -> RawResults | None:
    """Best-effort challenge x model x arm cache (config.load_results()) — never
    fatal, since every arm plot degrades gracefully without it.
    """
    try:
        raw = config.load_results()
    except Exception:  # noqa: BLE001 (arm plots are all optional)
        return None
    return raw or None


def _run_arm_plots(
    out_dir: Path,
    matrix: dict | None,
    matrix_path: Path,
    tasks: list[str],
    raw_results: RawResults | None,
) -> None:
    """N2/N4/N5/N6 — best-effort; each prints its own skip reason rather than
    aborting the whole report on one plot's failure.
    """
    if raw_results is None or matrix is None:
        print("  arm plots  : skipped (no results.csv cache available)")
        return
    raw_sampled = {cid: raw_results[cid] for cid in tasks if cid in raw_results}
    model_colors = _model_colors()
    arm_ranks = _arm_ranks()

    for label, fn, args in (
        (
            "Arm monotonicity",
            plot_arm_monotonicity,
            (raw_sampled, out_dir, model_colors, arm_ranks),
        ),
        ("Arm cost cloud  ", plot_arm_cloud, (raw_sampled, out_dir, model_colors, arm_ranks)),
        (
            "Chosen vs diff.  ",
            plot_chosen_arm_vs_difficulty,
            (raw_sampled, matrix, tasks, out_dir, model_colors, arm_ranks),
        ),
        ("Embedding map    ", plot_embedding_routing_map, (matrix, out_dir, model_colors)),
    ):
        try:
            result_path = fn(*args)
        except Exception as exc:  # noqa: BLE001 (each N-plot is independently optional)
            _step(label.strip(), f"skipped ({type(exc).__name__}: {exc})")
            continue
        _step(label.strip(), result_path or "skipped (no data)")
    _ = matrix_path  # kept for signature symmetry with the other plot entry points


def _report_imputation_outputs(
    im: ImputedMatrix,
    matrix: dict,
    tasks: list[str],
    out_dir: Path,
    strategies: list[object] | None = None,
) -> None:
    """Emit the five equal-coverage outputs: violation rate, coverage
    table, capability histogram, per-stratum winner, cascade overhead-leak."""
    from benchmark.routing import run_eval
    from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

    strategies = strategies if strategies is not None else run_eval.get_strategies()
    print(f"  {_violation_line(im)}")
    _rank, bands, order = _rank_bands()
    unknown_by_band = {r["band"]: r["unknown"] for r in coverage_rows(im, bands, order)}
    total_unknown = sum(unknown_by_band.values())
    print(
        f"  Residual UNKNOWN cells (excluded from scoring, NOT equal coverage yet): "
        f"{total_unknown} total; by band {unknown_by_band}"
    )
    print(f"  Coverage tbl : {write_coverage_table(im, out_dir)}")
    print(f"  Capab. evid. : {write_capability_evidence(out_dir, im)}")
    print(f"  Capability   : {plot_capability_distribution(im, out_dir)}")

    completed_for_strata, _im2 = summary.complete_scored_matrix(matrix)
    stats = stratum_reward_stats(completed_for_strata, im, strategies, tasks, config.gamma())
    print(f"  Per-stratum  : {plot_per_stratum_winrate(stats, out_dir)}")

    completed, _im = summary.complete_scored_matrix(matrix)
    overhead = cascade_overhead(completed, im, tasks, kNNCascadeStrategy(**config.knn_params()))
    print(f"  Cascade overhead-leak: USD {overhead:.4f}/task (failed cheaper probes before tau)")


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    # Line-buffered: a piped stdout otherwise buffers 8 KB, and an OOM SIGKILL
    # discards it — the first observed failure of this report printed NOTHING.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    config.load(config_path)

    ap = argparse.ArgumentParser(
        description="Generate comparison reports from routing benchmark results"
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--results",
        default=None,
        help="Optional: read a pre-derived strategy-summary CSV instead of deriving "
        "in-memory from results.csv (default: derive from the source of truth)",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmark/routing/reports",
        help="Output directory for reports (default: benchmark/routing/reports/)",
    )
    ap.add_argument(
        "--matrix",
        default=None,
        help="Path to task matrix JSON (enables per-task regret lines + heatmap)",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _memory_preflight()

    # ONE strategy set for the whole run. Each kNN-family strategy embeds every task
    # description and builds its own HNSW index on first select, so re-instantiating
    # the set per consumer cost GBs of RSS and OOM-killed the report.
    from benchmark.routing import run_eval

    strategies = run_eval.get_strategies()

    # ONE load of the matrix for the whole run: it was loaded twice (once to derive the
    # rows, once for the plots), holding two full copies alongside the embedders.
    matrix = load_matrix(matrix_path) if matrix_path and matrix_path.exists() else None

    if args.results:
        results = load_results(Path(args.results))
        source = args.results
        tasks: list[str] = []
    else:
        if matrix is None or not matrix.get("results"):
            print(
                "No results yet — results.csv holds no rows. "
                "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
            )
            return
        tasks = derive_tasks(matrix, config.benchmark_params().get("seed", 42))
        results = derive_rows(matrix, tasks, strategies)
        # Validate BEFORE writing: a degenerate matrix must not leave a misleading
        # summary CSV behind on its way to a crash in the first plot.
        problem = _validate_rows(results) if results else None
        if problem:
            print(f"Refusing to report: {problem}", file=sys.stderr)
            return
        # Write a human-readable copy to reports/ (gitignored) — never committed.
        summary.write_summary_csv(results, out_dir / "strategy_summary.csv")
        source = "results.csv (derived in-memory)"

    if not results:
        print("No strategy rows to plot.", file=sys.stderr)
        return

    problem = _validate_rows(results)
    if problem:
        print(f"Refusing to report: {problem}", file=sys.stderr)
        return

    print(f"Loaded {len(results)} strategies from {source}")

    print()
    print("Strategy ranking (derived from the results.csv source of truth):")
    print_summary(results)
    print()

    raw_results = _load_raw_results()
    matrix_for_plots = matrix

    # Complete the matrix ONCE and keep the completed copy: the measured-vs-projected
    # cells below used to re-run the same completion, doubling the work and the peak.
    imputed: ImputedMatrix | None = None
    completed: dict | None = None
    if matrix_for_plots is not None:
        completed, imputed = summary.complete_scored_matrix(matrix_for_plots)
    banner = _disclosure_banner(imputed, results)

    _step("Pareto", plot_pareto(results, out_dir, raw_results, len(tasks), banner))

    g = config.gamma()
    factories = _build_strategy_factories(g)
    _step(
        "Regret",
        plot_cumulative_regret(
            results,
            out_dir,
            matrix_path,
            gamma=g,
            strategy_factories=factories,
            raw_results=raw_results,
        ),
    )

    _step("Cost savings", plot_cost_savings(results, out_dir, banner))

    # Per-task selections with the imputed flag: what the measured-only kill-gate
    # panel and the measured-vs-projected breakdown are both built from.
    by_strategy: dict[str, tuple[StrategyCells, set[str]]] | None = None
    if completed is not None and tasks:
        by_strategy = strategy_cells(completed, tasks, strategies)

    _step("Cost=quality", plot_cost_quality_equal(results, out_dir, banner, by_strategy))

    if by_strategy:
        p5 = plot_measured_vs_imputed(by_strategy, out_dir)
        _step("Meas/imputed", p5 or "skipped (no scored cost)")

    if imputed is not None and matrix_for_plots is not None:
        _report_imputation_outputs(imputed, matrix_for_plots, tasks, out_dir, strategies)

    if matrix_path is not None:
        if matrix_path.exists():
            try:
                _step("Heatmap", plot_heatmap(matrix_path, out_dir, raw_results))
            except (FileNotFoundError, ValueError, KeyError) as exc:
                _step("Heatmap", f"skipped ({exc})")
        else:
            print(f"  Heatmap      : matrix file {matrix_path} not found, skipping")

    _run_arm_plots(out_dir, matrix_for_plots, matrix_path, tasks, raw_results)

    plt.close("all")
    print(f"Done. Peak RSS {_peak_rss_mb():,.0f} MB.")


if __name__ == "__main__":
    main()
