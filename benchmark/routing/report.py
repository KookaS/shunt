#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import impute, plot_style, summary
from benchmark.routing.impute import ImputedMatrix
from benchmark.routing.metrics import _reward
from benchmark.routing.plot_style import RawResults
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.oracle import OracleRewardAware


def _const_factory(strategy: Strategy) -> Callable[[], Strategy]:
    """Wrap an already-built strategy instance as a 0-arg factory — re-use, not
    rebuild, so a kNN strategy's embeddings/index aren't recomputed per plot.
    """
    return lambda: strategy


def _build_strategy_factories(gamma: float) -> dict[str, Callable[[], Strategy]]:
    """One source of truth for the regret plot's strategy set."""
    # The SAME config-enabled list run_eval.get_strategies reads for every
    # other plot (a hardcoded set here previously added Oracle-reward+Random
    # but omitted an enabled headline strategy like External-Prior — silently
    # absent from the regret plot while still shown everywhere else).
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


def _is_pareto(row: dict) -> bool:
    """Pareto flag, tolerant of bool (in-memory rows) or str (CSV override).

    A strategy with no scorable task is never Pareto-optimal: ($0, 0%) is
    un-dominated by construction, which is absence of evidence, not efficiency.
    """
    if int(float(row.get("n_tasks", 0) or 0)) <= 0:
        return False
    return row.get("Pareto") in (True, "True")


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
    scale_to_n_tasks: int | None = None,
) -> list[dict]:
    """One dict per (model, arm) column: cost/pass_rate/CI/size/color. Pass
    ``scale_to_n_tasks`` to extrapolate to a total-cost axis (K1's underlay);
    omit it for the per-task cost scale (N4's own axis).
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
        cost = stats.avg_cost * scale_to_n_tasks if scale_to_n_tasks else stats.avg_cost
        points.append(
            {
                "model": model,
                "arm": arm,
                "cost": cost,
                "pass_rate": stats.pass_rate * 100,
                "n": stats.n,
                "wilson": stats.wilson,
                "color": model_colors.get(model, "#9E9E9E"),
                "size": plot_style.arm_marker_size(rank, max_rank_by_model.get(model, 0)),
                "provisional": stats.provisional,
            }
        )
    return points


def _draw_arm_underlay(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    points: list[dict],
) -> None:
    """Faint (model, arm) dots behind the strategy markers — individual-cell
    context, extrapolated onto the same total-cost axis (avg cost x n_tasks).
    """
    for pt in points:
        face = pt["color"] if not pt["provisional"] else "none"
        ax.scatter(
            pt["cost"],
            pt["pass_rate"],
            s=max(18.0, pt["size"] * 0.3),
            facecolors=face,
            edgecolors=pt["color"],
            alpha=0.22,
            linewidth=0.6,
            zorder=1,
        )


def _hull_pareto_indices(names: list[str], pareto_map: dict[str, bool]) -> list[int]:
    """Indices of the Pareto-flagged rows entering the convex-hull/AIQ computation.

    Equal-coverage imputation makes the frontier fully measured, so the old
    partial-coverage sparse-frontier exclusion no longer applies (design S6).
    """
    return [i for i, name in enumerate(names) if pareto_map.get(name, False)]


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
            f"{empty} strategy row(s) have no scorable task and sit at $0 / 0% — "
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
        f"{plot_style.MIN_N_PROVISIONAL} tasks (drawn hollow) — their pass rate is provisional.",
    )


_PARETO_SPEC = FigureSpec(
    reading=(
        "x is the total dollars a routing strategy spent over every task it scored; y is its "
        "average pass rate in percent, with a 95% Wilson error bar. One big marker per strategy "
        "(orange diamond = Always-Frontier, green = Pareto-optimal, grey = neither: dominated, or "
        "having no scorable task and so parked at $0 / 0%). The faint "
        "small dots behind them are individual (model, arm) cells. The green dashed line is the "
        "convex hull through the Pareto strategies and the shaded area under it is what a router "
        "mixing two of them can reach."
    ),
    goal=(
        "Aim top-left: the same pass rate for fewer dollars. A strategy well to the LEFT of the "
        "orange diamond at a similar height is the win this benchmark is looking for."
    ),
    definitions=(
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
        ("Pareto-optimal", "no other strategy is both cheaper and better"),
        ("arm", "one reasoning-effort setting of a single model"),
        ("AIQ", "area under the frontier over the plotted-cost x 100% rectangle, 0-1"),
    ),
    notes=(
        _CI_NOTE,
        "The frontier is a convex hull (mixtures of two strategies), not a best-so-far staircase.",
    ),
    limitations=(
        "AIQ is normalised by the widest cost actually plotted, so it moves when the plotted "
        "set changes — read it within one figure, never across figures or runs.",
    ),
)


def _underlay_annotations(underlay: list[dict], raw: RawResults | None) -> Annotations:
    """The faint (model, arm) dots' provenance and thinness — omitted when none are drawn."""
    if not underlay:
        return Annotations()
    return Annotations(
        notes=("Faint-dot hues follow registry price order, cheapest model first.",),
        limitations=(
            "The faint (model, arm) dots are EXTRAPOLATED, not measured: average cost x task "
            "count, as if that one cell had run alone on every task.",
            *_provisional_limits(underlay, "underlay (model, arm) dots"),
            *_single_arm_limits(raw),
        ),
    )


def _pareto_annotations(
    ns: list[int], underlay: list[dict], raw: RawResults | None, banner: str | None, aiq: float
) -> Annotations:
    """Runtime footer content for K1: coverage, underlay thinness, imputation, arm reality."""
    hull_note = Annotations(
        notes=(
            (f"AIQ={aiq:.2f} of the cost-quality rectangle lies under the frontier.",)
            if aiq > 0
            else ()
        )
    )
    return _merge_annotations(
        _banner_annotations(banner),
        _row_coverage_annotations(ns),
        hull_note,
        _underlay_annotations(underlay, raw),
    )


def plot_pareto(
    results: list[dict[str, str]],
    out_dir: Path,
    raw_results: RawResults | None = None,
    n_tasks: int = 0,
    banner: str | None = None,
) -> Path:
    """K1 — cost-quality Pareto plane. Wilson CI per strategy, a convex-hull
    frontier (the region a mixture router can reach) with AIQ in the title, and a
    faint (model, arm) underlay for individual-cell context.
    """
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]
    pareto_map = {r["strategy"]: _is_pareto(r) for r in results}

    fig, ax = plt.subplots(figsize=(10, 6.5))

    underlay: list[dict] = []
    has_underlay = bool(raw_results)
    if raw_results:
        underlay = _arm_cloud_points(
            raw_results,
            _model_colors(),
            _arm_ranks(),
            scale_to_n_tasks=n_tasks or len(raw_results),
        )
        _draw_arm_underlay(ax, underlay)

    label_points: list[tuple[float, float, str]] = []
    for i, name in enumerate(names):
        is_frontier = name == "Always-Frontier"
        is_pareto = pareto_map.get(name, False)
        if is_frontier:
            color, size, marker = "#D55E00", 150, "D"
        elif is_pareto:
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
        label_points.append((float(costs[i]), float(perfs[i]), name))

    plot_style.label_points_with_leaders(ax, label_points)
    if has_underlay:
        ax.text(
            0.01,
            0.01,
            "faint dots: (model, arm) extrapolated to every task's cost, alone",
            transform=ax.transAxes,
            fontsize=7,
            color="#777777",
            va="bottom",
        )

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
            label=f"Pareto frontier (convex hull, AIQ={aiq:.2f})",
        )
        ax.fill_between(
            fx, 0, fy, alpha=0.06, color="#009E73", label="achievable region (mixture routing)"
        )

    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("Average pass rate (%)")
    ax.set_title(
        f"Cost vs quality — Pareto frontier spans AIQ={aiq:.2f} of the cost-quality rectangle",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "pareto_scatter.png",
        _PARETO_SPEC,
        extra=_pareto_annotations(ns, underlay, raw_results, banner, aiq),
    )


def _evaluate_strategies(
    factory_map: dict[str, Callable[[], Strategy]],
    matrix: dict,
    tasks: list[str],
) -> dict[str, tuple[list[tuple[str, str, bool, float]], set[str]]]:
    """Per strategy: ``(decisions, unscorable)`` where ``unscorable`` is the set of
    task ids whose chosen model was never measured (a coverage gap, NOT a real
    fail@$0 — callers exclude them, mirroring summary.evaluate)."""
    evaluated: dict[str, tuple[list[tuple[str, str, bool, float]], set[str]]] = {}
    for name, factory in factory_map.items():
        strategy = factory()
        decisions: list[tuple[str, str, bool, float]] = []
        unscorable: set[str] = set()
        for tid in tasks:
            task_meta = matrix.get("tasks", {}).get(tid, {})
            model = strategy.select(tid, task_meta, matrix)
            outcome = matrix.get("results", {}).get(tid, {}).get(model, {})
            if not outcome:
                unscorable.add(tid)
            passed = outcome.get("pass", False)
            cost = outcome.get("cost", 0.0)
            decisions.append((tid, model, passed, cost))
        evaluated[name] = (decisions, unscorable)
    return evaluated


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
                passed, cost = bool(row.get("pass")), float(row.get("cost", 0.0))
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
        passed, cost = bool(row.get("pass")), float(row.get("cost", 0.0))
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
        "scored — so the same x position is not the same task on two curves."
    ),
    goal=(
        "Look for the lowest, flattest curve. The title ranks routers by TOTAL regret, which is "
        "only comparable where two series span the same task count — when the counts below "
        "differ, prefer the SLOPE (average regret per task)."
    ),
    definitions=_REGRET_TERMS,
    notes=(
        "The Oracle-reward baseline is flat at 0 by construction: it is the reference every "
        "other curve is measured against, not a competitor.",
    ),
    limitations=(),
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
            "series because the chosen model was never measured — never scored fail@$0."
        )
    if len(set(lengths)) > 1:
        limits.append(
            f"Series span different task counts ({min(lengths)}-{max(lengths)}), so the "
            "endpoints are not directly comparable."
        )
    limits.extend(_single_arm_limits(raw))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


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
                            ax.plot(range(1, len(cumreg) + 1), cumreg, label=label, lw=1.5)
                            finals[name] = float(cumreg[-1])
                            lengths.append(len(cumreg))

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
                    # Headline the best deployable ROUTER, never an oracle.
                    # "Oracle" is measured against the near-identical Oracle-reward
                    # baseline (tautological ~0 regret), and "Arm-oracle" picks the
                    # best REALIZED arm per task in hindsight — both peek at outcomes
                    # a live router cannot see, so both are excluded from the pick
                    # regardless of arm count (they still show as reference lines).
                    excluded = {"Oracle", "Arm-oracle"}
                    candidates = {k: v for k, v in finals.items() if k not in excluded}
                    if candidates:
                        best_name, best_val = min(candidates.items(), key=lambda kv: kv[1])
                        ax.set_title(
                            f"{best_name} tracks the oracle closest among routers "
                            f"(regret={best_val:.2f} over {len(tasks)} tasks)"
                        )
                    else:
                        ax.set_title("Cumulative Regret vs Oracle (Per-Task)")
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
        "One bar per routing strategy. x is the strategy name (categorical, no order); y is the "
        "total dollars that strategy's SELECTIONS cost over every task it scored. The text above "
        "each bar repeats the cost, then the strategy's pass rate with its 95% Wilson interval "
        "in brackets. The red dotted line is Always-Frontier's total — the kill-gate baseline a "
        "router has to undercut."
    ),
    goal=(
        "Look for a short bar whose bracketed pass rate still overlaps Always-Frontier's: cheap "
        "is only interesting at matched quality."
    ),
    definitions=(
        ("Always-Frontier", "always route to the strongest enabled model"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
    ),
    notes=(
        "Bars are what each ROUTER SELECTS, not per-model data — a single bar can mix many models.",
        "Bar colours are a fixed semantic map (green oracle, red Always-Frontier, blue "
        "Always-Cheap, orange Random, grey everything else); they do not encode model identity.",
        _CI_NOTE,
    ),
    limitations=(
        "The y axis carries NO uncertainty: imputed cell costs enter as a per-model median point "
        "estimate with no interval, while the pass rates above the bars do have one.",
    ),
)


def _cost_savings_annotations(
    names: list[str], ns: list[int], costs: list[float], ref: float | None, banner: str | None
) -> Annotations:
    """Runtime footer content for the cost bars: coverage, imputation, headroom vs baseline."""
    notes: list[str] = []
    limits: list[str] = []
    # A row parked at $0 / n=0 is un-dominated by construction (see _is_pareto) —
    # calling it the cheapest strategy sells absence of evidence as efficiency.
    scored_costs = [c for c, n in zip(costs, ns, strict=True) if n > 0]
    if scored_costs and ref:
        notes.append(
            f"Cheapest strategy with a scorable task costs ${min(scored_costs):.4f} against the "
            f"${ref:.4f} Always-Frontier baseline."
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
    names = [r["strategy"] for r in results]
    costs = [float(r["TotalCost"]) for r in results]
    perfs = [float(r["AvgPerf%"]) for r in results]
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        "Oracle": "#4CAF50",
        "Oracle-reward": "#4CAF50",
        "Always-Frontier": "#F44336",
        "Always-Cheap": "#2196F3",
        "Random": "#FF9800",
    }
    colors = [color_map.get(n, "#9E9E9E") for n in names]

    bars = ax.bar(names, costs, color=colors, edgecolor="white")
    for bar, cost, perf, n, p in zip(bars, costs, perfs, ns, n_pass, strict=True):
        ci_str = ""
        if n > 0:
            lo, hi = plot_style.wilson_interval(p, n)
            ci_str = f"[{lo * 100:.0f}-{hi * 100:.0f}]"
        # Two lines, not one: bars with an identical cost (a real tie in this
        # data) sit adjacent, and a single wide line would bleed into the
        # neighbor's label — stacking keeps each bar's own text narrow.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"${cost:.4f}\n{perf:.0f}% {ci_str}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xlabel("Strategy (routing rule — bars are what each router SELECTS, not per-model data)")
    ax.set_ylabel("Total cost over all tasks ($)")
    ax.set_title(
        "Total cost by routing strategy — equal-quality framing\n"
        "(pass rate bracketed above each bar)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)

    ref_cost: float | None = None
    if len(names) >= 2:
        # Under equal coverage the frontier is measured on every task, so its total
        # cost is a valid reference line (no sparse-frontier guard needed — S6).
        ref_cost = next(
            (float(r["TotalCost"]) for r in results if r["strategy"] == "Always-Frontier"), None
        )
        if ref_cost is not None:
            ax.axhline(
                y=ref_cost,
                color="#F44336",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Always-Frontier cost = ${ref_cost:.4f}",
            )
            ax.legend()

    return plot_frame.save(
        fig,
        out_dir / "cost_savings.png",
        _COST_SAVINGS_SPEC,
        extra=_cost_savings_annotations(names, ns, costs, ref_cost, banner),
    )


_COST_QUALITY_SPEC = FigureSpec(
    reading=(
        "x is the total dollars a strategy spent; y is its pass rate in percent with a 95% "
        "Wilson error bar. The orange diamond is Always-Frontier and the orange horizontal band "
        "is ITS OWN interval — the 'equal quality' zone. A point is blue when its own interval "
        "overlaps that band and grey when it does not. A blue point far LEFT of the diamond is a "
        "strategy buying indistinguishable quality for less money."
    ),
    goal=(
        "Track the orange band leftwards: the leftmost BLUE point is the cheapest strategy this "
        "data cannot tell apart from Always-Frontier."
    ),
    definitions=(
        ("Always-Frontier", "always route to the strongest enabled model"),
        ("equal quality", "the two pass-rate intervals overlap; not proof they are equal"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
    ),
    notes=(
        "This is the kill-gate figure: the router has to reach the orange band at materially "
        "lower cost, or the project stops.",
        _CI_NOTE,
    ),
    limitations=(
        "Overlapping intervals at small n are WEAK evidence — the band widens as n shrinks and "
        "can swallow every point.",
        "This is an UNPAIRED proxy for the real gate, which is a paired per-task McNemar "
        "contrast on the same tasks.",
        "Frontier quality is partly imputed and imputation is conservative — it hands the "
        "frontier free passes, so the router's lead here is understated, not inflated.",
    ),
)


def _cost_quality_annotations(
    ns: list[int],
    frontier_idx: int | None,
    band: tuple[float, float] | None,
    overlaps: int,
    banner: str | None,
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
    )


def plot_cost_quality_equal(
    results: list[dict[str, str]], out_dir: Path, banner: str | None = None
) -> Path:
    """N1 (highest priority) — the kill-gate plot: pass% (Wilson CI) vs cost,
    Always-Frontier's CI as a horizontal band, annotating the cost cut at the
    cheapest CI-overlapping-quality strategy. Equal coverage makes the band valid.
    """
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6.5))

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
        overlap_count += int(overlaps and not is_frontier)
        color = "#D55E00" if is_frontier else ("#0072B2" if overlaps else "#9E9E9E")
        marker = "D" if is_frontier else "o"
        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=140 if is_frontier else 110,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.6,
        )
        label_points.append((float(costs[i]), float(perfs[i]), name))
        if overlaps and frontier_idx is not None and not is_frontier and costs[frontier_idx] > 0:
            cut = (1 - costs[i] / costs[frontier_idx]) * 100
            if best_cut is None or cut > best_cut[1]:
                best_cut = (name, cut, float(perfs[i]), float(perfs[frontier_idx]))
    plot_style.label_points_with_leaders(ax, label_points)

    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("Pass rate (%)")

    if best_cut is not None:
        name, cut, perf, fperf = best_cut
        title = (
            f"{name} matches Always-Frontier quality ({perf:.0f}% vs {fperf:.0f}%, "
            f"CIs overlap) at {max(cut, 0.0):.0f}% less cost"
        )
    else:
        title = (
            "No strategy's quality CI overlaps Always-Frontier's yet — "
            "no cost-equal-quality win to report"
        )
    ax.set_title(title, fontsize=10)
    if band is not None:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    return plot_frame.save(
        fig,
        out_dir / "cost_quality_equal.png",
        _COST_QUALITY_SPEC,
        extra=_cost_quality_annotations(ns, frontier_idx, band, overlap_count, banner),
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


def _heatmap_annotations(
    grid: np.ndarray, coverage: np.ndarray, glyphs: bool, raw: RawResults, synthesized: bool
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
            "At this size per-cell glyphs are off and task labels are thinned to ~40 evenly "
            "spaced rows, so most rows are unlabelled."
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

    # Thin task labels when there are too many to read; show ~40 evenly spaced.
    ystep = max(1, int(np.ceil(n_tasks / 40)))
    yticks = list(range(0, n_tasks, ystep))
    ax.set_yticks(yticks)
    ax.set_yticklabels([tasks[i].split("__")[-1] for i in yticks], fontsize=7)

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
        extra=_heatmap_annotations(grid, coverage, glyphs, raw, raw_results is None),
    )


_ARM_MONOTONICITY_SPEC = FigureSpec(
    reading=(
        "Small multiples, one facet per model. Inside a facet x is that model's average cost per "
        "task in dollars and y is its pass rate in percent with a 95% Wilson error bar; one point "
        "per sampled arm, connected in ascending within-model arm rank ('more effort'), labelled "
        "with the arm id and its n. A line that climbs left-to-right means more reasoning effort "
        "buys quality. Never compare x or y ACROSS facets as a rank scale: arm rank is ordinal "
        "within one model only, so rank 1 on a two-arm model is not rank 1 on a four-arm model."
    ),
    goal=(
        "Inside each facet, look for a line that rises. Flat or falling means the extra effort "
        "is buying cost and not pass rate."
    ),
    definitions=(
        ("arm", "one reasoning-effort setting of a single model"),
        ("arm rank", "ordinal effort order WITHIN one model, from the registry"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
    ),
    notes=(_CI_NOTE,),
    limitations=(
        "Points are not size- or fill-coded by sample size here: a point built on 1 task looks "
        "exactly like one built on 50, so read the n in each label.",
    ),
)


def _monotonicity_annotations(
    raw: RawResults, columns: list[tuple[str, str]], ordered_models: list[str]
) -> Annotations:
    """Runtime footer content for N2: how many facets have any arm variation at all."""
    stats = [plot_style.arm_stats(raw, m, a) for m, a in columns]
    sampled = [s for s in stats if s.n > 0]
    arms_per_model = {m: sum(1 for c in columns if c[0] == m) for m in ordered_models}
    lone = sum(1 for count in arms_per_model.values() if count <= 1)
    notes: list[str] = []
    limits: list[str] = []
    if sampled:
        ns = [s.n for s in sampled]
        notes.append(f"Per-arm sample sizes span n={min(ns)}-{max(ns)} across {len(sampled)} arms.")
        thin = sum(1 for s in sampled if s.provisional)
        if thin:
            limits.append(
                f"{thin} of {len(sampled)} plotted arms rest on fewer than "
                f"{plot_style.MIN_N_PROVISIONAL} tasks."
            )
    if lone:
        limits.append(
            f"{lone} of {len(ordered_models)} facets have a single sampled arm, so they are one "
            "dot with no slope to read."
        )
    limits.extend(_single_arm_limits(raw))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_arm_monotonicity(
    raw_results: RawResults,
    out_dir: Path,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
) -> Path | None:
    """N2 — small multiples, one facet per model, arms connected in rank order
    ("-> more effort"). The facet boundary structurally forbids a shared
    none/low/med/high axis across models (arms are ordinal within one model only).
    """
    columns = plot_style.arm_columns(raw_results)
    if not columns:
        return None
    ordered_models = [m for m in model_colors if any(c[0] == m for c in columns)]
    if not ordered_models:
        return None
    ncols = min(3, len(ordered_models))
    nrows = math.ceil(len(ordered_models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)

    for idx, model in enumerate(ordered_models):
        ax = axes[idx // ncols][idx % ncols]
        _draw_arm_monotonicity_facet(ax, raw_results, model, model_colors, arm_ranks, columns)

    for idx in range(len(ordered_models), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Arm monotonicity by model", fontsize=11)
    fig.tight_layout()
    return plot_frame.save(
        fig,
        out_dir / "arm_monotonicity.png",
        _ARM_MONOTONICITY_SPEC,
        extra=_monotonicity_annotations(raw_results, columns, ordered_models),
    )


def _draw_arm_monotonicity_facet(
    ax,  # noqa: ANN001 (matplotlib Axes; benchmark harness relaxed rung)
    raw_results: RawResults,
    model: str,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
    columns: list[tuple[str, str]],
) -> None:
    arms = sorted((a for m, a in columns if m == model), key=lambda a: arm_ranks.get((model, a), 0))
    color = model_colors.get(model, "#9E9E9E")
    xs, ys, downs, ups, ns = [], [], [], [], []
    for arm in arms:
        st = plot_style.arm_stats(raw_results, model, arm)
        lo, hi = st.wilson
        xs.append(st.avg_cost)
        ys.append(st.pass_rate * 100)
        down, up = plot_style.ci_yerr(st.pass_rate, lo, hi)
        downs.append(down * 100)
        ups.append(up * 100)
        ns.append(st.n)
    if xs:
        ax.errorbar(
            xs,
            ys,
            yerr=[downs, ups],
            fmt="o-",
            color=color,
            ecolor="#555555",
            capsize=3,
            markersize=7,
            zorder=3,
        )
        n_pts = len(xs)
        for i, (x, y, arm, n) in enumerate(zip(xs, ys, arms, ns, strict=True)):
            # Label toward the facet centre (leftmost point labels right, the
            # rightmost labels left) so text never runs off the facet edge; a
            # two-line label keeps it narrow and a translucent box lifts it off
            # the marker + error-bar cap it would otherwise sit on.
            last = n_pts > 1 and i == n_pts - 1
            ha = "right" if last else "left"
            dx = -7 if last else 7
            dy, va = (12, "bottom") if y < 82 else (-12, "top")
            # Arms of one model can share almost the same cost AND pass rate (adjacent
            # effort rungs often do), which stacks their labels; alternating the offset
            # by index separates the pair without moving the markers.
            if i % 2:
                dy = dy + (16 if dy > 0 else -16)
            ax.annotate(
                f"{arm}\n(n={n})",
                (x, y),
                fontsize=7,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
        if len(xs) >= 2:
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=1.2),
                zorder=2,
            )
            ax.text(
                0.5,
                -0.22,
                "→ more effort",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                color="#666666",
            )
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("avg cost/task ($)", fontsize=8)
    ax.set_ylabel("pass rate (%)", fontsize=8)
    ax.set_ylim(-5, 112)
    ax.margins(x=0.18)
    # Few, plain, rotated x-ticks: arm costs can be near-identical (e.g. the two
    # kimi-k2.5 arms differ by <$0.001), and the default locator then jams six
    # 6-digit labels into a collision — cap the count and rotate instead.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
        lbl.set_fontsize(7)
    ax.grid(True, alpha=0.3)


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
        "Hollow markers rest on a small sample. The dark dashed line is the convex hull through "
        "the non-dominated cells."
    ),
    goal=(
        "Aim top-left: most pass rate per dollar. The cells sitting on the dashed hull are the "
        "ones a router should be choosing between; everything below it is dominated."
    ),
    definitions=(
        ("arm", "one reasoning-effort setting of a single model"),
        ("arm rank", "ordinal effort order WITHIN one model, from the registry"),
        ("AIQ", "area under the frontier over the plotted-cost x 100% rectangle, 0-1"),
        ("Wilson CI", "confidence band on a pass rate that stays honest at small n"),
    ),
    notes=(_CI_NOTE,),
    limitations=(
        "AIQ is normalised by the widest cost actually plotted, so it moves when the plotted "
        "set changes — read it within one figure, never across figures or runs.",
        "x is an average over tasks, so it hides per-task cost variance entirely.",
        "A convex hull fitted through a handful of small-sample points is unstable — one more "
        "sample can move it.",
    ),
)


def _arm_cloud_annotations(
    points: list[dict], raw: RawResults, aiq: float, plotted_max_rank: int
) -> Annotations:
    """Runtime footer content for N4: cell count, sample thinness, whether size means anything."""
    ns = [p["n"] for p in points]
    notes = [
        f"{len(points)} measured (model, arm) cell(s) from "
        f"{len({p['model'] for p in points})} model(s); sample sizes span n={min(ns)}-{max(ns)}.",
        f"AIQ={aiq:.2f} of the cost-quality rectangle lies under the hull.",
    ]
    limits = list(_provisional_limits(points, "plotted cells"))
    if plotted_max_rank == 0:
        limits.append(
            "Every plotted cell is at arm rank 0, so the marker-size channel carries no "
            "information on this data — all markers are the base size."
        )
    limits.extend(_single_arm_limits(raw))
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


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

    pareto_pts = plot_style.pareto_prune([(p["cost"], p["pass_rate"]) for p in points])
    hull = plot_style.upper_hull(pareto_pts)
    aiq = plot_style.area_under_frontier(hull)
    if hull:
        fx = [p[0] for p in hull]
        fy = [p[1] for p in hull]
        if fx[0] > 0:
            fx = [0.0, *fx]
            fy = [fy[0], *fy]
        ax.plot(
            fx,
            fy,
            color="#333333",
            lw=1.6,
            linestyle="--",
            label=f"convex-hull frontier (AIQ={aiq:.2f})",
        )

    ax.set_xlabel("avg cost per task ($)")
    ax.set_ylabel("pass rate (%)")
    ax.set_title(
        f"(model, arm) cost-quality cloud — hue=model, size=arm rank (AIQ={aiq:.2f})",
        fontsize=10,
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
        extra=_arm_cloud_annotations(points, raw_results, aiq, plotted_max_rank),
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


def _chosen_arm_annotations(
    plotted: int, tasks: int, unmeasured: int, unsampled: int, chosen: set[str], omitted: list[str]
) -> Annotations:
    """Runtime footer content for N5: how many tasks silently fell out of the cloud."""
    notes = [
        f"{plotted} of {tasks} sampled task(s) are plotted; the router picked "
        f"{len(chosen)} distinct model(s)."
    ]
    limits: list[str] = []
    if unmeasured or unsampled:
        limits.append(
            f"{unmeasured + unsampled} task(s) are missing from this cloud: {unmeasured} where "
            f"the chosen (model, default-arm) cell was never measured and {unsampled} with no "
            "sampled cell at all."
        )
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
    chosen_models: set[str] = set()
    n_unsampled = 0
    n_unmeasured = 0
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
            n_unmeasured += 1
            continue
        xs.append(solved + jitter)
        ys.append(float(row.get("cost", 0.0)))
        colors.append(model_colors.get(model, "#9E9E9E"))
        chosen_models.add(model)
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
    ax.set_ylabel("chosen (model, default-arm) cost ($/task)")
    ax.set_title(
        "What kNN-cascade SELECTS: chosen model cost vs task solve-breadth\n"
        "(strategy-conditioned — one routed model per task, not per-model data; "
        "arm is always the default — live per-arm routing isn't wired up yet)",
        fontsize=9,
    )
    # Legend covers only the models this strategy actually routed to; models in the
    # palette it never selected are omitted (noted) rather than shown as dead keys.
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in model_colors.items()
        if m in chosen_models
    ]
    omitted = [m for m in model_colors if m not in chosen_models]
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
            len(xs), len(tasks), n_unmeasured, n_unsampled, chosen_models, omitted
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
    pca = PCA(n_components=2)
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
    """Ordinal band index per model (1 = weakest ... N = strongest). Adjacent ranked models
    merge into one band when their capability CIs overlap (cut only when prev.ci_hi <
    cur.ci_lo), so the band count is data-driven, not capped."""
    bands: dict[str, int] = {}
    band = 1
    prev_ev: config.ModelEvidence | None = None
    for rm in rank.ordered:
        ev = rank.evidence.get(rm.model)
        if prev_ev is not None and ev is not None and prev_ev.ci_hi < ev.ci_lo:
            band += 1
        bands[rm.model] = band
        if ev is not None:
            prev_ev = ev
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
    models, price range, pass-rate range) — never a semantic name."""
    m = meta.get(b, {})
    who = _cap_names(m.get("models") or [], 3)
    pr = m.get("price_range")
    price = f"${pr[0]:.0f}–${pr[1]:.0f}" if pr else "$?"
    cr = m.get("capability_range")
    rate = f"{cr['pass_rate_min'] * 100:.0f}–{cr['pass_rate_max'] * 100:.0f}%" if cr else "?"
    return f"band {b}\n{who}\n{price} · {rate}"


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
    return (
        head + axiom + " Imputation is conservative — the frontier gets free quality — so a "
        "broken axiom only widens the router's measured lead."
    )


_BAND_TERMS = (
    ("tau (crossover)", "the weakest capability band that solves a given task"),
    ("capability band", "adjacent ranked models merged when their capability CIs overlap"),
)

_CAPABILITY_SPEC = FigureSpec(
    reading=(
        "One bar per capability band, ordered weakest-first from the left, with "
        "'unsolvable' last. Each x tick is three lines of band metadata: the band index, its "
        "member models, and the band's price and pass-rate range. y is how many tasks that band "
        "is the WEAKEST to solve, labelled with the count and the share of the suite. A suite "
        "whose mass sits in the left-hand bands is mostly routable to cheap models."
    ),
    goal=(
        "Read where the mass sits. Mass in the weaker (left-hand) bands means most tasks fall to "
        "the cheapest models — that is the headroom a router monetises. Mass in 'unsolvable' is "
        "headroom no router has."
    ),
    definitions=(
        *_BAND_TERMS,
        ("Copeland score", "pairwise win count over tasks both models were measured on"),
    ),
    notes=(
        "Model rank is a Copeland score over CO-MEASURED tasks, deliberately not raw pass rate — "
        "the strongest models only ran on the hard subset.",
        "Band boundaries are data-driven: adjacent models merge when their capability intervals "
        "overlap, so the band count moves with the data and is not a fixed cap.",
    ),
    limitations=(
        "'unsolvable' conflates two different things: nothing solved the task, and nothing strong "
        "enough was ever run on it.",
        "tau is capped by what was sampled — it can never be stronger than the strongest model "
        "that actually ran on the task.",
        "Band membership labels are capped at three model names, so a wide band under-reports "
        "who is in it.",
    ),
)


def _capability_annotations(
    im: ImputedMatrix, rank: config.CapabilityRank, n_bands: int, total: int, unsolvable: int
) -> Annotations:
    """Runtime footer content for the capability histogram: imputation and rank provenance."""
    covered = im.n_real + im.n_imputed
    denom = total or 1  # safe division; the PRINTED count stays the true one
    notes = [
        f"{total} task(s) across {n_bands} capability band(s); {unsolvable} "
        f"({unsolvable / denom * 100:.0f}%) have no band that solves them.",
        f"{_violation_line(im)}.",
    ]
    if n_bands <= 1:
        notes.append(
            "The band axis has COLLAPSED to a single capability band on this data — the models "
            "never separated, so there is no left-to-right weak-to-strong gradient to read."
        )
    if covered:
        notes.append(
            f"{im.n_imputed} of {covered} cells ({im.n_imputed / covered * 100:.0f}%) are "
            f"monotone-imputed; {im.n_unknown} left UNKNOWN and excluded from tau."
        )
    limits: list[str] = []
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
    for model in im.tau.values():
        key = bands.get(model, "unsolvable") if model is not None else "unsolvable"
        counts[key] += 1
    total = sum(counts.values())
    denom = total or 1

    fig, ax = plt.subplots(figsize=(9, 5.5))
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
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("crossover capability band τ (weakest band that solves the task)")
    ax.set_ylabel("number of tasks")
    ax.set_title(
        f"Capability distribution — where the suite's difficulty lives ({total} tasks)\n"
        "τ = weakest capability band that solves each task ('unsolvable' = none does); "
        "bands are an ordinal, metadata-described grouping of the measured rank, "
        "not the routing unit",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    return plot_frame.save(
        fig,
        out_dir / "capability_distribution.png",
        _CAPABILITY_SPEC,
        extra=_capability_annotations(im, rank, len(order), total, counts["unsolvable"]),
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
        "One bar per task stratum — the tasks sharing a crossover capability band — ordered "
        "weakest to strongest with 'unsolvable' last; each x tick is band index, member models, "
        "price and pass-rate range. Bar HEIGHT is the winning deployable strategy's Reward SUMMED "
        "over that stratum's tasks, so it scales with how many tasks the stratum holds and is not "
        "comparable across bars. The label above each bar is the part that matters: which "
        "strategy won there, and its Reward."
    ),
    goal=(
        "Read the NAMES, not the heights. Different winners across strata means per-task routing "
        "has real headroom; the same winner everywhere means routing buys nothing here."
    ),
    definitions=(
        *_BAND_TERMS,
        ("Reward", "1 for a pass, 0 for a fail, minus gamma x cost, summed"),
        ("deployable", "a strategy a live router could run — oracles excluded"),
    ),
    notes=(
        "Oracles are excluded: they peek at realised outcomes and would win every stratum by "
        "construction.",
        "'unsolvable' strata normally go negative — nothing passes there, so only cost accrues.",
    ),
    limitations=(
        "No confidence interval on any bar, so a near-tie between two strategies reads exactly "
        "like a decisive win.",
        "Scored on the partly-imputed matrix with a reduced bootstrap, so these Rewards are "
        "coarser than the headline strategy rows.",
        "The winner is the largest SUMMED Reward and strategies in a stratum can be scored on "
        "different task counts, so wider coverage can win on volume rather than on quality.",
    ),
)


def _stratum_size(rows: list[dict]) -> int:
    """Tasks in a stratum — the largest scorable denominator any strategy row reports."""
    return max((int(float(r.get("n_tasks", 0) or 0)) for r in rows), default=0)


def _per_stratum_annotations(rows_by_stratum: dict, strata: list) -> Annotations:
    """Runtime footer content: stratum sizes, empty strata, and how close the wins are."""
    sizes = {s: _stratum_size(rows_by_stratum[s]) for s in strata}
    notes = ["Per-stratum task counts: " + ", ".join(f"{s}={n}" for s, n in sizes.items()) + "."]
    limits: list[str] = []
    if sizes and min(sizes.values()) < 10:
        limits.append(
            f"The smallest stratum holds only {min(sizes.values())} task(s), so its winner rests "
            "on almost no evidence."
        )
    empty = [s for s in strata if _stratum_winner(rows_by_stratum[s]) is None]
    if empty:
        limits.append(
            f"{len(empty)} stratum/strata have no deployable strategy row and are drawn as "
            "'none' at Reward 0."
        )
    margins = [_win_margin(rows_by_stratum[s]) for s in strata]
    real = [m for m in margins if m is not None]
    if real:
        limits.append(
            f"The narrowest win margin over the runner-up is {min(real):.2f} reward — read close "
            "bars as unresolved, not as a winner."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _win_margin(rows: list[dict]) -> float | None:
    """Reward gap between the best and second-best deployable strategy in a stratum."""
    deployable = sorted(
        (
            float(r.get("Reward", 0))
            for r in rows
            if r["strategy"] not in ("Oracle", "Oracle-reward")
            and int(float(r.get("n_tasks", 0) or 0)) > 0
        ),
        reverse=True,
    )
    return deployable[0] - deployable[1] if len(deployable) >= 2 else None


def plot_per_stratum_winrate(rows_by_stratum: dict, out_dir: Path) -> Path:
    """Which strategy wins per τ stratum (where routing helps vs where it can't) —
    one bar per capability-band stratum, height = winner's Reward, labelled with the winner."""
    rank, bands, order = _rank_bands()
    meta = band_metadata(rank, bands)
    strata = [s for s in [*order, "unsolvable"] if s in rows_by_stratum]
    winners = [(_stratum_winner(rows_by_stratum[s]) or ("none", 0.0)) for s in strata]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    heights = [w[1] for w in winners]
    labels = [_band_label(s, meta) if isinstance(s, int) else "unsolvable" for s in strata]
    bars = ax.bar(range(len(strata)), heights, color="#009E73", edgecolor="white")
    # Labels always sit ABOVE their bar, with headroom reserved for them: a
    # near-zero winner drawn below the axis collides with the 3-line tick text, and
    # the tallest bar's label is clipped by the axes edge without the margin.
    span = (max(heights) - min(min(heights), 0.0)) or 1.0
    pad = span * 0.06
    for bar, (name, reward) in zip(bars, winners, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            reward + pad if reward >= 0 else pad,
            f"{name}\n(R={reward:.2f})",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_ylim(min(min(heights), 0.0) - pad, max(heights) + span * 0.22)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("task stratum (crossover capability band τ)")
    ax.set_ylabel("winning strategy's Reward")
    ax.set_title(
        "Per-stratum winner — which router wins on easy vs hard tasks\n"
        "(oracles excluded; Reward = pass − γ·cost summed over the stratum)",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    return plot_frame.save(
        fig,
        out_dir / "per_stratum_winrate.png",
        _PER_STRATUM_SPEC,
        extra=_per_stratum_annotations(rows_by_stratum, strata),
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
        single = float(results.get(tid, {}).get(tau, {}).get("cost", 0.0))
        overheads.append(float(cascade_cost) - single)
    return sum(overheads) / len(overheads) if overheads else 0.0


def _per_stratum_rows(
    matrix: dict, im: ImputedMatrix, strategies: list[object], tasks: list[str], seed: int
) -> dict[int | str, list[dict]]:
    """Score every strategy within each τ capability-band stratum on the completed matrix."""
    _rank, bands, _order = _rank_bands()
    by_stratum: dict[int | str, list[str]] = {}
    for tid in tasks:
        tau = im.tau.get(tid)  # crossover MODEL; grouped into its ordinal band
        stratum: int | str = "unsolvable" if tau is None else bands.get(tau, "unsolvable")
        by_stratum.setdefault(stratum, []).append(tid)
    bm = config.benchmark_params()
    out: dict[int | str, list[dict]] = {}
    for stratum, stratum_tasks in by_stratum.items():
        out[stratum] = summary.compute_strategy_rows(
            matrix,
            stratum_tasks,
            strategies,
            gamma=config.gamma(),
            bootstrap=min(200, bm.get("bootstrap_iterations", 1000)),
            seed=seed,
        )
    return out


def derive_tasks(matrix: dict, seed: int) -> list[str]:
    """Covered tasks (present in results.csv) sampled by ``sample_size`` — matches run_eval."""
    # Evaluate over the MEASURED denominator, not the full suite: an uncovered task is
    # unmeasured, NOT a failure — scoring the 151 unrun-of-200 as fail@$0 diluted every
    # plotted pass-rate ~4x and made report.py disagree with run_eval/plot_strategies.
    return config.sample_tasks(sorted(matrix.get("results", {}).keys()), seed=seed)


def derive_rows(matrix: dict, tasks: list[str]) -> list[dict]:
    """Compute per-strategy summary rows in-memory from the results.csv cache —
    the single source of truth. Mirrors run_matrix.refresh_summary exactly (same
    strategies, gamma, bootstrap, seed, and task set — see derive_tasks).
    """
    from benchmark.routing import run_eval

    bm = config.benchmark_params()
    strategies = run_eval.get_strategies()
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
            print(f"  {label}: skipped ({type(exc).__name__}: {exc})")
            continue
        print(f"  {label}: {result_path}" if result_path else f"  {label}: skipped (no data)")
    _ = matrix_path  # kept for signature symmetry with the other plot entry points


def _report_imputation_outputs(
    im: ImputedMatrix, matrix: dict, tasks: list[str], out_dir: Path
) -> None:
    """Emit the five equal-coverage outputs: violation rate, coverage
    table, capability histogram, per-stratum winner, cascade overhead-leak."""
    from benchmark.routing import run_eval
    from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

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

    strategies = run_eval.get_strategies()
    seed = config.benchmark_params().get("seed", 42)
    rows_by_stratum = _per_stratum_rows(matrix, im, strategies, tasks, seed)
    print(f"  Per-stratum  : {plot_per_stratum_winrate(rows_by_stratum, out_dir)}")

    completed, _im = summary.complete_scored_matrix(matrix)
    overhead = cascade_overhead(completed, im, tasks, kNNCascadeStrategy(**config.knn_params()))
    print(f"  Cascade overhead-leak: ${overhead:.4f}/task (failed cheaper probes before τ)")


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
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

    if args.results:
        results = load_results(Path(args.results))
        source = args.results
        tasks: list[str] = []
    else:
        matrix = load_matrix(matrix_path)
        if matrix is None or not matrix.get("results"):
            print(
                "No results yet — results.csv holds no rows. "
                "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
            )
            return
        tasks = derive_tasks(matrix, config.benchmark_params().get("seed", 42))
        results = derive_rows(matrix, tasks)
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
    matrix_for_plots = load_matrix(matrix_path) if matrix_path and matrix_path.exists() else None

    # Complete the matrix ONCE for the diagnostics; the scored rows already came
    # through the same completion inside compute_strategy_rows.
    imputed: ImputedMatrix | None = None
    if matrix_for_plots is not None:
        _completed, imputed = summary.complete_scored_matrix(matrix_for_plots)
    banner = _disclosure_banner(imputed, results)

    p1 = plot_pareto(results, out_dir, raw_results, len(tasks), banner)
    print(f"  Pareto       : {p1}")

    g = config.gamma()
    factories = _build_strategy_factories(g)
    p2 = plot_cumulative_regret(
        results,
        out_dir,
        matrix_path,
        gamma=g,
        strategy_factories=factories,
        raw_results=raw_results,
    )
    print(f"  Regret       : {p2}")

    p3 = plot_cost_savings(results, out_dir, banner)
    print(f"  Cost savings : {p3}")

    p1n = plot_cost_quality_equal(results, out_dir, banner)
    print(f"  Cost=quality : {p1n}")

    if imputed is not None and matrix_for_plots is not None:
        _report_imputation_outputs(imputed, matrix_for_plots, tasks, out_dir)

    if matrix_path is not None:
        if matrix_path.exists():
            try:
                p4 = plot_heatmap(matrix_path, out_dir, raw_results)
                print(f"  Heatmap      : {p4}")
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"  Heatmap      : skipped ({exc})")
        else:
            print(f"  Heatmap      : matrix file {matrix_path} not found, skipping")

    _run_arm_plots(out_dir, matrix_for_plots, matrix_path, tasks, raw_results)

    plt.close("all")


if __name__ == "__main__":
    main()
