#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

from benchmark import config
from benchmark.routing import plot_style, summary
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
    """Okabe-Ito hue per model, fixed by tier/price order (cheap -> frontier)."""
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
    raw: RawResults,
    model_colors: dict[str, str],
    arm_ranks: dict[tuple[str, str], int],
    n_tasks: int,
) -> None:
    """Faint (model, arm) dots behind the strategy markers — individual-cell
    context, extrapolated onto the same total-cost axis (avg cost x n_tasks).
    """
    for pt in _arm_cloud_points(raw, model_colors, arm_ranks, scale_to_n_tasks=n_tasks):
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


def _hull_pareto_indices(names: list[str], pareto_map: dict[str, bool], phantom: bool) -> list[int]:
    """Indices entering the convex-hull/AIQ computation."""
    # Pareto-flagged rows, minus a phantom Always-Frontier — near-zero
    # cost/perf from partial coverage looks non-dominated to the generic
    # Pareto check but is never a real Pareto point (mirrors the guard N1 /
    # cost_savings already apply).
    return [
        i
        for i, name in enumerate(names)
        if pareto_map.get(name, False) and not (phantom and name == "Always-Frontier")
    ]


def plot_pareto(
    results: list[dict[str, str]],
    out_dir: Path,
    raw_results: RawResults | None = None,
    n_tasks: int = 0,
    matrix: dict | None = None,
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
    cov = _frontier_coverage(matrix)
    phantom = cov is not None and cov[1] < cov[2]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    has_underlay = bool(raw_results)
    if raw_results:
        model_colors = _model_colors()
        _draw_arm_underlay(ax, raw_results, model_colors, _arm_ranks(), n_tasks or len(raw_results))

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

    pareto_idx = _hull_pareto_indices(names, pareto_map, phantom)
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
        f"Cost vs quality — Pareto frontier spans AIQ={aiq:.2f} of the cost-quality rectangle\n"
        f"({plot_style.ci_footer()})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if phantom:
        assert cov is not None
        frontier, covered, total = cov
        if "Always-Frontier" in names:
            i = names.index("Always-Frontier")
            ax.annotate(
                f"⚠ phantom baseline (frontier covered on {covered}/{total} tasks)",
                xy=(costs[i], perfs[i]),
                xytext=(0.5, 1.12),
                textcoords="axes fraction",
                ha="center",
                fontsize=8,
                color="#B71C1C",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec="#B71C1C", alpha=0.95),
            )

    path = out_dir / "pareto_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _evaluate_strategies(
    factory_map: dict[str, Callable[[], Strategy]],
    matrix: dict,
    tasks: list[str],
) -> dict[str, list[tuple[str, str, bool, float]]]:
    evaluated: dict[str, list[tuple[str, str, bool, float]]] = {}
    for name, factory in factory_map.items():
        strategy = factory()
        decisions: list[tuple[str, str, bool, float]] = []
        for tid in tasks:
            task_meta = matrix.get("tasks", {}).get(tid, {})
            model = strategy.select(tid, task_meta, matrix)
            outcome = matrix.get("results", {}).get(tid, {}).get(model, {})
            passed = outcome.get("pass", False)
            cost = outcome.get("cost", 0.0)
            decisions.append((tid, model, passed, cost))
        evaluated[name] = decisions
    return evaluated


def _compute_per_task_regret(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    gamma: float = 0.1,
) -> np.ndarray:
    regrets: list[float] = []
    for sd, od in zip(strategy_decisions, oracle_decisions, strict=True):
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
            tasks = sorted(matrix.get("results", {}).keys())
            if tasks:
                all_decisions = _evaluate_strategies(strategy_factories, matrix, tasks)
                oracle_decisions = all_decisions.get("Oracle-reward")
                if oracle_decisions:
                    finals: dict[str, float] = {}
                    for name in [r["strategy"] for r in results]:
                        decisions = all_decisions.get(name)
                        if decisions and name != "Oracle-reward":
                            cumreg = _compute_per_task_regret(decisions, oracle_decisions, gamma)
                            label = name + f" (total={cumreg[-1]:.2f})"
                            ax.plot(range(1, len(cumreg) + 1), cumreg, label=label, lw=1.5)
                            finals[name] = float(cumreg[-1])

                    if raw_results:
                        raw_sampled = {cid: raw_results[cid] for cid in tasks if cid in raw_results}
                        for extra_name, fn, style in (
                            ("Arm-oracle", _arm_oracle_decisions, "--"),
                            ("Arm-bandit", _arm_bandit_decisions, ":"),
                        ):
                            extra_decisions = fn(raw_sampled, tasks, gamma)
                            cumreg = _compute_per_task_regret(
                                extra_decisions, oracle_decisions, gamma
                            )
                            label = f"{extra_name} (total={cumreg[-1]:.2f})"
                            ax.plot(
                                range(1, len(cumreg) + 1),
                                cumreg,
                                label=label,
                                lw=1.5,
                                linestyle=style,
                            )
                            finals[extra_name] = float(cumreg[-1])

                    single_arm = raw_results is not None and plot_style.is_single_arm(raw_results)
                    caveat = f" — {plot_style.ARM_SWEEP_PENDING_NOTE}" if single_arm else ""
                    ax.set_xlabel("Task (evaluation order)")
                    ax.set_ylabel(f"Cumulative regret vs Oracle-reward (γ={gamma}, reward units)")
                    # Exclude "Oracle" itself from the headline pick — it is compared
                    # against the (near-identical) Oracle-reward baseline, so its
                    # near-zero regret is tautological, not an informative takeaway.
                    # In the single-arm-only case, "Arm-oracle" is DEFINITIONALLY
                    # identical to Oracle-reward too (only one arm exists per model),
                    # so exclude it there as well — it only competes fairly once
                    # multiple arms are actually sampled.
                    excluded = {"Oracle", "Arm-oracle"} if single_arm else {"Oracle"}
                    candidates = {k: v for k, v in finals.items() if k not in excluded}
                    if candidates:
                        best_name, best_val = min(candidates.items(), key=lambda kv: kv[1])
                        ax.set_title(
                            f"{best_name} tracks the oracle closest among routers "
                            f"(regret={best_val:.2f} over {len(tasks)} tasks){caveat}"
                        )
                    else:
                        ax.set_title(f"Cumulative Regret vs Oracle (Per-Task){caveat}")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                    path = out_dir / "cumulative_regret.png"
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    return path

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
    ax.set_ylabel(f"Cumulative regret vs oracle (γ={gamma}, reward units)")
    ax.set_title("Cumulative Regret vs Oracle (Aggregate)")
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "cumulative_regret.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _frontier_coverage(matrix: dict | None) -> tuple[str, int, int] | None:
    """(frontier_model, tasks_with_data, total_tasks) for the most expensive
    evaluated model — the Always-Frontier / kill-gate baseline. ``None`` if no
    matrix. Exposes the sparse-frontier artifact that makes that baseline invalid.
    """
    if not matrix or not matrix.get("models"):
        return None
    results_data = matrix.get("results", {})
    total = len(results_data)
    if total == 0:
        return None
    frontier = max(matrix["models"], key=lambda m: _model_total_price(matrix["models"], m))
    covered = sum(1 for tr in results_data.values() if frontier in tr)
    return frontier, covered, total


def plot_cost_savings(
    results: list[dict[str, str]], out_dir: Path, matrix: dict | None = None
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

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title(
        f"Cost Comparison by Strategy — equal-quality framing "
        f"(pass rate, {plot_style.ci_footer()}, bracketed)"
    )
    ax.grid(True, axis="y", alpha=0.3)

    cov = _frontier_coverage(matrix)
    phantom = cov is not None and cov[1] < cov[2]
    if len(names) >= 2 and not phantom:
        # Only draw the frontier reference line when the frontier baseline is
        # actually measured on every task — otherwise it is not a valid reference.
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

    if phantom:
        assert cov is not None
        frontier, covered, total = cov
        if "Always-Frontier" in names:
            i = names.index("Always-Frontier")
            # Anchored in the top margin (axes-fraction y > 1, clip disabled) so
            # the box never overlaps a bar or its value label regardless of
            # which bar happens to be tallest.
            ax.annotate(
                f"⚠ PHANTOM BASELINE — {frontier} evaluated on {covered}/{total} tasks; "
                f"the other {total - covered} count as free $0 failures. Not comparable.",
                xy=(i, costs[i]),
                xytext=(0.5, 1.14),
                textcoords="axes fraction",
                ha="center",
                fontsize=8,
                color="#B71C1C",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec="#B71C1C", alpha=0.95),
            )

    path = out_dir / "cost_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_quality_equal(
    results: list[dict[str, str]], out_dir: Path, matrix: dict | None = None
) -> Path:
    """N1 (highest priority) — the kill-gate plot: pass% (Wilson CI) vs cost,
    Always-Frontier's CI as a horizontal band, annotating the cost cut at the
    cheapest CI-overlapping-quality strategy. Reuses the phantom-baseline guard.
    """
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    ns = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    n_pass = [int(float(r.get("n_pass", 0) or 0)) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    cov = _frontier_coverage(matrix)
    phantom = cov is not None and cov[1] < cov[2]
    frontier_idx = names.index("Always-Frontier") if "Always-Frontier" in names else None
    band: tuple[float, float] | None = None
    if frontier_idx is not None and not phantom and ns[frontier_idx] > 0:
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

    if phantom:
        assert cov is not None
        frontier, covered, total = cov
        title = (
            f"Always-Frontier ({frontier}) evaluated on only {covered}/{total} tasks — "
            "equal-quality comparison unmeasurable here"
        )
    elif best_cut is not None:
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
    ax.set_title(f"{title}\n({plot_style.ci_footer()})", fontsize=10)
    if band is not None:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    path = out_dir / "cost_quality_equal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
    single_arm_note = (
        f"  ({plot_style.ARM_SWEEP_PENDING_NOTE})"
        if plot_style.is_single_arm(raw)
        else f"  ({plot_style.UNEVEN_COVERAGE_NOTE})"
    )
    glyph_note = "" if glyphs else "  (colour-only at scale — see legend)"
    ax.set_title(
        f"Task × (model, arm) outcomes — ✓ pass · ✗ fail · gray = not sampled{glyph_note}\n"
        f"columns disagree on {len(split)} task(s): {_cap_names(split)} · "
        f"solved by none: {_cap_names(none_solve)} · "
        f"{frontier_model}/{frontier_arm} sampled on {frontier_n}/{n_tasks}{single_arm_note}",
        fontsize=9,
    )
    fig.tight_layout()

    path = out_dir / "model_complementarity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
    single_arm = plot_style.is_single_arm(raw_results)

    for idx, model in enumerate(ordered_models):
        ax = axes[idx // ncols][idx % ncols]
        _draw_arm_monotonicity_facet(ax, raw_results, model, model_colors, arm_ranks, columns)

    for idx in range(len(ordered_models), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    subtitle = f" — {plot_style.ARM_SWEEP_PENDING_NOTE}" if single_arm else ""
    fig.suptitle(f"Arm monotonicity by model{subtitle} ({plot_style.ci_footer()})", fontsize=11)
    fig.tight_layout()
    path = out_dir / "arm_monotonicity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
        )
        for x, y, arm, n in zip(xs, ys, arms, ns, strict=True):
            ax.annotate(
                f"{arm} (n={n})", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points"
            )
        if len(xs) >= 2:
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=1.2),
            )
            ax.text(
                0.5,
                -0.20,
                "→ more effort",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                color="#666666",
            )
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("avg cost/task ($)", fontsize=8)
    ax.set_ylabel("pass rate (%)", fontsize=8)
    ax.set_ylim(-5, 105)
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

    single_arm = plot_style.is_single_arm(raw_results)
    subtitle = f" — {plot_style.ARM_SWEEP_PENDING_NOTE}" if single_arm else ""
    ax.set_xlabel("avg cost per task ($)")
    ax.set_ylabel("pass rate (%)")
    ax.set_title(
        f"(model, arm) cost-quality cloud{subtitle} — hue=model, size=arm rank (AIQ={aiq:.2f})\n"
        f"({plot_style.ci_footer()})",
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
        model_legend = ax.legend(
            handles=handles, fontsize=7, loc="lower right", title="model (hue)"
        )
        ax.add_artist(model_legend)
    if size_handles:
        ax.legend(handles=size_handles, fontsize=6.5, loc="upper left", title="size = arm rank")
    ax.grid(True, alpha=0.3)

    path = out_dir / "arm_cost_quality_cloud.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
    for tid in tasks:
        per_model = raw_results.get(tid, {})
        sampled = [(m, a) for m, per_arm in per_model.items() for a in per_arm]
        if not sampled:
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
            continue
        xs.append(solved + jitter)
        ys.append(float(row.get("cost", 0.0)))
        colors.append(model_colors.get(model, "#9E9E9E"))
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
        "kNN-cascade's chosen model cost vs task solve-breadth\n"
        "(arm choice is always the default arm — live per-arm routing isn't wired up yet)",
        fontsize=9,
    )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in model_colors.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", title="model (hue)")
    ax.grid(True, alpha=0.3)

    path = out_dir / "chosen_arm_vs_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_embedding_routing_map(
    matrix: dict, out_dir: Path, model_colors: dict[str, str], k: int = 10
) -> Path | None:
    """N6 — 2-D PCA projection of the per-task feature vectors (reusing
    viz_knn's engineered model-outcome vectors; None if sklearn is unavailable),
    colored by the outcome-vector NN PROXY's (model, default-arm) — not kNNStrategy.
    """
    try:
        from sklearn.decomposition import PCA

        from benchmark.routing.scripts import viz_knn
    except ImportError:
        return None

    models_order = list(model_colors.keys())
    results = matrix.get("results", {})
    if not results or not models_order:
        return None
    task_ids, vecs = viz_knn.build_feature_vectors(results, models_order)
    if len(task_ids) < 3:
        return None

    k_eff = min(k, len(task_ids) - 1)
    selections = {
        tid: viz_knn.knn_select(vecs, i, models_order, k=k_eff) for i, tid in enumerate(task_ids)
    }
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)
    explained = pca.explained_variance_ratio_
    colors = [model_colors.get(selections[tid], "#9E9E9E") for tid in task_ids]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        coords[:, 0], coords[:, 1], c=colors, s=40, alpha=0.8, edgecolors="black", linewidth=0.3
    )
    seen = set(selections.values())
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in model_colors.items()
        if m in seen
    ]
    ax.legend(handles=handles, fontsize=7, title="chosen (model, default arm)")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(
        "Routing map — outcome-vector NN proxy selection per task "
        "(PCA of model-outcome vectors, NOT the routed kNN strategy)\n"
        "arm is always the default arm until the live executor wires per-arm routing",
        fontsize=9,
    )
    fig.tight_layout()

    path = out_dir / "embedding_routing_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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

    p1 = plot_pareto(results, out_dir, raw_results, len(tasks), matrix_for_plots)
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

    p3 = plot_cost_savings(results, out_dir, matrix_for_plots)
    print(f"  Cost savings : {p3}")

    p1n = plot_cost_quality_equal(results, out_dir, matrix_for_plots)
    print(f"  Cost=quality : {p1n}")

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
