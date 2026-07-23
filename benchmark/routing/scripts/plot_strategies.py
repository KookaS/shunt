#!/usr/bin/env python3
"""Strategy comparison Pareto scatter — rows derived exactly as report.py does."""

# Rows come from report.derive_tasks/derive_rows (i.e. summary.py), so this figure and
# report.py always print the SAME pass rates and costs off ONE denominator: coverage-gap
# cells are excluded, never scored as fail@$0.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config  # noqa: E402
from benchmark.routing import report  # noqa: E402
from benchmark.routing.metrics import compute_pareto  # noqa: E402

# Documented data-viz palette (light mode), the 6 best-separated categorical slots.
# Every strategy also carries a UNIQUE marker and a direct label, so identity never
# rests on hue alone — the required secondary encoding for a >3-series scatter, where
# no 6-hue subset can clear the all-pairs CVD floor (see dataviz color-formula).
_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#e34948")
_MARKERS = ("D", "s", "^", "v", "o", "P", "X", "*", "h", "<", ">", "d")
# Chart chrome & ink tokens (light surface #fcfcfb).
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"


def _style_maps(names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Assign color + marker per strategy, data-driven from a stable name sort.

    Enumerated from the results (never a hardcoded strategy list); a fixed sort keeps
    each strategy's hue/marker stable across runs (colour follows the entity).
    """
    ordered = sorted(names)
    colors = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(ordered)}
    markers = {n: _MARKERS[i % len(_MARKERS)] for i, n in enumerate(ordered)}
    return colors, markers


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def _coverage_note(matrix: dict, tasks: list[str], results: list[dict]) -> str | None:
    """Honest coverage caption: how many tasks the frontier model actually ran on."""
    # Plotted pass rates EXCLUDE unmeasured cells (summary.py drops them), so the
    # frontier number is real — but on a smaller denominator than its peers, and the
    # figure must say so.
    pricing = config.enabled_pricing()
    if not pricing:
        return None
    frontier_model = max(pricing, key=lambda m: config.cost_per_1m(m, pricing))
    covered = sum(1 for tid in tasks if frontier_model in matrix["results"].get(tid, {}))
    if covered >= len(tasks):
        return None
    row = next((r for r in results if r["strategy"] == "Always-Frontier"), None)
    scored = int(row["n_tasks"]) if row else covered
    return (
        f"Coverage caveat — Always-Frontier ({frontier_model}) ran on "
        f"{covered}/{len(tasks)} tasks; its pass rate is scored on the "
        f"{scored} measured tasks only (unmeasured cells excluded, not auto-failed)"
    )


def _declutter(fig, anns: list, step: float = 3.0, passes: int = 40) -> None:
    """Nudge overlapping point labels apart along y (in offset-point space)."""
    if len(anns) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(passes):
        boxes = [a.get_window_extent(renderer) for a in anns]
        pairs = [
            (i, j)
            for i in range(len(anns))
            for j in range(i + 1, len(anns))
            if boxes[i].overlaps(boxes[j])
        ]
        if not pairs:
            return
        for i, j in pairs:
            lo, hi = (i, j) if boxes[i].y0 <= boxes[j].y0 else (j, i)
            anns[lo].xyann = (anns[lo].xyann[0], anns[lo].xyann[1] - step)
            anns[hi].xyann = (anns[hi].xyann[0], anns[hi].xyann[1] + step)
        fig.canvas.draw()


def _label_offset(cost: float, perf: float, cost_mid: float, perf_hi: float) -> dict:
    """Point a label away from the crowded top-right and away from the top frame.

    High-pass points get their label BELOW the marker (never above, where it would
    collide with the title); high-cost points get it to the LEFT (into the plot).
    """
    right = cost >= cost_mid
    top = perf >= perf_hi
    dx = -10 if right else 10
    dy = -13 if top else 11
    return {
        "xytext": (dx, dy),
        "ha": "right" if dx < 0 else "left",
        "va": "top" if dy < 0 else "bottom",
    }


def plot_pareto(
    results: list[dict],
    out_path: Path,
    warning: str | None = None,
    frontier_cost: float | None = None,
) -> None:
    names = [r["strategy"] for r in results]
    colors, markers = _style_maps(names)
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)

    # Zero-evidence rows are excluded from the frontier: ($0, 0%) is un-dominated
    # by construction, so including them would draw "measured nothing" as optimal.
    pareto_map = compute_pareto(
        {r["strategy"]: r for r in results if int(float(r.get("n_tasks", 1) or 0)) > 0}
    )

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    point_labels: list = []

    # Geometric-mid cost (log axis) and a high-pass threshold steer label placement.
    cost_mid = float(np.sqrt(costs.min() * costs.max()))
    perf_hi = float(perfs.max()) - 12.0

    for i, name in enumerate(names):
        color = colors[name]
        marker = markers[name]
        is_pareto = pareto_map.get(name, False)

        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=170 if is_pareto else 95,
            marker=marker,
            zorder=6,
            edgecolors=_SURFACE,
            linewidth=1.2,
            label=name,
        )

        off = _label_offset(costs[i], perfs[i], cost_mid, perf_hi)
        label = f"{name}\n{perfs[i]:.0f}%, ${costs[i]:.2f}"
        point_labels.append(
            ax.annotate(
                label,
                (costs[i], perfs[i]),
                fontsize=8,
                color=_INK,
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": _MUTED, "lw": 0.6},
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": _SURFACE,
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=7,
                **off,
            )
        )

    # Pareto frontier — a constructed reference (recessive), not a measured series.
    pareto_idx = [i for i, name in enumerate(names) if pareto_map.get(name, False)]
    if pareto_idx:
        pts = sorted([(costs[i], perfs[i]) for i in pareto_idx], key=lambda p: p[0])
        fx: list[float] = [pts[0][0]]
        fy: list[float] = [pts[0][1]]
        for i in range(1, len(pts)):
            if pts[i][1] > fy[-1]:
                fx.append(pts[i][0])
                fy.append(pts[i][1])
        if fx[0] > 0:
            fx = [costs.min() * 0.6] + fx
            fy = [fy[0]] + fy
        ax.step(
            fx,
            fy,
            where="post",
            color=_INK2,
            linewidth=1.5,
            zorder=3,
            label="Pareto frontier (best pass rate per cost)",
        )

    # "What good looks like" — the top-left corner: oracle-level pass rate (top) at a
    # cost well below the fixed-frontier baseline (left of the vertical guide).
    best_perf = float(perfs.max())
    ax.axhline(
        y=best_perf,
        color=_MUTED,
        linestyle=(0, (1, 2)),
        linewidth=1.0,
        zorder=2,
        label=f"Best measured pass rate ({best_perf:.0f}%)",
    )
    if frontier_cost is not None and frontier_cost > 0:
        ax.axvline(
            x=frontier_cost,
            color=_MUTED,
            linestyle=(0, (1, 2)),
            linewidth=1.0,
            zorder=2,
            label=f"Frontier baseline cost (${frontier_cost:.2f})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total cost over the task suite  ($, log scale)", fontsize=11, color=_INK)
    ax.set_ylabel("Pass rate — % of tasks passing tests/typecheck", fontsize=11, color=_INK)
    ax.set_title(
        "Routing strategies — pass rate vs cost (cheaper is left, better is up)",
        fontsize=13,
        fontweight="bold",
        color=_INK,
        pad=12,
    )
    ax.set_xlim(left=costs.min() * 0.55, right=costs.max() * 1.7)
    ax.set_ylim(top=min(103.0, best_perf + 7.0))
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=1)
    leg.get_frame().set_edgecolor(_GRID)
    ax.grid(True, which="major", color=_GRID, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.tick_params(colors=_INK2)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_axisbelow(True)

    if warning:
        # Sparse-frontier honesty as a recessive footnote (not a loud banner): the
        # fixed-frontier strategy is scored on the subset its model actually ran on.
        fig.text(0.5, -0.02, warning, ha="center", va="top", fontsize=8.5, color=_INK2, wrap=True)

    _declutter(fig, point_labels)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)

    import argparse

    ap = argparse.ArgumentParser(description="Strategy comparison Pareto plot")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Path to matrix JSON (default: challenges.json)")
    ap.add_argument(
        "--output",
        default="benchmark/routing/reports/strategy_comparison.png",
        help="Output plot path",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_path = Path(args.output)

    matrix = load_matrix(matrix_path)

    if not matrix.get("results"):
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    # Same task set AND same metric derivation as report.py: one denominator for
    # both producers (report.derive_tasks/derive_rows go through summary.py, which
    # drops coverage-gap cells instead of scoring them as fail@$0).
    tasks = report.derive_tasks(matrix, config.benchmark_params().get("seed", 42))
    if not tasks:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    results = report.derive_rows(matrix, tasks)
    if not results:
        print("No strategies enabled — nothing to plot.")
        return

    print(f"Evaluating {len(results)} strategies on {len(tasks)} tasks")
    for row in results:
        print(
            f"  {row['strategy']:25}  pass={row['AvgPerf%']:>5.2f}%  "
            f"cost=${row['TotalCost']:<8.4f}  (n={row['n_tasks']}, "
            f"{row['n_unscorable']} unscorable excluded)"
        )

    # Fixed-frontier baseline cost anchors the "well left of frontier cost" guide
    # (data-driven: read from the Always-Frontier row if that baseline is present).
    frontier_row = next((r for r in results if r["strategy"] == "Always-Frontier"), None)
    frontier_cost = float(frontier_row["TotalCost"]) if frontier_row else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pareto(
        results,
        out_path,
        warning=_coverage_note(matrix, tasks, results),
        frontier_cost=frontier_cost,
    )
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
