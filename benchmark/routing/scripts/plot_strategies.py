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


def plot_pareto(
    results: list[dict],
    out_path: Path,
    display_name_key: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    warning: str | None = None,
) -> None:
    if display_name_key is None:
        display_name_key = {
            "Oracle": "oracle",
            "Oracle-reward": "oracle_reward",
            "Always-Cheap": "always_cheap",
            "Always-Frontier": "always_frontier",
            "Random": "random",
            "kNN": "knn",
            "kNN-cascade": "knn_cascade",
            "External-Prior": "external_prior",
            "kNN-blended": "knn_blended",
        }
    if colors is None:
        colors = {
            "oracle": "#0072B2",
            "oracle_reward": "#56B4E9",
            "always_cheap": "#009E73",
            "always_frontier": "#D55E00",
            "random": "#CC79A7",
            "knn": "#E69F00",
            "knn_cascade": "#F0E442",
            "external_prior": "#882255",
            "knn_blended": "#44AA99",
        }
    if markers is None:
        markers = {
            "oracle": "D",
            "oracle_reward": "s",
            "always_cheap": "^",
            "always_frontier": "v",
            "random": "o",
            "knn": "P",
            "knn_cascade": "X",
            "external_prior": "*",
            "knn_blended": "h",
        }
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)

    # Zero-evidence rows are excluded from the frontier: ($0, 0%) is un-dominated
    # by construction, so including them would draw "measured nothing" as optimal.
    pareto_map = compute_pareto(
        {r["strategy"]: r for r in results if int(float(r.get("n_tasks", 1) or 0)) > 0}
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    point_labels: list = []

    for i, name in enumerate(names):
        key = display_name_key.get(name, "random")
        color = colors.get(key, "#999999")
        marker = markers.get(key, "o")
        is_pareto = pareto_map.get(name, False)

        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=180 if is_pareto else 100,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.8,
            label=name,
        )

        label = f"{name}\n({perfs[i]:.0f}%, ${costs[i]:.2f})"
        point_labels.append(
            ax.annotate(
                label,
                (costs[i], perfs[i]),
                fontsize=8,
                xytext=(8, 8),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )
        )

    # Pareto frontier
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
            fx = [0.0] + fx
            fy = [fy[0]] + fy
        ax.step(
            fx,
            fy,
            where="post",
            color="#333333",
            linewidth=2,
            linestyle="--",
            label="Pareto frontier",
        )
        ax.fill_between(fx, 0, fy, step="post", alpha=0.05, color="#333333")

    # Reference line at frontier pass rate (highest pass rate among all)
    best_perf = max(perfs)
    ax.axhline(
        y=best_perf,
        color="#666666",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label=f"Best pass rate ({best_perf:.0f}%)",
    )

    # No "SWEET SPOT" annotation: it floated over an empty region of the axes,
    # implying a data point that does not exist. The Pareto step line already
    # shows where the cheap/high-pass corner is.

    ax.set_xscale("log")
    ax.set_xlabel("Total Cost ($, log scale)", fontsize=11)
    ax.set_ylabel("Average Performance (%)", fontsize=11)
    ax.set_title("Strategy Comparison — Pass Rate vs Cost", fontsize=13, fontweight="bold")
    ax.set_xlim(left=costs.min() * 0.7)
    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        ncol=1 if len(names) <= 4 else 2,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)

    if warning:
        # Sparse-frontier honesty: the fixed-frontier strategy is scored on the
        # subset its model actually ran on — a smaller denominator than its peers.
        ax.annotate(
            warning,
            xy=(0.5, 1.09),  # clear of the bold title (~1.06) and the lower-right legend
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            color="#B00020",
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#B00020", alpha=0.9
            ),
        )

    _declutter(fig, point_labels)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pareto(results, out_path, warning=_coverage_note(matrix, tasks, results))
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
