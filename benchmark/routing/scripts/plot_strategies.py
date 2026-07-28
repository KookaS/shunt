#!/usr/bin/env python3
"""Strategy comparison Pareto scatter — plotted straight from strategy_summary.csv."""

# This figure USED to re-derive its rows through report.derive_tasks/derive_rows. That was
# the single source of two defects:
#   1. STALENESS. The derivation is a second producer of the same numbers, so the PNG and
#      strategy_summary.csv could (and did) disagree on all 7 points — one by 174% on cost —
#      whenever the figure was skipped in a refresh sweep.
#   2. MEMORY. derive_rows instantiates the kNN, kNN-cascade and Tier-Classifier strategies,
#      each of which loads its own ONNX sentence-embedder and never releases it: >4 GB RSS,
#      OOM-killed (exit 137) on a 16 GB host.
# Reading the CSV that summary.py already wrote fixes both at once: one producer, so the
# figure cannot disagree with the summary, and no embedder is loaded at all (~80 MB, seconds).
# A freshness gate refuses to draw a figure older than the results it claims to describe.

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import plot_style, summary  # noqa: E402
from benchmark.routing.metrics import compute_pareto  # noqa: E402
from benchmark.routing.strategies.fixed import AlwaysFrontier  # noqa: E402

# Written by summary.write_summary_csv (report.py / run_matrix.py). Derived + gitignored:
# regenerate with `python -m benchmark.routing.report` or a run_matrix pass.
SUMMARY_CSV = Path("benchmark/routing/reports/strategy_summary.csv")
_NUMERIC_FIELDS = (
    "n_tasks",
    "n_unscorable",
    "n_pass",
    "AvgPerf%",
    "AvgPerf_ci_lower",
    "AvgPerf_ci_upper",
    "TotalCost",
    "AvgCost",
)

SPEC = FigureSpec(
    reading=(
        "x is what the whole task suite costs to run under that routing strategy, in USD on "
        "a log axis (one gridline = 10x). y is the share of those tasks whose patch passed "
        "the repo's own tests/typecheck. One mark is one strategy — colour and marker both "
        "identify it, and each point carries its own pass rate and cost. The dark step line "
        "is the Pareto frontier; the dotted guides mark the best pass rate any strategy "
        "reached (horizontal) and what always calling the frontier model costs (vertical)."
    ),
    goal=(
        "Aim top-left: a strategy at or above the pass-rate guide and well left of the "
        "frontier-cost line. That corner is the cost-at-equal-quality claim."
    ),
    definitions=(
        ("pass rate", "share of tasks whose patch passed the repo's tests/typecheck"),
        ("Pareto frontier", "strategies nothing else PLOTTED HERE beats on cost and pass rate"),
        ("error bar", "95% CI on the pass rate (cost has none)"),
        ("Always-Frontier", "baseline sending every task to the most expensive enabled model"),
    ),
    notes=(
        "Points are read directly from strategy_summary.csv, the file summary.py writes — "
        "one producer, so this figure cannot disagree with the summary table.",
        "This frontier is a best-so-far staircase over measured strategies; pareto_scatter.png "
        "draws a convex hull instead, which rides above this line wherever blending two "
        "strategies beats both.",
    ),
    limitations=(
        "A backtest over the recorded outcome matrix, not live runs; cost is the recorded "
        "per-cell cost, so cache effects are approximated rather than replayed.",
        "Cost carries no interval, and the vertical error bars are pass-rate CIs only — two "
        "strategies whose bars overlap are not distinguishable by this figure.",
        "PARETO MEMBERSHIP IS RELATIVE TO THE PLOTTED SET, and the set has changed: a "
        "strategy flagged Pareto-optimal here is only un-dominated by the strategies drawn "
        "beside it. External-Prior, which dominated kNN, was deleted from the repo — so kNN "
        "now reads as Pareto-optimal because its dominator was REMOVED, not because the "
        "router improved. Read the flag as 'nothing plotted beats it', never as progress.",
    ),
)

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
    """Footer limitation: how much of the Always-Frontier row is measured vs imputed."""
    # The row is scored on the COMPLETED matrix summary.py builds, so "measured" and
    # "scored" are different denominators: raw cells the frontier model really ran,
    # plus monotone-imputed cells, minus whatever stayed uncovered. Naming only the
    # raw count while quoting the completed one denies the imputation that produced it.
    row = next((r for r in results if r["strategy"] == "Always-Frontier"), None)
    if row is None:
        return None
    completed, _imputed = summary.complete_scored_matrix(matrix)
    # Derive the model the SAME way the strategy does (from the evaluated models on the
    # matrix it is scored against), not from the configured price list — the two
    # disagree whenever the priciest enabled model was never evaluated.
    strategy = AlwaysFrontier()
    picks = {tid: strategy.select(tid, completed["tasks"].get(tid, {}), completed) for tid in tasks}
    names = sorted(set(picks.values()))
    label = names[0] if len(names) == 1 else ", ".join(names)

    n = len(tasks)
    measured = sum(1 for tid in tasks if picks[tid] in matrix["results"].get(tid, {}))
    cells = [completed["results"].get(tid, {}).get(picks[tid]) for tid in tasks]
    imputed_cells = [cell for cell in cells if cell and cell.get("imputed")]
    imputed = len(imputed_cells)
    imputed_pass = sum(1 for cell in imputed_cells if cell.get("pass"))
    imputed_cost = sum(plot_style.row_real_cost(cell) for cell in imputed_cells)
    covered = sum(1 for cell in cells if cell)
    if measured >= n:
        return None
    scored = int(row["n_tasks"])
    head = (
        f"Coverage caveat — Always-Frontier routes to {label} (derived from the evaluated "
        f"models, as the strategy does), really measured on {measured}/{n} tasks"
    )
    if imputed:
        # The count alone is not the disclosure that matters: monotone imputation is
        # PASS-ONLY by construction, so it can never contribute a failure. Quoting
        # "78 cells completed" without that fact leaves the equal-quality claim circular.
        total_cost = float(row.get("TotalCost", 0.0) or 0.0)
        cost_share = (
            f", i.e. {imputed_cost / total_cost:.0%} of its {plot_style.usd(total_cost)}"
            if (total_cost > 0)
            else ""
        )
        return (
            f"{head}; monotone imputation completes {imputed} more ({imputed / n:.0%} of the "
            f"row), for {covered}/{n} completed cells. THE IMPUTATION IS PASS-ONLY: "
            f"{imputed_pass}/{imputed} imputed cells are filled pass=True and none can be a "
            f"failure, so {imputed_pass} of this row's {int(row.get('n_pass', 0) or 0)} passes "
            f"and {plot_style.usd(imputed_cost)} of its cost{cost_share} are synthetic. "
            f"The kill-gate "
            f"baseline this figure is measured against is therefore NOT a measured baseline; "
            f"its plotted pass rate rests on the {scored} task(s) that row scored — completed "
            f"cells, measured OR imputed. Cells left uncovered are excluded, not auto-failed"
        )
    return (
        f"{head}; its plotted pass rate rests on the {scored} measured task(s) only "
        f"(uncovered cells excluded, not auto-failed)"
    )


def _annotations(
    results: list[dict], coverage_note: str | None, stale_note: str | None = None
) -> Annotations:
    """Footer content that depends on the data: denominators and zero-evidence rows."""
    counts = [int(float(r.get("n_tasks", 0) or 0)) for r in results]
    scored = [c for c in counts if c > 0]
    notes: list[str] = []
    limits: list[str] = []
    if stale_note:
        limits.append(stale_note)
    if coverage_note:
        limits.append(coverage_note)
    if scored and min(scored) != max(scored):
        limits.append(
            f"Strategies are scored on different denominators ({min(scored)}-{max(scored)} "
            "tasks): a strategy whose chosen model was never measured on a task has that "
            "task excluded, not auto-failed"
        )
    n_zero = len(counts) - len(scored)
    if n_zero:
        notes.append(
            f"{n_zero} strategy row(s) measured 0 tasks and are excluded from the Pareto "
            "frontier — a $0 / 0% point is un-dominated by construction"
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _declutter(fig, ax, anns: list, step: float = 3.0, passes: int = 40) -> None:  # noqa: ANN001 (matplotlib Figure/Axes)
    """Nudge overlapping point labels apart along y (in offset-point space)."""
    if len(anns) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A label _label_offset placed BELOW its marker is there to stay clear of the title, so
    # it may never rise above its starting offset; the lower label of the pair then takes
    # the whole separation. A label placed ABOVE its marker has no such ceiling — giving it
    # one left `room` at 0 on the first pass for EVERY label, so the nudge was one-directional
    # and the whole cluster could only ever walk downwards.
    ceiling = [a.xyann[1] if a.xyann[1] < 0 else float("inf") for a in anns]
    for _ in range(passes):
        boxes = [_label_box(a, renderer) for a in anns]
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
            room = min(step, max(0.0, ceiling[hi] - anns[hi].xyann[1]))
            anns[hi].xyann = (anns[hi].xyann[0], anns[hi].xyann[1] + room)
            anns[lo].xyann = (anns[lo].xyann[0], anns[lo].xyann[1] - (2 * step - room))
        fig.canvas.draw()
        # The separation itself is unbounded — an unresolvable cluster keeps pushing every
        # pass — so the axes box, not the pass count, is what keeps a label on the figure.
        if _clamp_to_axes(ax, anns, renderer, ceiling):
            fig.canvas.draw()


def _clamp_to_axes(ax, anns: list, renderer, ceiling: list[float]) -> bool:  # noqa: ANN001 (matplotlib Axes/Renderer)
    """Pull any label that left the axes back inside; returns whether anything moved."""
    bounds = ax.get_window_extent(renderer)
    to_points = 72.0 / ax.figure.dpi
    moved = False
    for ann, top in zip(anns, ceiling, strict=True):
        box = _label_box(ann, renderer)
        if box.y0 < bounds.y0:
            shift = (bounds.y0 - box.y0) * to_points
        elif box.y1 > bounds.y1:
            shift = (bounds.y1 - box.y1) * to_points
        else:
            continue
        target = min(ann.xyann[1] + shift, top)
        if target != ann.xyann[1]:
            ann.xyann = (ann.xyann[0], target)
            moved = True
    return moved


def _label_box(ann, renderer):  # noqa: ANN001, ANN202 (matplotlib Annotation/Renderer)
    """The label's own drawn box — Annotation.get_window_extent spans the leader line too.

    Including the leader made a box grow as its label moved away, so every pair stayed
    'overlapping' however far apart the text got and the declutter never converged.
    """
    patch = ann.get_bbox_patch()
    return (patch or ann).get_window_extent(renderer)


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
    coverage_note: str | None = None,
    frontier_cost: float | None = None,
    stale_note: str | None = None,
) -> None:
    names = [r["strategy"] for r in results]
    colors, markers = _style_maps(names)
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    # The CIs were already in strategy_summary.csv and were simply not drawn. Without them
    # the top-left cluster reads as a ranking when the intervals overlap almost completely.
    ci_lo = np.array([float(r.get("AvgPerf_ci_lower", r["AvgPerf%"])) for r in results])
    ci_hi = np.array([float(r.get("AvgPerf_ci_upper", r["AvgPerf%"])) for r in results])

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

        ax.errorbar(
            costs[i],
            perfs[i],
            yerr=[[max(0.0, perfs[i] - ci_lo[i])], [max(0.0, ci_hi[i] - perfs[i])]],
            fmt="none",
            ecolor=color,
            elinewidth=1.3,
            capsize=4,
            alpha=0.85,
            zorder=5,
        )
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
    ax.set_ylim(bottom=max(0.0, float(ci_lo.min()) - 6.0), top=min(103.0, best_perf + 7.0))
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=1)
    leg.get_frame().set_edgecolor(_GRID)
    ax.grid(True, which="major", color=_GRID, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.tick_params(colors=_INK2)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_axisbelow(True)

    _declutter(fig, ax, point_labels)
    # The surface colour rides on the figure itself, so the frame's savefig picks it
    # up (savefig.facecolor defaults to the figure's own).
    plot_frame.save(fig, out_path, SPEC, extra=_annotations(results, coverage_note, stale_note))


def load_summary_rows(path: Path) -> list[dict]:
    """Read strategy_summary.csv, coercing the numeric columns the figure indexes."""
    rows: list[dict] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = dict(raw)
            for field in _NUMERIC_FIELDS:
                value = raw.get(field, "")
                row[field] = float(value) if value not in (None, "") else 0.0
            row["Pareto"] = str(raw.get("Pareto", "")).strip().lower() == "true"
            rows.append(row)
    return rows


def _staleness_limit(summary_csv: Path, results_csv: Path) -> str | None:
    """LIMITS line when the summary predates the results it claims to summarise."""
    # The defect this closes: the committed PNG once disagreed with strategy_summary.csv on
    # all seven points because it was skipped in a refresh sweep. Reading one file removes
    # the disagreement; saying so when the file itself is behind removes the silent version.
    if not results_csv.exists() or not summary_csv.exists():
        return None
    if summary_csv.stat().st_mtime >= results_csv.stat().st_mtime:
        return None
    return (
        f"STALE INPUT: {summary_csv.name} is older than {results_csv.name}, so these points "
        f"describe an earlier results set. Regenerate the summary "
        f"(python -m benchmark.routing.report) before trusting any number here"
    )


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Draw the strategy Pareto scatter from the summary CSV summary.py writes."""
    config.load(config_path)

    import argparse

    ap = argparse.ArgumentParser(description="Strategy comparison Pareto plot")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Path to matrix JSON (default: challenges.json)")
    ap.add_argument("--summary", default=None, help=f"Path to summary CSV (default: {SUMMARY_CSV})")
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
    summary_csv = Path(args.summary) if args.summary else SUMMARY_CSV

    matrix = load_matrix(matrix_path)

    if not matrix.get("results"):
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    if not summary_csv.exists():
        # Deliberately NOT falling back to an in-process re-derivation: a second producer of
        # these numbers is exactly what let the figure drift from the table it must match.
        print(
            f"No results yet — {summary_csv} has not been written. "
            "Generate it first: python -m benchmark.routing.report"
        )
        return

    results = load_summary_rows(summary_csv)
    if not results:
        print("No strategies enabled — nothing to plot.")
        return

    tasks = sorted(matrix["results"].keys())
    print(f"Plotting {len(results)} strategies from {summary_csv}")
    for row in results:
        print(
            f"  {row['strategy']:25}  pass={row['AvgPerf%']:>5.2f}%  "
            f"[{row['AvgPerf_ci_lower']:.2f}, {row['AvgPerf_ci_upper']:.2f}]  "
            f"cost=${row['TotalCost']:<8.4f}  (n={int(row['n_tasks'])}, "
            f"{int(row['n_unscorable'])} unscorable excluded)"
        )

    # Fixed-frontier baseline cost anchors the "well left of frontier cost" guide
    # (data-driven: read from the Always-Frontier row if that baseline is present).
    frontier_row = next((r for r in results if r["strategy"] == "Always-Frontier"), None)
    frontier_cost = float(frontier_row["TotalCost"]) if frontier_row else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pareto(
        results,
        out_path,
        coverage_note=_coverage_note(matrix, tasks, results),
        frontier_cost=frontier_cost,
        stale_note=_staleness_limit(summary_csv, config.results_csv_path()),
    )
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
