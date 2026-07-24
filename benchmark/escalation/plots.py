"""Detector figures on the routing visual system. Each function takes (data, Axes) and draws"""

# into the Axes — no file I/O here. Real PNGs arrive with collected data; this round the plot
# CODE is validated structurally on fixtures.
#
# The prevalence baseline is drawn on every precision-recall figure — ROC hides prevalence on an
# imbalanced problem, so an honest PR plot always shows the no-skill line.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmark.escalation import metrics
from benchmark.routing.plot_style import ci_footer, upper_hull

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from benchmark.calibration.labeler_metrics import ConfusionMatrix
    from benchmark.escalation.metrics import CellReport
    from benchmark.escalation.replay import SweepResult


def annotate_insufficient(ax: Axes, note: str) -> None:
    """Stamp a visible red 'insufficient data' box on a plot so a null result is never mistaken."""
    ax.text(
        0.5,
        0.5,
        f"insufficient data:\n{note}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="firebrick",
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "mistyrose", "edgecolor": "firebrick"},
    )


def annotate_no_skill(ax: Axes, note: str) -> None:
    """Orange box for a legible detector still at/below no-skill — never read as 'works'."""
    ax.text(
        0.5,
        0.32,
        f"no usable signal:\n{note}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="darkorange",
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "papayawhip", "edgecolor": "darkorange"},
    )


def pr_curve(
    scores: list[float],
    labels: list[bool],
    ax: Axes,
    *,
    note: str | None = None,
    no_skill_note: str | None = None,
) -> Axes:
    """Precision-recall curve with the prevalence (no-skill) baseline drawn as a dashed line."""
    points = _pr_points(scores, labels)
    recalls = [r for r, _ in points]
    precisions = [p for _, p in points]
    ax.plot(recalls, precisions, marker="o", label="detector")
    baseline = metrics.prevalence(labels)
    ax.axhline(baseline, linestyle="--", color="grey", label=f"prevalence={baseline:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AUPRC={metrics.auprc(scores, labels):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.text(0.0, -0.18, ci_footer(), transform=ax.transAxes, fontsize=7)
    if note:
        annotate_insufficient(ax, note)
    if no_skill_note:
        annotate_no_skill(ax, no_skill_note)
    return ax


def roc_curve(
    scores: list[float],
    labels: list[bool],
    ax: Axes,
    *,
    note: str | None = None,
    no_skill_note: str | None = None,
) -> Axes:
    """ROC curve (AUXILIARY — AUPRC is primary on this imbalanced problem) with the chance line."""
    points = _roc_points(scores, labels)
    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label="detector")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(f"ROC (AUROC={metrics.auroc(scores, labels):.3f}, auxiliary)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    if note:
        annotate_insufficient(ax, note)
    if no_skill_note:
        annotate_no_skill(ax, no_skill_note)
    return ax


def steps_to_detection_hist(sweep: SweepResult, ax: Axes, *, note: str | None = None) -> Axes:
    """Histogram of the first-escalation step index across the escalated trajectories."""
    detections = [
        d.first_escalation_index
        for point in sweep.points
        for d in point.decisions
        if d.first_escalation_index is not None
    ]
    if detections:
        # Integer-centered bins: one bar per distinct step value, so a narrow 1–2 range shows
        # as separate bars instead of matplotlib's default float binning collapsing to one block.
        lo, hi = min(detections), max(detections)
        # rwidth<1 + white edges: adjacent equal-height bars stay visually distinct instead of
        # fusing into one solid block (which is what a 1–2 step range otherwise looks like).
        bins = [i - 0.5 for i in range(lo, hi + 2)]
        ax.hist(detections, bins=bins, rwidth=0.8, edgecolor="white")
        ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel("first-escalation step")
    ax.set_ylabel("count")
    ax.set_title("steps to detection")
    if not detections:
        annotate_insufficient(ax, note or "0 escalations — trigger not exercised")
    return ax


def sweep_heatmap(
    cells: list[CellReport], ax: Axes, *, ladder: str, note: str | None = None
) -> Axes:
    """Heatmap of the objective metric (F1) over escalate_after_n x stale_window for one ladder."""
    ladder_cells = [c for c in cells if c.ladder == ladder]
    rows = sorted({c.escalate_after_n for c in ladder_cells})
    cols = sorted({c.stale_window for c in ladder_cells})
    grid = np.full((len(rows), len(cols)), np.nan)
    by_key = {(c.escalate_after_n, c.stale_window): c.f1 for c in ladder_cells}
    for r, n in enumerate(rows):
        for col, w in enumerate(cols):
            if (n, w) in by_key:
                grid[r, col] = by_key[(n, w)]
    ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)), [str(w) for w in cols])
    ax.set_yticks(range(len(rows)), [str(n) for n in rows])
    ax.set_xlabel("stale_window")
    ax.set_ylabel("escalate_after_n")
    ax.set_title(f"F1 sweep — ladder={ladder}")
    for r in range(len(rows)):
        for col in range(len(cols)):
            if not np.isnan(grid[r, col]):
                # 3 dp: on a low-F1 sweep, 2 dp collapses genuinely-different cells to one value,
                # making a real argmax look like an arbitrary tie.
                ax.text(col, r, f"{grid[r, col]:.3f}", ha="center", va="center", color="white")
    if note:
        annotate_insufficient(ax, note)
    return ax


def confusion_matrix_plot(cm: ConfusionMatrix, ax: Axes, *, note: str | None = None) -> Axes:
    """2x2 confusion heatmap (rows = actual risky/not, cols = flagged/not) with cell counts."""
    grid = np.array([[cm.tp, cm.fn], [cm.fp, cm.tn]], dtype=float)
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1], ["flagged", "not flagged"])
    ax.set_yticks([0, 1], ["risky", "not risky"])
    ax.set_title("confusion @ operating threshold")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(int(grid[r, c])), ha="center", va="center", color="black")
    if note:
        annotate_insufficient(ax, note)
    return ax


def cost_quality_frontier(perf: dict[str, float], cost: dict[str, float], ax: Axes) -> Axes:
    """Scatter of sweep configs in the (cost, quality) plane with the Pareto upper hull drawn."""
    names = list(perf)
    ax.scatter([cost[n] for n in names], [perf[n] for n in names])
    if len({cost[n] for n in names}) <= 1:
        # All points stack on one x — a vertical line, not a frontier. Label it honestly.
        annotate_insufficient(ax, "single cost point — no frontier to trace")
    else:
        hull = upper_hull([(cost[n], perf[n]) for n in names])
        if hull:
            ax.plot([x for x, _ in hull], [y for _, y in hull], linestyle="-", color="C1")
    ax.set_xlabel("cost of escalation")
    ax.set_ylabel("quality (AUPRC%)")
    ax.set_title("cost-quality frontier")
    return ax


def _pr_points(scores: list[float], labels: list[bool]) -> list[tuple[float, float]]:
    """(recall, precision) at each descending score threshold; (0,1) seed for a clean curve."""
    positives = sum(labels)
    if positives == 0:
        return [(0.0, 1.0)]
    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    out: list[tuple[float, float]] = [(0.0, 1.0)]
    tp = 0
    for seen, (_, is_pos) in enumerate(ranked, start=1):
        if is_pos:
            tp += 1
        out.append((tp / positives, tp / seen))
    return out


def _roc_points(scores: list[float], labels: list[bool]) -> list[tuple[float, float]]:
    """(fpr, tpr) at each descending score threshold; (0,0) seed for a clean ROC curve."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    out: list[tuple[float, float]] = [(0.0, 0.0)]
    tp = 0
    fp = 0
    for _, is_pos in ranked:
        if is_pos:
            tp += 1
        else:
            fp += 1
        out.append((fp / negatives, tp / positives))
    return out


__all__ = [
    "annotate_insufficient",
    "annotate_no_skill",
    "confusion_matrix_plot",
    "cost_quality_frontier",
    "pr_curve",
    "roc_curve",
    "steps_to_detection_hist",
    "sweep_heatmap",
]
