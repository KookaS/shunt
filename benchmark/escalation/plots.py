"""Detector figures on the routing visual system. Each function draws into a caller's Axes"""

# and returns the Annotations its own data earned — no file I/O here. The figure's static
# READ/GOAL/TERMS/NOTE/LIMITS text lives next to each function as a frozen FigureSpec; anything
# that depends on the data (counts, an empty histogram, a flat grid, a missing class) comes back
# through Annotations so it can never go stale as the corpus grows.
#
# The prevalence baseline is drawn on every precision-recall figure — ROC hides prevalence on an
# imbalanced problem, so an honest PR plot always shows the no-skill line.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from benchmark.escalation import metrics
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing.plot_style import upper_hull

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from benchmark.calibration.labeler_metrics import ConfusionMatrix
    from benchmark.escalation.metrics import CellReport
    from benchmark.escalation.replay import SweepResult

PR_CURVE_SPEC = FigureSpec(
    reading=(
        "x is recall and y is precision, both 0-1, over the per-prefix detector scores flattened "
        "across every trajectory. The curve is built by admitting one prefix at a time in "
        "descending score order, so each marker is one more prefix included, NOT one distinct "
        "threshold. The dashed grey line is prevalence, the precision a no-skill detector reaches "
        "by flagging everything. The title carries AUPRC, computed tie-block-wise — deliberately "
        "not the area under this polyline."
    ),
    goal=(
        "Look for the curve sitting well above the dashed prevalence line, and staying high "
        "as recall grows."
    ),
    definitions=(
        ("precision", "share of flagged prefixes that were really risky"),
        ("recall", "share of risky prefixes the detector flagged"),
        ("prevalence", "share of all prefixes that are positive; the no-skill precision"),
        ("AUPRC", "average precision over the ranked prefixes; above prevalence means signal"),
    ),
    notes=(
        "The detector score is binary: 0 until the policy first emits a non-HOLD directive, "
        "1 at and after it.",
    ),
    limitations=(
        "A binary score has only two real operating points, so every intermediate marker is a "
        "tie-ordering artifact and the drawn shape can disagree with the title's AUPRC, which "
        "resolves ties by block.",
        "Severe class imbalance: a handful of false alarms moves precision a long way.",
    ),
)

ROC_CURVE_SPEC = FigureSpec(
    reading=(
        "x is the false-positive rate and y the true-positive rate, both 0-1, over the same "
        "flattened per-prefix scores; each marker admits one more prefix in descending score "
        "order, not one distinct threshold. The dashed diagonal is chance. The title carries "
        "AUROC from the tie-averaged rank statistic, not from this polyline."
    ),
    goal=(
        "Look for the curve bowing above the diagonal toward the top-left; a curve lying on "
        "the diagonal is no signal."
    ),
    definitions=(
        ("true-positive rate", "share of risky prefixes the detector flagged"),
        ("false-positive rate", "share of safe prefixes the detector wrongly flagged"),
        ("AUROC", "tie-averaged rank statistic behind this curve; 0.5 is chance"),
    ),
    notes=(
        "Auxiliary only: AUPRC against prevalence is the primary measure, because ROC hides "
        "class imbalance.",
    ),
    limitations=(
        "ROC reads optimistically when positives are rare.",
        "The binary score makes intermediate markers tie-ordering artifacts, and this polyline "
        "breaks ties differently from the title's average-rank AUROC.",
    ),
)

STEPS_TO_DETECTION_SPEC = FigureSpec(
    reading=(
        "x is the step index at which a configuration first escalated a trajectory, in integer "
        "bins; y counts how many (trajectory, grid cell) pairs first escalated at that step. Bars "
        "exist only where an escalation actually fired."
    ),
    goal=(
        "Read bar position, not height: bars further left mean the router notices a failure "
        "loop earlier."
    ),
    definitions=(("detection", "the first non-HOLD directive, recorded at that step's index"),),
    notes=(
        "The trigger needs the same failing_check_id to recur across at least two steps, so "
        "length-1 trajectories can never produce a bar.",
    ),
    limitations=(
        "Counts are pooled over every grid cell, so one trajectory contributes up to one count "
        "per cell — y is not a number of runs.",
        "Pooling mixes configurations with different escalate_after_n, which structurally shifts "
        "the first-fire step.",
        "Shows when detection happened, never whether it was correct: a fast false alarm looks "
        "identical to a fast true catch.",
    ),
)

CONFUSION_MATRIX_SPEC = FigureSpec(
    reading=(
        "A 2x2 count grid. Rows are the truth (risky / not risky), columns are what the detector "
        "said (flagged / not flagged). Top-left is a correct catch, top-right a miss, bottom-left "
        "a false alarm, bottom-right a correct quiet. Read the printed counts."
    ),
    goal=(
        "Want the two diagonal cells (top-left, bottom-right) large and the two off-diagonal "
        "cells small."
    ),
    definitions=(("flagged", "detector score above the 0.5 operating threshold"),),
    notes=("Counts are per prefix, not per trajectory.",),
    limitations=(
        "The shading is auto-normalised with no colorbar, so shade is not comparable between "
        "runs — read the numbers.",
        "One arbitrary operating point (threshold 0.5, which merely splits the binary score), "
        "not a sweep.",
        "On imbalanced data the not-risky/not-flagged cell dominates and flattens every other "
        "shade.",
    ),
)

SWEEP_HEATMAP_SPEC = FigureSpec(
    reading=(
        "Rows are escalate_after_n, columns are stale_window, and colour is that cell's F1 on a "
        "fixed 0-1 scale; the F1 value is printed in each cell to three decimals. One cell is one "
        "swept configuration."
    ),
    goal=(
        "Find the brightest cell and read its row and column — that pair is the best "
        "configuration on this data."
    ),
    definitions=(
        ("F1", "harmonic mean of precision and recall at the operating threshold"),
        ("escalate_after_n", "same-key verified failures required before the router escalates"),
        ("stale_window", "decisions after which a failure that stops recurring is retired"),
    ),
    notes=(
        "Flat columns along stale_window are expected, not a bug: a success clears the failure "
        "log, so detection discriminates on escalate_after_n.",
    ),
    limitations=(
        "No colorbar: read the printed F1 values, not the shade.",
        "escalate_after_n has only two levels here — this is a small grid, not a landscape.",
    ),
)

COST_QUALITY_SPEC = FigureSpec(
    reading=(
        "x is the number of escalated trajectories a configuration produced, a cost PROXY rather "
        "than dollars; y is that configuration's F1 times 100. One dot is one swept configuration. "
        "When the configurations differ in cost, an orange line traces the upper convex hull "
        "through them; no dominance pruning runs first, so that line can fall as well as rise."
    ),
    goal=(
        "Aim up and to the left — the same quality for fewer escalations. A dot below the line "
        "is beaten by a MIXTURE of two configurations, which is weaker than being beaten "
        "outright by one."
    ),
    definitions=(
        ("F1", "harmonic mean of precision and recall at the operating threshold"),
        ("upper hull", "best quality reachable by mixing two configurations at each cost"),
    ),
    notes=(
        "Every swept cell appears here, both ladders, unlike the sweep heatmap.",
        "The hull is what a mixture of two configurations reaches, not a keep-max staircase.",
    ),
    limitations=(
        "Cost is a count of escalations, not measured spend: two escalations on different models "
        "cost wildly different amounts.",
        "The dots are unlabelled, so a good point cannot be traced back to its configuration from "
        "this figure alone.",
    ),
)


_NO_POSITIVES_PR = (
    "No positive prefixes in this data: the curve collapses to a single meaningless "
    "perfect-precision-at-zero-recall dot."
)
_ONE_CLASS_ROC = (
    "One class is absent, so the reported AUROC is the 0.5 no-information code rather than a "
    "measurement, and the curve degenerates to the bare diagonal."
)


def _scored_note(labels: list[bool]) -> str:
    """How much data is behind a curve — runtime, so it can never go stale as the corpus grows."""
    return f"scored on {len(labels)} prefixes, {sum(labels)} of them positive"


def pr_curve(scores: list[float], labels: list[bool], ax: Axes) -> Annotations:
    """Precision-recall curve with the prevalence (no-skill) baseline drawn as a dashed line."""
    points = _pr_points(scores, labels)
    ax.plot([r for r, _ in points], [p for _, p in points], marker="o", label="detector")
    baseline = metrics.prevalence(labels)
    ax.axhline(baseline, linestyle="--", color="grey", label=f"prevalence={baseline:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AUPRC={metrics.auprc(scores, labels):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    limits = () if any(labels) else (_NO_POSITIVES_PR,)
    return Annotations(notes=(_scored_note(labels),), limitations=limits)


def roc_curve(scores: list[float], labels: list[bool], ax: Axes) -> Annotations:
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
    one_class = sum(labels) in (0, len(labels))
    limits = (_ONE_CLASS_ROC,) if one_class else ()
    return Annotations(notes=(_scored_note(labels),), limitations=limits)


def steps_to_detection_hist(sweep: SweepResult, ax: Axes) -> Annotations:
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
        # One tick per step only while they still fit; past that, thin to ~20 evenly
        # spaced integer ticks — a real corpus reaches step 80+, where a tick per bar
        # overprints into an unreadable smear.
        step = max(1, math.ceil((hi - lo + 1) / 20))
        ax.set_xticks(range(lo, hi + 1, step))
    ax.set_xlabel("first-escalation step")
    ax.set_ylabel("count")
    ax.set_title("steps to detection")
    note = f"{len(detections)} detections pooled over {len(sweep.points)} grid cells"
    empty = "No escalation fired anywhere in the sweep: this histogram is empty, not flat."
    return Annotations(notes=(note,), limitations=() if detections else (empty,))


def sweep_heatmap(cells: list[CellReport], ax: Axes, *, ladder: str) -> Annotations:
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
                # making a real argmax look like an arbitrary tie. Ink flips on the bright end of
                # viridis so the value stays readable at every F1 — a caveat would not fix it.
                ink = "black" if grid[r, col] > 0.55 else "white"
                ax.text(col, r, f"{grid[r, col]:.3f}", ha="center", va="center", color=ink)
    return _heatmap_annotations(grid, shown=len(ladder_cells), total=len(cells), ladder=ladder)


def _heatmap_annotations(grid: np.ndarray, *, shown: int, total: int, ladder: str) -> Annotations:
    """Coverage, flatness and missing-cell caveats read off the drawn grid."""
    notes = (f"showing {shown} of {total} swept cells — only ladder={ladder}",)
    limitations: list[str] = []
    if shown < total:
        limitations.append(
            f"The other {total - shown} swept cells are on a different ladder and are not on "
            "this figure at all."
        )
    missing = int(np.isnan(grid).sum())
    if missing:
        limitations.append(f"{missing} cell(s) of this ladder have no result and are drawn blank.")
    finite = grid[~np.isnan(grid)]
    if finite.size and float(finite.min()) == float(finite.max()):
        limitations.append(
            f"Every cell is {float(finite.min()):.3f}: the grid is flat, there is no argmax, and "
            "any brightest cell a reader picks by eye is noise."
        )
    return Annotations(notes=notes, limitations=tuple(limitations))


def confusion_matrix_plot(cm: ConfusionMatrix, ax: Axes) -> Annotations:
    """2x2 confusion heatmap (rows = actual risky/not, cols = flagged/not) with cell counts."""
    grid = np.array([[cm.tp, cm.fn], [cm.fp, cm.tn]], dtype=float)
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1], ["flagged", "not flagged"])
    ax.set_yticks([0, 1], ["risky", "not risky"])
    ax.set_title("confusion @ operating threshold")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(int(grid[r, c])), ha="center", va="center", color="black")
    total = int(grid.sum())
    note = f"{total} prefixes scored at threshold {metrics.DETECTION_THRESHOLD}"
    return Annotations(notes=(note,))


def cost_quality_frontier(perf: dict[str, float], cost: dict[str, float], ax: Axes) -> Annotations:
    """Scatter of sweep configs in the (cost, quality) plane with the Pareto upper hull drawn."""
    names = list(perf)
    ax.scatter([cost[n] for n in names], [perf[n] for n in names])
    distinct = len({cost[n] for n in names})
    limitations: list[str] = []
    if distinct <= 1:
        # All points stack on one x — a vertical line, not a frontier. Label it honestly.
        limitations.append(
            "Every configuration produced the same escalation count: the dots stack in one "
            "column and there is no frontier to trace."
        )
    else:
        hull = upper_hull([(cost[n], perf[n]) for n in names])
        if hull:
            ax.plot([x for x, _ in hull], [y for _, y in hull], linestyle="-", color="C1")
        if distinct == 2:
            limitations.append(
                "Only two distinct cost values, so the drawn line is the two raw points rather "
                "than a computed hull."
            )
    ax.set_xlabel("cost of escalation")
    ax.set_ylabel("quality (F1 x 100)")
    ax.set_title("cost-quality frontier")
    note = f"{len(names)} configurations at {distinct} distinct escalation counts"
    return Annotations(notes=(note,), limitations=tuple(limitations))


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
    "CONFUSION_MATRIX_SPEC",
    "COST_QUALITY_SPEC",
    "PR_CURVE_SPEC",
    "ROC_CURVE_SPEC",
    "STEPS_TO_DETECTION_SPEC",
    "SWEEP_HEATMAP_SPEC",
    "confusion_matrix_plot",
    "cost_quality_frontier",
    "pr_curve",
    "roc_curve",
    "steps_to_detection_hist",
    "sweep_heatmap",
]
