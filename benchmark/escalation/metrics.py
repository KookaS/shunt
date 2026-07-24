"""Detector metrics: prefix labelling + AUPRC-vs-prevalence + full suite per grid cell."""

# The primary metric is AUPRC against the prevalence baseline — ROC hides prevalence on an
# imbalanced problem, so AUROC is reported AUXILIARY only. The confusion / precision / recall /
# F1 / FPR / kappa suite is reused from the calibration labeler (no cost model needed); the
# per-cell CellReport + best-config selection turn the sweep into an actual hyperparameter choice.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.calibration.labeler_metrics import ConfusionMatrix, LabelerMetrics, compute_metrics

if TYPE_CHECKING:
    from benchmark.escalation.schema import Trajectory

# The operating threshold for turning the cumulative-detection score into a hard flag. The score
# is already binary 0/1 (cumulative detection), so 0.5 splits flagged (1.0) from not (0.0).
DETECTION_THRESHOLD = 0.5


def label_prefixes(traj: Trajectory, horizon: int) -> list[bool]:
    """PrefixGuard p_t = 1[y=0 AND t >= T-H]: a prefix is positive iff the trajectory failed
    terminally and the step is within `horizon` decisions of the end. Recomputed from the
    terminal label + step index, so authenticity can cross-check it.
    """
    n = len(traj.steps)
    failed = not traj.header.terminal_resolved
    return [failed and (t >= n - horizon) for t in range(n)]


def prevalence(labels: list[bool]) -> float:
    """The positive rate — the no-skill AUPRC baseline drawn on every PR plot."""
    return sum(labels) / len(labels) if labels else 0.0


def auprc(scores: list[float], labels: list[bool]) -> float:
    """Area under the precision-recall curve (average precision), matching sklearn's
    step-function definition: sum of (R_n - R_{n-1}) * P_n over decreasing score thresholds.
    """
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    ap = 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    index = 0
    total = len(ranked)
    while index < total:
        threshold = ranked[index][0]
        while index < total and ranked[index][0] == threshold:
            if ranked[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic — AUXILIARY only.

    ROC hides prevalence on an imbalanced problem (AUPRC is primary); reported for completeness.
    Returns 0.5 (chance) when one class is absent, matching the metric's no-information point.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    # Rank-sum (average ranks for ties) → AUC = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        run = index
        while run < len(order) and scores[order[run]] == scores[order[index]]:
            run += 1
        avg_rank = (index + run + 1) / 2.0  # 1-based average rank across the tie block
        for k in range(index, run):
            ranks[order[k]] = avg_rank
        index = run
    rank_sum_pos = sum(ranks[i] for i, lab in enumerate(labels) if lab)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def detection_metrics(
    scores: list[float], labels: list[bool], *, threshold: float = DETECTION_THRESHOLD
) -> LabelerMetrics:
    """Confusion + precision/recall/F1/FPR/Cohen-kappa at the operating threshold.

    Reuses the calibration labeler's `compute_metrics`; positive class = the risky/flagged prefix.
    """
    owner = {str(i): ("good" if lab else "bad") for i, lab in enumerate(labels)}
    auto = {str(i): ("good" if s >= threshold else "bad") for i, s in enumerate(scores)}
    return compute_metrics(owner, auto)


@dataclass(frozen=True)
class CellReport:
    """The full metric suite for one hyperparameter cell of the sweep."""

    escalate_after_n: int
    stale_window: int
    ladder: str
    confusion: ConfusionMatrix
    precision: float
    recall: float
    f1: float
    fpr: float
    cohen_kappa: float
    auprc: float
    auroc: float
    prevalence: float
    n_escalated: int
    mean_steps_to_detection: float | None

    def to_dict(self) -> dict[str, object]:
        cm = self.confusion
        return {
            "escalate_after_n": self.escalate_after_n,
            "stale_window": self.stale_window,
            "ladder": self.ladder,
            "confusion": {"tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn},
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "auprc": round(self.auprc, 4),
            "auroc": round(self.auroc, 4),
            "prevalence": round(self.prevalence, 4),
            "n_escalated": self.n_escalated,
            "mean_steps_to_detection": (
                None
                if self.mean_steps_to_detection is None
                else round(self.mean_steps_to_detection, 3)
            ),
        }


def cell_report(
    *,
    escalate_after_n: int,
    stale_window: int,
    ladder: str,
    scores: list[float],
    labels: list[bool],
    n_escalated: int,
    mean_steps_to_detection: float | None,
) -> CellReport:
    """Compute one cell's full metric suite from its cumulative-detection scores + prefix labels."""
    m = detection_metrics(scores, labels)
    return CellReport(
        escalate_after_n=escalate_after_n,
        stale_window=stale_window,
        ladder=ladder,
        confusion=m.confusion,
        precision=m.precision,
        recall=m.recall,
        f1=m.f1,
        fpr=m.fpr,
        cohen_kappa=m.cohen_kappa,
        auprc=auprc(scores, labels),
        auroc=auroc(scores, labels),
        prevalence=prevalence(labels),
        n_escalated=n_escalated,
        mean_steps_to_detection=mean_steps_to_detection,
    )


# An objective maps a cell to a sort key (higher is better) — a named, swappable function.
Objective = Callable[[CellReport], tuple[float, ...]]


def objective_max_f1(cell: CellReport) -> tuple[float, ...]:
    """Default objective: maximise F1, tie-broken by EARLIER detection (lower steps-to-detection).

    A None steps-to-detection (never detected) sorts last on the tie-break (treated as +inf).
    """
    steps = float("inf") if cell.mean_steps_to_detection is None else cell.mean_steps_to_detection
    return (cell.f1, -steps)


def select_best_config(
    cells: list[CellReport], *, objective: Objective = objective_max_f1
) -> CellReport | None:
    """The argmax cell under `objective`, or None when no cell discriminates (insufficient data).

    Returns None when the cells are empty, all degenerate (zero escalations everywhere), or all
    share one objective value — an arbitrary pick would misrepresent a non-result as a choice.
    """
    if not cells or all(c.n_escalated == 0 for c in cells):
        return None
    if len({objective(c) for c in cells}) <= 1:
        return None
    return max(cells, key=objective)


__all__ = [
    "CellReport",
    "DETECTION_THRESHOLD",
    "Objective",
    "auprc",
    "auroc",
    "cell_report",
    "detection_metrics",
    "label_prefixes",
    "objective_max_f1",
    "prevalence",
    "select_best_config",
]
