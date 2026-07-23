"""Validate the automatic outcome labeler against an owner-labeled ground-truth set.

Precision / recall / F1 / false-positive-rate + Cohen's kappa (positive class 'good');
a labeler over the pre-registered FPR bound is FLAGGED, not silently trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Pre-registered ceiling on the labeler's false-positive rate. A false positive is a
# fabricated "verified success" that biases the kill-gate the dangerous way, so it is
# bounded tightly. Changing this is a reviewed pre-registration change, not a tuning knob.
PREREGISTERED_FPR_BOUND = 0.10

_GOOD = "good"
_BAD = "bad"
_VALID_LABELS = frozenset({_GOOD, _BAD})


@dataclass(frozen=True)
class ConfusionMatrix:
    """2x2 counts with positive class = 'good'. n is the number of compared items."""

    tp: int  # owner good, labeler good
    fp: int  # owner bad,  labeler good  <- the dangerous cell
    fn: int  # owner good, labeler bad
    tn: int  # owner bad,  labeler bad

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


@dataclass(frozen=True)
class LabelerMetrics:
    """Auto-labeler quality vs the owner set, plus the pre-registered FPR flag."""

    confusion: ConfusionMatrix
    precision: float
    recall: float
    f1: float
    fpr: float
    cohen_kappa: float
    fpr_bound: float
    flagged: bool
    n_compared: int


def confusion_matrix(owner_labels: dict[str, str], auto_labels: dict[str, str]) -> ConfusionMatrix:
    """Build the 2x2 confusion over sessions present AND validly labeled in BOTH maps."""
    tp = fp = fn = tn = 0
    for session_id, owner in owner_labels.items():
        auto = auto_labels.get(session_id)
        if owner not in _VALID_LABELS or auto not in _VALID_LABELS:
            continue
        if owner == _GOOD and auto == _GOOD:
            tp += 1
        elif owner == _BAD and auto == _GOOD:
            fp += 1
        elif owner == _GOOD and auto == _BAD:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _cohen_kappa(cm: ConfusionMatrix) -> float:
    """Cohen's kappa for the 2x2 agreement; 0.0 when undefined (no variance)."""
    n = cm.n
    if n == 0:
        return 0.0
    p_observed = (cm.tp + cm.tn) / n
    p_owner_good = (cm.tp + cm.fn) / n
    p_auto_good = (cm.tp + cm.fp) / n
    p_expected = p_owner_good * p_auto_good + (1.0 - p_owner_good) * (1.0 - p_auto_good)
    if math.isclose(p_expected, 1.0):
        # Perfect chance agreement (a degenerate all-one-class set): kappa undefined ->
        # report 1.0 iff observed also perfect, else 0.0, rather than dividing by zero.
        return 1.0 if math.isclose(p_observed, 1.0) else 0.0
    return (p_observed - p_expected) / (1.0 - p_expected)


def compute_metrics(
    owner_labels: dict[str, str],
    auto_labels: dict[str, str],
    *,
    fpr_bound: float = PREREGISTERED_FPR_BOUND,
) -> LabelerMetrics:
    """Precision/recall/F1/FPR + Cohen's kappa; flag when FPR exceeds the pre-reg bound."""
    cm = confusion_matrix(owner_labels, auto_labels)
    precision = _safe_ratio(cm.tp, cm.tp + cm.fp)
    recall = _safe_ratio(cm.tp, cm.tp + cm.fn)
    f1 = _safe_ratio(2 * cm.tp, 2 * cm.tp + cm.fp + cm.fn)
    fpr = _safe_ratio(cm.fp, cm.fp + cm.tn)
    return LabelerMetrics(
        confusion=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        cohen_kappa=_cohen_kappa(cm),
        fpr_bound=fpr_bound,
        flagged=fpr > fpr_bound,
        n_compared=cm.n,
    )


def format_metrics(metrics: LabelerMetrics) -> str:
    """Human-readable report; names the FLAG verdict against the pre-registered bound."""
    cm = metrics.confusion
    flag = (
        f"FLAGGED — FPR {metrics.fpr:.3f} exceeds the pre-registered bound "
        f"{metrics.fpr_bound:.3f}; do NOT trust this labeler's positives silently"
        if metrics.flagged
        else f"OK — FPR {metrics.fpr:.3f} within the pre-registered bound {metrics.fpr_bound:.3f}"
    )
    return "\n".join(
        [
            f"Compared items : {metrics.n_compared}",
            f"Confusion      : tp={cm.tp} fp={cm.fp} fn={cm.fn} tn={cm.tn}",
            f"Precision      : {metrics.precision:.3f}",
            f"Recall         : {metrics.recall:.3f}",
            f"F1             : {metrics.f1:.3f}",
            f"FPR            : {metrics.fpr:.3f}",
            f"Cohen's kappa  : {metrics.cohen_kappa:.3f}",
            f"Verdict        : {flag}",
        ]
    )
