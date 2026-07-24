"""Metrics: AUPRC matches sklearn, prefix labels recompute, prevalence baseline is available."""

from __future__ import annotations

import random

import pytest

from benchmark.escalation import metrics
from tests.escalation.factories import make_step, make_trajectory

sklearn_ap = pytest.importorskip("sklearn.metrics").average_precision_score


def test_auprc_matches_sklearn_on_a_fixture() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.4, 0.3, 0.1]
    labels = [True, True, False, True, False, False, True, False, False, False]
    assert metrics.auprc(scores, labels) == pytest.approx(sklearn_ap(labels, scores), abs=1e-9)


def test_auprc_matches_sklearn_random() -> None:
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(5, 40)
        scores = [rng.random() for _ in range(n)]
        labels = [rng.random() < 0.3 for _ in range(n)]
        if not any(labels):
            continue
        assert metrics.auprc(scores, labels) == pytest.approx(sklearn_ap(labels, scores), abs=1e-9)


def test_auprc_zero_when_no_positives() -> None:
    assert metrics.auprc([0.5, 0.9], [False, False]) == 0.0


def test_label_prefixes_recompute_from_terminal_and_index() -> None:
    steps = [make_step(step_index=i, decision_index=i) for i in range(5)]
    failed = make_trajectory(steps, terminal_resolved=False)
    # horizon 2 over a length-5 failed trajectory → last two prefixes positive.
    assert metrics.label_prefixes(failed, horizon=2) == [False, False, False, True, True]
    resolved = make_trajectory(steps, terminal_resolved=True)
    assert metrics.label_prefixes(resolved, horizon=2) == [False] * 5  # a solved run has no risk


def test_prevalence_is_the_positive_rate() -> None:
    assert metrics.prevalence([True, False, False, False]) == pytest.approx(0.25)
    assert metrics.prevalence([]) == 0.0


sklearn_roc = pytest.importorskip("sklearn.metrics").roc_auc_score


def test_auroc_matches_sklearn_random() -> None:
    rng = random.Random(1)
    for _ in range(20):
        n = rng.randint(5, 40)
        scores = [rng.random() for _ in range(n)]
        labels = [rng.random() < 0.4 for _ in range(n)]
        if not any(labels) or all(labels):
            continue
        assert metrics.auroc(scores, labels) == pytest.approx(sklearn_roc(labels, scores), abs=1e-9)


def test_auroc_is_chance_when_one_class_absent() -> None:
    assert metrics.auroc([0.1, 0.9, 0.3], [False, False, False]) == 0.5
    assert metrics.auroc([0.1, 0.9, 0.3], [True, True, True]) == 0.5


def test_detection_metrics_hand_computed_confusion_and_f1() -> None:
    # Positive class = risky/flagged prefix. Threshold 0.5 on the binary cumulative score.
    #   score  label   -> cell
    #   1.0    True     tp
    #   1.0    True     tp
    #   1.0    False    fp
    #   0.0    True     fn
    #   0.0    False    tn
    #   0.0    False    tn
    scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    labels = [True, True, False, True, False, False]
    m = metrics.detection_metrics(scores, labels)
    assert (m.confusion.tp, m.confusion.fp, m.confusion.fn, m.confusion.tn) == (2, 1, 1, 2)
    assert m.precision == pytest.approx(2 / 3)  # tp / (tp+fp)
    assert m.recall == pytest.approx(2 / 3)  # tp / (tp+fn)
    assert m.f1 == pytest.approx(2 / 3)  # 2tp / (2tp+fp+fn) = 4/6
    assert m.fpr == pytest.approx(1 / 3)  # fp / (fp+tn)


def _cell(f1: float, *, n_escalated: int = 5, steps: float | None = 3.0) -> metrics.CellReport:
    from benchmark.calibration.labeler_metrics import ConfusionMatrix

    return metrics.CellReport(
        escalate_after_n=2,
        stale_window=5,
        ladder="effort_then_tier",
        confusion=ConfusionMatrix(tp=1, fp=0, fn=0, tn=1),
        precision=1.0,
        recall=1.0,
        f1=f1,
        fpr=0.0,
        cohen_kappa=0.0,
        auprc=0.0,
        auroc=0.5,
        prevalence=0.5,
        n_escalated=n_escalated,
        mean_steps_to_detection=steps,
    )


def test_select_best_config_argmax_on_f1() -> None:
    cells = [_cell(0.4), _cell(0.9), _cell(0.6)]
    best = metrics.select_best_config(cells)
    assert best is not None
    assert best.f1 == pytest.approx(0.9)


def test_select_best_config_tie_broken_by_earlier_detection() -> None:
    early = _cell(0.8, steps=2.0)
    late = _cell(0.8, steps=9.0)
    assert metrics.select_best_config([late, early]) is early  # same F1 → earlier detection wins


def test_select_best_config_null_when_all_degenerate() -> None:
    cells = [_cell(0.0, n_escalated=0), _cell(0.0, n_escalated=0)]
    assert metrics.select_best_config(cells) is None


def test_select_best_config_null_when_no_discrimination() -> None:
    cells = [_cell(0.5), _cell(0.5)]  # identical objective → arbitrary pick refused
    assert metrics.select_best_config(cells) is None
