"""Metrics: statistics match sklearn, drawn curves match their own titles, nulls are real."""

from __future__ import annotations

import random

import pytest

from benchmark.escalation import metrics

sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_ap = sklearn_metrics.average_precision_score
sklearn_roc = sklearn_metrics.roc_auc_score


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


def test_auroc_matches_sklearn_random_including_heavy_ties() -> None:
    rng = random.Random(1)
    for _ in range(20):
        n = rng.randint(5, 40)
        # Coarse quantisation forces large tie blocks — the case the drawn curve used to mishandle.
        scores = [float(rng.randint(0, 2)) for _ in range(n)]
        labels = [rng.random() < 0.4 for _ in range(n)]
        if not any(labels) or all(labels):
            continue
        assert metrics.auroc(scores, labels) == pytest.approx(sklearn_roc(labels, scores), abs=1e-9)


def test_auroc_is_chance_when_one_class_absent() -> None:
    assert metrics.auroc([0.1, 0.9, 0.3], [False, False, False]) == 0.5
    assert metrics.auroc([0.1, 0.9, 0.3], [True, True, True]) == 0.5


def test_prevalence_is_the_positive_rate() -> None:
    assert metrics.prevalence([True, False, False, False]) == pytest.approx(0.25)
    assert metrics.prevalence([]) == 0.0


def test_detection_metrics_hand_computed_confusion_and_f1() -> None:
    #   score  label   -> cell
    #   1.0    True     tp        0.0    True     fn
    #   1.0    True     tp        0.0    False    tn
    #   1.0    False    fp        0.0    False    tn
    scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    labels = [True, True, False, True, False, False]
    m = metrics.detection_metrics(scores, labels)
    assert (m.confusion.tp, m.confusion.fp, m.confusion.fn, m.confusion.tn) == (2, 1, 1, 2)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.fpr == pytest.approx(1 / 3)


def test_operating_threshold_is_reachable_on_a_calibrated_score() -> None:
    # THE BUG THIS FIXES: a fixed 0.5 cut on a calibrated probability whose base rate is ~0.38
    # is unreachable — the score centres near 0.38, so almost nothing is ever flagged. The
    # threshold must instead come from the scores, spending a flag budget equal to prevalence.
    scores = [0.30 + 0.001 * i for i in range(100)]  # 0.300 .. 0.399: NOTHING crosses 0.5
    labels = [i < 38 for i in range(100)]
    assert sum(s >= 0.5 for s in scores) == 0, "the fixed cut really is unreachable here"
    threshold = metrics.operating_threshold(scores, labels)
    assert sum(s >= threshold for s in scores) == 38  # exactly the base-rate flag budget
    assert metrics.prevalence(labels) == pytest.approx(0.38)


def test_operating_threshold_honours_an_explicit_flag_budget() -> None:
    scores = [float(i) / 100 for i in range(100)]
    labels = [i < 38 for i in range(100)]
    tenth = metrics.operating_threshold(scores, labels, flag_rate=0.1)
    assert sum(s >= tenth for s in scores) == 10
    # A zero budget flags nothing rather than silently flagging the top row.
    none_at_all = metrics.operating_threshold(scores, labels, flag_rate=0.0)
    assert sum(s >= none_at_all for s in scores) == 0


def test_detection_metrics_derives_its_threshold_when_none_is_given() -> None:
    scores = [0.30 + 0.001 * i for i in range(100)]
    labels = [i >= 62 for i in range(100)]  # the 38 highest-scoring rows are the failures
    m = metrics.detection_metrics(scores, labels)
    assert (m.confusion.tp, m.confusion.fp) == (38, 0)  # perfectly ranked, so the budget is exact
    assert metrics.detection_metrics(scores, labels, threshold=0.5).confusion.tp == 0


def _polyline_area(points: list[tuple[float, float]]) -> float:
    """Trapezoidal area under a polyline — what a reader integrates by eye."""
    ordered = sorted(points)
    return sum(
        (b[0] - a[0]) * (a[1] + b[1]) / 2.0 for a, b in zip(ordered, ordered[1:], strict=False)
    )


def test_drawn_roc_area_equals_the_reported_auroc() -> None:
    # The committed figure drew area 0.554 under a title reading 0.450, because the old builder
    # admitted one tied ROW at a time in corpus order. Collapsing ties to distinct thresholds makes
    # the drawn shape and the reported statistic the same number, by construction.
    rng = random.Random(5)
    for _ in range(20):
        scores = [float(rng.randint(0, 3)) for _ in range(200)]
        labels = [rng.random() < 0.3 for _ in range(200)]
        if not any(labels) or all(labels):
            continue
        drawn = _polyline_area(metrics.roc_operating_points(scores, labels))
        assert drawn == pytest.approx(metrics.auroc(scores, labels), abs=1e-9)


def test_drawn_roc_is_invariant_to_input_row_order() -> None:
    # The old polyline moved from 0.554 to ~0.450 when the rows were shuffled: the shape was the
    # corpus's ordering, not the detector. The tie-collapsed curve cannot depend on row order.
    rng = random.Random(9)
    scores = [float(rng.randint(0, 2)) for _ in range(300)]
    labels = [rng.random() < 0.35 for _ in range(300)]
    before = metrics.roc_operating_points(scores, labels)
    pairs = list(zip(scores, labels, strict=True))
    rng.shuffle(pairs)
    after = metrics.roc_operating_points([s for s, _ in pairs], [lab for _, lab in pairs])
    assert before == after


def test_roc_and_pr_collapse_a_binary_score_to_its_real_operating_points() -> None:
    scores = [1.0] * 40 + [0.0] * 60
    labels = [i % 3 == 0 for i in range(100)]
    # A binary score has exactly two thresholds, so ROC has 3 vertices: (0,0), the point, (1,1).
    assert len(metrics.roc_operating_points(scores, labels)) == 3
    assert len(metrics.pr_operating_points(scores, labels)) == 3


def test_permutation_null_places_a_real_signal_outside_the_band() -> None:
    labels = [i < 50 for i in range(100)]
    scores = [1.0 if lab else 0.0 for lab in labels]  # perfect
    null = metrics.permute_statistic(scores, labels, metrics.auroc, n_permutations=200)
    assert null.observed == pytest.approx(1.0)
    assert null.mean == pytest.approx(0.5, abs=0.05)
    assert null.beats_null
    assert null.p_value < 0.01


def test_permutation_null_keeps_pure_noise_inside_the_band() -> None:
    # The old gate was a bare `auprc > prevalence` with no noise floor, so a +0.0008 excess against
    # a null sd of 0.00055 reported `status: OK`. A null must not clear its own band.
    rng = random.Random(11)
    labels = [rng.random() < 0.4 for _ in range(300)]
    scores = [rng.random() for _ in range(300)]
    null = metrics.permute_statistic(scores, labels, metrics.auroc, n_permutations=200)
    assert not null.beats_null
    assert null.p_value > 0.05


def test_permutation_null_refuses_too_few_draws() -> None:
    with pytest.raises(ValueError, match="permutation null needs"):
        metrics.permutation_null(0.9, [0.5] * 10)


def test_permutation_p_value_can_never_be_zero() -> None:
    # An exact permutation test has resolution 1/(n+1); reporting p=0 would overclaim.
    null = metrics.permutation_null(99.0, [0.0] * 200)
    assert null.p_value == pytest.approx(1 / 201)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = metrics.wilson_interval(37, 100)
    assert low < 0.37 < high
    # Oracle: the closed-form Wilson bounds for p=0.37, n=100 (cross-checked against an
    # independent evaluation with scipy's exact z, which agrees to 5 dp).
    assert (low, high) == pytest.approx((0.281824, 0.467795), abs=1e-4)
    # Degenerate counts stay inside [0, 1] instead of a nonsense normal-approx interval.
    assert metrics.wilson_interval(0, 10)[0] == 0.0
    assert metrics.wilson_interval(10, 10)[1] == 1.0
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)


def test_grouped_bootstrap_resamples_groups_not_rows() -> None:
    # 10 challenges of 10 correlated rows each. A row-level bootstrap would report a spuriously
    # tight interval on this structure; the grouped one must stay wide enough to contain the point.
    rng = random.Random(3)
    groups = [f"c{i // 10}" for i in range(100)]
    labels = [i // 10 < 5 for i in range(100)]
    scores = [rng.random() for _ in range(100)]
    low, high = metrics.grouped_bootstrap_ci(scores, labels, groups, metrics.auroc, n_resamples=200)
    assert low <= metrics.auroc(scores, labels) <= high
    assert high - low > 0.05
