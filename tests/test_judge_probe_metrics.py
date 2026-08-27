"""Unit tests for the judge-probe metrics script (no live calls, no mocks)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from benchmark.routing.scripts.judge_probe_metrics import (
    decide_verdict,
    fleiss_kappa,
    label_r2_auc,
    shuffled_null,
)


def _loo_brute(x: np.ndarray, y: np.ndarray) -> float:
    """Independent brute-force leave-one-out R^2 oracle for the analytic version."""
    preds = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        preds[i] = LinearRegression().fit(x[mask], y[mask]).predict(x[i : i + 1])[0]
    return float(r2_score(y, preds))


def test_label_r2_auc_in_sample_and_loo() -> None:
    labels = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    pass_by = {"a": False, "b": False, "c": True, "d": True}
    res = label_r2_auc(labels, pass_by)
    assert res["n"] == 4
    x = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert res["r2"] == pytest.approx(r2_score(y, LinearRegression().fit(x, y).predict(x)))
    assert res["r2_loo"] == pytest.approx(_loo_brute(x, y))


def test_label_r2_auc_nonmonotone_matches_brute_loo() -> None:
    labels = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    pass_by = {"a": True, "b": False, "c": True, "d": False}
    res = label_r2_auc(labels, pass_by)
    x = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert res["r2"] == pytest.approx(r2_score(y, LinearRegression().fit(x, y).predict(x)))
    assert res["r2_loo"] == pytest.approx(_loo_brute(x, y))


def test_label_r2_auc_insufficient_data_is_none() -> None:
    single_class = {"a": 1.0, "b": 2.0, "c": 3.0}
    pass_by = {"a": True, "b": True, "c": True}
    res = label_r2_auc(single_class, pass_by)
    assert res["r2"] is None and res["r2_loo"] is None
    too_short = label_r2_auc({"a": 1.0}, {"a": True})
    assert too_short["r2"] is None and too_short["r2_loo"] is None


def test_shuffled_null_band_contains_mean_and_zero() -> None:
    rng = np.random.default_rng(3)
    labels = {f"t{i}": float(rng.normal()) for i in range(24)}
    pass_by = {f"t{i}": bool(i % 3 == 0) for i in range(24)}
    lo, hi, mean = shuffled_null(labels, pass_by, n_perm=300, loo=True)
    assert lo == lo and hi == hi and mean == mean  # no NaN
    assert lo <= mean <= hi  # band contains the null mean
    assert lo <= 0.0 <= hi  # the LOO null brackets zero
    lo_ins, hi_ins, mean_ins = shuffled_null(labels, pass_by, n_perm=300)
    assert lo_ins <= mean_ins <= hi_ins
    assert abs(mean_ins) < 0.1  # in-sample null mean is ~0 (slightly positive by overfit)


def test_shuffled_null_perfect_signal_clears_band() -> None:
    n = 20
    labels = {f"t{i}": float(i) for i in range(n)}
    pass_by = {f"t{i}": bool(i < n // 2) for i in range(n)}
    res = label_r2_auc(labels, pass_by)
    lo, hi, _mean = shuffled_null(labels, pass_by, n_perm=300, loo=True)
    assert res["r2_loo"] is not None
    assert res["r2_loo"] > hi


def test_fleiss_kappa_perfect_agreement_is_one() -> None:
    matrix = np.array([[2, 0], [0, 2], [2, 0], [0, 2]], dtype=float)
    assert fleiss_kappa(matrix) == pytest.approx(1.0)


def test_fleiss_kappa_maximal_disagreement() -> None:
    matrix = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=float)
    assert fleiss_kappa(matrix) == pytest.approx(-1.0)


def test_fleiss_kappa_mixed_gives_zero() -> None:
    matrix = np.array([[2, 0], [1, 1], [0, 2], [1, 1]], dtype=float)
    assert fleiss_kappa(matrix) == pytest.approx(0.0)


def test_fleiss_kappa_single_category_is_nan() -> None:
    matrix = np.array([[2, 0], [2, 0], [2, 0]], dtype=float)
    assert np.isnan(fleiss_kappa(matrix))


def test_decide_verdict_fail_inside_null() -> None:
    verdict = decide_verdict(
        r2_loo=0.01, n=50, null_loo_hi=0.02, human_r2_loo=0.10, positive_control_ok=True
    )
    assert verdict.startswith("FAIL")


def test_decide_verdict_marginal_above_null() -> None:
    verdict = decide_verdict(
        r2_loo=0.05, n=50, null_loo_hi=0.02, human_r2_loo=0.10, positive_control_ok=True
    )
    assert verdict.startswith("MARGINAL")


def test_decide_verdict_invalid_without_positive_control() -> None:
    verdict = decide_verdict(
        r2_loo=0.05, n=50, null_loo_hi=0.02, human_r2_loo=0.10, positive_control_ok=False
    )
    assert verdict.startswith("INVALID")


def test_decide_verdict_inconclusive_when_sparse() -> None:
    assert (
        decide_verdict(
            r2_loo=None, n=50, null_loo_hi=0.02, human_r2_loo=0.10, positive_control_ok=True
        )
        == "INCONCLUSIVE"
    )
    assert (
        decide_verdict(
            r2_loo=0.05, n=10, null_loo_hi=0.02, human_r2_loo=0.10, positive_control_ok=True
        )
        == "INCONCLUSIVE"
    )


def test_pipeline_marginal_end_to_end() -> None:
    n = 30
    labels = {f"t{i}": float(i) for i in range(n)}
    pass_by = {f"t{i}": bool(i >= 15) for i in range(n)}
    human = {f"t{i}": (0.0 if i < 15 else 1.0) for i in range(n)}

    anchor = label_r2_auc(labels, pass_by)
    control = label_r2_auc(human, pass_by)
    null_loo_hi = shuffled_null(labels, pass_by, n_perm=300, loo=True)[1]
    control_null_hi = shuffled_null(human, pass_by, n_perm=300, loo=True)[1]

    assert anchor["r2_loo"] is not None and control["r2_loo"] is not None
    control_ok = control["r2_loo"] > control_null_hi
    assert control_ok
    verdict = decide_verdict(
        anchor["r2_loo"], anchor["n"], null_loo_hi, control["r2_loo"], control_ok
    )
    assert verdict.startswith("MARGINAL")
