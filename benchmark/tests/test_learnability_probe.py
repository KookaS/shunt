"""Unit tests for the learnability-probe statistics and controls (no corpus, no live calls)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.routing.scripts import knn_nulls
from benchmark.routing.scripts.learnability_probe import (
    _auc,
    _planted_sims,
    _repo_onehot,
    _same_tag_sims,
    destroyed_signal_score,
    evaluate_knn,
    evaluate_lr,
    knn_loo_scores,
    logistic_loo_predictions,
    r2_of,
)


def _binary_y(n: int, rate: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < rate).astype(float)


def test_r2_perfect_and_constant() -> None:
    y = _binary_y(80, 0.5)
    assert r2_of(y, y) == pytest.approx(1.0)
    assert r2_of(y, np.full(80, y.mean())) == pytest.approx(0.0)


def test_r2_worse_than_constant_is_negative() -> None:
    y = _binary_y(80, 0.5)
    bad = 1.0 - y  # anti-perfect
    assert r2_of(y, bad) < 0.0


def test_knn_loo_scores_matches_knn_nulls_r2() -> None:
    rng = np.random.default_rng(1)
    n = 60
    sims = rng.normal(size=(n, n))
    sims = (sims + sims.T) / 2
    y = _binary_y(n, 0.6, seed=2)
    for k in (1, 5, 20):
        pred = knn_loo_scores(sims, y, k)
        assert r2_of(y, pred) == pytest.approx(knn_nulls.loo_r2(sims, y, k), abs=1e-12)


def test_planted_similarity_recovers_y() -> None:
    """The positive control: similarity IS the outcome, so LOO kNN recovers it exactly."""
    y = _binary_y(120, 0.5, seed=3)
    sims = _planted_sims(y)
    for k in (5, 20, 40):
        pred = knn_loo_scores(sims, y, k)
        assert r2_of(y, pred) > 0.99
        assert _auc(pred, y) > 0.99


def test_logistic_loo_recovers_planted_feature() -> None:
    """An LR whose only feature is the label must separate it in leave-one-out."""
    y = _binary_y(120, 0.7, seed=4)
    pred = logistic_loo_predictions(y[:, None], y)
    assert r2_of(y, pred) > 0.9
    assert _auc(pred, y) > 0.99


def test_knn_null_centres_at_chance() -> None:
    """On outcome-free data the kNN LOO R2 null band brackets zero."""
    rng = np.random.default_rng(5)
    n = 60
    sims = rng.normal(size=(n, n))
    sims = (sims + sims.T) / 2
    y = _binary_y(n, 0.6, seed=6)
    res = evaluate_knn(sims, y, n_perm=60, seed=0)
    assert -0.5 < res.r2_null.mean < 0.5
    assert res.r2_null.lo <= 0.0 <= res.r2_null.hi
    assert len(res.r2_grid) == len(res.k_grid)


def test_lr_null_centres_at_chance() -> None:
    """On a noise-only design the logistic LOO R2 null band brackets zero."""
    rng = np.random.default_rng(7)
    n = 60
    x = rng.normal(size=(n, 4))
    y = _binary_y(n, 0.6, seed=8)
    res = evaluate_lr(x, y, n_perm=60, seed=0)
    assert -0.5 < res.r2_null.mean < 0.5
    assert res.r2_null.lo <= 0.0 <= res.r2_null.hi


def test_destroyed_signal_draw_sits_in_the_null_band() -> None:
    """The destroyed-signal leg is a real pipeline draw, and it lands inside its null."""
    rng = np.random.default_rng(11)
    n = 60
    sims = rng.normal(size=(n, n))
    sims = (sims + sims.T) / 2
    y = _binary_y(n, 0.6, seed=12)
    res = evaluate_knn(sims, y, n_perm=60, seed=0)
    destroyed = destroyed_signal_score(sims, y, k=20, seed=13)
    assert res.r2_null.lo <= destroyed <= res.r2_null.hi
    assert destroyed != pytest.approx(res.r2_null.mean)


def test_same_tag_similarity_is_symmetric_and_bounded() -> None:
    tags = np.array([0, 0, 1, 1, 2], dtype=float)
    tiebreak = np.ones((5, 5))
    sims = _same_tag_sims(tags, tiebreak)
    assert sims.shape == (5, 5)
    assert np.allclose(sims, sims.T)
    assert sims[0, 1] > sims[0, 2]  # same tag beats a different tag
    assert np.all(sims <= 1.0 + 1e-3)


def test_repo_onehot_shape() -> None:
    repos = np.array(["a/b", "a/b", "c/d"])
    onehot = _repo_onehot(repos)
    assert onehot.shape == (3, 2)
    assert onehot.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_auc_single_class_reads_as_chance() -> None:
    y = np.zeros(20, dtype=int)
    score = np.random.default_rng(9).random(20)
    assert _auc(score, y.astype(float)) == pytest.approx(0.5)
