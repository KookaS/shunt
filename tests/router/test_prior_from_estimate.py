"""Empirical-Bayes Beta prior derived from an offline success-rate estimate + strength."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shunt.router.exploration import (
    CandidateStats,
    ThompsonSampler,
    beta_prior_from_estimate,
)


def _mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def test_strong_estimate_seeds_prior_mean_near_estimate() -> None:
    # High strength ⇒ prior mean pulled to the estimate, not the flat 0.5.
    alpha, beta = beta_prior_from_estimate(0.9, strength=1000.0, strength_cap=1000.0)
    assert _mean(alpha, beta) == pytest.approx(0.9, abs=0.01)


def test_zero_strength_reproduces_flat_prior() -> None:
    # No offline evidence ⇒ exactly Beta(base_alpha, base_beta) = flat Beta(1,1).
    assert beta_prior_from_estimate(0.9, strength=0.0) == (1.0, 1.0)


def test_tiny_sample_stays_regularized_near_half() -> None:
    # A confident-looking estimate from ONE observation must not produce an overconfident
    # prior: mean stays close to 0.5, far from the raw 1.0 estimate.
    alpha, beta = beta_prior_from_estimate(1.0, strength=1.0)
    assert _mean(alpha, beta) == pytest.approx(2.0 / 3.0)  # (1+1)/(2+1)
    assert _mean(alpha, beta) < 0.7


def test_strength_capped() -> None:
    # Strength above the cap is clamped, so a huge history can't swamp the neighborhood.
    capped = beta_prior_from_estimate(0.8, strength=10_000.0, strength_cap=20.0)
    at_cap = beta_prior_from_estimate(0.8, strength=20.0, strength_cap=20.0)
    assert capped == at_cap


def test_estimate_clamped_to_unit_interval() -> None:
    alpha, beta = beta_prior_from_estimate(1.5, strength=10.0)
    assert beta == pytest.approx(1.0)  # (1-clamped(1.5)=0) * k ⇒ base_beta only


def test_non_finite_inputs_stay_finite_and_neutral() -> None:
    # Unreachable via the DB (confidences can't be NaN), but a NaN must never reach
    # np.random.beta (it would crash). Non-finite estimate → neutral 0.5 mean; non-finite
    # strength → flat prior (no evidence trusted).
    alpha, beta = beta_prior_from_estimate(float("nan"), strength=10.0)
    assert math.isfinite(alpha) and math.isfinite(beta)
    assert _mean(alpha, beta) == pytest.approx(0.5)
    assert beta_prior_from_estimate(0.9, strength=float("nan")) == (1.0, 1.0)
    assert beta_prior_from_estimate(0.9, strength=float("inf")) == (1.0, 1.0)


def test_sampler_uses_informative_prior_when_stats_carry_estimate() -> None:
    # A model with NO neighborhood counts but a strong offline estimate should still draw
    # high — the flat prior would leave it at Beta(1,1) (mean 0.5).
    informed = CandidateStats(
        "m", successes=0.0, failures=0.0, cost=1.0, prior_estimate=0.95, prior_strength=500.0
    )
    flat = CandidateStats("m", successes=0.0, failures=0.0, cost=1.0)
    informed_mean = np.mean(
        [
            ThompsonSampler(np.random.default_rng(s), prior_strength_cap=1000.0)
            .select([informed], 0.6, 0)
            .sampled_rates["m"]
            for s in range(400)
        ]
    )
    flat_mean = np.mean(
        [
            ThompsonSampler(np.random.default_rng(s)).select([flat], 0.6, 0).sampled_rates["m"]
            for s in range(400)
        ]
    )
    assert informed_mean > 0.85
    assert flat_mean == pytest.approx(0.5, abs=0.05)
