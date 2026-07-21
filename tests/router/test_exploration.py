from __future__ import annotations

import numpy as np
import pytest

from shunt.router.exploration import CandidateStats, ThompsonSampler


def _sampler(seed: int = 0) -> ThompsonSampler:
    return ThompsonSampler(np.random.default_rng(seed))


def test_select_requires_candidates() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        _sampler().select([], threshold=0.6)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_prior_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        ThompsonSampler(np.random.default_rng(0), prior_alpha=bad)


def test_dominant_candidate_wins_with_high_propensity() -> None:
    cands = [
        CandidateStats("cheap", successes=50, failures=1, cost=0.1),
        CandidateStats("pricey", successes=1, failures=50, cost=1.0),
    ]
    dec = _sampler().select(cands, threshold=0.6, propensity_mc_samples=200)
    assert dec.model == "cheap"
    assert dec.propensity > 0.9
    assert dec.is_exploratory is False


def test_cost_aware_prefers_cheaper_when_both_qualify() -> None:
    # Both strongly-passing; cheapest-above-threshold should win the vast majority.
    cands = [
        CandidateStats("cheap", successes=80, failures=2, cost=0.1),
        CandidateStats("pricey", successes=80, failures=2, cost=1.0),
    ]
    picks = [
        _sampler(s).select(cands, threshold=0.6, propensity_mc_samples=0).model for s in range(40)
    ]
    assert picks.count("cheap") > picks.count("pricey")


def test_thin_neighborhood_has_higher_posterior_variance() -> None:
    # The core "explore more when data is thin" property: same posterior MEAN, but the
    # thin (few-sample) candidate's sampled rates are more spread than the dense one's.
    thin = CandidateStats("m", successes=2, failures=1, cost=1.0)  # Beta(3,2), mean 0.6
    dense = CandidateStats("m", successes=200, failures=100, cost=1.0)  # Beta(201,101), ~0.6
    thin_draws = [
        ThompsonSampler(np.random.default_rng(s)).select([thin], 0.6, 0).sampled_rates["m"]
        for s in range(3000)
    ]
    dense_draws = [
        ThompsonSampler(np.random.default_rng(s)).select([dense], 0.6, 0).sampled_rates["m"]
        for s in range(3000)
    ]
    assert float(np.var(thin_draws)) > float(np.var(dense_draws))


def test_propensity_bounds() -> None:
    cands = [
        CandidateStats("a", successes=5, failures=5, cost=0.1),
        CandidateStats("b", successes=5, failures=5, cost=0.2),
    ]
    dec = _sampler().select(cands, threshold=0.6, propensity_mc_samples=100)
    assert 0.0 < dec.propensity <= 1.0


def test_zero_mc_samples_yields_unit_propensity() -> None:
    cands = [CandidateStats("a", successes=1, failures=1, cost=0.1)]
    dec = _sampler().select(cands, threshold=0.6, propensity_mc_samples=0)
    assert dec.propensity == 1.0
    assert dec.model == "a"


def test_propensity_estimation_does_not_perturb_decision_stream() -> None:
    # Regression: propensity_mc_samples is a LOGGING knob and must not change which model
    # future decisions get. Two samplers, same seed, differing only in mc_samples → the
    # NEXT decision's draws must be identical.
    cands = [CandidateStats("a", 5, 5, 0.1), CandidateStats("b", 5, 5, 0.2)]
    s1 = ThompsonSampler(np.random.default_rng(0))
    s2 = ThompsonSampler(np.random.default_rng(0))
    s1.select(cands, 0.6, propensity_mc_samples=100)
    s2.select(cands, 0.6, propensity_mc_samples=0)
    assert s1.select(cands, 0.6, 0).sampled_rates == s2.select(cands, 0.6, 0).sampled_rates


def test_nan_cost_never_selected_as_cheapest() -> None:
    # Regression: NaN comparisons are all False, so a naive min(key=cost) can return the
    # NaN model. A non-finite cost must sort last, never cheapest.
    cands = [
        CandidateStats("nan", successes=80, failures=2, cost=float("nan")),
        CandidateStats("ok", successes=80, failures=2, cost=1.0),
    ]
    picks = {
        _sampler(s).select(cands, threshold=0.6, propensity_mc_samples=0).model for s in range(40)
    }
    assert picks == {"ok"}  # the finite-cost model wins every cheapest-above-threshold pick
