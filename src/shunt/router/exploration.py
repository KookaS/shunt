"""Cost-aware Thompson-sampling exploration over the kNN neighborhood.

Draws a pass-probability per model from ``Beta(s+prior_a, f+prior_b)`` (neighborhood
counts, so thin neighborhoods explore more), then picks the cheapest clearing threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _cost_key(stats: CandidateStats) -> float:
    """Sort key for cheapest-selection; a non-finite cost sorts LAST (never cheapest)."""
    return stats.cost if math.isfinite(stats.cost) else math.inf


# Default cap on the informative prior's pseudo-count strength: even a model with a huge
# offline history contributes at most this many prior pseudo-observations, so the prior
# regularizes the sparse local neighborhood rather than swamping it.
_DEFAULT_PRIOR_STRENGTH_CAP = 20.0


def beta_prior_from_estimate(
    estimate: float,
    strength: float,
    *,
    base_alpha: float = 1.0,
    base_beta: float = 1.0,
    strength_cap: float = _DEFAULT_PRIOR_STRENGTH_CAP,
) -> tuple[float, float]:
    """Map an offline success-rate ``estimate`` + effective ``strength`` to a Beta prior."""
    # Beta(base_alpha + est·k, base_beta + (1-est)·k) with k = min(strength, cap) clamped to
    # [0, cap]. The prior mean (base_alpha + est·k)/(base_alpha+base_beta+k) → estimate as k
    # grows and stays near 0.5 (the flat base) for a tiny sample — so a strong estimate seeds
    # near itself while a thin one stays regularized. strength=0 reproduces flat Beta(a, b).
    # Coerce non-finite inputs to neutral values before the min/max clamps (NaN would else
    # propagate through both and hand np.random.beta a NaN parameter → crash). Unreachable
    # today — the priors derive from DB confidences the NOT NULL constraint keeps non-NaN —
    # but keep the map total, like exploration._cost_key's finite guard.
    safe_strength = strength if math.isfinite(strength) else 0.0
    safe_estimate = estimate if math.isfinite(estimate) else 0.5
    k = min(max(safe_strength, 0.0), strength_cap)
    clamped = min(max(safe_estimate, 0.0), 1.0)
    return (base_alpha + clamped * k, base_beta + (1.0 - clamped) * k)


@dataclass(frozen=True)
class CandidateStats:
    """Per-model verified evidence aggregated from the neighborhood."""

    # prior_estimate/prior_strength carry the offline kNN aggregate (the model's global
    # weighted success rate + its effective sample size) to seed an informative Beta prior;
    # a None estimate falls back to the sampler's flat prior.
    model: str
    successes: float  # weighted count of verified passes nearby
    failures: float  # weighted count of verified failures nearby
    cost: float  # representative cost (cheaper models sort first)
    prior_estimate: float | None = None  # offline global weighted success rate (empirical Bayes)
    prior_strength: float = 0.0  # effective sample size backing prior_estimate


@dataclass(frozen=True)
class ExplorationDecision:
    """Outcome of a Thompson draw: the chosen model + data for provenance/off-policy eval."""

    model: str
    greedy_model: str  # deterministic exploit pick — reference for the conservative gate
    sampled_rates: dict[str, float]
    propensity: float
    is_exploratory: bool


class ThompsonSampler:
    """Beta-Bernoulli Thompson sampler with cost-aware, threshold-gated selection."""

    def __init__(
        self,
        rng: np.random.Generator,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        prior_strength_cap: float = _DEFAULT_PRIOR_STRENGTH_CAP,
    ) -> None:
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("prior_alpha and prior_beta must be > 0 (Beta needs positive params)")
        self._rng = rng
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._prior_strength_cap = prior_strength_cap

    def _prior(self, stats: CandidateStats) -> tuple[float, float]:
        """The Beta prior for a model: informative (from its offline estimate) or flat.

        With an offline aggregate, shrink toward the model's global rate (empirical Bayes);
        with none, fall back to the flat ``Beta(prior_alpha, prior_beta)``.
        """
        if stats.prior_estimate is None:
            return (self._prior_alpha, self._prior_beta)
        return beta_prior_from_estimate(
            stats.prior_estimate,
            stats.prior_strength,
            base_alpha=self._prior_alpha,
            base_beta=self._prior_beta,
            strength_cap=self._prior_strength_cap,
        )

    def _posterior(self, stats: CandidateStats) -> tuple[float, float]:
        """Beta posterior (alpha, beta) from prior + neighborhood counts (>=0 clamped)."""
        prior_alpha, prior_beta = self._prior(stats)
        return (
            prior_alpha + max(0.0, stats.successes),
            prior_beta + max(0.0, stats.failures),
        )

    def _draw(self, stats: CandidateStats) -> float:
        alpha, beta = self._posterior(stats)
        return float(self._rng.beta(alpha, beta))

    def _argmax_greedy(self, candidates: list[CandidateStats], threshold: float) -> str:
        """Deterministic cost-aware greedy pick using posterior *means* (no sampling).

        The exploit baseline: cheapest whose mean pass-rate clears threshold, else the
        highest-mean candidate. Used to decide whether a sampled choice was exploratory.
        """
        means = {c.model: self._mean(c) for c in candidates}
        qualifying = [c for c in candidates if means[c.model] >= threshold]
        if qualifying:
            return min(qualifying, key=_cost_key).model
        return max(candidates, key=lambda c: means[c.model]).model

    def _mean(self, stats: CandidateStats) -> float:
        alpha, beta = self._posterior(stats)
        return alpha / (alpha + beta)

    def _select_from_draws(
        self,
        candidates: list[CandidateStats],
        draws: dict[str, float],
        threshold: float,
    ) -> str:
        qualifying = [c for c in candidates if draws[c.model] >= threshold]
        if qualifying:
            return min(qualifying, key=_cost_key).model
        return max(candidates, key=lambda c: draws[c.model]).model

    def select(
        self,
        candidates: list[CandidateStats],
        threshold: float,
        propensity_mc_samples: int = 100,
    ) -> ExplorationDecision:
        """Thompson-sample one model, cost-aware and threshold-gated."""
        # Returns the choice + sampled rates (provenance) + a Monte-Carlo propensity (for
        # off-policy eval) + whether it diverged from the greedy pick (was exploratory).
        if not candidates:
            raise ValueError("select() requires at least one candidate")
        draws = {c.model: self._draw(c) for c in candidates}
        chosen = self._select_from_draws(candidates, draws, threshold)
        greedy = self._argmax_greedy(candidates, threshold)
        propensity = self._estimate_propensity(candidates, threshold, chosen, propensity_mc_samples)
        return ExplorationDecision(
            model=chosen,
            greedy_model=greedy,
            sampled_rates=draws,
            propensity=propensity,
            is_exploratory=(chosen != greedy),
        )

    def _estimate_propensity(
        self,
        candidates: list[CandidateStats],
        threshold: float,
        chosen: str,
        mc_samples: int,
    ) -> float:
        """Monte-Carlo estimate of P(select == chosen) — logged for off-policy eval."""
        # No closed-form propensity for TS selection → resample. Clamped away from 0 so
        # importance weights stay finite.
        if mc_samples <= 0:
            return 1.0
        # Resampling must NOT advance the decision stream (else a pure logging knob would
        # change which model future requests get). Snapshot + restore the RNG state.
        state = self._rng.bit_generator.state
        try:
            hits = 0
            for _ in range(mc_samples):
                draws = {c.model: self._draw(c) for c in candidates}
                if self._select_from_draws(candidates, draws, threshold) == chosen:
                    hits += 1
        finally:
            self._rng.bit_generator.state = state
        return max(hits / mc_samples, 1.0 / (mc_samples + 1))
