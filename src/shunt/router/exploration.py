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


@dataclass(frozen=True)
class CandidateStats:
    """Per-model verified evidence aggregated from the neighborhood."""

    model: str
    successes: float  # weighted count of verified passes nearby
    failures: float  # weighted count of verified failures nearby
    cost: float  # representative cost (cheaper models sort first)


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
    ) -> None:
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("prior_alpha and prior_beta must be > 0 (Beta needs positive params)")
        self._rng = rng
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    def _posterior(self, stats: CandidateStats) -> tuple[float, float]:
        """Beta posterior (alpha, beta) from prior + neighborhood counts (>=0 clamped)."""
        return (
            self._prior_alpha + max(0.0, stats.successes),
            self._prior_beta + max(0.0, stats.failures),
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
