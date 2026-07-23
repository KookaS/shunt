"""nₑ = (Σw)²/Σ(w²) primitive + its backward-compat invariant (uniform ⇒ raw count)."""

from __future__ import annotations

import pytest

from shunt.router.selection import effective_sample_size


def test_uniform_weights_equal_raw_count() -> None:
    # The backward-compat invariant: all-equal weights ⇒ nₑ == N, for any equal value.
    assert effective_sample_size([1.0] * 7) == pytest.approx(7.0)
    assert effective_sample_size([0.5] * 7) == pytest.approx(7.0)


def test_empty_and_all_zero_are_zero() -> None:
    assert effective_sample_size([]) == 0.0
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_mixed_weights_below_raw_count() -> None:
    # Five outcomes, four barely-confident: nₑ well under 5.
    weights = [1.0, 0.1, 0.1, 0.1, 0.1]
    ne = effective_sample_size(weights)
    # (1.4)²/(1+4*0.01) = 1.96/1.04 = 1.8846…
    assert ne == pytest.approx(1.96 / 1.04)
    assert ne < len(weights)


def test_single_dominant_weight_collapses_to_one() -> None:
    # One heavy + noise ⇒ nₑ → 1 (one outcome carries essentially all the evidence).
    assert effective_sample_size([100.0, 0.01, 0.01]) < 1.02


def test_negative_weights_dropped() -> None:
    # The confidence clamp never yields negatives, but guard anyway: they must not count.
    assert effective_sample_size([1.0, 1.0, -5.0]) == pytest.approx(2.0)


def test_non_finite_weights_dropped() -> None:
    # Unreachable in practice (the DB NOT NULL constraint keeps confidences non-NaN), but the
    # pure function stays total: a NaN/inf weight drops out instead of poisoning nₑ to NaN.
    assert effective_sample_size([1.0, 1.0, float("nan")]) == pytest.approx(2.0)
    assert effective_sample_size([1.0, 1.0, float("inf")]) == pytest.approx(2.0)
    assert effective_sample_size([float("nan"), float("inf")]) == 0.0
