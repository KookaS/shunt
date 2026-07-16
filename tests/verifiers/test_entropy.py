import math

import pytest

from shunt.verifiers.aggregator import VerifierAggregator, _bernoulli_entropy
from shunt.verifiers.base import VerifierResult


class TestBernoulliEntropy:
    def test_zero_at_certainty(self) -> None:
        assert _bernoulli_entropy(0.0) == 0.0
        assert _bernoulli_entropy(1.0) == 0.0

    def test_max_at_fifty_fifty(self) -> None:
        h = _bernoulli_entropy(0.5)
        assert h == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        assert _bernoulli_entropy(0.3) == pytest.approx(_bernoulli_entropy(0.7))

    def test_known_value(self) -> None:
        expected = -0.25 * math.log2(0.25) - 0.75 * math.log2(0.75)
        assert _bernoulli_entropy(0.25) == pytest.approx(expected)

    def test_inconsistent_draws_exceed_threshold(self) -> None:
        agg = VerifierAggregator()
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="failure", confidence=0.3),
        ]
        agg.aggregate(results)
        p = 2 / 3
        expected_entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        assert expected_entropy > 0.8

    def test_consistent_draws_below_threshold(self) -> None:
        agg = VerifierAggregator()
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="success", confidence=0.8),
            VerifierResult(outcome="success", confidence=0.4),
        ]
        agg.aggregate(results)
        expected_entropy = 0.0
        assert expected_entropy <= 0.8

    def test_aggregator_detail_includes_entropy(self) -> None:
        agg = VerifierAggregator()
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="failure", confidence=0.3),
        ]
        result = agg.aggregate(results)
        assert "entropy=" in result.detail
