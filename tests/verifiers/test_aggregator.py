import pytest

from shunt.verifiers.aggregator import VerifierAggregator
from shunt.verifiers.base import VerifierResult


class TestVerifierAggregator:
    def setup_method(self) -> None:
        self.agg = VerifierAggregator()

    def test_empty_results(self) -> None:
        result = self.agg.aggregate([])
        assert result.outcome == "unknown"
        assert result.confidence == 0.0

    def test_single_success(self) -> None:
        results = [VerifierResult(outcome="success", confidence=0.4)]
        result = self.agg.aggregate(results)
        assert result.outcome == "success"
        assert result.confidence == pytest.approx(2.0 / 3.0)

    def test_all_successes(self) -> None:
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="success", confidence=0.8),
            VerifierResult(outcome="weak_success", confidence=0.2),
        ]
        result = self.agg.aggregate(results)
        alpha = 1 + 3
        beta = 1 + 0
        assert result.confidence == pytest.approx(alpha / (alpha + beta))
        assert result.outcome == "success"

    def test_mixed_success_failure(self) -> None:
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="failure", confidence=0.3),
            VerifierResult(outcome="success", confidence=0.8),
        ]
        result = self.agg.aggregate(results)
        alpha = 1 + 2
        beta = 1 + 1
        expected = alpha / (alpha + beta)
        assert result.confidence == pytest.approx(expected)

    def test_all_failures(self) -> None:
        results = [
            VerifierResult(outcome="failure", confidence=0.3),
            VerifierResult(outcome="failure", confidence=0.7),
        ]
        result = self.agg.aggregate(results)
        alpha = 1 + 0
        beta = 1 + 2
        expected = alpha / (alpha + beta)
        assert result.confidence == pytest.approx(expected)
        assert result.outcome == "failure"

    def test_should_index_above_threshold(self) -> None:
        result = VerifierResult(outcome="success", confidence=0.5)
        assert self.agg.should_index(result)

    def test_should_index_below_threshold(self) -> None:
        result = VerifierResult(outcome="unknown", confidence=0.0)
        assert not self.agg.should_index(result)

    def test_should_index_at_exactly_threshold(self) -> None:
        result = VerifierResult(outcome="success", confidence=0.3)
        assert not self.agg.should_index(result)

    def test_custom_k_and_threshold(self) -> None:
        agg = VerifierAggregator(k=5, index_threshold=0.5)
        result = VerifierResult(outcome="success", confidence=0.6)
        assert agg.should_index(result)
        result2 = VerifierResult(outcome="unknown", confidence=0.4)
        assert not agg.should_index(result2)

    def test_aggregate_detail_contains_beta_info(self) -> None:
        results = [
            VerifierResult(outcome="success", confidence=0.4),
            VerifierResult(outcome="success", confidence=0.8),
        ]
        result = self.agg.aggregate(results)
        assert "Beta" in result.detail
        assert "entropy" in result.detail

    def test_best_outcome_wins_on_tie(self) -> None:
        results = [
            VerifierResult(outcome="weak_success", confidence=0.2),
            VerifierResult(outcome="success", confidence=0.4),
        ]
        result = self.agg.aggregate(results)
        assert result.outcome == "success"
