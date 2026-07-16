from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult

from .conftest import FakeModelPool


class MockOutcomeIndex:
    def __init__(
        self,
        count: int = 0,
        neighbors: list[NeighborResult] | None = None,
    ) -> None:
        self._count = count  # Tier-2 labeled count
        self._neighbors = neighbors or []
        self.queries: list[Any] = []

    def count_labeled(self) -> int:
        return self._count

    def count_total_labeled(self) -> int:
        return self._count

    def query(self, embedding: Any, k: int = 20) -> list[NeighborResult]:
        self.queries.append((embedding, k))
        return self._neighbors


class RecordingEmbedder(Embedder):
    """Embedder that records calls instead of actually embedding."""

    def __init__(self) -> None:
        super().__init__(lazy=True)
        self.call_count = 0

    def embed(self, text: str) -> Any:
        self.call_count += 1
        return np.array([0.1] * 768, dtype=np.float32)


def _neighbor(
    model: str,
    outcome: bool = True,
    cost: float = 1.0,
    confidence: float = 0.9,
    distance: float = 0.1,
) -> NeighborResult:
    return NeighborResult(
        model=model,
        outcome=outcome,
        cost=cost,
        verification_confidence=confidence,
        distance=distance,
        session_id="test",
        truncation_rate=0.0,
    )


class TestDecide:
    def test_cold_start_returns_default_model(self):
        index = MockOutcomeIndex(count=5)
        pool = FakeModelPool("qwen3.7-plus", "gpt4")
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
            cold_start_threshold=20,
        )
        model, reason, provenance = engine.decide("session-1", "some prompt")
        assert model == "qwen3.7-plus"
        assert reason == "cold_start"
        assert embedder.call_count == 0  # no embedding needed
        assert provenance["model_chosen"] == "qwen3.7-plus"
        assert provenance["selection_rule_used"] == "cold_start"
        assert provenance["fallback_chain_triggered"] is False

    def test_cold_start_threshold_exact_boundary(self):
        index = MockOutcomeIndex(count=20)
        pool = FakeModelPool("untested", "other-model")
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            cold_start_threshold=20,
        )
        # At exactly 20 sessions, cold start should be INACTIVE (20 < 20 is False)
        model, reason, provenance = engine.decide("session-1", "some prompt")
        assert model != "qwen3.7-plus"
        assert reason != "cold_start"
        assert provenance["model_chosen"] == model

    def test_cold_start_active_below_threshold(self):
        index = MockOutcomeIndex(count=19)
        pool = FakeModelPool("qwen3.7-plus", "model-a")
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            cold_start_threshold=20,
        )
        model, reason, provenance = engine.decide("session-1", "some prompt")
        assert model == "qwen3.7-plus"
        assert reason == "cold_start"
        assert provenance["selection_rule_used"] == "cold_start"

    def test_normal_flow_with_neighbors(self):
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.9, distance=0.1)
            for _ in range(5)
        ]
        index = MockOutcomeIndex(count=50, neighbors=neighbors)
        pool = FakeModelPool("model-a", "model-b")
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            cold_start_threshold=20,
        )
        model, reason, provenance = engine.decide("session-1", "some prompt")
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"
        assert provenance["selection_rule_used"] == "cheapest_above_threshold"
        assert provenance["fallback_chain_triggered"] is False
        assert len(provenance["top_k_neighbor_ids"]) == 5
        assert len(provenance["neighbor_confidence_scores"]) == 5

    def test_queries_outcome_index_with_embedding(self):
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.9, distance=0.1)
            for _ in range(5)
        ]
        index = MockOutcomeIndex(count=50, neighbors=neighbors)
        pool = FakeModelPool("model-a")
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            cold_start_threshold=20,
        )
        engine.decide("session-1", "prompt text")
        assert len(index.queries) == 1
        emb, k = index.queries[0]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)
        assert k == 20


class TestCaching:
    def test_same_session_caches_embedding(self):
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.9, distance=0.1)
            for _ in range(5)
        ]
        index = MockOutcomeIndex(count=50, neighbors=neighbors)
        pool = FakeModelPool("model-a")
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
        )

        engine.decide("session-1", "same prompt")
        engine.decide("session-1", "same prompt")
        assert embedder.call_count == 1

    def test_different_sessions_separate_cache(self):
        neighbors = [
            _neighbor("model-a", outcome=True, cost=1.0, confidence=0.9, distance=0.1)
            for _ in range(5)
        ]
        index = MockOutcomeIndex(count=50, neighbors=neighbors)
        pool = FakeModelPool("model-a")
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
        )

        engine.decide("session-1", "prompt")
        engine.decide("session-2", "prompt")
        assert embedder.call_count == 2


class TestGetNeighbors:
    def test_returns_neighbors_from_index(self):
        expected = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.9, distance=0.1),
            _neighbor("model-b", outcome=False, cost=1.0, confidence=0.8, distance=0.2),
        ]
        index = MockOutcomeIndex(count=50, neighbors=expected)
        engine = RouterEngine(
            model_pool=FakeModelPool("model-a", "model-b"),
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
        )

        result = engine.get_neighbors("session-1", "prompt", k=10)
        assert result == expected

    def test_get_neighbors_caches_embedding(self):
        index = MockOutcomeIndex(count=50, neighbors=[])
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=FakeModelPool("model-a"),
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
        )

        engine.get_neighbors("session-1", "prompt")
        engine.get_neighbors("session-1", "prompt")
        assert embedder.call_count == 1
