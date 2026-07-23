from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult
from shunt.router.strategies.fixed import AlwaysCheapStrategy

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

    def effective_labeled(self) -> float:
        # Uniform weights ⇒ nₑ == raw count, so the effective gate matches the count gate.
        return float(self._count)

    def effective_tier2(self) -> float:
        return float(self._count)

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

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


class TestStaleEmbeddingSpace:
    def test_stale_space_short_circuits_to_cold_start_no_embed_no_query(self):
        # Kill-gate: a fingerprint mismatch (trust_neighbors=False) must route via the
        # cold-start default WITHOUT embedding or querying the foreign-space index.
        index = MockOutcomeIndex(count=100, neighbors=[_neighbor("gpt4", outcome=True)])
        pool = FakeModelPool("qwen3.7-plus", "gpt4")
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
            trust_neighbors=False,
        )
        model, reason, provenance = engine.decide("session-1", "some prompt")
        assert model == "qwen3.7-plus"
        assert reason == "stale_embedding_space"
        assert provenance["selection_rule_used"] == "stale_embedding_space"
        # No embed, no index query — the whole point of refusing a foreign space.
        assert embedder.call_count == 0
        assert index.queries == []

    def test_trusted_space_still_queries(self):
        index = MockOutcomeIndex(count=100, neighbors=[_neighbor("gpt4", outcome=True)])
        pool = FakeModelPool("qwen3.7-plus", "gpt4")
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            trust_neighbors=True,
        )
        engine.decide("session-1", "some prompt")
        assert len(index.queries) == 1


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
        # Cold-start still embeds: its whole purpose is gathering the kNN bootstrap
        # corpus, so the session must be indexable even though routing ignores neighbors.
        assert embedder.call_count == 1
        assert engine.cached_embedding("session-1") is not None
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


class TestStrategyAndNeighborK:
    def test_injected_strategy_overrides_selection(self):
        # A frontier-favouring neighbourhood, but always_cheap must ignore it.
        neighbors = [_neighbor("model-b", outcome=True, cost=9.0) for _ in range(5)]
        index = MockOutcomeIndex(count=50, neighbors=neighbors)
        pool = FakeModelPool("model-a", "model-b")  # both land in the cheap tier
        engine = RouterEngine(
            model_pool=pool,
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            strategy=AlwaysCheapStrategy(),
        )
        model, reason, _ = engine.decide("session-1", "prompt")
        assert model == "model-a"
        assert reason == "always_cheap"

    def test_fixed_strategy_bypasses_cold_start_and_query(self):
        # Empty store (cold-start would be active for knn); a fixed strategy must still
        # route to its model and never touch the embedder or the index.
        index = MockOutcomeIndex(count=0, neighbors=[])
        embedder = RecordingEmbedder()
        engine = RouterEngine(
            model_pool=FakeModelPool("model-a", "model-b"),
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=embedder,
            strategy=AlwaysCheapStrategy(),
        )
        model, reason, _ = engine.decide("session-1", "prompt")
        assert reason == "always_cheap"
        assert model == "model-a"
        assert embedder.call_count == 0  # no embedding
        assert index.queries == []  # no kNN query

    def test_neighbor_k_drives_query(self):
        index = MockOutcomeIndex(count=50, neighbors=[])
        engine = RouterEngine(
            model_pool=FakeModelPool("model-a"),
            session_manager=MagicMock(),
            outcome_index=index,
            embedder=RecordingEmbedder(),
            neighbor_k=7,
        )
        engine.decide("session-1", "prompt")
        assert index.queries[0][1] == 7


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
