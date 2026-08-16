"""Startup must not load the ~600MB embedding model for a strategy that never embeds."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult, SelectionRule
from shunt.router.strategies.fixed import AlwaysCheapStrategy, AlwaysFrontierStrategy
from shunt.router.strategies.knn import KnnStrategy

from .conftest import FakeModelPool


class WarmRecordingEmbedder:
    """EmbedderProtocol fake that records warm()/embed() instead of loading ONNX."""

    def __init__(self) -> None:
        self.warm_calls = 0
        self.embed_calls = 0

    def warm(self) -> None:
        self.warm_calls += 1

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        self.embed_calls += 1
        return np.zeros(768, dtype=np.float32)


class StubIndex:
    def count_labeled(self) -> int:
        return 0

    def count_total_labeled(self) -> int:
        return 0

    def effective_labeled(self) -> float:
        return 0.0

    def effective_tier2(self) -> float:
        return 0.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: Any, k: int = 20) -> list[NeighborResult]:
        return []


def _engine(strategy: Any, embedder: WarmRecordingEmbedder) -> RouterEngine:
    return RouterEngine(
        model_pool=FakeModelPool("cheap", "frontier"),
        session_manager=MagicMock(),
        outcome_index=StubIndex(),
        embedder=embedder,
        strategy=strategy,
    )


class TestEngineWarmIsStrategyAware:
    def test_always_cheap_never_loads_the_embedding_model(self) -> None:
        embedder = WarmRecordingEmbedder()
        engine = _engine(AlwaysCheapStrategy(), embedder)

        engine.warm()

        assert embedder.warm_calls == 0
        assert engine.needs_embeddings is False

    def test_always_frontier_never_loads_the_embedding_model(self) -> None:
        embedder = WarmRecordingEmbedder()

        _engine(AlwaysFrontierStrategy(), embedder).warm()

        assert embedder.warm_calls == 0

    def test_knn_still_warms_exactly_as_before(self) -> None:
        embedder = WarmRecordingEmbedder()
        engine = _engine(KnnStrategy(SelectionRule()), embedder)

        engine.warm()

        assert embedder.warm_calls == 1
        assert engine.needs_embeddings is True
