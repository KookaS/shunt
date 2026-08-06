"""A fixed strategy must not spawn the background warm thread, and must say why."""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from shunt.proxy.server import _warm_embedder_in_background
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult, SelectionRule
from shunt.router.strategies.fixed import AlwaysCheapStrategy
from shunt.router.strategies.knn import KnnStrategy

_WARM_THREAD_NAME = "shunt-embedder-warm"


class WarmRecordingEmbedder:
    def __init__(self) -> None:
        self.warm_calls = 0

    def warm(self) -> None:
        self.warm_calls += 1

    def embed(self, text: str) -> npt.NDArray[np.float32]:
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
        model_pool=MagicMock(),
        session_manager=MagicMock(),
        outcome_index=StubIndex(),
        embedder=embedder,
        strategy=strategy,
    )


def _join_warm_threads() -> None:
    for thread in threading.enumerate():
        if thread.name == _WARM_THREAD_NAME:
            thread.join(timeout=5)


def test_fixed_strategy_skips_the_warm_thread_and_discloses_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    embedder = WarmRecordingEmbedder()
    with caplog.at_level(logging.INFO, logger="shunt.proxy.server"):
        _warm_embedder_in_background(_engine(AlwaysCheapStrategy(), embedder))
    _join_warm_threads()

    assert embedder.warm_calls == 0
    assert "embedding model NOT loaded" in caplog.text
    assert "does not consult neighbours" in caplog.text
    assert "Embedding model ready" not in caplog.text


def test_knn_strategy_still_warms_in_the_background(
    caplog: pytest.LogCaptureFixture,
) -> None:
    embedder = WarmRecordingEmbedder()
    with caplog.at_level(logging.INFO, logger="shunt.proxy.server"):
        _warm_embedder_in_background(_engine(KnnStrategy(SelectionRule()), embedder))
        _join_warm_threads()

    assert embedder.warm_calls == 1
    assert "Embedding model ready" in caplog.text
