"""An escalation to a model with no neighbourhood history must not be booked as free."""

from __future__ import annotations

from unittest.mock import MagicMock

from shunt.router.engine import RouterEngine
from shunt.router.policy import ExplorationPolicy

from .conftest import FakeModelPool
from .test_engine import MockOutcomeIndex, RecordingEmbedder, _neighbor


def _engine(neighbors: list[object]) -> RouterEngine:
    return RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b", "frontier"),
        session_manager=MagicMock(),
        outcome_index=MockOutcomeIndex(count=50, neighbors=neighbors),
        embedder=RecordingEmbedder(),
        cold_start_threshold=20,
        exploration=ExplorationPolicy(enabled=False),
    )


def test_known_model_uses_its_own_weighted_cost() -> None:
    neighbors = [_neighbor("model-a", outcome=True, cost=2.0) for _ in range(4)]

    assert _engine(neighbors)._representative_cost(neighbors, "model-a") == 2.0


def test_untested_model_falls_back_to_the_priciest_known_cost() -> None:
    # "no history" is exactly the escalation case (exploration_untested / safe_fallback) —
    # the most expensive routes. Returning 0.0 booked those as free, which both understated
    # spend and let a never-seen model read as the cheapest option.
    neighbors = [
        _neighbor("model-a", outcome=True, cost=1.0),
        _neighbor("model-b", outcome=True, cost=7.5),
    ]

    assert _engine(neighbors)._representative_cost(neighbors, "frontier") == 7.5


def test_no_neighbours_at_all_is_still_zero() -> None:
    assert _engine([])._representative_cost([], "frontier") == 0.0
