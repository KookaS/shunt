"""Unit tests for OutcomeIndexAdapter — the store→engine read-back seam.

No network, no GPU: rows are inserted with precomputed vectors, never the real embedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore


@pytest.fixture
def store(tmp_path) -> OutcomeStore:  # type: ignore[no-untyped-def]
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _vec(seed: float, dim: int = 8) -> np.ndarray:
    return np.full(dim, seed, dtype=np.float32)


def _add_labeled(  # noqa: PLR0913 (keyword-only test-row builder, one arg per column)
    store: OutcomeStore,
    sid: str,
    *,
    model: str,
    vec: np.ndarray,
    cost: float,
    tier1: str = "success",
    tier1_conf: float = 0.7,
    tier2: str | None = None,
    tier2_conf: float | None = None,
    aggregated: float = 0.9,
) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=f"prompt {sid}",
        embedding=vec,
        model_chosen=model,
        cost=cost,
        cache_stats={},
        duration=1.0,
    )
    store.store_outcome(
        session_id=sid,
        tier1_outcome=tier1,
        tier1_confidence=tier1_conf,
        tier2_outcome=tier2,
        tier2_confidence=tier2_conf,
        aggregated_confidence=aggregated,
    )


def test_counts_empty_store(store: OutcomeStore) -> None:
    adapter = OutcomeIndexAdapter(store)
    assert adapter.count_total_labeled() == 0
    assert adapter.count_labeled() == 0


def test_count_total_vs_verified(store: OutcomeStore) -> None:
    _add_labeled(store, "s1", model="cheap", vec=_vec(0.1), cost=1.0)  # tier1 only
    _add_labeled(
        store, "s2", model="mid", vec=_vec(0.2), cost=2.0, tier2="success", tier2_conf=0.95
    )
    adapter = OutcomeIndexAdapter(store)
    assert adapter.count_total_labeled() == 2
    assert adapter.count_labeled() == 1  # only s2 is Tier-2 verified


def test_query_returns_labeled_neighbors(store: OutcomeStore) -> None:
    _add_labeled(store, "s1", model="cheap", vec=_vec(0.5), cost=1.5, aggregated=0.8)
    adapter = OutcomeIndexAdapter(store)
    neighbors = adapter.query(_vec(0.5), k=5)
    assert len(neighbors) == 1
    n = neighbors[0]
    assert n.model == "cheap"
    assert n.outcome is True
    assert n.cost == 1.5
    assert n.verification_confidence == 0.8
    assert n.session_id == "s1"
    assert n.distance >= 0.0


def test_query_maps_failure_to_false(store: OutcomeStore) -> None:
    _add_labeled(store, "s1", model="cheap", vec=_vec(0.5), cost=1.0, tier1="failure")
    adapter = OutcomeIndexAdapter(store)
    (n,) = adapter.query(_vec(0.5), k=5)
    assert n.outcome is False


def test_tier2_label_overrides_tier1(store: OutcomeStore) -> None:
    # Tier-1 said success, but the verified Tier-2 label says failure — Tier-2 wins.
    _add_labeled(
        store,
        "s1",
        model="cheap",
        vec=_vec(0.5),
        cost=1.0,
        tier1="success",
        tier2="failure",
        tier2_conf=0.9,
    )
    adapter = OutcomeIndexAdapter(store)
    (n,) = adapter.query(_vec(0.5), k=5)
    assert n.outcome is False


def test_query_skips_sessions_without_outcome(store: OutcomeStore) -> None:
    # An embedded session with no outcome row contributes no learning signal.
    store.store_session(
        session_id="unlabeled",
        prompt_text="p",
        embedding=_vec(0.5),
        model_chosen="cheap",
        cost=1.0,
        cache_stats={},
        duration=1.0,
    )
    adapter = OutcomeIndexAdapter(store)
    assert adapter.query(_vec(0.5), k=5) == []


def test_explicit_zero_confidence_is_not_upgraded(store: OutcomeStore) -> None:
    # An explicitly stored 0.0 means "no confidence in this label"; truthiness testing
    # used to silently upgrade it to the fallback, over-trusting the neighbor.
    _add_labeled(
        store, "s1", model="cheap", vec=_vec(0.5), cost=1.0, tier1_conf=0.0, aggregated=0.0
    )
    adapter = OutcomeIndexAdapter(store, default_confidence=0.33)
    (n,) = adapter.query(_vec(0.5), k=5)
    assert n.verification_confidence == 0.0


def test_confidence_falls_back_to_default_when_absent(store: OutcomeStore) -> None:
    # The default applies only when the row carries no confidence value at all —
    # `aggregated_confidence` is NOT NULL with a 0.0 store default, so zero there
    # still falls through.
    adapter = OutcomeIndexAdapter(store, default_confidence=0.33)
    assert adapter._resolve_confidence({"aggregated_confidence": 0.0}) == 0.33
    assert adapter._resolve_confidence({}) == 0.33
    partial = {"aggregated_confidence": 0.0, "tier2_confidence": 0.7}
    assert adapter._resolve_confidence(partial) == 0.7


def test_query_empty_store_returns_empty(store: OutcomeStore) -> None:
    adapter = OutcomeIndexAdapter(store)
    assert adapter.query(_vec(0.1), k=5) == []


def test_unembedded_outcomes_do_not_end_cold_start(tmp_path: Path) -> None:
    """A flagged session that was never embedded can never be anyone's neighbour."""
    # Any fixed strategy returns before embedding, so `store_session` skips the index.
    # Counting those outcomes ended cold start while the kNN index was still empty.
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    for i in range(30):
        store.store_session(
            session_id=f"s{i}",
            prompt_text="p",
            embedding=None,
            model_chosen="m",
            cost=0.0,
            cache_stats={},
            duration=1.0,
        )
        store.store_outcome(
            session_id=f"s{i}",
            tier1_outcome=True,
            tier1_confidence=1.0,
            tier2_outcome=True,
        )

    assert store.count_outcomes() == 0
    assert store.count_verified_outcomes() == 0


# ── an UNKNOWN cost surfaces as +inf, never a real 0.0 that sorts cheapest ──
def test_unknown_cost_neighbor_surfaces_as_inf(store: OutcomeStore) -> None:
    import math

    from shunt.db.store import SessionProvenance

    store.store_session(
        session_id="u",
        prompt_text="p",
        embedding=_vec(0.1),
        model_chosen="frontier",
        cost=0.0,  # stored zero, but UNKNOWN — must NOT read as cheapest
        cache_stats={},
        duration=1.0,
        provenance=SessionProvenance(cost_known=False),
    )
    store.store_outcome("u", "success", 0.9, aggregated_confidence=0.9)

    neighbors = OutcomeIndexAdapter(store).query(_vec(0.1), k=5)
    assert len(neighbors) == 1
    assert math.isinf(neighbors[0].cost)


def test_known_zero_cost_neighbor_stays_zero(store: OutcomeStore) -> None:
    # A genuinely reported 0.0 (fully cached / free) stays 0.0 — only UNKNOWN maps to inf.
    store.store_session(
        session_id="z",
        prompt_text="p",
        embedding=_vec(0.2),
        model_chosen="cheap",
        cost=0.0,
        cache_stats={},
        duration=1.0,
    )
    store.store_outcome("z", "success", 0.9, aggregated_confidence=0.9)

    neighbors = OutcomeIndexAdapter(store).query(_vec(0.2), k=5)
    assert len(neighbors) == 1
    assert neighbors[0].cost == 0.0
