"""Store + adapter effective-sample-size gate and offline per-model prior aggregates.

Rows are inserted with precomputed vectors (no embedder/network); the adapter reuses the
same confidence weighting the neighbour path applies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.router.selection import effective_sample_size


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:  # type: ignore[misc]
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _vec(seed: float, dim: int = 8) -> np.ndarray:
    return np.full(dim, seed, dtype=np.float32)


def _add_tier2(
    store: OutcomeStore, sid: str, *, model: str, outcome: str, confidence: float
) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=f"p {sid}",
        embedding=_vec(hash(sid) % 100 / 100.0 + 0.01),
        model_chosen=model,
        cost=1.0,
        cache_stats={},
        duration=1.0,
    )
    store.append_outcome_event(
        OutcomeEvent(
            session_id=sid,
            tier=2,
            source="auto_tier2",
            outcome=outcome,
            confidence=confidence,
            run_signature=f"sig-{sid}",
        )
    )


class TestEffectiveSampleSize:
    def test_uniform_confidence_equals_raw_count(self, store: OutcomeStore) -> None:
        # Backward-compat: confidence 1.0 everywhere ⇒ nₑ == count_verified_outcomes.
        adapter = OutcomeIndexAdapter(store)
        for i in range(5):
            _add_tier2(store, f"s{i}", model="m", outcome="success", confidence=1.0)
        assert store.count_verified_outcomes() == 5
        assert adapter.effective_tier2() == pytest.approx(5.0)
        assert adapter.effective_labeled() == pytest.approx(5.0)

    def test_mixed_confidence_below_raw_count(self, store: OutcomeStore) -> None:
        adapter = OutcomeIndexAdapter(store)
        confidences = [1.0, 0.1, 0.1, 0.1, 0.1]
        for i, c in enumerate(confidences):
            _add_tier2(store, f"s{i}", model="m", outcome="success", confidence=c)
        expected = effective_sample_size(confidences)
        assert adapter.effective_tier2() == pytest.approx(expected)
        assert adapter.effective_tier2() < store.count_verified_outcomes()


class TestModelPriors:
    def test_per_model_weighted_success_rate_and_strength(self, store: OutcomeStore) -> None:
        adapter = OutcomeIndexAdapter(store)
        # model "good": 3 passes / 1 fail, uniform confidence ⇒ est 0.75, strength nₑ = 4.
        _add_tier2(store, "g0", model="good", outcome="success", confidence=1.0)
        _add_tier2(store, "g1", model="good", outcome="success", confidence=1.0)
        _add_tier2(store, "g2", model="good", outcome="weak_success", confidence=1.0)
        _add_tier2(store, "g3", model="good", outcome="failure", confidence=1.0)
        # model "bad": 1 pass / 1 fail ⇒ est 0.5.
        _add_tier2(store, "b0", model="bad", outcome="success", confidence=1.0)
        _add_tier2(store, "b1", model="bad", outcome="failure", confidence=1.0)

        priors = adapter.model_priors()
        assert priors["good"][0] == pytest.approx(0.75)
        assert priors["good"][1] == pytest.approx(4.0)
        assert priors["bad"][0] == pytest.approx(0.5)
        assert priors["bad"][1] == pytest.approx(2.0)

    def test_empty_store_yields_no_priors(self, store: OutcomeStore) -> None:
        assert OutcomeIndexAdapter(store).model_priors() == {}
