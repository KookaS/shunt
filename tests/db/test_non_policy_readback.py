"""B7: an auto-escalated session's outcome is real neighbour signal but not a policy prior."""

# An imposed escalation is not a router *choice*, so it must not seed a Thompson prior — but
# the escalated model's verified pass/fail IS real capability signal, so it stays a kNN
# neighbour and the selection rule can converge. Precomputed vectors, real OutcomeStore.

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore


@pytest.fixture
def store(tmp_path: Any) -> Any:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _vec(seed: float, dim: int = 8) -> np.ndarray:
    return np.full(dim, seed, dtype=np.float32)


def _add(
    store: OutcomeStore,
    sid: str,
    *,
    vec: np.ndarray,
    auto_escalated: bool,
) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=f"p {sid}",
        embedding=vec,
        model_chosen="cheap",
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance={"auto_escalated": True} if auto_escalated else {"downshift": False},
    )
    store.store_outcome(
        session_id=sid,
        tier1_outcome="success",
        tier1_confidence=0.7,
        tier2_outcome="success",
        tier2_confidence=0.95,
        aggregated_confidence=0.9,
    )


def test_escalated_session_kept_as_knn_neighbour(store: OutcomeStore) -> None:
    _add(store, "normal", vec=_vec(0.1), auto_escalated=False)
    _add(store, "escalated", vec=_vec(0.11), auto_escalated=True)
    adapter = OutcomeIndexAdapter(store)
    ids = {n.session_id for n in adapter.query(_vec(0.1), k=10)}
    assert "normal" in ids
    assert "escalated" in ids  # escalated outcome is real capability signal for the selection rule


def test_escalated_success_is_visible_but_seeds_no_prior(store: OutcomeStore) -> None:
    # A hard neighbourhood where the ONLY verified evidence is an escalated SUCCESS on "cheap":
    # the selection rule sees it (routes toward the winning model), but it seeds no Thompson prior.
    _add(store, "escalated", vec=_vec(0.5), auto_escalated=True)
    adapter = OutcomeIndexAdapter(store)
    neighbours = adapter.query(_vec(0.5), k=10)
    assert any(n.session_id == "escalated" and n.outcome for n in neighbours)  # visible + success
    assert "cheap" not in adapter.model_priors()  # but NOT a policy-attributable prior


def test_escalated_session_excluded_from_model_priors(store: OutcomeStore) -> None:
    # Only the escalated session exists for "cheap" → its prior must not be seeded from it.
    _add(store, "escalated", vec=_vec(0.2), auto_escalated=True)
    adapter = OutcomeIndexAdapter(store)
    assert "cheap" not in adapter.model_priors()


def test_normal_session_still_seeds_priors(store: OutcomeStore) -> None:
    _add(store, "normal", vec=_vec(0.3), auto_escalated=False)
    adapter = OutcomeIndexAdapter(store)
    assert "cheap" in adapter.model_priors()  # control: a genuine policy pick is still counted
