"""Session-persist wiring for decision-time provenance (cost_known and model fingerprint).

Drives ``_store_session_with_provenance`` with a real store + engine-less ProxyRouter,
asserting the schema-v2 provenance columns are threaded through, not left at their defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shunt.db.store import OutcomeStore
from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter
from shunt.proxy.server import _store_session_with_provenance
from shunt.router.provenance import build_provenance
from shunt.session import Session, SessionManager

# A model that exists in the shipped registry, so the fingerprint resolves to a real tag.
_REGISTRY_MODEL = "qwen3.7-plus"
_EXPECTED_FINGERPRINT = "alibaba/qwen3.7-plus@qwen3.7-plus"


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def router() -> ProxyRouter:
    return ProxyRouter(model_pool=ModelPool(), session_manager=SessionManager(), retry_count=1)


@pytest.fixture
def session() -> Session:
    s = Session(session_id="sess-1", tool_identity="t", start_time=_now())
    s.metadata["last_prompt"] = "hi"
    return s


def _now():  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ── cost_known is threaded from the cost_unreported signal ────────────────────
def test_unreported_cost_persists_cost_known_false(
    store: OutcomeStore, router: ProxyRouter, session: Session
) -> None:
    session.total_cost = 0.0
    session.metadata["cost_unreported"] = True  # provider reported no usage.cost
    _store_session_with_provenance(store, router, session, _REGISTRY_MODEL, "cold_start")

    row = store.get_session("sess-1")
    assert row is not None
    # UNKNOWN, distinct from a real cost=0.0 with cost_known=1.
    assert row["cost_known"] == 0
    assert row["cost"] == 0.0


def test_reported_cost_persists_cost_known_true(
    store: OutcomeStore, router: ProxyRouter, session: Session
) -> None:
    session.total_cost = 0.0  # a genuine, reported zero — no cost_unreported flag
    _store_session_with_provenance(store, router, session, _REGISTRY_MODEL, "cold_start")

    row = store.get_session("sess-1")
    assert row is not None
    assert row["cost_known"] == 1


# ── every persisted row carries non-null propensity + fingerprint ─────────────
def test_propensity_and_fingerprint_from_decision_provenance(
    store: OutcomeStore, router: ProxyRouter, session: Session
) -> None:
    session.decision_provenance = build_provenance(
        model_chosen=_REGISTRY_MODEL,
        selection_rule_used="cheapest_above_threshold",
        router_propensity=0.3,
    )
    _store_session_with_provenance(store, router, session, _REGISTRY_MODEL, "knn")

    row = store.get_session("sess-1")
    assert row is not None
    assert row["selection_propensity"] == pytest.approx(0.3)
    assert row["model_fingerprint"] == _EXPECTED_FINGERPRINT


def test_synthesized_provenance_still_carries_non_null_values(
    store: OutcomeStore, router: ProxyRouter, session: Session
) -> None:
    # Engine-less path: no decision_provenance → one is synthesized (propensity 1.0),
    # and the fingerprint still resolves from the model name — NO null rows allowed.
    session.decision_provenance = None
    _store_session_with_provenance(store, router, session, _REGISTRY_MODEL, "always-cheap")

    row = store.get_session("sess-1")
    assert row is not None
    assert row["selection_propensity"] is not None
    assert row["selection_propensity"] == pytest.approx(1.0)
    assert row["model_fingerprint"] == _EXPECTED_FINGERPRINT


def test_fingerprint_is_none_for_unknown_model(
    store: OutcomeStore, router: ProxyRouter, session: Session
) -> None:
    # A model not in the registry cannot be fingerprinted — None, never a fabricated tag.
    _store_session_with_provenance(store, router, session, "not-a-real-model", "knn")

    row = store.get_session("sess-1")
    assert row is not None
    assert row["model_fingerprint"] is None
