"""Integration tests for the RouterEngine → ProxyRouter live wiring (Step 6).

All upstream calls are mocked — no live/paid API traffic. The embedder is faked so
no ONNX model loads; the kNN path is driven by an in-memory fake OutcomeIndex.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore
from shunt.models import TIER_ORDER
from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter
from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
_COLD_START_MODEL = "qwen3.7-plus"
_KNN_MODEL = "deepseek-v4-flash"


class FakeEmbedder(Embedder):
    """Returns a fixed vector without loading a real model."""

    def __init__(self) -> None:
        super().__init__(lazy=True)
        self.call_count = 0

    def embed(self, text: str) -> Any:
        self.call_count += 1
        return np.full(768, 0.1, dtype=np.float32)


class FakeOutcomeIndex:
    """In-memory OutcomeIndex returning canned neighbors past cold-start."""

    def __init__(self, count: int, neighbors: list[NeighborResult]) -> None:
        self._count = count
        self._neighbors = neighbors
        self.queries: list[Any] = []

    def count_labeled(self) -> int:
        return self._count

    def count_total_labeled(self) -> int:
        return self._count

    def effective_labeled(self) -> float:
        return float(self._count)  # uniform weights ⇒ nₑ == raw count

    def effective_tier2(self) -> float:
        return float(self._count)

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: Any, k: int = 20) -> list[NeighborResult]:
        self.queries.append((embedding, k))
        return self._neighbors


def _dummy_response(model: str) -> MagicMock:
    resp = MagicMock()
    resp.id = "cmpl-dummy"
    resp.model = model
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 10
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message.content = "dummy"
    choice.message.role = "assistant"
    resp.choices = [choice]
    resp.model_dump.return_value = {
        "id": "cmpl-dummy",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        "model": model,
    }
    return resp


def _knn_neighbors() -> list[NeighborResult]:
    return [
        NeighborResult(
            model=_KNN_MODEL,
            outcome=True,
            cost=1.0,
            verification_confidence=0.9,
            distance=0.1,
            session_id=f"s{i}",
        )
        for i in range(3)
    ]


@pytest.fixture
def model_pool() -> ModelPool:
    return ModelPool()


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(inactivity_timeout=900, grace_period=120)


@pytest.fixture
def session(session_manager: SessionManager) -> Session:
    return session_manager.create_session("test-tool")


def _make_router(
    model_pool: ModelPool,
    session_manager: SessionManager,
    outcome_index: Any,
    *,
    exploration: Any = None,
) -> ProxyRouter:
    engine = RouterEngine(
        model_pool=model_pool,
        session_manager=session_manager,
        outcome_index=outcome_index,
        embedder=FakeEmbedder(),
        exploration=exploration,
    )
    return ProxyRouter(
        model_pool=model_pool,
        session_manager=session_manager,
        retry_count=2,
        engine=engine,
    )


# ── (a) empty store → engine cold-starts, not the hard-code ──────────────────


@pytest.mark.asyncio
async def test_empty_store_routes_via_engine_cold_start(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "empty.db"))
    try:
        router = _make_router(model_pool, session_manager, OutcomeIndexAdapter(store))
        body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
        with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _dummy_response(_COLD_START_MODEL)
            _, model_name, _ = await router.route_chat_completion(body, session)
        assert model_name == _COLD_START_MODEL
        # reason from the ENGINE ("cold_start"), never the hard-code fallback source.
        assert session.metadata["model_source"] == "cold_start"
        assert mock_acompletion.call_args.args[0].name == _COLD_START_MODEL
    finally:
        store.close()


# ── (b) neighbors present → engine takes the kNN path ────────────────────────


@pytest.mark.asyncio
async def test_knn_path_routes_to_engine_choice(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index)
    body = {"messages": [{"role": "user", "content": "Refactor this"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _dummy_response(_KNN_MODEL)
        _, model_name, _ = await router.route_chat_completion(body, session)
    assert model_name == _KNN_MODEL
    assert session.metadata["model_source"] == "cheapest_above_threshold"
    # The model actually invoked upstream matches the engine's decision.
    assert mock_acompletion.call_args.args[0].name == _KNN_MODEL


@pytest.mark.asyncio
async def test_decision_locked_once_per_session(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index)
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _dummy_response(_KNN_MODEL)
        await router.route_chat_completion(body, session)
        await router.route_chat_completion(body, session)
    # Cache-safety: exactly one routing decision (one index query) for the session.
    assert len(index.queries) == 1


# ── (c) exploration disabled → deterministic kNN, unchanged behaviour ─────────


@pytest.mark.asyncio
async def test_exploration_disabled_is_plain_knn(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index, exploration=None)
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _dummy_response(_KNN_MODEL)
        _, model_name, _ = await router.route_chat_completion(body, session)
    assert model_name == _KNN_MODEL
    # No exploration draw — the reason is a pure selection-rule verdict.
    assert "exploration" not in session.metadata["model_source"]
    assert session.metadata["model_source"] == "cheapest_above_threshold"


def test_store_persists_engine_provenance_and_embedding(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    # Regression: the server must persist the engine's REAL provenance/reason and the
    # embedding it already computed — not a synthetic rebuild with embedding=None.
    from shunt.proxy.server import _store_session_with_provenance

    store = OutcomeStore(db_path=str(tmp_path / "prov.db"))
    try:
        index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
        router = _make_router(model_pool, session_manager, index)
        router._get_or_lock_model(session, "refactor this")  # runs engine.decide
        _store_session_with_provenance(
            store, router, session, session.model_chosen or "", "model-name-not-reason"
        )
        stored = store.get_session(session.session_id)
        assert stored is not None
        prov = json.loads(stored["decision_provenance"])
        # Engine's real reason survives, not the model-name passed as `reason`.
        assert prov["selection_rule_used"] == "cheapest_above_threshold"
        assert prov["candidate_model_scores"]  # rich provenance preserved
        # Embedding persisted → the session is queryable by the kNN read-back.
        assert stored["embedding_blob"] is not None
    finally:
        store.close()


# ── (d) server _build_engine honors router.strategy + KnnPolicy knobs ────────


def test_build_engine_honors_always_cheap_strategy(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    from shunt.proxy.server import _build_engine
    from shunt.router.policy import RouterPolicy

    store = OutcomeStore(db_path=str(tmp_path / "cheap.db"))
    try:
        policy = RouterPolicy(strategy="always_cheap")
        engine = _build_engine(model_pool, session_manager, store, policy)
        index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
        engine._outcome_index = index
        engine._embedder = FakeEmbedder()
        model, reason, _ = engine.decide(session.session_id, "refactor this")
        # always_cheap ignores neighbors and picks the lowest-tier model of the pool.
        assert reason == "always_cheap"
        assert model == model_pool.get_tier_models(TIER_ORDER[0])[0].name
    finally:
        store.close()


def test_always_frontier_honored_on_empty_store(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    # Regression: with an empty store, cold-start would be active — but a fixed strategy
    # must route to its model from the first turn, not the cheap cold-start default.
    from shunt.proxy.server import _build_engine
    from shunt.router.policy import RouterPolicy

    store = OutcomeStore(db_path=str(tmp_path / "frontier.db"))
    try:
        policy = RouterPolicy(strategy="always_frontier")
        engine = _build_engine(model_pool, session_manager, store, policy)
        model, reason, _ = engine.decide(session.session_id, "hard task")
        assert reason == "always_frontier"
        assert model != _COLD_START_MODEL
    finally:
        store.close()


def test_build_engine_passes_knn_policy_k_to_query(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    from shunt.proxy.server import _build_engine
    from shunt.router.policy import KnnPolicy, RouterPolicy

    store = OutcomeStore(db_path=str(tmp_path / "k.db"))
    try:
        policy = RouterPolicy(policy=KnnPolicy(k=9))
        engine = _build_engine(model_pool, session_manager, store, policy)
        index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
        engine._outcome_index = index
        engine._embedder = FakeEmbedder()
        engine.decide(session.session_id, "refactor this")
        # KnnPolicy.k (not a hardcoded 20) reaches the outcome-index query.
        assert index.queries[0][1] == 9
    finally:
        store.close()


def test_build_engine_disables_exploration_for_fixed_strategy(
    model_pool: ModelPool,
    session_manager: SessionManager,
    tmp_path: Any,
) -> None:
    from shunt.proxy.server import _build_engine
    from shunt.router.policy import ExplorationPolicy, RouterPolicy

    store = OutcomeStore(db_path=str(tmp_path / "expl.db"))
    try:
        # Exploration enabled in config, but a fixed strategy must not explore.
        policy = RouterPolicy(strategy="always_cheap", exploration=ExplorationPolicy(enabled=True))
        engine = _build_engine(model_pool, session_manager, store, policy)
        assert engine._sampler is None
    finally:
        store.close()


def test_no_engine_falls_back_to_hardcode(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    # Engine-less construction preserves the legacy hard-coded default (existing tests).
    router = ProxyRouter(model_pool=model_pool, session_manager=session_manager)
    model = router._get_or_lock_model(session, "prompt")
    assert model == _COLD_START_MODEL
    assert session.metadata["model_source"] == "cold-start-always-cheap"


# ── Regression: cost, cold-start indexing, registry parity, decision locking ──


class _Usage:
    """Minimal OpenAI-shaped usage object carrying a provider-reported charge."""

    def __init__(self, cost: float | None) -> None:
        self.prompt_tokens = 5
        self.completion_tokens = 10
        self.prompt_tokens_details = None
        if cost is not None:
            self.cost = cost


def _stream_chunk(model: str, usage: Any) -> Any:
    """One OpenAI-shaped streaming chunk; usage rides only on the final one."""
    chunk = MagicMock()
    chunk.model = model
    chunk.usage = usage
    choice = MagicMock()
    choice.finish_reason = "stop" if usage is not None else None
    choice.delta.model_dump.return_value = {"role": "assistant", "content": "x"}
    chunk.choices = [choice]
    chunk.model_dump.return_value = {"model": model, "choices": [{"index": 0}]}
    return chunk


def _priced_response(model: str, cost: float | None) -> Any:
    resp = _dummy_response(model)
    resp.usage = _Usage(cost)
    return resp


def _run_one_session(
    router: ProxyRouter,
    store: OutcomeStore,
    session: Session,
    model: str,
    cost: float | None,
) -> None:
    """Route one non-streaming request and persist the session, as the server does."""
    from shunt.proxy.server import _store_session_with_provenance

    body = {"messages": [{"role": "user", "content": "do the thing"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _priced_response(model, cost)
        _, model_name, reason = asyncio.run(router.route_chat_completion(body, session))
    _store_session_with_provenance(store, router, session, model_name, reason)


def test_persisted_session_records_provider_reported_cost(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    # Regression: `total_cost` was declared but never assigned, so every persisted row
    # (and therefore every NeighborResult.cost) was structurally 0.0.
    store = OutcomeStore(db_path=str(tmp_path / "cost.db"))
    try:
        index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
        router = _make_router(model_pool, session_manager, index)
        _run_one_session(router, store, session, _KNN_MODEL, 0.0003504)
        stored = store.get_session(session.session_id)
        assert stored is not None
        assert stored["cost"] > 0
        assert stored["cost"] == pytest.approx(0.0003504)
    finally:
        store.close()


def test_cost_unreported_leaves_total_at_zero(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    # A provider that omits usage.cost must not be back-filled with a guessed price.
    index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index)
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _priced_response(_KNN_MODEL, None)
        asyncio.run(router.route_chat_completion(body, session))
    assert session.total_cost == 0.0
    assert session.metadata["cost_unreported"] is True


def test_cold_start_sessions_are_embedded_and_indexed(
    model_pool: ModelPool,
    session_manager: SessionManager,
    tmp_path: Any,
) -> None:
    # Regression: cold-start returned before embedding, so the corpus it exists to
    # gather was persisted with NULL embeddings and the HNSW index stayed empty.
    store = OutcomeStore(db_path=str(tmp_path / "cold.db"))
    try:
        router = _make_router(model_pool, session_manager, OutcomeIndexAdapter(store))
        n_sessions = 5
        session_ids: list[str] = []
        for i in range(n_sessions):
            session = session_manager.create_session(f"tool-{i}")
            _run_one_session(router, store, session, _COLD_START_MODEL, 0.001)
            assert session.metadata["model_source"] == "cold_start"
            session_ids.append(session.session_id)
        # The corpus is what cold start exists to gather, so the embeddings must be
        # durable...
        assert len(store.get_all_embeddings()) == n_sessions
        # ...but an unlabeled session is not yet a usable neighbour, so it stays out of
        # the index until an outcome arrives. Indexing it earlier is what let ordinary
        # traffic crowd the labeled sessions out of the k nearest.
        assert store.get_stats()["index_size"] == 0

        for session_id in session_ids:
            store.store_outcome(session_id, tier1_outcome="success", tier1_confidence=0.9)
        assert store.get_stats()["index_size"] == n_sessions
    finally:
        store.close()


class _FpEmbedder(FakeEmbedder):
    """FakeEmbedder carrying a chosen fingerprint (no ONNX load)."""

    def __init__(self, repo: str) -> None:
        super().__init__()
        self._fp = {"repo": repo, "dim": 768, "max_chars": 4000, "revision": None}

    def fingerprint(self) -> dict[str, Any]:
        return self._fp


def test_missing_stored_fingerprint_is_adopted_and_trusted(tmp_path: Any) -> None:
    from shunt.proxy.server import _resolve_embedding_trust

    store = OutcomeStore(db_path=str(tmp_path / "fp1.db"))
    try:
        embedder = _FpEmbedder("jinaai/jina-embeddings-v2-base-code")
        assert store.load_embedding_fingerprint() is None
        trust = _resolve_embedding_trust(embedder, store)  # type: ignore[arg-type]
        assert trust is True
        # Adopted: the current config is now the stored space.
        assert store.load_embedding_fingerprint() == embedder.fingerprint()
    finally:
        store.close()


def test_matching_fingerprint_trusts_neighbours(tmp_path: Any) -> None:
    from shunt.proxy.server import _resolve_embedding_trust

    store = OutcomeStore(db_path=str(tmp_path / "fp2.db"))
    try:
        embedder = _FpEmbedder("jinaai/jina-embeddings-v2-base-code")
        store.save_embedding_fingerprint(embedder.fingerprint())
        assert _resolve_embedding_trust(embedder, store) is True  # type: ignore[arg-type]
    finally:
        store.close()


def test_mismatched_fingerprint_refuses_neighbours(tmp_path: Any) -> None:
    from shunt.proxy.server import _resolve_embedding_trust

    store = OutcomeStore(db_path=str(tmp_path / "fp3.db"))
    try:
        store.save_embedding_fingerprint(
            {
                "repo": "Snowflake/snowflake-arctic-embed-m-long",
                "dim": 768,
                "max_chars": 4000,
                "revision": None,
            }
        )
        embedder = _FpEmbedder("jinaai/jina-embeddings-v2-base-code")
        assert _resolve_embedding_trust(embedder, store) is False  # type: ignore[arg-type]
        # Refusal does not overwrite the stored fingerprint — only `shunt reindex` does.
        assert store.load_embedding_fingerprint()["repo"].startswith("Snowflake")
    finally:
        store.close()


def test_mismatch_engine_serves_cold_start_no_query(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
    tmp_path: Any,
) -> None:
    # End-to-end: a stale fingerprint → _build_engine(trust_neighbors=False) → the engine
    # serves cold-start and never queries the foreign-space index.
    from shunt.proxy.server import _build_engine
    from shunt.router.policy import RouterPolicy

    store = OutcomeStore(db_path=str(tmp_path / "fp4.db"))
    try:
        index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
        engine = _build_engine(
            model_pool,
            session_manager,
            store,
            RouterPolicy(),
            embedder=FakeEmbedder(),
            trust_neighbors=False,
        )
        engine._outcome_index = index
        _, reason, _ = engine.decide(session.session_id, "refactor this")
        assert reason == "stale_embedding_space"
        assert index.queries == []
    finally:
        store.close()


def test_live_strategies_match_builder_registry() -> None:
    # Structural parity: router.yaml may name any LIVE_STRATEGIES entry, so every one
    # must have a builder — previously only a convention.
    from shunt.router.policy import LIVE_STRATEGIES
    from shunt.router.strategies.registry import _BUILDERS

    assert set(LIVE_STRATEGIES) == set(_BUILDERS)


def test_concurrent_first_turns_decide_once(
    model_pool: ModelPool,
    session_manager: SessionManager,
    session: Session,
) -> None:
    # Cache-safety: concurrent first turns must produce exactly one routing decision.
    class SlowIndex(FakeOutcomeIndex):
        def query(self, embedding: Any, k: int = 20) -> list[NeighborResult]:
            time.sleep(0.05)
            return super().query(embedding, k)

    index = SlowIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: router._get_or_lock_model(session, "hi"), range(8)))

    assert len(index.queries) == 1
    assert len(set(results)) == 1


def test_streaming_session_persists_cost_after_stream_drains(
    model_pool: ModelPool,
    session_manager: SessionManager,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    # Regression: the server persisted the session BEFORE the stream was consumed, so
    # usage (which only arrives on the final chunk) was always recorded as zero.
    from fastapi.testclient import TestClient

    from shunt.proxy import server

    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    store = OutcomeStore(db_path=str(tmp_path / "stream.db"))
    index = FakeOutcomeIndex(count=100, neighbors=_knn_neighbors())
    router = _make_router(model_pool, session_manager, index)

    async def _chunks(*_a: Any, **_kw: Any) -> Any:
        async def gen() -> Any:
            yield _stream_chunk(_KNN_MODEL, usage=None)
            yield _stream_chunk(_KNN_MODEL, usage=_Usage(0.00042))

        return gen()

    try:
        with patch(_ACOMPLETION_PATCH, new=_chunks), TestClient(server.app) as client:
            # Override the lifespan-built collaborators with the mocked ones.
            server.app.state.session_manager = session_manager
            server.app.state.model_pool = model_pool
            server.app.state.router = router
            server.app.state.outcome_store = store
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert resp.status_code == 200
            resp.read()
            session_id = resp.headers["X-Shunt-Session-Id"]
        stored = store.get_session(session_id)
        assert stored is not None
        assert stored["cost"] == pytest.approx(0.00042)
    finally:
        store.close()
