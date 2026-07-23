"""End-to-end embedding pipeline (embed → store → HNSW index → kNN query) on the fake."""

# The CI check that the real embedding SETUP works, hermetically (no ONNX, no download):
# the faithful FakeEmbedder (distinct vectors per prompt) runs through the REAL OutcomeStore
# and OutcomeIndexAdapter, so retrieval discrimination is genuinely exercised — unlike the
# fixed-vector stubs the wiring tests use.

from __future__ import annotations

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore
from tests.fake_embedder import FakeEmbedder


@pytest.fixture
def store(tmp_path):  # type: ignore[no-untyped-def]
    s = OutcomeStore(db_path=str(tmp_path / "pipeline.db"))
    yield s
    s.close()


def _label(store: OutcomeStore, sid: str, prompt: str, model: str, emb: np.ndarray) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=prompt,
        embedding=emb,
        model_chosen=model,
        cost=1.0,
        cache_stats={},
        duration=1.0,
    )
    store.store_outcome(
        session_id=sid,
        tier1_outcome="success",
        tier1_confidence=0.7,
        tier2_outcome="success",
        tier2_confidence=0.95,
        aggregated_confidence=0.9,
    )


def test_embed_store_index_query_roundtrip(store: OutcomeStore) -> None:
    fe = FakeEmbedder()
    prompts = {
        "s_parser": "fix the null pointer in the argument parser",
        "s_docs": "add a docstring to the config loader",
        "s_perf": "optimize the hot loop in the tokenizer",
    }
    for sid, prompt in prompts.items():
        _label(store, sid, prompt, model="deepseek-v4-flash", emb=fe.embed(prompt))

    adapter = OutcomeIndexAdapter(store)
    # Query with a stored prompt's real embedding — the same session must come back nearest.
    neighbors = adapter.query(fe.embed(prompts["s_docs"]), k=3)
    assert neighbors, "index returned no neighbours — the embed→store→index chain is broken"
    assert neighbors[0].session_id == "s_docs"
    assert neighbors[0].distance == pytest.approx(0.0, abs=1e-4)
    assert neighbors[0].outcome is True


def test_index_discriminates_between_distinct_prompts(store: OutcomeStore) -> None:
    fe = FakeEmbedder()
    _label(
        store,
        "cheap_task",
        "rename a local variable",
        "deepseek-v4-flash",
        fe.embed("rename a local variable"),
    )
    _label(
        store,
        "hard_task",
        "design a distributed consensus protocol",
        "kimi-k3",
        fe.embed("design a distributed consensus protocol"),
    )

    adapter = OutcomeIndexAdapter(store)
    near_cheap = adapter.query(fe.embed("rename a local variable"), k=2)
    # Distinct prompts get distinct vectors, so the matching session ranks first — a
    # fixed-vector stub would make these two equidistant and this assertion meaningless.
    assert near_cheap[0].session_id == "cheap_task"
    assert near_cheap[0].model == "deepseek-v4-flash"


def test_unmeasured_query_still_returns_labeled_neighbours(store: OutcomeStore) -> None:
    fe = FakeEmbedder()
    _label(
        store,
        "s1",
        "write a unit test for the cache",
        "deepseek-v4-flash",
        fe.embed("write a unit test for the cache"),
    )
    adapter = OutcomeIndexAdapter(store)
    # A brand-new prompt (never stored) still resolves to the nearest labeled session.
    neighbors = adapter.query(fe.embed("a completely different unseen request"), k=5)
    assert len(neighbors) == 1
    assert neighbors[0].session_id == "s1"
