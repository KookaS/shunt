"""_resolve_embedding_trust must not blind-trust a legacy pre-fingerprint corpus."""

# A DB created before embedding fingerprints existed holds vectors of an UNRECORDED space.
# Adopting the currently-configured fingerprint over it and trusting kNN would silently
# route on foreign-space neighbours (same dims, different space). Only a truly empty corpus
# is safe to adopt+trust; a legacy corpus must cold-start until `shunt reindex` re-stamps.

from __future__ import annotations

import numpy as np

from shunt.db.store import OutcomeStore
from shunt.proxy.server import _resolve_embedding_trust
from tests.fake_embedder import FakeEmbedder


def _label(store: OutcomeStore, sid: str, emb: np.ndarray) -> None:
    store.store_session(
        session_id=sid,
        prompt_text="t",
        embedding=emb,
        model_chosen="m",
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


def test_fresh_corpus_adopts_and_trusts(tmp_path):  # type: ignore[no-untyped-def]
    store = OutcomeStore(db_path=str(tmp_path / "fresh.db"))
    try:
        emb = FakeEmbedder(repo="embedder-A")
        assert _resolve_embedding_trust(emb, store) is True
        assert store.load_embedding_fingerprint() == emb.fingerprint()
    finally:
        store.close()


def test_legacy_corpus_without_fingerprint_is_not_trusted(tmp_path):  # type: ignore[no-untyped-def]
    store = OutcomeStore(db_path=str(tmp_path / "legacy.db"))
    try:
        a = FakeEmbedder(repo="embedder-A")
        _label(store, "s1", a.embed("hello world"))
        # A pre-fingerprint DB: embeddings exist, no fingerprint was ever stamped.
        assert store.load_embedding_fingerprint() is None

        b = FakeEmbedder(repo="embedder-B")  # a different (custom) space, same 768 dims
        assert _resolve_embedding_trust(b, store) is False
        # It must NOT silently adopt B's fingerprint over A's corpus.
        assert store.load_embedding_fingerprint() is None
    finally:
        store.close()
