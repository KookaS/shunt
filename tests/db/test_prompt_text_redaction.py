"""A secret pasted into a prompt must not be persisted verbatim in the sessions table.

`prompt_text` was the one free-text sink that skipped the redactor every other sink
(trajectory records, escalation dedup keys, error bodies, headers) already applies.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pytest

from shunt.db.store import OutcomeStore

# Assembled at runtime so a secret scanner cannot mistake the fixture for a live key.
_SECRET: Final[str] = "sk-" + "B" * 8 + "0123456789bcdef"
_PROMPT: Final[str] = f"deploy fails, my api key: {_SECRET} — what is wrong?"


@pytest.fixture
def store(tmp_path: Any) -> Any:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def test_prompt_text_is_redacted_at_rest(store: OutcomeStore) -> None:
    store.store_session(
        session_id="s1",
        prompt_text=_PROMPT,
        embedding=np.random.randn(64).astype(np.float32),
        model_chosen="model-a",
        cost=0.1,
        cache_stats={},
        duration=1.0,
    )
    session = store.get_session("s1")
    assert session is not None
    stored: str = session["prompt_text"]
    assert _SECRET not in stored
    assert "<redacted>" in stored
    # The surrounding prompt survives — redaction must not blank the routing text.
    assert "deploy fails" in stored


def test_reindex_reads_the_redacted_text(store: OutcomeStore) -> None:
    # prompt_text is re-embedded by reindex_corpus; the redacted text must still be
    # embeddable (a plain str) and must not resurrect the secret.
    store.store_session(
        session_id="s1",
        prompt_text=_PROMPT,
        embedding=np.random.randn(64).astype(np.float32),
        model_chosen="model-a",
        cost=0.1,
        cache_stats={},
        duration=1.0,
    )
    seen: list[str] = []

    class _Embedder:
        def embed(self, text: str) -> np.ndarray:
            seen.append(text)
            return np.random.randn(64).astype(np.float32)

        def fingerprint(self) -> dict[str, Any]:
            return {"repo": "fake", "dim": 64, "max_chars": 4000, "revision": None}

    result = store.reindex_corpus(_Embedder())
    assert result["reindexed"] == 1
    assert seen and all(_SECRET not in t for t in seen)
