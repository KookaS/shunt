"""The ~600MB embedding model must be cached somewhere durable."""

# fastembed's default cache is a temp dir. In a container only the data volume
# survives a restart, so every `docker compose up` re-downloaded the model from
# HuggingFace: a slow first request, and a hard failure with no network.

from __future__ import annotations

import os

import pytest

from shunt.router.embedder import EmbedderUnavailableError, embedding_cache_dir


def test_it_defaults_under_the_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_EMBED_CACHE_DIR", raising=False)
    monkeypatch.setenv("SHUNT_DATA_DIR", "/data")
    assert embedding_cache_dir() == os.path.join("/data", "models")


def test_an_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_DATA_DIR", "/data")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", "/elsewhere")
    assert embedding_cache_dir() == "/elsewhere"


def test_without_a_data_dir_it_uses_a_stable_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Still stable across restarts — the point is never to land in a temp dir.
    monkeypatch.delenv("SHUNT_EMBED_CACHE_DIR", raising=False)
    monkeypatch.delenv("SHUNT_DATA_DIR", raising=False)
    path = embedding_cache_dir()
    assert path.endswith(os.path.join(".cache", "shunt", "models"))
    assert not path.startswith("/tmp")


def test_config_cache_dir_used_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # embedding.yaml cache_dir sits below the env override but above the data-dir default.
    monkeypatch.delenv("SHUNT_EMBED_CACHE_DIR", raising=False)
    monkeypatch.setenv("SHUNT_DATA_DIR", "/data")
    assert embedding_cache_dir("/from-yaml") == "/from-yaml"


def test_env_cache_dir_still_beats_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", "/env-wins")
    assert embedding_cache_dir("/from-yaml") == "/env-wins"


def test_a_failed_load_names_the_download_and_the_cache_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Offline, the raw error surfaced as a bare 502 naming neither.
    from shunt.router import embedder as embedder_module

    monkeypatch.delenv("SHUNT_DISALLOW_REAL_EMBEDDER", raising=False)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", "/nope")
    instance = embedder_module.Embedder.__new__(embedder_module.Embedder)
    instance._model = None
    instance._repo = "jinaai/jina-embeddings-v2-base-code"
    instance._cache_dir_cfg = None

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("Network is unreachable")

    monkeypatch.setattr("fastembed.TextEmbedding", _boom)
    with pytest.raises(EmbedderUnavailableError) as raised:
        instance._load_model()

    message = str(raised.value)
    assert "HuggingFace" in message
    assert "/nope" in message
    assert "SHUNT_EMBED_CACHE_DIR" in message
