from __future__ import annotations

from pathlib import Path

import pytest

from shunt.router.embedding_config import (
    EmbeddingConfig,
    EmbeddingModel,
    load_embedding_config,
)

_JINA_REPO = "jinaai/jina-embeddings-v2-base-code"
_ARCTIC_REPO = "Snowflake/snowflake-arctic-embed-m-long"


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


_VALID = """
embedding:
  active: jina-code
  max_chars: 4000
  models:
    jina-code: { repo: jinaai/jina-embeddings-v2-base-code, dim: 768, context_length: 8192 }
    arctic: { repo: Snowflake/snowflake-arctic-embed-m-long, dim: 768, context_length: 2048 }
  cache_dir: null
"""


class TestPrecedence:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        cfg = load_embedding_config(_write(tmp_path / "embedding.yaml", _VALID))
        assert cfg.active == "jina-code"
        assert cfg.resolve_active({}).repo == _JINA_REPO

    def test_config_dir_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "embedding.yaml", _VALID.replace("active: jina-code", "active: arctic"))
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        cfg = load_embedding_config()
        assert cfg.active == "arctic"

    def test_packaged_default_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHUNT_CONFIG_DIR", raising=False)
        # Point config_dir at an empty dir so the packaged default is what loads.
        cfg = load_embedding_config()
        assert cfg.resolve_active({}).repo == _JINA_REPO
        assert cfg.max_chars == 4000


class TestForbidExtra:
    def test_unknown_top_key_raises(self, tmp_path: Path) -> None:
        bad = _VALID + "  bogus: 1\n"
        with pytest.raises(Exception):  # noqa: B017,PT011 (pydantic ValidationError, fail-loud)
            load_embedding_config(_write(tmp_path / "embedding.yaml", bad))

    def test_unknown_model_key_raises(self, tmp_path: Path) -> None:
        bad = _VALID.replace(
            "dim: 768, context_length: 8192", "dim: 768, context_length: 8192, x: 1"
        )
        with pytest.raises(Exception):  # noqa: B017,PT011
            load_embedding_config(_write(tmp_path / "embedding.yaml", bad))


class TestResolveActive:
    def _cfg(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            active="jina-code",
            max_chars=4000,
            models={
                "jina-code": EmbeddingModel(repo=_JINA_REPO, dim=768, context_length=8192),
                "arctic": EmbeddingModel(repo=_ARCTIC_REPO, dim=768, context_length=2048),
            },
            cache_dir=None,
        )

    def test_active_key_default(self) -> None:
        assert self._cfg().resolve_active({}).repo == _JINA_REPO

    def test_env_key_match(self) -> None:
        assert self._cfg().resolve_active({"SHUNT_EMBEDDER_MODEL": "arctic"}).repo == _ARCTIC_REPO

    def test_env_repo_match(self) -> None:
        # Back-compat: an operator who set the full HF repo string still resolves.
        got = self._cfg().resolve_active({"SHUNT_EMBEDDER_MODEL": _ARCTIC_REPO})
        assert got.repo == _ARCTIC_REPO

    def test_unresolvable_env_raises_listing_keys(self) -> None:
        with pytest.raises(ValueError) as e:
            self._cfg().resolve_active({"SHUNT_EMBEDDER_MODEL": "nope"})
        assert "jina-code" in str(e.value) and "arctic" in str(e.value)

    def test_active_key_must_exist(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingConfig(
                active="ghost",
                max_chars=4000,
                models={"jina-code": EmbeddingModel(repo=_JINA_REPO, dim=768, context_length=8192)},
                cache_dir=None,
            )


class TestFingerprint:
    def test_fields(self) -> None:
        model = EmbeddingModel(repo=_JINA_REPO, dim=768, context_length=8192)
        fp = model.fingerprint(max_chars=4000)
        assert fp == {"repo": _JINA_REPO, "dim": 768, "max_chars": 4000, "revision": None}

    def test_revision_carried(self) -> None:
        model = EmbeddingModel(repo=_JINA_REPO, dim=768, context_length=8192)
        assert model.fingerprint(max_chars=4000, revision="abc")["revision"] == "abc"

    def test_max_chars_changes_fingerprint(self) -> None:
        model = EmbeddingModel(repo=_JINA_REPO, dim=768, context_length=8192)
        assert model.fingerprint(max_chars=4000) != model.fingerprint(max_chars=8000)
