from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shunt.router.embedder import (
    DISALLOW_REAL_EMBEDDER_ENV,
    Embedder,
    RealEmbedderBlockedError,
)

_JINA_REPO = "jinaai/jina-embeddings-v2-base-code"
_ARCTIC_REPO = "Snowflake/snowflake-arctic-embed-m-long"


class TestModelProperties:
    def test_active_model_default(self) -> None:
        e = Embedder(lazy=True)
        assert e.model_name == _JINA_REPO
        assert e.dims == 768
        assert e.context_length == 8192

    def test_env_selects_by_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHUNT_EMBEDDER_MODEL", "arctic")
        e = Embedder(lazy=True)
        assert e.model_name == _ARCTIC_REPO
        assert e.context_length == 2048

    def test_env_selects_by_repo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHUNT_EMBEDDER_MODEL", _ARCTIC_REPO)
        e = Embedder(lazy=True)
        assert e.model_name == _ARCTIC_REPO

    def test_unknown_env_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHUNT_EMBEDDER_MODEL", "not-a-model")
        with pytest.raises(ValueError):
            Embedder(lazy=True)

    def test_model_name_arg_resolves_by_repo(self) -> None:
        e = Embedder(model_name=_ARCTIC_REPO, lazy=True)
        assert e.model_name == _ARCTIC_REPO
        assert e.context_length == 2048


class TestFingerprint:
    def test_default_fingerprint(self) -> None:
        assert Embedder(lazy=True).fingerprint() == {
            "repo": _JINA_REPO,
            "dim": 768,
            "max_chars": 4000,
            "revision": None,
        }

    def test_max_chars_env_changes_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHUNT_EMBED_MAX_CHARS", "8000")
        assert Embedder(lazy=True).fingerprint()["max_chars"] == 8000

    def test_model_change_changes_fingerprint(self) -> None:
        jina = Embedder(lazy=True).fingerprint()
        arctic = Embedder(model_name=_ARCTIC_REPO, lazy=True).fingerprint()
        assert jina != arctic


class TestRealLoadGuard:
    def test_load_raises_when_disallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The autouse fixture already set the flag; a lazy Embedder that then loads must fail.
        monkeypatch.setenv(DISALLOW_REAL_EMBEDDER_ENV, "1")
        e = Embedder(lazy=True)
        with pytest.raises(RealEmbedderBlockedError):
            e.warm()

    def test_construction_stays_hermetic_under_guard(self) -> None:
        # Building a lazy Embedder reads only config — no ONNX load, so no guard trip.
        assert Embedder(lazy=True).model_name == _JINA_REPO


class TestEmbed:
    def test_embed_returns_float32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISALLOW_REAL_EMBEDDER_ENV, raising=False)
        fake_vec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = iter([fake_vec])
            mock_cls.return_value = instance
            e = Embedder(lazy=False)
            result = e.embed("hello")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_embed_batch_returns_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISALLOW_REAL_EMBEDDER_ENV, raising=False)
        fake_vecs = [
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3, 0.4], dtype=np.float64),
        ]
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = iter(fake_vecs)
            mock_cls.return_value = instance
            e = Embedder(lazy=False)
            results = e.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(r.dtype == np.float32 for r in results)

    def test_lazy_load_on_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISALLOW_REAL_EMBEDDER_ENV, raising=False)
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = iter([np.array([0.1])])
            mock_cls.return_value = instance
            e = Embedder(lazy=True)
            assert e._model is None
            e.embed("hello")  # triggers load
            mock_cls.assert_called_once()


class TestTruncationRate:
    def test_short_text_returns_zero(self) -> None:
        e = Embedder(lazy=True)
        assert e.truncation_rate("Hello world") == 0.0

    def test_long_text_truncated(self) -> None:
        e = Embedder(lazy=True, model_name=_ARCTIC_REPO)
        # Measured against the BINDING limit — the char clip, not the model context.
        assert e.truncation_rate("x" * (e.max_chars // 2)) == 0.0
        assert e.truncation_rate("x" * e.max_chars) == 0.0
        rate = e.truncation_rate("x" * (e.max_chars * 10))
        assert rate == pytest.approx(0.9)
        assert 0.0 < rate <= 1.0

    def test_very_long_text_caps_at_one(self) -> None:
        e = Embedder(lazy=True)
        rate = e.truncation_rate("x" * 1_000_000)
        assert rate < 1.0
        assert rate > 0.9

    def test_truncation_rate_never_requires_model_load(self) -> None:
        e = Embedder(lazy=True)
        assert e._model is None
        e.truncation_rate("test")
        assert e._model is None  # still lazy
