from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from shunt.router.embedder import (
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    Embedder,
)


class TestModelProperties:
    def test_primary_model_default(self):
        e = Embedder(lazy=True)
        assert e.model_name == PRIMARY_MODEL

    def test_primary_model_dims(self):
        e = Embedder(lazy=True)
        assert e.dims == 768

    def test_primary_model_context(self):
        e = Embedder(lazy=True)
        assert e.context_length == 8192

    def test_fallback_properties(self):
        e = Embedder(model_name=FALLBACK_MODEL, lazy=True)
        assert e.model_name == FALLBACK_MODEL
        assert e.dims == 768
        assert e.context_length == 2048

    def test_custom_model_env_var(self, monkeypatch):
        monkeypatch.setenv("SHUNT_EMBEDDER_MODEL", "custom-model")
        e = Embedder(lazy=True)
        assert e.model_name == "custom-model"
        assert e.dims == 768  # fallback defaults
        assert e.context_length == 8192  # fallback defaults


class TestEmbed:
    def test_embed_returns_float32(self):
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

    def test_embed_batch_returns_list(self):
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

    def test_lazy_load_on_first_call(self):
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = iter([np.array([0.1])])
            mock_cls.return_value = instance

            e = Embedder(lazy=True)
            assert e._model is None

            e.embed("hello")  # triggers load
            mock_cls.assert_called_once()


class TestTruncationRate:
    def test_short_text_returns_zero(self):
        e = Embedder(lazy=True)
        assert e.truncation_rate("Hello world") == 0.0

    def test_long_text_truncated(self):
        e = Embedder(lazy=True, model_name=FALLBACK_MODEL)  # 2048 context
        # ~5000 chars → ~1250 tokens < 2048 → 0.0
        assert e.truncation_rate("x" * 5000) == 0.0

        # ~50000 chars → ~12500 tokens > 2048 → truncated
        rate = e.truncation_rate("x" * 50000)
        assert rate > 0.0
        assert rate <= 1.0

    def test_very_long_text_caps_at_one(self):
        e = Embedder(lazy=True)

        # character-based estimate: len(text) // 4
        # For text length 1_000_000: 250000 tokens, 8192 context
        rate = e.truncation_rate("x" * 1_000_000)
        assert rate < 1.0
        assert rate > 0.9

    def test_truncation_rate_never_requires_model_load(self):
        e = Embedder(lazy=True)
        assert e._model is None
        e.truncation_rate("test")
        assert e._model is None  # still lazy
