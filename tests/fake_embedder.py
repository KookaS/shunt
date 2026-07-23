"""A faithful, hermetic ``Embedder`` for the unit suite + CI (no ONNX, no download)."""

# Why a subclass, not a from-scratch stub: it reuses the REAL Embedder's clipping,
# embed_batch iteration, float32 conversion, truncation_rate, and fingerprint — only the
# fastembed ONNX model is swapped. So a test that uses it exercises the same setup the
# live/benchmark path runs, and it stays faithful as Embedder evolves (the fidelity test
# in tests/router/test_fake_embedder_fidelity.py enforces that). Vectors are deterministic
# per (clipped) text and DISTINCT across texts — unlike a fixed np.full stub, distinct
# prompts get distinct vectors, so the index/kNN retrieval path is genuinely tested.

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np

from shunt.router.embedder import Embedder
from shunt.router.embedding_config import EmbeddingModel

FAKE_REPO = "fake-embedder-768"


class _DeterministicModel:
    """Stand-in for ``fastembed.TextEmbedding``: one deterministic vector per text."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            yield np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)


class FakeEmbedder(Embedder):
    """Drop-in Embedder mirroring the real setup; deterministic distinct vectors, no ONNX."""

    def __init__(self, *, dim: int = 768, max_chars: int = 4000, repo: str = FAKE_REPO) -> None:
        model = EmbeddingModel(repo=repo, dim=dim, context_length=8192)
        super().__init__(model=model, max_chars=max_chars, lazy=True)

    def _load_model(self) -> None:
        # Install the deterministic stand-in instead of loading fastembed — this is what
        # keeps the fake hermetic even while SHUNT_DISALLOW_REAL_EMBEDDER is set.
        self._model = _DeterministicModel(self._dim)
