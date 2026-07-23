"""Fidelity contract: the FakeEmbedder must mirror the real Embedder's setup.

If this drifts, tests/CI would "pass" against an embedder that does not match production —
exactly the failure this guards. No ONNX is loaded here (the fake overrides the load path).
"""

from __future__ import annotations

import numpy as np
import pytest

from shunt.router.embedder import DISALLOW_REAL_EMBEDDER_ENV, Embedder, RealEmbedderBlockedError
from tests.fake_embedder import FakeEmbedder


def test_fake_covers_the_real_public_surface() -> None:
    real = {a for a in dir(Embedder) if not a.startswith("_")}
    fake = {a for a in dir(FakeEmbedder) if not a.startswith("_")}
    missing = real - fake
    assert not missing, f"FakeEmbedder is missing real Embedder surface: {sorted(missing)}"


def test_dim_and_dtype_match_real_setup() -> None:
    fe = FakeEmbedder()
    assert fe.dims == 768
    vec = fe.embed("fix the flaky parser test")
    assert vec.shape == (768,)
    assert vec.dtype == np.float32


def test_distinct_texts_get_distinct_vectors_deterministically() -> None:
    fe = FakeEmbedder()
    a1, a2, b = fe.embed("alpha"), fe.embed("alpha"), fe.embed("beta")
    assert np.array_equal(a1, a2), "same text must embed identically (deterministic)"
    assert not np.allclose(a1, b), "distinct texts must differ — a fixed-vector stub would not"


def test_embed_batch_agrees_with_embed() -> None:
    fe = FakeEmbedder()
    texts = ["one", "two", "three"]
    batch = fe.embed_batch(texts)
    assert len(batch) == len(texts)
    for text, bvec in zip(texts, batch, strict=True):
        assert np.array_equal(bvec, fe.embed(text))


def test_clipping_matches_real_setup() -> None:
    fe = FakeEmbedder(max_chars=10)
    assert fe.max_chars == 10
    # Both inputs clip to the same first 10 chars, so they must embed identically.
    assert np.array_equal(fe.embed("x" * 10), fe.embed("x" * 50))
    assert fe.truncation_rate("x" * 50) == pytest.approx(0.8)


def test_fingerprint_has_the_real_shape() -> None:
    fp = FakeEmbedder(max_chars=4000).fingerprint()
    assert set(fp) >= {"repo", "dim", "max_chars"}
    assert fp["dim"] == 768
    assert fp["max_chars"] == 4000


def test_fake_works_under_the_real_load_guard() -> None:
    # The fake must stay usable while SHUNT_DISALLOW_REAL_EMBEDDER blocks the REAL model —
    # that is what lets CI exercise the pipeline hermetically.
    import os

    os.environ[DISALLOW_REAL_EMBEDDER_ENV] = "1"
    try:
        assert FakeEmbedder().embed("hello").shape == (768,)
        # And the REAL embedder must refuse to load under the same guard (no 600MB download).
        with pytest.raises(RealEmbedderBlockedError):
            Embedder(model_name=None, lazy=False)
    finally:
        os.environ.pop(DISALLOW_REAL_EMBEDDER_ENV, None)
