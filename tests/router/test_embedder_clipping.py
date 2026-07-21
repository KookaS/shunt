"""Embedding input is bounded — an unbounded prompt OOM-killed the router."""

from __future__ import annotations

import numpy as np
import pytest

from shunt.router.embedder import DEFAULT_MAX_EMBED_CHARS, Embedder


class _RecordingModel:
    """Stand-in for the ONNX encoder: records what it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        self.seen.extend(texts)
        return iter([np.zeros(768, dtype=np.float32) for _ in texts])


@pytest.fixture
def embedder() -> tuple[Embedder, _RecordingModel]:
    e = Embedder()
    model = _RecordingModel()
    e._model = model
    return e, model


def test_long_prompt_is_clipped_before_the_encoder(
    embedder: tuple[Embedder, _RecordingModel],
) -> None:
    # Measured in the shipped container: 20k chars peaked at 13.7 GB and 60k was
    # OOM-killed. A coding agent's system prompt alone exceeds 20k, so without this the
    # first real request from Claude Code or opencode takes the whole router down.
    e, model = embedder
    e.embed("x" * 50_000)

    assert len(model.seen[0]) == DEFAULT_MAX_EMBED_CHARS


def test_short_prompt_is_untouched(embedder: tuple[Embedder, _RecordingModel]) -> None:
    e, model = embedder
    e.embed("def foo(): pass")

    assert model.seen[0] == "def foo(): pass"


def test_batch_clips_every_member(embedder: tuple[Embedder, _RecordingModel]) -> None:
    e, model = embedder
    e.embed_batch(["y" * 50_000, "short"])

    assert [len(t) for t in model.seen] == [DEFAULT_MAX_EMBED_CHARS, len("short")]


def test_cap_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EMBED_MAX_CHARS", "128")
    e = Embedder()
    model = _RecordingModel()
    e._model = model

    e.embed("z" * 5_000)

    assert len(model.seen[0]) == 128


def test_a_zero_or_negative_cap_never_yields_an_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A misconfigured cap must not send "" to the encoder — that embeds every prompt
    # identically, which silently makes every neighbourhood meaningless. Clamping to 1
    # avoided the empty string but kept the same degeneracy, so this now fails loud.
    monkeypatch.setenv("SHUNT_EMBED_MAX_CHARS", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        Embedder()


def test_a_non_integer_cap_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EMBED_MAX_CHARS", "8k")
    with pytest.raises(ValueError, match="must be an integer"):
        Embedder()


def test_an_unset_cap_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_EMBED_MAX_CHARS", raising=False)
    assert Embedder().max_chars == DEFAULT_MAX_EMBED_CHARS
