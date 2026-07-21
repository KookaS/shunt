from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "jinaai/jina-embeddings-v2-base-code"
FALLBACK_MODEL = "Snowflake/snowflake-arctic-embed-m-long"


class EmbedderUnavailableError(RuntimeError):
    """The embedding model could not be loaded — actionable, unlike the raw cause."""


def embedding_cache_dir() -> str:
    """Where the ONNX model is cached on disk — durable, not a temp dir."""
    # Defaults under SHUNT_DATA_DIR so the ~600MB download survives a restart. The
    # library default is a temp dir, which in a container meant re-downloading from
    # HuggingFace on every start — slow, and a hard failure with no network.
    if override := os.environ.get("SHUNT_EMBED_CACHE_DIR"):
        return override
    data_dir = os.environ.get("SHUNT_DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "models")
    return os.path.join(os.path.expanduser("~"), ".cache", "shunt", "models")


# Hard cap on the characters handed to the ONNX encoder. Attention is O(n^2), so an
# unbounded prompt is an unbounded allocation — measured in the shipped container:
# 200 chars 888 MB (model load), 4k 1.3 GB, 8k 2.8 GB, 12k 7.2 GB, 20k 13.7 GB, 60k
# OOM-killed. A coding agent's system prompt alone exceeds 20k, so an uncapped embed
# takes the whole router down on the FIRST real request from Claude Code or opencode.
#
# 4000 is not arbitrary: re-embedding the routing corpus at a 4000-char cap raised the
# held-out correlation from 0.068 to 0.113, so it is the value the routing evidence
# already points at, and it costs ~400 MB over the model itself.
DEFAULT_MAX_EMBED_CHARS: Final[int] = 4000

MODEL_METADATA: Final[dict[str, tuple[int, int]]] = {
    "jinaai/jina-embeddings-v2-base-code": (768, 8192),
    "jina-embeddings-v2-base-code": (768, 8192),
    "Snowflake/snowflake-arctic-embed-m-long": (768, 2048),
    "arctic-embed-m-long": (768, 2048),
}


def _parse_max_chars(raw: str | None) -> int:
    """Resolve SHUNT_EMBED_MAX_CHARS, failing loud on a value that is not an int."""
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_EMBED_CHARS
    try:
        value = int(raw)
    except ValueError:
        # A typo ("8k") previously raised a bare ValueError from deep inside __init__.
        raise ValueError(
            f"SHUNT_EMBED_MAX_CHARS must be an integer number of characters, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"SHUNT_EMBED_MAX_CHARS must be >= 1, got {value}")
    return value


class Embedder:
    """Fastembed wrapper for prompt embedding (default
    jina-embeddings-v2-base-code, 768d; override via ``SHUNT_EMBEDDER_MODEL``).
    Lazy-loads the ONNX model on the first ``embed()`` call.
    """

    def __init__(
        self,
        model_name: str | None = None,
        lazy: bool = True,
    ) -> None:
        self._model_name = model_name or os.environ.get("SHUNT_EMBEDDER_MODEL", PRIMARY_MODEL)
        dim, ctx = MODEL_METADATA.get(self._model_name, (768, 8192))
        self._dim = dim
        self._context_length = ctx
        self._max_chars = _parse_max_chars(os.environ.get("SHUNT_EMBED_MAX_CHARS"))
        self._model: Any = None
        if not lazy:
            self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return self._dim

    @property
    def context_length(self) -> int:
        return self._context_length

    def _load_model(self) -> None:
        from fastembed import TextEmbedding

        cache_dir = embedding_cache_dir()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=cache_dir)
        except Exception as exc:
            # The first load downloads ~600MB from HuggingFace. Offline, the raw error
            # surfaces as a bare 502 that names neither the download nor the cache path.
            raise EmbedderUnavailableError(
                f"could not load embedding model {self._model_name!r} (cache: {cache_dir}). "
                "The first run downloads it from HuggingFace — check network access, or "
                "pre-populate the cache dir. Set SHUNT_EMBED_CACHE_DIR to relocate it."
            ) from exc

    def warm(self) -> None:
        """Load the model now, so the first request does not pay for it."""
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is None:
            self._load_model()

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def _clip(self, text: str) -> str:
        """Bound the encoder input so one long prompt cannot exhaust memory."""
        if len(text) <= self._max_chars:
            return text
        logger.debug(
            "Prompt clipped for embedding: %d chars -> %d (routing signal only; the full "
            "prompt is still forwarded upstream untouched)",
            len(text),
            self._max_chars,
        )
        return text[: self._max_chars]

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string, returning a float32 array."""
        self._ensure_model()
        return np.array(next(self._model.embed([self._clip(text)])), dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        """Embed a batch of texts, returning a list of float32 arrays."""
        self._ensure_model()
        clipped = [self._clip(t) for t in texts]
        return [np.array(e, dtype=np.float32) for e in self._model.embed(clipped)]

    def truncation_rate(self, text: str) -> float:
        """Fraction of *text* the embedder actually discards, in [0, 1]."""
        # Measured against the BINDING limit. The char clip (default 4000) bites long
        # before the model's 8192-token context does, so comparing against the context
        # window reported ~0.67 for a prompt that was in fact 96% discarded — and this
        # feeds NeighborResult.truncation_rate, i.e. a routing signal.
        kept = min(len(text), self._max_chars)
        return 0.0 if len(text) == 0 else 1.0 - (kept / len(text))
