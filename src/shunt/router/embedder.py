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

MODEL_METADATA: Final[dict[str, tuple[int, int]]] = {
    "jinaai/jina-embeddings-v2-base-code": (768, 8192),
    "jina-embeddings-v2-base-code": (768, 8192),
    "Snowflake/snowflake-arctic-embed-m-long": (768, 2048),
    "arctic-embed-m-long": (768, 2048),
}


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

        self._model = TextEmbedding(model_name=self._model_name)

    def _ensure_model(self) -> None:
        if self._model is None:
            self._load_model()

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string, returning a float32 array."""
        self._ensure_model()
        return np.array(next(self._model.embed([text])), dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        """Embed a batch of texts, returning a list of float32 arrays."""
        self._ensure_model()
        return [np.array(e, dtype=np.float32) for e in self._model.embed(texts)]

    def truncation_rate(self, text: str) -> float:
        """Estimate the prompt fraction exceeding the context window (0.0 if it
        fits, else ``1 - ctx/estimated_tokens`` clamped to [0, 1]). Uses a
        char-based heuristic, so it never triggers a model load.
        """
        ctx = self._context_length
        estimated = max(1, len(text) // 4)
        if estimated <= ctx:
            return 0.0
        return min(1.0, 1.0 - (ctx / estimated))
