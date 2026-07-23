"""Schema + loader for ``embedding.yaml`` — the swap-safe embedding-model config.

The active model plus ``max_chars`` form the corpus fingerprint; a mismatch against the
stored one makes the router refuse foreign-space kNN neighbours until ``shunt reindex``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shunt.models.config import strict_yaml_load

logger = logging.getLogger(__name__)

_CONFIG_DIR_ENV: Final[str] = "SHUNT_CONFIG_DIR"
_CONFIG_FILENAME: Final[str] = "embedding.yaml"
_MODEL_ENV: Final[str] = "SHUNT_EMBEDDER_MODEL"


class EmbeddingModel(BaseModel):
    """One named embedding model: the HF repo plus its geometric shape."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    dim: int = Field(gt=0)
    context_length: int = Field(gt=0)

    def fingerprint(self, max_chars: int, revision: str | None = None) -> dict[str, object]:
        """The tuple that fully determines the vector space, as a JSON-able dict."""
        return {"repo": self.repo, "dim": self.dim, "max_chars": max_chars, "revision": revision}


class EmbeddingConfig(BaseModel):
    """Top-level ``embedding.yaml`` schema: the active model keyed into a closed set."""

    model_config = ConfigDict(extra="forbid")

    active: str
    max_chars: int = Field(default=4000, gt=0)
    models: dict[str, EmbeddingModel]
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _check_active(self) -> EmbeddingConfig:
        if self.active not in self.models:
            valid = ", ".join(sorted(self.models))
            raise ValueError(f"embedding.active {self.active!r} is not a known model key ({valid})")
        return self

    def resolve_active(self, env: Mapping[str, str]) -> EmbeddingModel:
        """The model to run: env ``SHUNT_EMBEDDER_MODEL`` (key or repo) wins over ``active``."""
        override = env.get(_MODEL_ENV)
        return self.resolve(override) if override else self.models[self.active]

    def resolve(self, value: str) -> EmbeddingModel:
        """Resolve *value* against the closed set: exact key, else a ``repo`` match, else error."""
        if value in self.models:
            return self.models[value]
        for model in self.models.values():
            if model.repo == value:
                return model
        valid = ", ".join(sorted(self.models))
        raise ValueError(
            f"embedding model {value!r} is not a known key or repo. Valid keys: {valid}. "
            "It never silently falls back to a default — fix the value or add the model."
        )


def parse_embedding_config(data: dict[str, object] | None) -> EmbeddingConfig:
    """Validate an ``embedding:`` config mapping into an EmbeddingConfig."""
    if not data:
        raise ValueError("embedding config is empty; expected an `embedding:` section")
    section = data.get("embedding", data)
    return EmbeddingConfig.model_validate(section)


def _user_config_path() -> Path:
    config_dir = os.environ.get(_CONFIG_DIR_ENV)
    base = Path(config_dir) if config_dir else Path.home() / ".config" / "shunt"
    return base / _CONFIG_FILENAME


def packaged_embedding_path() -> Path:
    """Path to the embedding config shipped inside the package."""
    import importlib.resources

    ref = importlib.resources.files("shunt.config") / _CONFIG_FILENAME
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def load_embedding_config(path: str | Path | None = None) -> EmbeddingConfig:
    """Explicit path → $SHUNT_CONFIG_DIR/embedding.yaml → packaged embedding.yaml."""
    resolved = Path(path) if path is not None else _user_config_path()
    if not resolved.exists():
        logger.debug("embedding config: %s absent, falling back to packaged", resolved)
        resolved = packaged_embedding_path()
    logger.debug("embedding config: loaded from %s", resolved)
    return parse_embedding_config(strict_yaml_load(resolved.read_text()))
