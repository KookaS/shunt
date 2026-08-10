"""Shared no-I/O fakes for the escalation tests (session, index, embedder, model pools)."""

# Extracted so the escalation suites agree on ONE fake router surface: a divergent copy of
# these silently changes what a test proves (a pool without ``get_model`` reports no effort
# headroom at all, which reads as "the ladder stepped rank" rather than "the fake has no arms").

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from shunt.models.config import ModelConfig, ReasoningArm, ReasoningConfig


@dataclass
class Session:
    tool_identity: str


class SessionManager:
    """Every session id resolves to one fixed tool identity."""

    def __init__(self, tool_identity: str = "toolA") -> None:
        self._tool = tool_identity

    def get_session(self, session_id: str) -> Session:
        return Session(tool_identity=self._tool)


class EchoSessionManager:
    """Echoes the session id back as the identity, so a resolver can key by task."""

    def get_session(self, session_id: str) -> Session:
        return Session(tool_identity=session_id)


class Index:
    """Warm index with an empty neighbourhood → base selection is the cheapest model."""

    def count_labeled(self) -> int:
        return 100

    def count_total_labeled(self) -> int:
        return 100

    def effective_labeled(self) -> float:
        return 100.0

    def effective_tier2(self) -> float:
        return 100.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: np.ndarray, k: int = 20) -> list[Any]:
        return []


class Embedder:
    """Constant vector — the routing decision under test never depends on the embedding."""

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


def model_config(name: str, reasoning: ReasoningConfig | None = None) -> ModelConfig:
    """A minimal registry entry: only the name and the reasoning ladder matter here."""
    return ModelConfig(
        name=name,
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=reasoning,
    )


def reasoning_ladder(low: str, high: str) -> ReasoningConfig:
    """A two-arm effort ladder, *low* being the model's default rung."""
    return ReasoningConfig(
        default_arm=low,
        arms=[
            ReasoningArm(id=low, rank=0, api={"reasoning_effort": low}),
            ReasoningArm(id=high, rank=1, api={"reasoning_effort": high}),
        ],
    )


class RankedReasoningPool:
    """Ranked models, each carrying its OWN effort-arm vocabulary (weakest -> strongest)."""

    # Distinct vocabularies per model mirror the real registry (deepseek {nothink,high,max},
    # qwen {nothink,think}), so a rank step is forced to reset a FOREIGN arm rather than
    # coincidentally finding the same id on the next model.

    def __init__(
        self, ladders: dict[str, ReasoningConfig | None], unhealthy: set[str] | None = None
    ) -> None:
        self._models = {name: model_config(name, arms) for name, arms in ladders.items()}
        self._ranked = list(self._models.values())
        # Named models report unhealthy for the whole test. The rank ladder's degrade path only
        # exists when a rung it aims at is unavailable, so it is unreachable without this.
        self._unhealthy = unhealthy or set()

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def ranked_models(self) -> list[ModelConfig]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        for i, m in enumerate(self._ranked):
            if m.name == name:
                return i
        return None

    def models_from_rank(self, i: int) -> list[ModelConfig]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        # Membership, not a blanket True: the cold-start strategy probes for its hardcoded
        # default first, and a pool that claims to be healthy for a model it does not hold
        # returns a name no other collaborator can rank.
        return name in self._models and name not in self._unhealthy
