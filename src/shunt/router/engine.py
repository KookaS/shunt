from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from shunt.router.cold_start import ColdStartStrategy
from shunt.router.embedder import Embedder
from shunt.router.provenance import build_provenance
from shunt.router.selection import ModelPoolProtocol, NeighborResult, SelectionRule


class SessionManagerProtocol(Protocol):
    """Minimal session-manager interface the engine holds a reference to."""

    def get_session(self, session_id: str) -> object: ...


class EmbedderProtocol(Protocol):
    """Minimal embedder interface: text in, vector out."""

    def embed(self, text: str) -> npt.NDArray[np.float32]: ...


logger = logging.getLogger(__name__)


class OutcomeIndex(Protocol):
    """Interface the RouterEngine expects from the outcome + HNSW index.

    Implemented by the DB module (separate build) — the engine never
    hardcodes DB paths or storage internals.
    """

    def count_labeled(self) -> int:
        """Return the number of sessions with Tier-2 (verified) outcomes."""

    def count_total_labeled(self) -> int:
        """Return the total number of sessions with any labeled outcome (Tier-1 or Tier-2)."""

    def query(
        self,
        embedding: npt.NDArray[np.float32],
        k: int = 20,
    ) -> list[NeighborResult]:
        """Return the *k* nearest labeled sessions to *embedding*."""


class RouterEngine:
    """Top-level router decision engine, tying together the embedder, outcome
    index (kNN), and selection rule. Embeddings are cached per session.
    """

    def __init__(  # noqa: PLR0913 (config-heavy constructor wiring router collaborators)
        self,
        model_pool: ModelPoolProtocol,
        session_manager: SessionManagerProtocol,
        outcome_index: OutcomeIndex,
        embedder: EmbedderProtocol | None = None,
        cold_start_threshold: int = 20,
        selection_rule: SelectionRule | None = None,
        cold_start_strategy: ColdStartStrategy | None = None,
    ) -> None:
        self._model_pool = model_pool
        self._session_manager = session_manager
        self._outcome_index = outcome_index
        self._embedder = embedder or Embedder()
        self._cold_start_strategy = cold_start_strategy or ColdStartStrategy()
        self._selection_rule = selection_rule or SelectionRule(
            cold_start_threshold=cold_start_threshold,
        )
        self._cold_start_threshold = cold_start_threshold
        self._cache: dict[str, npt.NDArray[np.float32]] = {}

    def _compute_candidate_model_scores(
        self,
        neighbors: list[NeighborResult],
    ) -> dict[str, float]:
        """Compute per-model weighted success rates from *neighbors*."""
        from shunt.router.selection import _confidence_weight

        groups: dict[str, list[NeighborResult]] = {}
        for n in neighbors:
            groups.setdefault(n.model, []).append(n)

        scores: dict[str, float] = {}
        for model, group in groups.items():
            weights = [_confidence_weight(n) for n in group]
            total_weight = sum(weights)
            if total_weight <= 0:
                scores[model] = 0.0
                continue
            weighted_success = (
                sum(w * (1.0 if n.outcome else 0.0) for w, n in zip(weights, group, strict=True))
                / total_weight
            )
            scores[model] = weighted_success
        return scores

    def decide(self, session_id: str, prompt_text: str) -> tuple[str, str, dict[str, Any]]:
        """Return ``(model_name, reason, provenance_dict)`` for *prompt_text*.

        Checks cold-start; embeds and queries the outcome index; applies the
        selection rule.
        """
        n_total = self._outcome_index.count_total_labeled()
        n_tier2 = self._outcome_index.count_labeled()
        cold_start_active = self._cold_start_strategy.is_active(n_total, n_tier2)

        if cold_start_active:
            model_name = self._cold_start_strategy.select(self._model_pool)
            provenance = build_provenance(
                model_chosen=model_name,
                selection_rule_used="cold_start",
                fallback_chain_triggered=False,
                router_propensity=1.0,
                candidate_model_scores={},
            )
            return (model_name, "cold_start", provenance)

        embedding = self._get_embedding(session_id, prompt_text)
        neighbors = self._outcome_index.query(embedding, k=20)

        model_name, reason = self._selection_rule.select(
            neighbors,
            self._model_pool,
            cold_start_active=False,
        )

        candidate_scores = self._compute_candidate_model_scores(neighbors)
        fallback = reason in ("exploration_untested", "safe_fallback")

        provenance = build_provenance(
            model_chosen=model_name,
            selection_rule_used=reason,
            neighbors=neighbors,
            fallback_chain_triggered=fallback,
            tier_escalation_reason=reason if fallback else None,
            router_propensity=candidate_scores.get(model_name, 1.0),
            candidate_model_scores=candidate_scores,
        )
        return (model_name, reason, provenance)

    def get_neighbors(
        self,
        session_id: str,
        prompt_text: str,
        k: int = 20,
    ) -> list[NeighborResult]:
        """Return raw kNN neighbors for *prompt_text* (for ``SHUNT EXPLAIN``)."""
        embedding = self._get_embedding(session_id, prompt_text)
        return self._outcome_index.query(embedding, k=k)

    def _get_embedding(self, session_id: str, prompt_text: str) -> npt.NDArray[np.float32]:
        if session_id in self._cache:
            return self._cache[session_id]
        embedding = self._embedder.embed(prompt_text)
        self._cache[session_id] = embedding
        return embedding
