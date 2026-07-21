"""Adapts :class:`OutcomeStore` (+ its HNSW index) to the engine's ``OutcomeIndex``.

The engine reads verified outcomes back through this seam; the store keeps its own
storage internals (SQL + hnswlib), the router keeps its ``NeighborResult`` domain type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shunt.router.selection import NeighborResult

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from shunt.db.store import OutcomeStore

# A stored outcome counts as a success iff its verified label is one of these.
_SUCCESS_LABELS = ("success", "weak_success")
# Fallback confidence when a neighbor has an outcome row carrying no confidence value
# at all (no aggregated score and both tier fields absent) — a weak-but-nonzero prior so
# the neighbor still contributes, without over-trusting an unquantified label.
_DEFAULT_CONFIDENCE = 0.5


class OutcomeIndexAdapter:
    """Read-back seam: exposes stored, labeled sessions to the ``RouterEngine``."""

    def __init__(
        self, store: OutcomeStore, default_confidence: float = _DEFAULT_CONFIDENCE
    ) -> None:
        self._store = store
        self._default_confidence = default_confidence

    def count_labeled(self) -> int:
        """Sessions with a Tier-2 (verified) outcome."""
        return self._store.count_verified_outcomes()

    def count_total_labeled(self) -> int:
        """Sessions with any labeled outcome (Tier-1 or Tier-2)."""
        return self._store.count_outcomes()

    def query(self, embedding: npt.NDArray[np.float32], k: int = 20) -> list[NeighborResult]:
        """Nearest *labeled* sessions to *embedding*.

        Fetches the *k* nearest embedded sessions from the index and keeps only those
        with a stored outcome, so the result may hold fewer than *k* neighbors.
        """
        results: list[NeighborResult] = []
        for session_id, distance in self._store.query_index(embedding, k):
            session = self._store.get_session(session_id)
            outcome = self._store.get_outcome(session_id)
            if session is None or outcome is None:
                continue
            results.append(self._to_neighbor(session, outcome, distance))
        return results

    def _to_neighbor(
        self,
        session: dict[str, Any],
        outcome: dict[str, Any],
        distance: float,
    ) -> NeighborResult:
        """Join a session row + its outcome row into a ``NeighborResult``."""
        return NeighborResult(
            model=str(session["model_chosen"]),
            outcome=self._resolve_success(outcome),
            cost=float(session["cost"]),
            verification_confidence=self._resolve_confidence(outcome),
            distance=distance,
            session_id=str(session["session_id"]),
        )

    @staticmethod
    def _resolve_success(outcome: dict[str, Any]) -> bool:
        """Verified Tier-2 label wins; else Tier-1. Non-success labels ⇒ ``False``."""
        label = outcome.get("tier2_outcome") or outcome.get("tier1_outcome")
        return label in _SUCCESS_LABELS

    def _resolve_confidence(self, outcome: dict[str, Any]) -> float:
        """Prefer the aggregated confidence; fall back to Tier-2, Tier-1, then default."""
        # `aggregated_confidence` is NOT NULL with a store-side default of 0.0, so a zero
        # there means "never aggregated" and must fall through. The tier fields carry no
        # such sentinel: an explicit 0.0 is a real "no confidence" and must NOT be
        # silently upgraded to the default.
        aggregated = outcome.get("aggregated_confidence")
        if aggregated:
            return float(aggregated)
        for key in ("tier2_confidence", "tier1_confidence"):
            value = outcome.get(key)
            if value is not None:
                return float(value)
        return self._default_confidence
