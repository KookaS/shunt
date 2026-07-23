"""Adapts :class:`OutcomeStore` (+ its HNSW index) to the engine's ``OutcomeIndex``.

The engine reads verified outcomes back through this seam; the store keeps its own
storage internals (SQL + hnswlib), the router keeps its ``NeighborResult`` domain type.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from shunt.router.selection import NeighborResult, effective_sample_size

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from shunt.db.store import OutcomeStore

# A stored outcome counts as a success iff its verified label is one of these.
_SUCCESS_LABELS = ("success", "weak_success")


def _is_non_policy(row: dict[str, Any]) -> bool:
    """True when the session's decision was an imposed auto-escalation, not a policy pick."""
    # An escalated turn ran a model/arm the failure signal forced, not one the router sampled —
    # so its verified outcome must not train that model as a free (policy-attributable) choice.
    raw = row.get("decision_provenance")
    if not isinstance(raw, str):
        return False
    try:
        provenance = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(provenance, dict) and bool(provenance.get("auto_escalated"))


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

    def effective_labeled(self) -> float:
        """Effective sample size ``nₑ`` over all labeled outcomes, weighted by confidence.

        Reuses the same per-outcome confidence the neighbour path weights with, so uniform
        confidences make ``nₑ`` equal the raw ``count_total_labeled`` (backward-compat).
        """
        return self._effective_sample_size(tier2_only=False)

    def effective_tier2(self) -> float:
        """Effective sample size ``nₑ`` over Tier-2 (verified) outcomes only."""
        return self._effective_sample_size(tier2_only=True)

    def _effective_sample_size(self, *, tier2_only: bool) -> float:
        rows = self._store.labeled_outcome_rows(tier2_only=tier2_only)
        return effective_sample_size([self._resolve_confidence(r) for r in rows])

    def model_priors(self) -> dict[str, tuple[float, float]]:
        """Per-model ``(estimate, strength)`` offline Tier-2 aggregate, for prior seeding."""
        # estimate = the model's global confidence-weighted success rate; strength = the
        # effective sample size nₑ of that history. Seeds an informative Thompson prior
        # (empirical-Bayes shrinkage) so a model with evidence doesn't restart at Beta(1,1)
        # each decision. Verified (Tier-2) outcomes only — the trusted population.
        # Exclude imposed auto-escalations — the Thompson prior is a policy signal too, so an
        # escalated model must not seed its own prior as if the router had chosen it.
        all_rows = self._store.labeled_outcome_rows(tier2_only=True)
        rows = [r for r in all_rows if not _is_non_policy(r)]
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(str(row["model_chosen"]), []).append(row)
        priors: dict[str, tuple[float, float]] = {}
        for model, model_rows in by_model.items():
            weights = [self._resolve_confidence(r) for r in model_rows]
            total = sum(weights)
            if total <= 0.0:
                continue
            weighted_success = sum(
                w for w, r in zip(weights, model_rows, strict=True) if self._resolve_success(r)
            )
            priors[model] = (weighted_success / total, effective_sample_size(weights))
        return priors

    def query(self, embedding: npt.NDArray[np.float32], k: int = 20) -> list[NeighborResult]:
        """Nearest *labeled* sessions to *embedding*.

        Fetches the *k* nearest embedded sessions from the index and keeps only those
        with a stored outcome, so the result may hold fewer than *k* neighbors.
        """
        # Auto-escalated sessions ARE kept as neighbours — the escalated (bigger) model's
        # verified pass/fail on that task is real capability signal the selection rule needs to
        # converge. They are withheld only from the policy-attributable aggregates (`model_priors`
        # + the nulled propensity), never from the neighbour set.
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
            cost=self._resolve_cost(session),
            verification_confidence=self._resolve_confidence(outcome),
            distance=distance,
            session_id=str(session["session_id"]),
        )

    @staticmethod
    def _resolve_cost(session: dict[str, Any]) -> float:
        """UNKNOWN cost (``cost_known=0``) surfaces as ``+inf`` so it never sorts cheapest."""
        # A stored ``0.0`` with ``cost_known=1`` is a genuine free/fully-cached call and stays
        # ``0.0``; only an unreported cost — indistinguishable from a real zero in ``cost`` alone
        # — is lifted to ``+inf``, reusing the engine's existing non-finite guards.
        if session.get("cost_known", 1) == 0:
            return math.inf
        return float(session["cost"])

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
