from __future__ import annotations

from typing import Any

from shunt.router.selection import NeighborResult


def build_provenance(  # noqa: PLR0913 (config-heavy provenance builder, one arg per field)
    model_chosen: str,
    selection_rule_used: str,
    neighbors: list[NeighborResult] | None = None,
    fallback_chain_triggered: bool = False,
    tier_escalation_reason: str | None = None,
    router_propensity: float = 1.0,
    candidate_model_scores: dict[str, float] | None = None,
    downshift: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable decision provenance dict."""
    top_k_neighbor_ids: list[str] = []
    neighbor_confidence_scores: list[float] = []
    if neighbors:
        top_k_neighbor_ids = [n.session_id for n in neighbors]
        neighbor_confidence_scores = [n.verification_confidence for n in neighbors]

    return {
        "top_k_neighbor_ids": top_k_neighbor_ids,
        "neighbor_confidence_scores": neighbor_confidence_scores,
        "model_chosen": model_chosen,
        "selection_rule_used": selection_rule_used,
        "fallback_chain_triggered": fallback_chain_triggered,
        "tier_escalation_reason": tier_escalation_reason,
        "router_propensity": router_propensity,
        "candidate_model_scores": candidate_model_scores or {},
        # Decided-once at route time; read back at capture to feed the ConservativeGate,
        # which banks slack only from verified *downshift*-exploration outcomes.
        "downshift": downshift,
    }
