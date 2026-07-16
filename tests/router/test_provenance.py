from __future__ import annotations

from shunt.router.provenance import build_provenance
from shunt.router.selection import NeighborResult


def _neighbor(
    model: str = "model-a",
    outcome: bool = True,
    cost: float = 1.0,
    confidence: float = 0.9,
    distance: float = 0.1,
    session_id: str = "s1",
) -> NeighborResult:
    return NeighborResult(
        model=model,
        outcome=outcome,
        cost=cost,
        verification_confidence=confidence,
        distance=distance,
        session_id=session_id,
        truncation_rate=0.0,
    )


class TestBuildProvenance:
    def test_minimal_provenance(self):
        prov = build_provenance(
            model_chosen="model-a",
            selection_rule_used="cold_start",
        )
        assert prov["model_chosen"] == "model-a"
        assert prov["selection_rule_used"] == "cold_start"
        assert prov["fallback_chain_triggered"] is False
        assert prov["tier_escalation_reason"] is None
        assert prov["top_k_neighbor_ids"] == []
        assert prov["neighbor_confidence_scores"] == []
        assert prov["candidate_model_scores"] == {}
        assert prov["router_propensity"] == 1.0

    def test_with_neighbors(self):
        neighbors = [
            _neighbor(session_id="s1", confidence=0.9),
            _neighbor(session_id="s2", confidence=0.8),
        ]
        prov = build_provenance(
            model_chosen="model-a",
            selection_rule_used="cheapest_above_threshold",
            neighbors=neighbors,
            fallback_chain_triggered=False,
            router_propensity=0.95,
            candidate_model_scores={"model-a": 0.95, "model-b": 0.70},
        )
        assert prov["top_k_neighbor_ids"] == ["s1", "s2"]
        assert prov["neighbor_confidence_scores"] == [0.9, 0.8]
        assert prov["candidate_model_scores"]["model-a"] == 0.95
        assert prov["router_propensity"] == 0.95
        assert prov["fallback_chain_triggered"] is False

    def test_fallback_provenance(self):
        prov = build_provenance(
            model_chosen="frontier-model",
            selection_rule_used="safe_fallback",
            fallback_chain_triggered=True,
            tier_escalation_reason="safe_fallback",
            router_propensity=0.5,
            candidate_model_scores={"cheap": 0.3, "mid": 0.5},
        )
        assert prov["fallback_chain_triggered"] is True
        assert prov["tier_escalation_reason"] == "safe_fallback"
        assert prov["router_propensity"] == 0.5

    def test_empty_neighbors(self):
        prov = build_provenance(
            model_chosen="model-a",
            selection_rule_used="exploration_untested",
            neighbors=[],
            fallback_chain_triggered=True,
            tier_escalation_reason="exploration_untested",
        )
        assert prov["top_k_neighbor_ids"] == []
        assert prov["neighbor_confidence_scores"] == []

    def test_serializable(self):
        prov = build_provenance(
            model_chosen="m",
            selection_rule_used="cheapest_above_threshold",
            neighbors=[_neighbor(session_id="s1")],
            fallback_chain_triggered=False,
            router_propensity=0.85,
            candidate_model_scores={"m": 0.85, "n": 0.60},
        )
        import json

        serialized = json.dumps(prov)
        parsed = json.loads(serialized)
        assert parsed["model_chosen"] == "m"
        assert parsed["top_k_neighbor_ids"] == ["s1"]
        assert parsed["router_propensity"] == 0.85
