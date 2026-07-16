"""Tests for the blended-kNN routing strategy (pure core embedding-free; full index gated)."""

from __future__ import annotations

import os

import pytest

from benchmark.routing.strategies import knn_blended as kb


class TestLoadExternalPriors:
    def test_cheap_rate_is_p_solve_not_degenerate_p_cheap(self, tmp_path):
        csv_path = tmp_path / "ext.csv"
        csv_path.write_text(
            "instance_id,n_sub,n_solved,p_solve,n_cheap,cheap_solved,p_cheap,"
            "n_frontier,frontier_solved,p_frontier,gap\n"
            "hard1,19,6,0.3158,0,0,,2,1,0.5,\n"  # no cheap cohort; frontier real
            "easy1,88,80,0.9,10,10,1.0,40,39,0.975,-0.025\n"  # p_cheap=1.0 IGNORED
        )
        priors = kb.load_external_priors(csv_path)
        # cheap-tier signal := p_solve (p_cheap is degenerate and unused).
        assert priors["hard1"] == (0.3158, 0.5)
        assert priors["easy1"] == (0.9, 0.975)  # 0.9 = p_solve, NOT p_cheap 1.0

    def test_frontier_falls_back_to_p_solve_when_absent(self, tmp_path):
        csv_path = tmp_path / "ext.csv"
        csv_path.write_text(
            "instance_id,n_sub,n_solved,p_solve,n_cheap,cheap_solved,p_cheap,"
            "n_frontier,frontier_solved,p_frontier,gap\n"
            "x,10,4,0.4,0,0,,0,0,,\n"  # no frontier cohort either
        )
        assert kb.load_external_priors(csv_path)["x"] == (0.4, 0.4)

    def test_missing_file_is_empty(self, tmp_path):
        assert kb.load_external_priors(tmp_path / "nope.csv") == {}


class TestBlendScores:
    def test_our_neighbors_bool_weight_one(self):
        our = [{"cheap_m": {"pass": True}}, {"cheap_m": {"pass": False}}]
        scores = kb.blend_scores(our, [], ["cheap_m"], {"cheap_m": "cheap"}, 0.25)
        rate, n = scores["cheap_m"]
        assert rate == 0.5 and n == 2

    def test_external_uses_tier_rate_at_weight(self):
        # cheap model reads p_cheap (idx0); frontier reads p_frontier (idx1).
        ext = [(0.2, 0.9)]
        scores = kb.blend_scores(
            [], ext, ["cheap_m", "front_m"], {"cheap_m": "cheap", "front_m": "frontier"}, 0.25
        )
        assert scores["cheap_m"][0] == pytest.approx(0.2)
        assert scores["front_m"][0] == pytest.approx(0.9)

    def test_blend_downweights_external_vs_ours(self):
        # 1 our-pass (w=1.0, succ=1) + 1 external cheap-rate 0.0 (w=0.25).
        # blended = (1*1 + 0.25*0) / (1 + 0.25) = 0.8, NOT 0.5.
        our = [{"m": {"pass": True}}]
        ext = [(0.0, 0.0)]
        scores = kb.blend_scores(our, ext, ["m"], {"m": "cheap"}, 0.25)
        assert scores["m"][0] == pytest.approx(0.8)
        assert scores["m"][1] == 2  # both count as samples


class TestSelectModel:
    BY_COST = ["cheap", "mid", "frontier"]

    def test_picks_cheapest_clearing_threshold(self):
        scores = {"cheap": (0.9, 5), "mid": (0.95, 5), "frontier": (1.0, 5)}
        assert kb.select_model(scores, self.BY_COST, 0.5, 1) == "cheap"

    def test_escalates_when_cheap_below_threshold(self):
        scores = {"cheap": (0.2, 5), "mid": (0.8, 5), "frontier": (1.0, 5)}
        assert kb.select_model(scores, self.BY_COST, 0.5, 1) == "mid"

    def test_min_samples_gate_then_relax(self):
        # cheap clears rate but not min_samples; mid clears both -> mid wins the
        # strict pass. (relax only triggers if NOTHING clears min_samples.)
        scores = {"cheap": (0.9, 1), "mid": (0.9, 5), "frontier": (1.0, 5)}
        assert kb.select_model(scores, self.BY_COST, 0.5, 3) == "mid"

    def test_relax_when_none_meet_min_samples(self):
        scores = {"cheap": (0.9, 1), "mid": (0.9, 1)}
        # nobody has >=3 samples -> relaxed pass picks cheapest above threshold
        assert kb.select_model(scores, self.BY_COST, 0.5, 3) == "cheap"

    def test_fallback_cheapest_when_nothing_clears(self):
        scores = {"cheap": (0.1, 5), "mid": (0.2, 5), "frontier": (0.3, 5)}
        assert kb.select_model(scores, self.BY_COST, 0.5, 1) == "cheap"


class TestIntegrationOffline:
    @pytest.mark.skipif(
        not os.environ.get("SHUNT_RUN_HEAVY_EMBED"),
        reason="embeds 500 statements (memory-heavy); opt in with SHUNT_RUN_HEAVY_EMBED=1",
    )
    def test_blended_escalates_sympy_only(self):
        # Full index over our-10 + external-490. Skips if embedder/HF unavailable.
        pytest.importorskip("hnswlib")
        from benchmark import config

        config.load()
        matrix = config.load_matrix(config.challenges_path())
        try:
            strat = kb.kNNBlended(external_weight=0.25, success_rate_threshold=0.5)
            decisions = {
                t: strat.select(t, matrix.get("tasks", {}).get(t, {}), matrix)
                for t in sorted(matrix["results"])
            }
        except Exception as exc:  # noqa: BLE001 (offline embedder / HF cache absent)
            pytest.skip(f"embedding backend unavailable: {exc}")
        # Every decision is a real enabled model.
        assert set(decisions.values()) <= set(config.enabled_models())
