"""Tests for the out-of-sample held-out generalization eval.

Light pieces (reward, tier costs, AUC, neighbour aggregation over a synthetic
index) run always; the full 490-instance embed is gated behind SHUNT_RUN_HEAVY_EMBED.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from benchmark.routing import heldout_eval as he


class TestReward:
    def test_reward_penalizes_cost(self):
        assert he._reward(1.0, 0.0, 0.1) == 1.0
        assert he._reward(1.0, 5.0, 0.1) == pytest.approx(0.5)


class TestTierCosts:
    def test_cheap_le_escalated(self):
        cheap, esc = he._tier_costs()
        assert cheap > 0 and esc > 0
        assert cheap <= esc  # cheapest cheap-tier model is no pricier than escalation


class TestRankAuc:
    def test_perfect_and_none(self):
        # Low score flags label 1: scores [0,1,2,3], labels [1,1,0,0] → perfect (1.0).
        assert he._rank_auc(np.array([0.0, 1, 2, 3]), np.array([1, 1, 0, 0])) == 1.0
        # Reversed → 0.0; single-class → nan.
        assert he._rank_auc(np.array([3.0, 2, 1, 0]), np.array([1, 1, 0, 0])) == 0.0
        assert np.isnan(he._rank_auc(np.array([1.0, 2, 3]), np.array([1, 1, 1])))


class TestNeighbourMean:
    def _index(self):
        import hnswlib

        emb = np.array([[1.0, 0.0], [0.9, 0.44], [0.0, 1.0]], dtype=np.float32)
        idx = hnswlib.Index(space="cosine", dim=2)
        idx.init_index(max_elements=3, ef_construction=50, M=8)
        idx.add_items(emb, np.arange(3))
        idx.set_ef(10)
        return idx, emb

    def test_self_excluded_uses_neighbours(self):
        idx, emb = self._index()
        vals = np.array([0.9, 0.1, 0.5])
        # k=1: self (row 0) excluded; nearest real neighbour is row 1 → 0.1, not 0.9.
        assert he._neighbour_mean(idx, emb, 0, k=1, vals=vals) == pytest.approx(0.1)

    def test_fallback_is_global_mean_not_self(self):
        # If EVERY neighbour is distance-excluded, fall back to the GLOBAL mean,
        # never vals[i] (which would leak the leave-one-out target).
        import hnswlib

        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # identical → both excluded
        idx = hnswlib.Index(space="cosine", dim=2)
        idx.init_index(max_elements=2, ef_construction=50, M=8)
        idx.add_items(emb, np.arange(2))
        idx.set_ef(10)
        vals = np.array([0.9, 0.1])
        # Row 0: its only neighbour (row 1) is distance ~0 and excluded → global mean 0.5,
        # NOT its own 0.9.
        assert he._neighbour_mean(idx, emb, 0, k=1, vals=vals) == pytest.approx(0.5)


class TestFullEval:
    @pytest.mark.skipif(
        not os.environ.get("SHUNT_RUN_HEAVY_EMBED"),
        reason="embeds ~490 statements; opt in with SHUNT_RUN_HEAVY_EMBED=1",
    )
    def test_generalization_signal_is_near_zero(self):
        from benchmark import config

        config.load()
        rep = he.evaluate_heldout()
        names = {r.strategy for r in rep.rows}
        assert {"Always-Cheap", "Neighbour", "Oracle-tier-acc", "Reward-Oracle"} == names
        by = {r.strategy: r for r in rep.rows}
        # Reward-Oracle is the true upper bound → beats always-cheap on reward.
        assert by["Reward-Oracle"].avg_reward >= by["Always-Cheap"].avg_reward
        # Tier-accuracy-oracle is perfect on accuracy by construction.
        assert by["Oracle-tier-acc"].accuracy == pytest.approx(1.0)
        # The honest finding: difficulty does NOT cluster → near-zero predictive signal.
        assert abs(rep.corr) < 0.25
        assert rep.auc < 0.6
