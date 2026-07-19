from __future__ import annotations

from shunt.models import TIER_ORDER, ModelPool
from shunt.router.selection import NeighborResult, SelectionRule

from .conftest import FakeModel, FakeModelPool


def _neighbor(
    model: str,
    outcome: bool = True,
    cost: float = 1.0,
    confidence: float = 0.9,
    distance: float = 0.1,
) -> NeighborResult:
    return NeighborResult(
        model=model,
        outcome=outcome,
        cost=cost,
        verification_confidence=confidence,
        distance=distance,
        session_id="test",
        truncation_rate=0.0,
    )


class TestColdStart:
    def test_cold_start_returns_cold_start_model(self):
        rule = SelectionRule(cold_start_threshold=20)
        pool = FakeModelPool("qwen3.7-plus", "gpt4")
        model, reason = rule.select([], pool, cold_start_active=True)
        assert model == "qwen3.7-plus"
        assert reason == "cold_start"

    def test_cold_start_custom_model(self):
        rule = SelectionRule(cold_start_threshold=20, cold_start_model="claude-sonnet-4")
        pool = FakeModelPool("claude-sonnet-4", "gpt4")
        model, reason = rule.select([], pool, cold_start_active=True)
        assert model == "claude-sonnet-4"

    def test_not_cold_start_with_enough_sessions(self):
        rule = SelectionRule(cold_start_threshold=5)
        pool = FakeModelPool("model-a", "model-b")
        model, reason = rule.select(
            [
                _neighbor("model-a", outcome=True, cost=1.0),
                _neighbor("model-a", outcome=True, cost=1.0),
                _neighbor("model-a", outcome=True, cost=1.0),
            ],
            pool,
            cold_start_active=False,
        )
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"


class TestEligibility:
    def test_empty_neighbors_no_cold_start_escalates(self):
        rule = SelectionRule()
        pool = FakeModelPool("cheap-model", "mid-model")
        model, reason = rule.select([], pool, cold_start_active=False)
        assert model == "cheap-model"
        assert reason == "exploration_untested"

    def test_single_model_meets_threshold(self):
        rule = SelectionRule()
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [  # 3 samples > threshold
            _neighbor("model-a", outcome=True, cost=5.0),
            _neighbor("model-a", outcome=True, cost=5.0),
            _neighbor("model-a", outcome=True, cost=5.0),
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"

    def test_below_min_samples_not_eligible(self):
        rule = SelectionRule(min_samples=3)
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [  # only 2 samples
            _neighbor("model-a", outcome=True, cost=5.0),
            _neighbor("model-a", outcome=True, cost=5.0),
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        # model-a is tested (in neighbors) but below min_samples.
        # Escalation should pick the first untested model.
        assert model == "model-b"
        assert reason == "exploration_untested"

    def test_low_success_rate_not_eligible(self):
        rule = SelectionRule(min_success_rate=0.7)
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0),
            _neighbor("model-a", outcome=False, cost=5.0),
            _neighbor("model-a", outcome=False, cost=5.0),
        ]
        # success rate = 1/3 ≈ 0.33 < 0.7
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert reason == "exploration_untested"

    def test_exact_threshold_eligible(self):
        rule = SelectionRule(min_success_rate=0.7)
        pool = FakeModelPool("model-a", "model-b")
        # 7 successes out of 10 = 0.7 exactly → >= 0.7 qualifies
        neighbors = [
            _neighbor("model-a", outcome=True, cost=1.0, confidence=1.0, distance=0.0)
            for _ in range(7)
        ] + [
            _neighbor("model-a", outcome=False, cost=1.0, confidence=1.0, distance=0.0)
            for _ in range(3)
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"

    def test_zero_total_weight_skips_group(self):
        rule = SelectionRule()
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.0, distance=1.0),
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.0, distance=0.5),
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        # model-a has zero total weight → skipped, escalation
        assert reason in ("exploration_untested", "safe_fallback")


class TestCheapestAboveThreshold:
    def test_cheapest_among_eligible_wins(self):
        rule = SelectionRule()
        pool = FakeModelPool("cheap", "mid", "expensive")
        neighbors = (
            [_neighbor("expensive", outcome=True, cost=10.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("mid", outcome=True, cost=5.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("cheap", outcome=True, cost=2.0, confidence=0.9) for _ in range(5)]
        )
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "cheap"
        assert reason == "cheapest_above_threshold"

    def test_mixed_eligibility_picks_cheapest_eligible(self):
        rule = SelectionRule()
        pool = FakeModelPool("cheap-fail", "cheap-pass", "mid", "expensive")
        neighbors = (
            [_neighbor("cheap-fail", outcome=False, cost=1.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("cheap-pass", outcome=True, cost=2.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("expensive", outcome=True, cost=10.0, confidence=0.9) for _ in range(5)]
        )
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "cheap-pass"


class TestEscalation:
    def test_untested_model_explored_first(self):
        rule = SelectionRule()
        pool = FakeModelPool("untested-cheap", "untested-mid", "tested")
        neighbors = [_neighbor("tested", outcome=False, cost=5.0, confidence=0.9) for _ in range(5)]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "untested-cheap"
        assert reason == "exploration_untested"

    def test_all_tested_none_eligible_safe_fallback(self):
        rule = SelectionRule()
        pool = FakeModelPool("cheap-fail", "mid-fail", "frontier-fail")
        neighbors = (
            [_neighbor("cheap-fail", outcome=False, cost=1.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("mid-fail", outcome=False, cost=2.0, confidence=0.9) for _ in range(5)]
            + [
                _neighbor("frontier-fail", outcome=False, cost=10.0, confidence=0.9)
                for _ in range(5)
            ]
        )
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        # All tested and none eligible — should fall back to smartest
        assert reason == "safe_fallback"
        assert model == "frontier-fail"

    def test_mixed_tier_pool(self):
        pool = FakeModelPool.__new__(FakeModelPool)
        pool.models = {
            "cheap": [FakeModel("cheapy")],
            "mid": [FakeModel("middy")],
            "frontier": [FakeModel("fronty")],
        }
        rule = SelectionRule()
        # No neighbors → escalate
        model, reason = rule.select([], pool, cold_start_active=False)
        assert model == "cheapy"
        assert reason == "exploration_untested"

        # All tested → safe fallback to frontier
        neighbors = (
            [_neighbor("cheapy", outcome=False, cost=1.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("middy", outcome=False, cost=2.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("fronty", outcome=False, cost=3.0, confidence=0.9) for _ in range(5)]
        )
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "fronty"
        assert reason == "safe_fallback"

    def test_safe_fallback_on_the_real_pool_is_opus(self):
        # _escalate returns get_tier_models(tier)[-1] — the LAST FRONTIER ROW IN
        # YAML ORDER, not the smartest model. The two coincide only because
        # claude-opus-4-6 is last in default_config.yaml. Appending a frontier
        # model after it would silently redirect every safe_fallback to it, and
        # the single-model-per-tier fakes above cannot catch that. This pins it.
        rule = SelectionRule()
        pool = ModelPool()
        names = [m.name for t in TIER_ORDER for m in pool.get_tier_models(t)]
        neighbors = [
            _neighbor(n, outcome=False, cost=1.0, confidence=0.9) for n in names for _ in range(5)
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert reason == "safe_fallback"
        assert model == "claude-opus-4-6"


class TestConfidenceWeighting:
    def test_higher_confidence_more_influence(self):
        rule = SelectionRule(min_samples=2)
        pool = FakeModelPool("model-a", "model-b")
        # Both have equal outcomes, but model-a has high confidence
        # and model-b has low confidence outcomes
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=1.0, distance=0.0),
            _neighbor("model-a", outcome=True, cost=5.0, confidence=1.0, distance=0.0),
            _neighbor("model-b", outcome=True, cost=3.0, confidence=0.1, distance=0.0),
            _neighbor("model-b", outcome=True, cost=3.0, confidence=0.1, distance=0.0),
        ]
        # Both are eligible, model-b is cheaper, should win
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "model-b"
        assert reason == "cheapest_above_threshold"

    def test_low_confidence_failures_less_influential(self):
        rule = SelectionRule(min_samples=2)
        pool = FakeModelPool("model-a", "model-b")
        # model-a: one high-confidence success, many low-confidence failures
        # model-b: one moderate success
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=1.0, distance=0.0),
            _neighbor("model-a", outcome=False, cost=5.0, confidence=0.05, distance=0.7),
            _neighbor("model-a", outcome=False, cost=5.0, confidence=0.05, distance=0.9),
            _neighbor("model-a", outcome=False, cost=5.0, confidence=0.05, distance=0.8),
            _neighbor("model-b", outcome=True, cost=3.0, confidence=0.9, distance=0.0),
            _neighbor("model-b", outcome=True, cost=3.0, confidence=0.9, distance=0.0),
        ]
        # Weighted success for model-a: high-confidence success dominates
        # model-b is cheaper and eligible
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "model-b"
