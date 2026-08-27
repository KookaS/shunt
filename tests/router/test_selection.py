from __future__ import annotations

from shunt.models import ModelPool
from shunt.router.selection import NeighborResult, SelectionRule

from .conftest import FakeModelPool


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
        pool = FakeModelPool("deepseek-v4-flash", "gpt4")
        model, reason = rule.select([], pool, cold_start_active=True)
        assert model == "deepseek-v4-flash"
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
        # All tested and none eligible — cost-minimal fallback, NOT the strongest
        # (the regression this pins: it used to return the frontier model, the driver
        # of the single-shot control's over-provisioning on the completed matrix).
        assert reason == "safe_fallback"
        assert model == "cheap-fail"

    def test_all_tested_fallback_prefers_cheapest_over_strongest(self):
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
        assert model == "cheap-fail"
        assert reason == "safe_fallback"

    def test_mixed_rank_pool(self):
        # weakest -> strongest by rank
        pool = FakeModelPool("cheapy", "middy", "fronty")
        rule = SelectionRule()
        # No neighbors → escalate
        model, reason = rule.select([], pool, cold_start_active=False)
        assert model == "cheapy"
        assert reason == "exploration_untested"

        # All tested → cost-minimal safe fallback to the cheapest
        neighbors = (
            [_neighbor("cheapy", outcome=False, cost=1.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("middy", outcome=False, cost=2.0, confidence=0.9) for _ in range(5)]
            + [_neighbor("fronty", outcome=False, cost=3.0, confidence=0.9) for _ in range(5)]
        )
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert model == "cheapy"
        assert reason == "safe_fallback"

    def test_safe_fallback_on_the_real_pool_is_the_cheapest_model(self):
        # `_escalate` returns ranked_models()[0] — the cheapest model — when every model
        # is tested and none qualifies. Over the full registry that is deepseek-v4-flash
        # ($0.42/M total), the bottom rank.
        rule = SelectionRule()
        pool = ModelPool()
        names = [m.name for m in pool.ranked_models()]
        neighbors = [
            _neighbor(n, outcome=False, cost=1.0, confidence=0.9) for n in names for _ in range(5)
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert reason == "safe_fallback"
        assert model == names[0]
        assert model == "deepseek-v4-flash"


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

    def test_imputed_passes_do_not_vote_as_measured(self):
        # The benchmark wires imputed cells (monotone-ladder pass=True fills) to
        # verification_confidence 0.0, so a model whose ONLY evidence is imputed passes
        # must not clear the bar on them — its group gets zero total weight and is skipped.
        rule = SelectionRule(min_samples=2, min_success_rate=0.6)
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [
            _neighbor("model-a", outcome=True, cost=1.0, confidence=0.0, distance=0.0)
            for _ in range(5)
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        assert reason in ("exploration_untested", "safe_fallback")
        assert model == "model-b"  # the untested/cheapest model, not the imputed-pass one

    def test_min_samples_counts_only_measured_neighbours(self):
        # min_samples must reflect MEASUREMENT: zero-weight (imputed / distance>=1)
        # neighbours carry no signal and must not satisfy the sample floor. Here
        # model-a has 3 raw neighbours but only 2 measured, so min_samples=3 rejects it.
        rule = SelectionRule(min_samples=3, min_success_rate=0.6)
        pool = FakeModelPool("model-a", "model-b")
        neighbors = [
            _neighbor("model-a", outcome=True, cost=5.0, confidence=1.0, distance=0.0),
            _neighbor("model-a", outcome=True, cost=5.0, confidence=1.0, distance=0.0),
            _neighbor("model-a", outcome=True, cost=5.0, confidence=0.0, distance=0.0),
        ]
        model, reason = rule.select(neighbors, pool, cold_start_active=False)
        # model-a is below the measured floor → escalate to the untested model
        assert model == "model-b"
        assert reason == "exploration_untested"


class TestUnknownCostNeverSortsCheapest:
    """With realized costs the cheap model wins; an UNKNOWN cost (surfaced as +inf
    by the read-back seam) must never sort cheapest and invert the router upward."""

    def _rule(self) -> SelectionRule:
        return SelectionRule(min_success_rate=0.6, min_samples=3)

    def test_cheap_wins_when_costs_realized(self) -> None:
        pool = FakeModelPool("cheap", "frontier")
        neighbors = [_neighbor("cheap", outcome=True, cost=1.0) for _ in range(3)] + [
            _neighbor("frontier", outcome=True, cost=5.0) for _ in range(3)
        ]
        model, reason = self._rule().select(neighbors, pool, cold_start_active=False)
        assert model == "cheap"
        assert reason == "cheapest_above_threshold"

    def test_unknown_cost_model_does_not_sort_cheapest(self) -> None:
        import math

        pool = FakeModelPool("cheap", "frontier")
        # frontier's cost is UNKNOWN → +inf; a 0.0 here (the cost-unknown bug) makes it cheapest.
        neighbors = [_neighbor("cheap", outcome=True, cost=2.0) for _ in range(3)] + [
            _neighbor("frontier", outcome=True, cost=math.inf) for _ in range(3)
        ]
        model, _reason = self._rule().select(neighbors, pool, cold_start_active=False)
        assert model == "cheap"

    def test_zero_weight_unknown_cost_neighbour_does_not_poison_sort(self) -> None:
        import math

        # Regression: a zero-weight neighbour (distance>=1.0 → weight 0) with UNKNOWN cost
        # (+inf) makes the term `0 * inf = nan`, poisoning the group's weighted_cost. Since
        # every nan comparison is False, an unpatched sort leaves the frontier group (inserted
        # FIRST) as "cheapest_above_threshold" over a genuinely cheaper known-cost model — the
        # cost-inversion the engine already guards but SelectionRule did not.
        pool = FakeModelPool("cheap", "frontier")
        frontier = [
            _neighbor("frontier", outcome=True, cost=math.inf, distance=0.1),
            _neighbor("frontier", outcome=True, cost=math.inf, distance=0.1),
            _neighbor("frontier", outcome=True, cost=math.inf, distance=1.0),  # weight 0 → nan term
        ]
        cheap = [_neighbor("cheap", outcome=True, cost=1.0, distance=0.1) for _ in range(3)]
        # frontier inserted first so a nan-poisoned group would win the stable sort.
        model, reason = self._rule().select(frontier + cheap, pool, cold_start_active=False)
        assert model == "cheap"
        assert reason == "cheapest_above_threshold"
