"""Tests for the report plot helpers — the phantom-frontier guard in particular."""

from __future__ import annotations

from benchmark import config
from benchmark.routing import report, run_eval
from benchmark.routing.impute import ImputedMatrix


class TestStrategyFactoriesMatchEnabledSet:
    """The regret plot's strategy set must derive from the same config-enabled
    source every other plot uses (run_eval.get_strategies) — no headline
    strategy silently dropped, no strategy added that isn't config-enabled."""

    def test_every_enabled_strategy_has_a_factory(self):
        config.load("benchmark/benchmark.yaml")
        enabled_names = {s.name for s in run_eval.get_strategies()}
        factories = report._build_strategy_factories(config.gamma())
        assert enabled_names <= factories.keys()

    def test_shipped_knn_is_not_silently_dropped(self):
        # knn is enabled in benchmark.yaml's strategies.enabled — the shipped
        # algorithm, and the headline strategy that must appear on the regret plot.
        config.load("benchmark/benchmark.yaml")
        factories = report._build_strategy_factories(config.gamma())
        assert "kNN" in factories

    def test_oracle_reward_always_present_as_internal_reference(self):
        # Oracle-reward is the regret plot's baseline every strategy is scored
        # against — required even when benchmark.yaml comments it out of `enabled`.
        config.load("benchmark/benchmark.yaml")
        assert "oracle_reward" not in config.strategies().get("enabled", [])
        factories = report._build_strategy_factories(config.gamma())
        assert "Oracle-reward" in factories

    def test_no_strategy_added_beyond_enabled_plus_oracle_reward(self):
        config.load("benchmark/benchmark.yaml")
        enabled_names = {s.name for s in run_eval.get_strategies()}
        factories = report._build_strategy_factories(config.gamma())
        assert factories.keys() == enabled_names | {"Oracle-reward"}


class TestDisabledModelExcluded:
    def test_disabled_model_cannot_leak_via_stray_row(self):
        # A disabled model (opus) with a stray results row must never re-enter the
        # matrix — otherwise it silently re-promotes to frontier (opus $30 > k3 $18).
        config.load("benchmark/benchmark.yaml")
        stray = {
            "t1": {
                "deepseek-v4-flash": {"pass": True, "cost": 0.01},
                "claude-opus-4-6": {"pass": True, "cost": 0.2},  # disabled in config
            }
        }
        assert "claude-opus-4-6" not in config.models_matrix(stray)


def _matrix(results: dict) -> dict:
    # Two models; opus is far more expensive so it is the "frontier" pick.
    return {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.2},
            "opus": {"input_price": 5.0, "output_price": 25.0},
        },
        "results": results,
    }


def _imputed(n_multi_observed: int, violations: list | None = None) -> ImputedMatrix:
    return ImputedMatrix(
        matrix={},
        violations=list(violations or []),
        n_real=0,
        n_imputed=0,
        n_unknown=0,
        tau={},
        n_multi_observed=n_multi_observed,
    )


class TestSyntheticArmCacheMakesNoArmClaim:
    """_synthesize_raw collapses every model to one 'default' arm when no per-arm
    cache exists — that is missing data, not a measured single-arm sweep."""

    _single = {"t1": {"m1": {"default": {"pass": True, "cost": 0.01}}}}

    def test_synthesized_cache_suppresses_the_single_arm_claim(self):
        assert report._single_arm_limits(self._single, synthesized=True) == ()

    def test_loaded_single_arm_cache_still_makes_the_claim(self):
        limits = report._single_arm_limits(self._single, synthesized=False)
        assert len(limits) == 1
        assert "exactly one sampled arm" in limits[0]


class TestMonotonicityUnmeasuredIsNotMeasured:
    """violation_ci returns (0, 0, 0) at n=0 — a vacuous denominator, not a perfect run."""

    def test_zero_multi_observed_makes_no_measured_claim(self):
        line = report._violation_line(_imputed(0))
        assert "UNMEASURED" in line
        assert "measured, not assumed" not in line
        assert "v̂=0.000" not in line

    def test_real_denominator_still_reports_the_measured_rate(self):
        line = report._violation_line(_imputed(8))
        assert "v̂=0.000" in line
        assert "8 multi-observed tasks" in line
        assert "measured, not assumed" in line

    def test_disclosure_banner_does_not_assert_100_percent_on_zero_observations(self):
        banner = report._disclosure_banner(_imputed(0), [{"n_tasks": 3}])
        assert banner is not None
        assert "UNVERIFIED" in banner
        assert "measured, not assumed" not in banner
