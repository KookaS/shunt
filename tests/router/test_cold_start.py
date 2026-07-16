from __future__ import annotations

from unittest.mock import MagicMock

from shunt.router.cold_start import ColdStartStrategy

from .conftest import FakeModel, FakeModelPool


def _make_pool(*model_names: str) -> FakeModelPool:
    pool = FakeModelPool.__new__(FakeModelPool)
    pool.models = {
        "cheap": [FakeModel(n) for n in model_names],
        "mid": [],
        "frontier": [],
    }
    return pool


class TestColdStartStrategyDefaults:
    def test_default_thresholds(self):
        strategy = ColdStartStrategy()
        assert strategy.threshold_tier2 == 20
        assert strategy.threshold_tier1 == 50

    def test_default_fallback_models(self):
        strategy = ColdStartStrategy()
        assert strategy.fallback_models == ["deepseek-v4-flash", "zai-glm-5.2"]

    def test_env_override_thresholds(self, monkeypatch):
        monkeypatch.setenv("SHUNT_COLD_START_THRESHOLD_TIER2", "10")
        monkeypatch.setenv("SHUNT_COLD_START_THRESHOLD_TIER1", "25")
        strategy = ColdStartStrategy()
        assert strategy.threshold_tier2 == 10
        assert strategy.threshold_tier1 == 25


class TestColdStartIsActive:
    def test_active_when_below_both_thresholds(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=5, count_tier2=0) is True

    def test_inactive_when_tier2_meets_threshold(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=30, count_tier2=20) is False

    def test_inactive_when_tier2_exceeds_threshold(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=30, count_tier2=25) is False

    def test_inactive_when_total_labeled_meets_threshold(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=50, count_tier2=0) is False

    def test_inactive_when_total_labeled_exceeds_threshold(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=100, count_tier2=5) is False

    def test_active_at_zero_sessions(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=0, count_tier2=0) is True

    def test_tier2_overrides_low_total(self):
        strategy = ColdStartStrategy(threshold_tier2=3, threshold_tier1=50)
        assert strategy.is_active(count_labeled=3, count_tier2=3) is False

    def test_high_total_overrides_low_tier2(self):
        strategy = ColdStartStrategy(threshold_tier2=100, threshold_tier1=10)
        assert strategy.is_active(count_labeled=10, count_tier2=0) is False

    def test_exact_boundary_tier2_inactive(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=19, count_tier2=20) is False

    def test_exact_boundary_total_inactive(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=50, count_tier2=19) is False

    def test_one_below_tier2_threshold_active(self):
        strategy = ColdStartStrategy(threshold_tier2=20, threshold_tier1=50)
        assert strategy.is_active(count_labeled=19, count_tier2=19) is True


class TestColdStartSelect:
    def test_prefers_qwen3_5_plus_when_healthy(self):
        pool = MagicMock()
        pool.is_healthy.return_value = True
        strategy = ColdStartStrategy()
        model = strategy.select(pool)
        assert model == "qwen3.7-plus"
        pool.is_healthy.assert_called_once_with("qwen3.7-plus")

    def test_falls_back_when_qwen_unhealthy(self):
        pool = MagicMock()
        pool.is_healthy.side_effect = lambda m: m != "qwen3.7-plus"
        strategy = ColdStartStrategy()
        model = strategy.select(pool)
        assert model == "deepseek-v4-flash"

    def test_falls_back_through_chain(self):
        pool = MagicMock()
        pool.is_healthy.side_effect = lambda m: m == "zai-glm-5.2"
        strategy = ColdStartStrategy()
        model = strategy.select(pool)
        assert model == "zai-glm-5.2"

    def test_escalates_when_all_fallback_exhausted(self):
        pool = MagicMock()
        # Fallback models unhealthy, but escalation models are healthy
        pool.is_healthy.side_effect = lambda m: (
            m == "deepseek-v4-flash" or m == "emergency-fallback"
        )
        pool.get_tier_models.side_effect = lambda tier: {
            "cheap": [FakeModel("deepseek-v4-flash"), FakeModel("emergency-fallback")],
            "mid": [],
            "frontier": [],
        }.get(tier, [])
        strategy = ColdStartStrategy()
        model = strategy.select(pool)
        assert model == "deepseek-v4-flash"

    def test_ultimate_fallback_returns_default(self):
        pool = MagicMock()
        pool.is_healthy.return_value = False
        pool.get_tier_models.return_value = []
        strategy = ColdStartStrategy()
        model = strategy.select(pool)
        assert model == "qwen3.7-plus"

    def test_custom_fallback_models(self):
        pool = MagicMock()
        pool.is_healthy.side_effect = lambda m: m == "custom-fallback"
        strategy = ColdStartStrategy(
            fallback_models=["custom-fallback"],
        )
        model = strategy.select(pool)
        assert model == "custom-fallback"
