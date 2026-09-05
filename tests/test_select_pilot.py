"""Price-band bucketing in select_pilot — pins the escalation band by price terciles."""

from benchmark.runner.select_pilot import _escalation_models, _price_bands, classify_pattern


def _r(**passes: bool) -> dict:
    return {name: {"pass": ok} for name, ok in passes.items()}


class TestPriceBands:
    def test_bands_split_enabled_pool_into_thirds(self) -> None:
        # 7 enabled models, price ascending: deepseek-v4-flash, deepseek-v4-pro, qwen,
        # gpt-5-mini, kimi-k2.5, zai-glm, kimi-k3 → cheap {deepseek-v4-flash,
        # deepseek-v4-pro}, mid {qwen, gpt-5-mini, kimi-k2.5}, escalation
        # {zai-glm, kimi-k3}. deepseek-v4-pro joined the pool 2026-09-04 and prices into
        # the cheap tercile, which pushed qwen3.7-plus up into mid.
        cheap, mid, escalation = _price_bands()
        assert cheap == {"deepseek-v4-flash", "deepseek-v4-pro"}
        assert mid == {"qwen3.7-plus", "gpt-5-mini", "kimi-k2.5"}
        assert escalation == {"zai-glm-5.2", "kimi-k3"}

    def test_escalation_band_is_the_top_price_tercile(self) -> None:
        assert _escalation_models() == {"zai-glm-5.2", "kimi-k3"}

    def test_buckets_are_disjoint_and_cover_the_enabled_pool(self) -> None:
        cheap, mid, escalation = _price_bands()
        buckets = [cheap, mid, escalation]
        for i, a in enumerate(buckets):
            for b in buckets[i + 1 :]:
                assert a & b == set()
        assert cheap | mid | escalation == {
            "qwen3.7-plus",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "gpt-5-mini",
            "kimi-k2.5",
            "kimi-k3",
            "zai-glm-5.2",
        }


class TestClassifyPattern:
    def test_top_band_alone_passing_is_frontier_only(self) -> None:
        results = _r(
            **{
                "qwen3.7-plus": False,
                "deepseek-v4-flash": False,
                "gpt-5-mini": False,
                "kimi-k2.5": False,
                "kimi-k3": False,
                "zai-glm-5.2": True,
            }
        )
        assert classify_pattern("task-1", results) == "frontier-only"

    def test_all_failing_is_other(self) -> None:
        results = _r(
            **{
                "qwen3.7-plus": False,
                "deepseek-v4-flash": False,
                "gpt-5-mini": False,
                "kimi-k2.5": False,
                "kimi-k3": False,
                "zai-glm-5.2": False,
            }
        )
        assert classify_pattern("task-1", results) == "other"

    def test_cheap_and_mid_passing_is_all_pass(self) -> None:
        results = _r(
            **{
                "qwen3.7-plus": True,
                "deepseek-v4-flash": True,
                "gpt-5-mini": True,
                "kimi-k2.5": True,
            }
        )
        assert classify_pattern("task-1", results) == "all-pass"
