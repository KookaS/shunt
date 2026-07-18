"""Tier bucketing in select_pilot — pins the frontier bucket after the high→frontier migration."""

from benchmark.runner.select_pilot import _models_by_tier, classify_pattern


def _r(**passes: bool) -> dict:
    return {name: {"pass": ok} for name, ok in passes.items()}


class TestTierBuckets:
    def test_frontier_bucket_includes_glm(self) -> None:
        # zai-glm-5.2 used to carry tier "high", which matched NO bucket here, so
        # its results were silently dropped from every pattern. An earlier tier
        # vocabulary migration folded high into frontier; this pins that it now counts.
        assert _models_by_tier("frontier") == {"kimi-k3", "zai-glm-5.2"}

    def test_buckets_are_disjoint_and_cover_the_enabled_pool(self) -> None:
        cheap, mid, frontier = (_models_by_tier(t) for t in ("cheap", "mid", "frontier"))
        assert cheap & mid == set()
        assert cheap & frontier == set()
        assert mid & frontier == set()
        # No enabled model lands in an unmatched tier the way glm silently did.
        assert cheap | mid | frontier == {
            "qwen3.7-plus",
            "deepseek-v4-flash",
            "gpt-5-mini",
            "kimi-k2.5",
            "kimi-k3",
            "zai-glm-5.2",
        }


class TestClassifyPattern:
    def test_glm_alone_passing_is_frontier_only_not_other(self) -> None:
        # The behaviour the tier migration changed: glm passing while kimi-k3
        # fails. Pre-migration glm was invisible here, so frontier_pass was False
        # and this classified as "other".
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

    def test_all_frontier_failing_is_other(self) -> None:
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
