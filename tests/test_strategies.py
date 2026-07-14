"""Tests for routing strategies: Oracle, AlwaysCheap, AlwaysFrontier, Random."""

import sys
from pathlib import Path

_routing_dir = Path(__file__).resolve().parent.parent / "benchmark" / "routing"
sys.path.insert(0, str(_routing_dir))

from strategies.fixed import AlwaysCheap, AlwaysFrontier, Random  # noqa: E402
from strategies.oracle import Oracle  # noqa: E402


def make_matrix():
    return {
        "models": {
            "cheap-model": {"input_price": 0.10, "output_price": 0.10},
            "mid-model": {"input_price": 1.00, "output_price": 1.00},
            "frontier-model": {"input_price": 5.00, "output_price": 5.00},
            "failing-cheap": {"input_price": 0.10, "output_price": 0.10},
        },
        "results": {
            "task-cheapest-passes": {
                "cheap-model": {"pass": True, "cost": 1.0},
                "mid-model": {"pass": True, "cost": 2.5},
                "frontier-model": {"pass": True, "cost": 10.0},
                "failing-cheap": {"pass": False, "cost": 0.5},
            },
            "task-only-frontier-passes": {
                "failing-cheap": {"pass": False, "cost": 0.5},
                "frontier-model": {"pass": True, "cost": 10.0},
            },
            "task-all-fail": {
                "model-a": {"pass": False, "cost": 2.0},
                "model-b": {"pass": False, "cost": 3.0},
            },
            "task-no-results": {},
        }
    }


class TestOracle:
    def test_cheapest_passing_model(self):
        oracle = Oracle()
        matrix = make_matrix()
        chosen = oracle.select("task-cheapest-passes", {}, matrix)
        assert chosen == "cheap-model"

    def test_sole_passing_model(self):
        oracle = Oracle()
        matrix = make_matrix()
        chosen = oracle.select("task-only-frontier-passes", {}, matrix)
        assert chosen == "frontier-model"

    def test_fallback_to_cheapest_when_all_fail(self):
        oracle = Oracle()
        matrix = make_matrix()
        chosen = oracle.select("task-all-fail", {}, matrix)
        assert chosen == "model-a"

    def test_empty_results_returns_empty(self):
        oracle = Oracle()
        matrix = make_matrix()
        chosen = oracle.select("task-no-results", {}, matrix)
        assert chosen == ""


class TestAlwaysCheap:
    def test_returns_explicit_model_when_configured(self):
        strategy = AlwaysCheap(model="qwen3.5-plus")
        assert strategy.select("any-task", {}, {}) == "qwen3.5-plus"

    def test_derives_cheapest_from_matrix(self):
        strategy = AlwaysCheap()
        matrix = make_matrix()
        chosen = strategy.select("any-task", {}, matrix)
        # matrix has cheap-model at cost 1.0, mid-model at 2.5, frontier at 10.0
        assert chosen == "cheap-model"

    def test_falls_back_without_matrix(self):
        strategy = AlwaysCheap()
        chosen = strategy.select("any-task", {}, {})
        assert chosen == "deepseek-v4-flash"

    def test_name_is_always_cheap(self):
        strategy = AlwaysCheap()
        assert strategy.name == "Always-Cheap"


class TestAlwaysFrontier:
    def test_derives_frontier_from_matrix(self):
        strategy = AlwaysFrontier()
        matrix = make_matrix()
        chosen = strategy.select("any-task", {}, matrix)
        assert chosen == "frontier-model"


class TestRandom:
    def test_returns_valid_model_name(self):
        strategy = Random(seed=42)
        matrix = make_matrix()
        chosen = strategy.select("task-cheapest-passes", {}, matrix)
        valid_names = {"cheap-model", "mid-model", "frontier-model", "failing-cheap"}
        assert chosen in valid_names

    def test_deterministic(self):
        strategy = Random(seed=42)
        matrix = make_matrix()
        assert strategy.select("task-cheapest-passes", {}, matrix) == strategy.select(
            "task-cheapest-passes", {}, matrix
        )

    def test_different_seed_different_result(self):
        matrix = make_matrix()
        a = Random(seed=42).select("task-cheapest-passes", {}, matrix)
        b = Random(seed=99).select("task-cheapest-passes", {}, matrix)
        assert a != b
