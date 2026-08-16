"""bootstrap_ci may resample cache-aware cost because it is scoped per task.

A task's cache cost is invariant to the set/order of the other tasks, so a whole-task
resample preserves each task's attempt adjacency — and a guard re-arms the refusal if that breaks.
"""

from __future__ import annotations

import pytest

from benchmark.routing import cache_cost
from benchmark.routing import metrics as metrics_mod
from benchmark.routing.cache_cost import CachePrice, cache_aware_total
from benchmark.routing.metrics import bootstrap_ci
from benchmark.routing.strategies.fixed import AlwaysFrontier
from benchmark.routing.summary import SUMMARY_FIELDS, compute_strategy_rows


def _price(model: str, share: float, discount: float) -> CachePrice:
    return CachePrice(
        model=model,
        input_share=share,
        discount=discount,
        hit_rate=1.0,
        provenance=cache_cost.MEASURED,
        share_provenance=cache_cost.MEASURED,
    )


class TestCacheCostPerTaskScoping:
    """A task's cache cost is invariant to the set and order of the other tasks."""

    def test_a_tasks_cache_cost_is_invariant_to_the_other_tasks(self):
        prices = {
            "m": _price("m", 1.0, 0.5),
            "frontier": _price("frontier", 1.0, 1.0),
        }
        # t1 holds the ONLY within-task repeat — the only place a discount may fire.
        # t2 routes to the same model as t1 but is an INDEPENDENT session: cold prefix.
        attempts = {
            "t1": [("m", 10.0), ("m", 10.0)],
            "t2": [("m", 10.0)],
            "t3": [("frontier", 10.0)],
            "t4": [("m", 5.0)],
        }

        def cost(tid: str) -> float:
            return cache_aware_total(attempts[tid], prices)

        def total(order: list[str]) -> float:
            return sum(cost(t) for t in order)

        # The within-task repeat is priced once, whatever else is in the sample.
        assert cost("t1") == 15.0
        # t2's single attempt is full price even right after t1 on the SAME model: two
        # sessions are two cold prefixes (the old flat model banked a discount here).
        assert total(["t1", "t2", "t3"]) == 15.0 + 10.0 + 10.0
        assert total(["t3", "t1", "t4"]) == 10.0 + 15.0 + 5.0
        assert total(["t2", "t4", "t3"]) == 10.0 + 5.0 + 10.0
        assert total(["t1", "t2", "t4"]) == 15.0 + 10.0 + 5.0


class TestCacheAwareBootstrapCI:
    """The refusal is lifted; cache-aware cost carries a paired bootstrap CI."""

    def test_ci_is_emitted_and_brackets_the_cache_aware_point_estimate(self):
        prices = {"m": _price("m", 1.0, 0.5)}
        # t1 repeats m within the task (second attempt discounted: 15); t2/t3 single (10 each).
        decisions = [("t1", "m", True, 20.0), ("t2", "m", True, 10.0), ("t3", "m", True, 10.0)]
        attempts = {
            "t1": [("m", 10.0), ("m", 10.0)],
            "t2": [("m", 10.0)],
            "t3": [("m", 10.0)],
        }
        cis = bootstrap_ci(decisions, decisions, n_bootstrap=300, attempts=attempts, prices=prices)
        point = cache_aware_total(attempts["t1"], prices) + 10.0 + 10.0
        assert point < sum(d[3] for d in decisions)  # the discount moved the statistic
        assert cis.total_cost_cacheaware != (0.0, 0.0)
        assert cis.total_cost_cacheaware[0] <= point <= cis.total_cost_cacheaware[1]
        assert cis.avg_cost_cacheaware[0] <= point / 3 <= cis.avg_cost_cacheaware[1]

    def test_old_contract_without_a_price_map_stays_zero(self):
        decisions = [(f"t{i}", "m", True, float(i)) for i in range(20)]
        cis = bootstrap_ci(decisions, decisions, n_bootstrap=200)
        assert cis.total_cost_cacheaware == (0.0, 0.0)
        assert cis.avg_cost_cacheaware == (0.0, 0.0)

    def test_half_specified_cache_scope_is_refused(self):
        decisions = [("t1", "m", True, 10.0)]
        attempts = {"t1": [("m", 10.0)]}
        with pytest.raises(ValueError, match="BOTH"):
            bootstrap_ci(decisions, decisions, n_bootstrap=10, attempts=attempts)

    def test_summary_row_carries_the_cache_aware_ci(self, monkeypatch):
        monkeypatch.setattr(
            "benchmark.routing.summary.config.impute_config", lambda: {"enabled": False}
        )
        monkeypatch.setattr(
            "benchmark.routing.summary.cache_prices",
            lambda models, shares=None: {"cheap": _price("cheap", 1.0, 0.5)},
        )
        rows = compute_strategy_rows(_matrix(), ["t1", "t2"], [AlwaysFrontier()], bootstrap=200)
        row = rows[0]
        for field in (
            "TotalCost_cacheaware_ci_lower",
            "TotalCost_cacheaware_ci_upper",
            "AvgCost_cacheaware_ci_lower",
            "AvgCost_cacheaware_ci_upper",
        ):
            assert field in row, f"missing {field}"
            assert isinstance(row[field], float)
            assert field in SUMMARY_FIELDS
        assert row["TotalCost_cacheaware_ci_lower"] <= row["TotalCost_cacheaware_ci_upper"]
        assert (
            row["TotalCost_cacheaware_ci_lower"]
            <= row["TotalCost_cacheaware"]
            <= row["TotalCost_cacheaware_ci_upper"]
        )


class TestScopingGuard:
    """The guard re-fires if a future cross-run cache model breaks per-task scoping."""

    def test_refuses_when_cache_cost_is_not_scoped_per_task(self, monkeypatch):
        state: dict[str, str | None] = {"prev": None}

        def flat_cache_cost(attempts, prices):
            # A cross-run cache model: the discount bleeds between tasks through hidden
            # state — task B inherits task A's warm prefix, as the old flat implementation did.
            total = sum(c for _m, c in attempts)
            model = attempts[-1][0] if attempts else None
            if state["prev"] is not None and model is not None and state["prev"] == model:
                total -= 1.0
            state["prev"] = model
            return total

        monkeypatch.setattr(metrics_mod, "cache_aware_total", flat_cache_cost)
        decisions = [("t1", "m", True, 10.0), ("t2", "m", True, 10.0), ("t3", "m", True, 10.0)]
        attempts = {"t1": [("m", 10.0)], "t2": [("m", 10.0)], "t3": [("m", 10.0)]}
        with pytest.raises(RuntimeError, match="not scoped per task"):
            bootstrap_ci(
                decisions,
                decisions,
                n_bootstrap=10,
                attempts=attempts,
                prices={"m": _price("m", 1.0, 0.5)},
            )

    def test_guard_stays_silent_on_scoped_data(self):
        decisions = [("t1", "m", True, 10.0), ("t2", "m", True, 10.0)]
        attempts = {"t1": [("m", 10.0)], "t2": [("m", 10.0)]}
        cis = bootstrap_ci(
            decisions,
            decisions,
            n_bootstrap=20,
            attempts=attempts,
            prices={"m": _price("m", 1.0, 0.5)},
        )
        assert cis.total_cost_cacheaware != (0.0, 0.0)


def _matrix() -> dict:
    """Two tasks, both scored on the frontier cell (no coverage selection)."""
    return {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.1},
            "frontier": {"input_price": 5.0, "output_price": 5.0},
        },
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {"cheap": {"pass": False, "cost": 1.0}, "frontier": {"pass": True, "cost": 10.0}},
            "t2": {"cheap": {"pass": True, "cost": 1.0}, "frontier": {"pass": True, "cost": 10.0}},
        },
    }
