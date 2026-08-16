"""The four instrument defects that produced a wrong cost verdict, each pinned by a test.

Cache-blind cost, a price-ratio proxy for the token mix, a cost point estimate with no CI, and
a coverage-selected subset that announced itself nowhere.
"""

from __future__ import annotations

import pytest

from benchmark.routing import cache_cost, selection_guard, session_cascade_control
from benchmark.routing.metrics import bootstrap_ci
from benchmark.routing.strategies.price_cascade import PriceCascade
from benchmark.routing.summary import SUMMARY_FIELDS, compute_strategy_rows, evaluate_billed


def _price(model: str, share: float, discount: float) -> cache_cost.CachePrice:
    return cache_cost.CachePrice(
        model=model,
        input_share=share,
        discount=discount,
        hit_rate=1.0,
        provenance=cache_cost.MEASURED,
        share_provenance=cache_cost.MEASURED,
    )


class TestCacheAwareCost:
    def test_a_repeat_of_the_same_model_banks_its_discount(self):
        prices = {"m": _price("m", share=1.0, discount=0.5)}
        # Two attempts on one model: the second is half price at share 1.0, hit 1.0, discount 0.5.
        assert cache_cost.cache_aware_total([("m", 10.0), ("m", 10.0)], prices) == 15.0

    def test_a_switch_banks_nothing(self):
        prices = {"m": _price("m", 1.0, 0.5), "n": _price("n", 1.0, 0.5)}
        assert cache_cost.cache_aware_total([("m", 10.0), ("n", 10.0)], prices) == 20.0

    def test_only_adjacent_repeats_bank(self):
        prices = {"m": _price("m", 1.0, 0.5), "n": _price("n", 1.0, 0.5)}
        # m, n, m is three cold prefixes — the discount is not a per-model tally.
        assert cache_cost.cache_aware_total([("m", 10.0), ("n", 10.0), ("m", 10.0)], prices) == 30.0

    def test_an_unpriced_model_is_charged_in_full_not_dropped(self):
        assert cache_cost.cache_aware_total([("ghost", 4.0), ("ghost", 4.0)], {}) == 8.0


class TestMeasuredInputShare:
    """Defect 2: the share is a fact about the measured token mix, not about the price list."""

    def test_share_is_cost_weighted_over_measured_tokens(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "challenge_id,model,in_tok,out_tok\na,m,900,100\nb,m,100,0\n",
        )
        monkeypatch.setattr(
            cache_cost.config,
            "load_pricing",
            lambda *a, **k: {"m": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 4.0}},
        )
        # 1000 in x 1.0 = 1000; 100 out x 4.0 = 400 -> 1000/1400.
        assert cache_cost.measured_input_shares(csv_path)["m"] == pytest.approx(1000 / 1400)

    def test_the_price_ratio_proxy_is_not_what_is_reported(self, tmp_path, monkeypatch):
        # The deleted proxy would have read 1/(1+4) = 0.2 here, regardless of what ran.
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("challenge_id,model,in_tok,out_tok\na,m,900,100\nb,m,100,0\n")
        monkeypatch.setattr(
            cache_cost.config,
            "load_pricing",
            lambda *a, **k: {"m": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 4.0}},
        )
        assert cache_cost.measured_input_shares(csv_path)["m"] > 0.5

    def test_a_model_with_no_measured_tokens_is_flagged_assumed(self, monkeypatch):
        monkeypatch.setattr(cache_cost, "measured_input_shares", lambda *a, **k: {})
        monkeypatch.setattr(cache_cost.config, "load_pricing", lambda *a, **k: {})
        price = cache_cost.cache_prices(["unseen"])["unseen"]
        assert price.share_provenance == cache_cost.ASSUMED
        assert price.input_share == cache_cost.ASSUMED_INPUT_SHARE


class TestCostConfidenceInterval:
    """Defect 3: cost is heavy-tailed here, so a point total may not travel alone."""

    def test_bootstrap_returns_a_cost_ci_that_brackets_the_point_estimate(self):
        decisions = [(f"t{i}", "m", True, float(i)) for i in range(20)]
        cis = bootstrap_ci(decisions, decisions, n_bootstrap=200)
        total = sum(d[3] for d in decisions)
        assert cis.total_cost[0] <= total <= cis.total_cost[1]
        assert cis.avg_cost[0] <= total / len(decisions) <= cis.avg_cost[1]

    def test_a_heavy_tail_widens_the_cost_ci(self):
        flat = [(f"t{i}", "m", True, 1.0) for i in range(30)]
        spiky = [(f"t{i}", "m", True, 30.0 if i == 0 else 0.0) for i in range(30)]
        flat_ci = bootstrap_ci(flat, flat, n_bootstrap=300)
        spiky_ci = bootstrap_ci(spiky, spiky, n_bootstrap=300)
        flat_width = flat_ci.total_cost[1] - flat_ci.total_cost[0]
        spiky_width = spiky_ci.total_cost[1] - spiky_ci.total_cost[0]
        assert spiky_width > flat_width

    def test_no_decisions_yields_zero_bounds_not_a_crash(self):
        cis = bootstrap_ci([], [], n_bootstrap=10)
        assert cis.total_cost == (0.0, 0.0)


def _matrix() -> dict:
    """Two tasks; only `t1` carries the frontier cell, so scoring it selects on coverage."""
    return {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.1},
            "frontier": {"input_price": 5.0, "output_price": 5.0},
        },
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {
                "cheap": {"pass": False, "cost": 1.0},
                "frontier": {"pass": True, "cost": 10.0},
            },
            "t2": {"cheap": {"pass": True, "cost": 1.0}},
        },
    }


class TestSubsetSelectionGuard:
    """Defect 4: a coverage-selected score must say so, on the row itself."""

    def test_warns_when_the_scored_set_is_smaller_than_the_sample(self):
        m = _matrix()
        sel = selection_guard.assess(["t1"], ["t1", "t2"], m, "cheap")
        assert sel.is_subset
        # cheap fails t1 and passes the dropped t2 -> the leftover is measurably easier.
        assert sel.bias_pp == pytest.approx(-100.0)
        assert sel.biased
        assert "difficulty-biased" in sel.note

    def test_stays_silent_on_a_full_sample(self):
        m = _matrix()
        sel = selection_guard.assess(["t1", "t2"], ["t1", "t2"], m, "cheap")
        assert not sel.is_subset
        assert not sel.biased
        assert sel.note == ""
        assert selection_guard.footer({"S": sel.note}) == []

    def test_reference_model_is_the_cheapest_priced(self):
        assert selection_guard.reference_model(_matrix()) == "cheap"

    def test_summary_rows_carry_the_flag_and_the_note(self, monkeypatch):
        monkeypatch.setattr(
            "benchmark.routing.summary.config.impute_config", lambda: {"enabled": False}
        )
        monkeypatch.setattr(
            "benchmark.routing.summary.cache_prices", lambda models, shares=None: {}
        )
        from benchmark.routing.strategies.fixed import AlwaysFrontier

        rows = compute_strategy_rows(_matrix(), ["t1", "t2"], [AlwaysFrontier()], bootstrap=20)
        row = next(r for r in rows if r["strategy"] == "Always-Frontier")
        assert row["subset_selected"] is True
        assert "selected by coverage" in row["subset_note"]
        assert selection_guard.rows_footer(rows)

    def test_every_new_column_is_declared_in_summary_fields(self):
        for field in (
            "TotalCost_cacheaware",
            "AvgCost_cacheaware",
            "TotalCost_ci_lower",
            "AvgCost_ci_upper",
            "Pareto_naive",
            "subset_selected",
            "subset_note",
        ):
            assert field in SUMMARY_FIELDS


class TestBilledAttempts:
    def test_the_summary_does_not_bank_a_discount_across_tasks(self):
        # Two tasks routed to one model are two cold prefixes, not one warm conversation. A model
        # that concatenated the whole run would discount the second task and flatter every
        # fixed-model baseline.
        from benchmark.routing import summary as summary_mod

        prices = {"frontier": _price("frontier", 1.0, 1.0)}
        decisions = [("t1", "frontier", True, 10.0), ("t2", "frontier", True, 10.0)]
        attempts = {"t1": [("frontier", 10.0)], "t2": [("frontier", 10.0)]}
        assert summary_mod._cache_aware_cost(decisions, attempts, prices) == 20.0

    def test_the_summary_does_bank_a_within_task_repeat(self):
        from benchmark.routing import summary as summary_mod

        prices = {"m": _price("m", 1.0, 1.0)}
        decisions = [("t1", "m", True, 20.0)]
        attempts = {"t1": [("m", 10.0), ("m", 10.0)]}
        assert summary_mod._cache_aware_cost(decisions, attempts, prices) == 10.0

    def test_a_cascade_publishes_every_attempt_not_just_the_total(self):
        m = _matrix()
        _dec, _uns, attempts = evaluate_billed(PriceCascade(max_tries=1), m, ["t1"])
        assert attempts["t1"] == [("cheap", 1.0), ("frontier", 10.0)]

    def test_attempts_reconcile_with_the_collapsed_total(self):
        # Two accounting paths for one bill is how a cache-aware column silently disagrees with
        # the naive one it sits beside; this pins them together.
        strategy = PriceCascade(max_tries=1)
        strategy.select("t1", {}, _matrix())
        assert sum(c for _m, c in strategy.cascade_attempts) == pytest.approx(
            strategy.cascade_total_cost
        )

    def test_a_single_shot_decision_is_one_attempt(self):
        from benchmark.routing.strategies.fixed import AlwaysFrontier

        _dec, _uns, attempts = evaluate_billed(AlwaysFrontier(), m := _matrix(), ["t1"])
        assert m and attempts["t1"] == [("frontier", 10.0)]


class TestLadderCertificationBlock:
    """Defect 5: a ladder the positive control never exercised cannot produce a quotable row."""

    def test_the_certified_ladder_is_allowed(self):
        session_cascade_control.assert_ladder_quotable("rank_only")

    def test_an_uncertified_ladder_is_refused(self):
        with pytest.raises(RuntimeError, match="no positive control"):
            session_cascade_control.assert_ladder_quotable("effort_then_rank")

    def test_the_run_path_refuses_before_it_evaluates(self, monkeypatch):
        from benchmark.routing import run_eval

        monkeypatch.setattr(
            run_eval.config,
            "strategies",
            lambda: {
                "enabled": ["session_cascade"],
                "session_cascade": {"ladder": "effort_then_rank"},
            },
        )
        with pytest.raises(RuntimeError, match="no positive control"):
            run_eval.get_strategies()
