"""Tests for the kill-gate measurement (benchmark/runner/kill_gate.py): BUG 1
(unmeasured cells are UNSCORABLE, excluded from the equal-pass pairing) and BUG 3
(the verdict is driven by the real kNN-semantic-cascade router, and router errors surface)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark import config, validate_results
from benchmark.routing import cache_cost, censoring
from benchmark.routing import metrics as metrics_mod
from benchmark.routing.cache_cost import CachePrice
from benchmark.routing.strategies import BilledAttempt, knn
from benchmark.runner import kill_gate


def _d(
    tid: str, model: str, passed: bool, cost: float, *, scorable: bool = True
) -> kill_gate.Decision:
    """Build an 8-field Decision tuple (last field = scorable)."""
    return (tid, model, passed, cost, 100, 50, 1, scorable)


# ---------------------------------------------------------------------------
# BUG 1 — unscorable cells never enter the equal-pass comparison
# ---------------------------------------------------------------------------


class TestUnscorableCellsExcluded:
    def test_make_decision_marks_missing_cell_unscorable(self):
        present = kill_gate._make_decision("t1", "m", {"pass": True, "cost": 1.0})
        missing = kill_gate._make_decision("t2", "m", {})
        assert present[7] is True
        assert missing[7] is False

    def test_make_decision_marks_censored_cell_unscorable(self):
        # A censored control cell (resource-limit stop) must NOT count as a clean pass=False in
        # the quality denominator — scoring it would understate the frontier baseline's quality.
        censored = kill_gate._make_decision("t1", "m", {"pass": False, "stop_reason": "step_limit"})
        legacy_timeout = kill_gate._make_decision("t2", "m", {"pass": False, "timeout_flag": True})
        real_fail = kill_gate._make_decision("t3", "m", {"pass": False, "stop_reason": "unsolved"})
        assert censored[7] is False  # excluded
        assert legacy_timeout[7] is False  # legacy censored row excluded too
        assert real_fail[7] is True  # a genuine uncensored fail stays scorable

    def test_evaluate_control_marks_missing_frontier_unscorable(self):
        matrix = {
            "results": {
                "t1": {"frontier": {"pass": True, "cost": 1.0}},
                "t2": {"cheap": {"pass": True, "cost": 0.1}},  # frontier NOT measured
            }
        }
        control = kill_gate.evaluate_control(matrix, ["t1", "t2"], "frontier")
        by_task = {d[0]: d for d in control}
        assert by_task["t1"][7] is True
        assert by_task["t2"][7] is False


class TestCensoringGuardFiresWhenImputeIsOff:
    """The censoring guard's fallback path (impute off) must actually fire and be reported."""

    # With the shipped `impute.enabled: true` the matrix is completed FIRST, so a censored cell
    # is imputed and the guard never fires on it; with impute off the guard is the whole
    # mechanism. Pin that it fires and is reported, so it cannot silently rot into dead code.

    def test_censored_cell_is_unscorable_and_reported_when_impute_off(self, monkeypatch):
        monkeypatch.setattr(
            config, "impute_config", lambda: {"enabled": False, "drop_unsolvable": False}
        )
        tasks = ["t1", "t2"]
        matrix = _frontier_matrix(tasks)
        # t1's frontier cell stopped on a resource limit: its true pass/fail is unknown, so it
        # is NOT a clean capability fail. With imputation off, `is_censored` must fire on it.
        matrix["results"]["t1"]["frontier"] = {
            "pass": False,
            "cost": 1.0,
            "stop_reason": "step_limit",
        }
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "router", True, 0.5) for t in task_ids]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        censored_cell = matrix["results"]["t1"]["frontier"]
        assert censoring.is_censored(censored_cell)
        decision = kill_gate._make_decision("t1", "frontier", censored_cell)
        assert decision[7] is False  # the censoring guard excluded it from the pairing

        _, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert "Unscorable (cov gap): 1" in report
        # The report names the handler, so a reader cannot mistake impute-off exclusion for
        # impute-on completion.
        assert "impute.enabled=false" in report

    def test_bootstrap_cost_delta_drops_phantom_zero_fail(self):
        # t2 test-arm cell is unmeasured: recorded (pass=False, cost=0.0). It must
        # NOT pair with control's real (pass=False, cost=5.0) as a fake-cheap point.
        control = [_d("t1", "f", True, 1.0), _d("t2", "f", False, 5.0)]
        test = [_d("t1", "c", True, 0.5), _d("t2", "c", False, 0.0, scorable=False)]
        res = kill_gate.bootstrap_cost_delta(control, test, n_iterations=200)
        assert res["n_eq"] == 1  # only t1, the phantom t2 pair excluded
        assert res["mean"] == pytest.approx(-0.5)  # 0.5 - 1.0, not polluted by -5.0

    def test_bootstrap_cost_ratio_drops_unscorable(self):
        control = [_d("t1", "f", True, 1.0), _d("t2", "f", True, 4.0)]
        test = [_d("t1", "c", True, 0.5), _d("t2", "c", True, 1.0, scorable=False)]
        res = kill_gate.bootstrap_cost_ratio(control, test, n_iterations=200)
        assert res["n_eq"] == 1

    def test_bootstrap_pass_rate_delta_drops_unscorable(self):
        # Two scorable tasks + one unscorable. Pass-rate delta must be computed
        # over the two scorable pairs only.
        control = [_d("t1", "f", True, 1.0), _d("t2", "f", True, 1.0), _d("t3", "f", True, 1.0)]
        test = [
            _d("t1", "c", True, 0.5),
            _d("t2", "c", False, 0.5),
            _d("t3", "c", False, 0.0, scorable=False),
        ]
        res = kill_gate.bootstrap_pass_rate_delta(control, test, n_iterations=200)
        # scorable pairs: t1 (T vs T -> 0), t2 (F vs T -> -1); mean = -0.5
        assert res["mean"] == pytest.approx(-0.5)


class TestBootstrapRejectsZeroIterations:
    """A 0-iteration bootstrap is a caller error — it raises cleanly instead of
    indexing an empty resample list (which crashed with a bare IndexError)."""

    @staticmethod
    def _arms() -> tuple[list[kill_gate.Decision], list[kill_gate.Decision]]:
        control = [_d("t1", "f", True, 1.0), _d("t2", "f", False, 5.0)]
        test = [_d("t1", "c", True, 0.5), _d("t2", "c", False, 1.0)]
        return control, test

    def test_cost_delta_rejects_zero_iterations(self):
        control, test = self._arms()
        with pytest.raises(ValueError, match="at least 1 iteration"):
            kill_gate.bootstrap_cost_delta(control, test, n_iterations=0)

    def test_pass_rate_delta_rejects_zero_iterations(self):
        control, test = self._arms()
        with pytest.raises(ValueError, match="at least 1 iteration"):
            kill_gate.bootstrap_pass_rate_delta(control, test, n_iterations=0)

    def test_cost_ratio_rejects_zero_iterations(self):
        control, test = self._arms()
        with pytest.raises(ValueError, match="at least 1 iteration"):
            kill_gate.bootstrap_cost_ratio(control, test, n_iterations=0)


def test_empty_corpus_exits_untested_not_passed(monkeypatch, tmp_path) -> None:
    # A gate that measured nothing must exit UNTESTED (1 — the same code as the
    # coverage-floor path), never the green 0 an automated consumer reads as PASS.
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps({"results": {}, "tasks": {}, "models": {}}))
    monkeypatch.setattr(config, "load", lambda path=None: {})
    monkeypatch.setattr(config, "benchmark_params", lambda: {})
    monkeypatch.setattr(config, "challenges_path", lambda: matrix_path)
    monkeypatch.setattr(validate_results, "gate", lambda csv, pricing: ("", False))
    monkeypatch.setattr(config, "load_matrix", lambda path=None: {"results": {}})
    monkeypatch.setattr(config, "enabled_pricing", lambda: {})
    monkeypatch.setattr(config, "frontier_model", lambda: "frontier")
    monkeypatch.setattr(config, "sample_tasks", lambda tasks, seed=42: list(tasks))
    monkeypatch.setattr(config, "impute_config", lambda: {"enabled": False})
    monkeypatch.setattr(sys, "argv", ["kill_gate", "--matrix", str(matrix_path), "--n", "20"])
    with pytest.raises(SystemExit) as excinfo:
        kill_gate.main()
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# The pass-rate bootstrap is PAIRED — both arms resampled on the same tasks
# ---------------------------------------------------------------------------


def _unpaired_pass_rate_ci(
    control: list[kill_gate.Decision], test: list[kill_gate.Decision], n_iterations: int
) -> tuple[float, float]:
    """Reference implementation of the WRONG (arms-resampled-independently) estimator."""
    import random as _random

    control_pass = [1 if cd[2] else 0 for cd in control]
    test_pass = [1 if td[2] else 0 for td in test]
    n = len(control_pass)
    rng = _random.Random(42)
    boot = []
    for _ in range(n_iterations):
        c_s = [rng.choice(control_pass) for _ in range(n)]
        t_s = [rng.choice(test_pass) for _ in range(n)]
        boot.append((sum(t_s) - sum(c_s)) / n)
    boot.sort()
    return boot[int(0.05 * n_iterations)], boot[int(0.95 * n_iterations)]


class TestPassRateBootstrapIsPaired:
    """Both arms score the SAME tasks, so the resampling unit is the task. Independent
    per-arm resampling discards the across-arm correlation and inflates the CI."""

    @staticmethod
    def _correlated_arms() -> tuple[list[kill_gate.Decision], list[kill_gate.Decision]]:
        # 40 tasks. The router reproduces the control's outcome everywhere except two
        # tasks it loses — strong positive correlation, small true delta (-2/40 = -0.05).
        control_pass = [i < 30 for i in range(40)]
        test_pass = [p and i not in (0, 1) for i, p in enumerate(control_pass)]
        control = [_d(f"t{i}", "f", p, 1.0) for i, p in enumerate(control_pass)]
        test = [_d(f"t{i}", "c", p, 0.5) for i, p in enumerate(test_pass)]
        return control, test

    def test_paired_ci_is_materially_narrower_than_unpaired(self):
        control, test = self._correlated_arms()
        res = kill_gate.bootstrap_pass_rate_delta(control, test, n_iterations=4000)
        paired_width = res["ci_upper"] - res["ci_lower"]
        lo, hi = _unpaired_pass_rate_ci(control, test, n_iterations=4000)
        unpaired_width = hi - lo
        assert paired_width < 0.6 * unpaired_width, (
            f"paired width {paired_width} not materially narrower than {unpaired_width}"
        )

    def test_point_estimate_is_unchanged_by_pairing(self):
        # Pairing affects the VARIANCE of the delta, never its mean.
        control, test = self._correlated_arms()
        res = kill_gate.bootstrap_pass_rate_delta(control, test, n_iterations=1000)
        assert res["mean"] == pytest.approx(-2 / 40)

    def test_perfectly_correlated_pair_collapses_ci_to_zero_width(self):
        # A synthetic PERFECTLY-correlated pair — both arms agree on every task — must
        # collapse the CI to zero width. Every per-task delta is 0, so every resample
        # draws the same mean. The WRONG (independent-arms) estimator cannot produce
        # this: it resamples each arm's pass/fail separately, so two arms that agree
        # task-for-task still scatter the resampled delta and open a wide CI.
        control_pass = [i % 2 == 0 for i in range(40)]  # 20 passes / 20 fails
        test_pass = list(control_pass)  # identical outcome on every task -> delta = 0
        control = [_d(f"t{i}", "f", p, 1.0) for i, p in enumerate(control_pass)]
        test = [_d(f"t{i}", "c", p, 0.5) for i, p in enumerate(test_pass)]
        res = kill_gate.bootstrap_pass_rate_delta(control, test, n_iterations=4000)
        assert res["ci_lower"] == res["ci_upper"] == 0.0, (
            f"perfectly-correlated pair must give a zero-width CI, got "
            f"[{res['ci_lower']}, {res['ci_upper']}]"
        )
        assert res["std"] == 0.0


# ---------------------------------------------------------------------------
# BUG 3 — the router arm drives the verdict; router errors surface
# ---------------------------------------------------------------------------


class TestDecideVerdictCacheGuard:
    """The cache-aware guard: a naive-cost PASS is only allowed when the router is
    also cheaper once caching is priced (cache-aware ratio < 1.0)."""

    @staticmethod
    def _pass_eligible() -> tuple[dict, dict]:
        # Naive CI strictly below zero (router cheaper) + quality intact.
        cost_delta = {"ci_lower": -0.5, "ci_upper": -0.1, "mean": -0.3, "n_eq": 4}
        pr_delta = {"ci_lower": 0.0, "ci_upper": 0.0, "mean": 0.0}
        return cost_delta, pr_delta

    def test_passes_when_cache_aware_cheaper(self):
        cost_delta, pr_delta = self._pass_eligible()
        code, label = kill_gate.decide_verdict(cost_delta, pr_delta, 0.6, 4, cache_aware_ratio=0.8)
        assert code == 0
        assert "PASS" in label

    def test_guard_blocks_when_cache_aware_not_cheaper(self):
        cost_delta, pr_delta = self._pass_eligible()
        code, label = kill_gate.decide_verdict(cost_delta, pr_delta, 0.6, 4, cache_aware_ratio=1.05)
        assert code == 1
        assert "cache" in label.lower()

    def test_guard_blocks_at_ratio_exactly_one(self):
        cost_delta, pr_delta = self._pass_eligible()
        code, _ = kill_gate.decide_verdict(cost_delta, pr_delta, 0.6, 4, cache_aware_ratio=1.0)
        assert code == 1


class TestComputeCacheCostsPerTaskScoping:
    """Per-task scoping defect: cache-aware cost is scoped PER TASK (one task = one session), so
    a repeat-model discount fires only within a task's own attempt sequence — never between
    independent tasks that happen to route to the same model."""

    def test_between_task_repeats_get_no_discount(self):
        # Three INDEPENDENT tasks all served by the same model. Under per-task scoping each
        # task has a single attempt, so no discount applies anywhere and the cache-aware total
        # equals the naive total. (The old flat implementation banked 2 spurious discounts.)
        decisions = [
            _d("t1", "kimi-k3", True, 10.0),
            _d("t2", "kimi-k3", True, 10.0),
            _d("t3", "kimi-k3", True, 10.0),
        ]
        pricing = {"kimi-k3": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0}}
        naive, cache_aware = kill_gate.compute_cache_costs(decisions, pricing)
        assert naive == pytest.approx(30.0)
        assert cache_aware == pytest.approx(naive)  # no between-task discount

    def test_within_task_repeat_still_banks_its_discount(self):
        # Two attempts WITHIN one task on the same model are one session: the second attempt's
        # prefix is still warm, so the discount fires (this is the cascade case the model exists
        # to price — it must not be removed by the scoping fix).
        decisions = [
            _d("t1", "kimi-k3", True, 10.0),
            _d("t1", "kimi-k3", True, 10.0),
        ]
        pricing = {"kimi-k3": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0}}
        naive, cache_aware = kill_gate.compute_cache_costs(decisions, pricing)
        assert naive == pytest.approx(20.0)
        assert cache_aware < naive  # the in-session repeat is cheaper

    def test_control_arm_flat_sequence_no_longer_banks_spurious_discount(self):
        # The control arm: 20 consecutive same-model decisions (one per task). The flat
        # implementation banked 19 spurious discounts. Per-task scoping must produce NO discount.
        decisions = [_d(f"t{i}", "kimi-k3", True, 10.0) for i in range(20)]
        pricing = {"kimi-k3": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0}}
        naive, cache_aware = kill_gate.compute_cache_costs(decisions, pricing)
        assert naive == pytest.approx(200.0)
        assert cache_aware == pytest.approx(naive)


class TestCoverageFloorPreRegistered:
    """The pre-registered coverage floor: a verdict is quotable only when >= 90% of BOTH arms'
    cells are MEASURED."""

    def test_imputed_cells_are_unscorable(self):
        # A cell completed by the monotone-ladder imputer (`imputed: True`) is NOT a real
        # observation. Scoring it as a clean measured cell is exactly the failure this floor
        # exists to prevent.
        imputed = kill_gate._make_decision("t1", "m", {"pass": True, "cost": 1.0, "imputed": True})
        assert imputed[7] is False  # unscorable
        real = kill_gate._make_decision("t1", "m", {"pass": True, "cost": 1.0})
        assert real[7] is True

    def test_run_kill_gate_returns_untested_below_the_floor(self, monkeypatch):
        # 10 tasks, but only 5 of the control cells are real (the rest imputed): 50% < 90%,
        # so the verdict is UNTESTED regardless of the underlying cost/quality numbers.
        def fake_control(m, task_ids, frontier_model):
            out = []
            for i, t in enumerate(task_ids):
                imputed = i >= 5
                out.append(
                    kill_gate._make_decision(
                        t, "frontier", {"pass": True, "cost": 1.0, "imputed": imputed}
                    )
                )
            return out

        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "a", True, 0.5) for t in task_ids]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_control", fake_control)
        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix={"results": {}},
            pricing={"frontier": {"input": 5.0, "output": 5.0}},
            task_ids=[f"t{i}" for i in range(10)],
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 1
        assert "UNTESTED" in report
        assert "coverage floor" in report.lower()

    def test_run_kill_gate_passes_above_the_floor(self, monkeypatch):
        # All cells real (100% >= 90%), router cheaper, quality intact -> PASS.
        def fake_control(m, task_ids, frontier_model):
            return [_d(t, "frontier", True, 1.0) for t in task_ids]

        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "a", True, 0.5) for t in task_ids]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_control", fake_control)
        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix={"results": {}},
            pricing={"frontier": {"input": 5.0, "output": 5.0}},
            task_ids=[f"t{i}" for i in range(10)],
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 0
        assert "PASS" in report


def _frontier_matrix(tasks: list[str]) -> dict:
    return {
        "results": {
            t: {"frontier": {"pass": True, "cost": 1.0, "in_tok": 100, "out_tok": 50, "calls": 1}}
            for t in tasks
        },
        "models": {"frontier": {"input_price": 5.0, "output_price": 5.0}},
        "tasks": {t: {} for t in tasks},
    }


class TestVerdictDrivenByRouter:
    def test_verdict_uses_router_not_oracle(self, monkeypatch):
        tasks = ["t1", "t2", "t3", "t4"]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        # Router is MORE expensive than control on every task (2.0 vs 1.0), all pass.
        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "router", True, 2.0) for t in task_ids]

        # Oracle would be far CHEAPER (0.1) and passing -> would yield PASS if it
        # drove the verdict. It must not.
        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        # Router is strictly more expensive at equal quality -> FAIL.
        assert exit_code == 1
        assert "FAIL" in report

    def test_router_failure_surfaces(self, monkeypatch):
        tasks = ["t1", "t2"]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        def boom(m, task_ids, strategy=None):
            raise RuntimeError("router exploded")

        monkeypatch.setattr(kill_gate, "evaluate_router", boom)

        with pytest.raises(RuntimeError, match="router exploded"):
            kill_gate.run_kill_gate(
                matrix=matrix,
                pricing=pricing,
                task_ids=tasks,
                verifier_threshold=0.6,
                frontier_model="frontier",
                n_iterations=200,
            )

    def test_router_cheaper_passes(self, monkeypatch):
        tasks = ["t1", "t2", "t3", "t4"]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        # Single-model router, cheaper than control on every task -> cache-aware
        # ratio < 1 and naive CI below zero -> PASS.
        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "router", True, 0.5) for t in task_ids]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 0
        assert "PASS" in report

    def test_ci_crosses_zero_is_inconclusive(self, monkeypatch):
        tasks = ["t1", "t2", "t3", "t4"]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        # Router cost straddles control (0.9/1.1 vs 1.0) -> delta CI crosses zero.
        def fake_router(m, task_ids, strategy=None):
            costs = [0.9, 1.1, 0.9, 1.1]
            return [_d(t, "router", True, c) for t, c in zip(task_ids, costs, strict=True)]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, _ = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 2

    def test_cache_guard_does_not_block_a_naive_cheaper_router_under_per_task_scoping(
        self, monkeypatch
    ):
        # The per-task scoping fix corrected the flat implementation's semantics: the fixed-model
        # control banks NO between-task cache discount (each task is one session, one attempt),
        # so the cache-aware ratio collapses to the naive ratio and a router that is genuinely
        # cheaper on naive cost is NOT spuriously blocked. (The old flat scoping handed the
        # control a discount on every task after the first, manufacturing a cache-costlier
        # reading out of independent tasks.)
        tasks = [f"t{i}" for i in range(10)]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        # Router is cheaper on NAIVE cost (0.93 vs 1.0) and switches model every task.
        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "a" if i % 2 == 0 else "b", True, 0.93) for i, t in enumerate(task_ids)]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 0
        assert "PASS" in report

    def test_phantom_unscorable_cell_does_not_corrupt_the_verdict(self, monkeypatch):
        # Regression: unscorable ($0 coverage-gap) cells must be excluded from the cost basis
        # exactly like the bootstrap pairing does. A task where the ROUTER is unmeasured ($0)
        # but the CONTROL is measured ($1) must not be counted — it is not a real fail@$0. The
        # corrected per-task scoping makes cache-aware == naive, so the honest verdict for a
        # genuinely cheaper router is PASS; the phantom must not flip it (either direction).
        scorable = [f"t{i}" for i in range(10)]
        phantom = "tphantom"
        tasks = scorable + [phantom]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        def make_router(include_phantom: bool):
            def fake_router(m, task_ids, strategy=None):
                out = []
                for i, t in enumerate(task_ids):
                    if include_phantom and t == phantom:
                        out.append(_d(t, "a", False, 0.0, scorable=False))
                    else:
                        out.append(_d(t, "a" if i % 2 == 0 else "b", True, 0.93))
                return out

            return fake_router

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        def run(include_phantom: bool) -> tuple[int, str]:
            monkeypatch.setattr(kill_gate, "evaluate_router", make_router(include_phantom))
            monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)
            return kill_gate.run_kill_gate(
                matrix=matrix,
                pricing=pricing,
                task_ids=tasks,
                verifier_threshold=0.6,
                frontier_model="frontier",
                n_iterations=200,
            )

        with_phantom = run(include_phantom=True)
        without_phantom = run(include_phantom=False)
        # The phantom pair must not open (or close) the guard: same verdict with and without.
        assert with_phantom[0] == without_phantom[0]
        assert with_phantom[0] == 0
        assert "PASS" in with_phantom[1]

    def test_report_surfaces_unscorable_count(self, monkeypatch):
        tasks = ["t1", "t2", "t3"]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        def fake_router(m, task_ids, strategy=None):
            # t3 lands on an unmeasured cell -> unscorable coverage gap.
            return [
                _d("t1", "router", True, 0.5),
                _d("t2", "router", True, 0.5),
                _d("t3", "router", False, 0.0, scorable=False),
            ]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        _, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert "nscorable" in report or "overage" in report
        assert "1" in report


# ---------------------------------------------------------------------------
# The verdict is a TRACKED, deterministic artifact
# ---------------------------------------------------------------------------


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-embedding: sha256 of the text -> a 32-dim vector."""
    # The suite runs under SHUNT_DISALLOW_REAL_EMBEDDER (autouse fixture), so the real ONNX
    # model cannot load here. ONNX inference is deterministic given fixed weights, so this
    # stand-in is faithful for the DETERMINISM property this test asserts — the committed
    # corpus is real, only the embedding function is swapped.
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = digest * 4  # 32 bytes -> 128 bytes -> 32 float32
        out[i] = np.frombuffer(raw, dtype=np.float32)
    return out


class TestVerdictArtifact:
    """The verdict artifact is tracked and deterministic (no
    timestamps, no paths), and carries the numbers a reviewer needs to see move."""

    @staticmethod
    def _committed_run(
        tmp_path: pytest.TempPathFactory, name: str = "kill_gate_verdict.json"
    ) -> tuple[Path, dict]:
        # Regenerate from the committed corpus exactly as main() selects tasks: the
        # matrix + pricing come from committed files; the router arm runs the REAL
        # kNN selection over the REAL matrix with only the embedding function swapped
        # for a deterministic stand-in (see _fake_embed_texts).
        config.load("benchmark/benchmark.yaml")
        matrix = config.load_matrix()
        pricing = config.enabled_pricing()
        frontier = config.frontier_model()
        all_tasks = sorted(matrix.get("results", {}).keys())
        sample = config.sample_tasks(all_tasks, seed=42)
        task_ids = kill_gate.select_tasks({"results": {t: {} for t in sample}}, 20, seed=42)

        verdict_path = tmp_path / name
        kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=task_ids,
            verifier_threshold=0.6,
            frontier_model=frontier,
            n_iterations=500,
            verdict_path=verdict_path,
            task_seed=42,
        )
        payload = json.loads(verdict_path.read_text())
        return verdict_path, payload

    def test_two_regenerations_are_byte_identical(self, tmp_path, monkeypatch):
        # Determinism: two consecutive regenerations from the committed corpus produce a
        # byte-identical artifact — a verdict nobody can diff is a verdict nobody can
        # audit, so determinism is the whole point of the tracked file.
        if not config.results_csv_path().exists():
            pytest.skip("no committed results.csv in this checkout")
        monkeypatch.setattr(knn, "_embed_texts", _fake_embed_texts)
        first, _ = self._committed_run(tmp_path)
        # Same committed inputs, regenerated from scratch (fresh matrix load): the
        # emission path must be a pure function of them — byte-identical, or the
        # tracked artifact would churn on every run and be un-auditable.
        config.load("benchmark/benchmark.yaml")
        matrix = config.load_matrix()
        pricing = config.enabled_pricing()
        all_tasks = sorted(matrix.get("results", {}).keys())
        sample = config.sample_tasks(all_tasks, seed=42)
        task_ids = kill_gate.select_tasks({"results": {t: {} for t in sample}}, 20, seed=42)
        second = tmp_path / "kill_gate_verdict_2.json"
        kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=task_ids,
            verifier_threshold=0.6,
            frontier_model=config.frontier_model(),
            n_iterations=500,
            verdict_path=second,
            task_seed=42,
        )
        assert first.read_bytes() == second.read_bytes()

    def test_required_keys_present(self, tmp_path, monkeypatch):
        # Key presence: the artifact carries verdict, cache-aware ratio, naive ratio, n, and the
        # coverage/subset guard — the numbers a reviewer needs to see move.
        monkeypatch.setattr(knn, "_embed_texts", _fake_embed_texts)
        _, payload = self._committed_run(tmp_path)
        assert "verdict" in payload
        assert "cache_aware_ratio" in payload
        assert "naive_ratio" in payload
        assert "n" in payload
        # The cache-aware ratio carries its own paired-task bootstrap
        # CI in the tracked artifact, wherever the ratio is published.
        assert "cache_aware_ratio_ci" in payload
        ratio_ci = payload["cache_aware_ratio_ci"]
        assert "mean" in ratio_ci
        assert "ci_lower" in ratio_ci
        assert "ci_upper" in ratio_ci
        assert ratio_ci["ci_lower"] <= ratio_ci["ci_upper"]
        assert payload["cache_aware_ratio"] == pytest.approx(ratio_ci["mean"])
        assert "coverage" in payload
        coverage = payload["coverage"]
        assert "floor" in coverage
        assert "control_measured" in coverage
        assert "router_measured" in coverage
        assert "control_coverage" in coverage
        assert "router_coverage" in coverage
        assert "tripped" in coverage
        # Subset guard: how much of the scored task set actually entered the comparison.
        assert "n_scorable" in payload
        assert "n_unscorable" in payload
        assert payload["n_scorable"] + payload["n_unscorable"] == payload["n"]

    def test_no_timestamp_or_absolute_path(self, tmp_path, monkeypatch):
        # No run-local noise: timestamps and absolute paths make every run a diff and
        # train reviewers to ignore the file, so they must not exist here.
        monkeypatch.setattr(knn, "_embed_texts", _fake_embed_texts)
        verdict_path, _ = self._committed_run(tmp_path)
        text = verdict_path.read_text()
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)  # dates
        assert not re.search(r"\d{2}:\d{2}(:\d{2})?", text)  # clock times
        # epoch seconds/millis (a standalone large integer — NOT a decimal fraction
        # like a cost's 10 fractional digits, which the lookarounds exclude)
        assert not re.search(r"(?<![\d.])[1-9]\d{8}(?![\d.])", text)
        assert "/" not in text  # no absolute or rooted paths
        assert "\\" not in text  # no Windows paths
        assert "~" not in text  # no home-relative paths


# ---------------------------------------------------------------------------
# The kill-gate copy of the cache-aware bootstrap refusal is re-ruled
# ---------------------------------------------------------------------------


def _cache_price(model: str) -> CachePrice:
    return CachePrice(
        model=model,
        input_share=1.0,
        discount=0.5,
        hit_rate=1.0,
        provenance="assumed",
        share_provenance="assumed",
    )


class TestCacheAwareRatioCI:
    """The cache-aware ratio carries a paired-task bootstrap CI wherever it is
    printed — the tracked verdict artifact, the report, and the verdict label."""

    @staticmethod
    def _fake_arms():
        def fake_control(m, task_ids, frontier_model):
            return [_d(t, "frontier", True, 1.0) for t in task_ids]

        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "a", True, 0.5) for t in task_ids]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        return fake_control, fake_router, fake_oracle

    def test_verdict_artifact_carries_the_cache_aware_ci(self, tmp_path, monkeypatch):
        fake_control, fake_router, fake_oracle = self._fake_arms()
        monkeypatch.setattr(kill_gate, "evaluate_control", fake_control)
        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        verdict_path = tmp_path / "kill_gate_verdict.json"
        exit_code, _ = kill_gate.run_kill_gate(
            matrix={"results": {}},
            pricing={"frontier": {"input": 5.0, "output": 5.0}},
            task_ids=[f"t{i}" for i in range(8)],
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
            verdict_path=verdict_path,
        )
        assert exit_code == 0
        payload = json.loads(verdict_path.read_text())
        ci = payload["cache_aware_ratio_ci"]
        # Router 0.5 / control 1.0 on every task -> ratio 0.5, and every resample is
        # the same, so the CI collapses onto the point estimate (which is the ratio of
        # totals the payload publishes).
        assert payload["cache_aware_ratio"] == pytest.approx(0.5)
        assert ci["mean"] == pytest.approx(payload["cache_aware_ratio"])
        assert ci["ci_lower"] == pytest.approx(0.5)
        assert ci["ci_upper"] == pytest.approx(0.5)

    def test_report_prints_the_cache_aware_ci_next_to_the_ratio(self, monkeypatch):
        fake_control, fake_router, fake_oracle = self._fake_arms()
        monkeypatch.setattr(kill_gate, "evaluate_control", fake_control)
        monkeypatch.setattr(kill_gate, "evaluate_router", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        _, report = kill_gate.run_kill_gate(
            matrix={"results": {}},
            pricing={"frontier": {"input": 5.0, "output": 5.0}},
            task_ids=[f"t{i}" for i in range(8)],
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        # The ratio line and the CI line sit together, and the verdict label repeats
        # the ratio WITH its interval — no reader is shown the ratio without the CI.
        assert "cache-aware ratio 0.5000" in report
        assert "90% CI (bootstrap)" in report
        assert "[0.5000, 0.5000]" in report
        assert "cache-aware ratio 0.5000 CI [0.5000, 0.5000]" in report
        # The old refusal is gone from the printed report.
        assert "not bootstrappable" not in report
        assert "on NAIVE cost" not in report

    def test_bootstrap_cache_aware_ratio_matches_the_point_ratio(self):
        control = [_d(f"t{i}", "frontier", True, 10.0) for i in range(10)]
        test = [_d(f"t{i}", "a" if i % 2 else "b", True, 5.0) for i in range(10)]
        res = kill_gate.bootstrap_cache_aware_ratio(
            control, test, {"frontier": {}, "a": {}, "b": {}}, n_iterations=300
        )
        assert res["n_eq"] == 10
        assert res["mean"] == pytest.approx(0.5)
        assert res["ci_lower"] == pytest.approx(0.5)
        assert res["ci_upper"] == pytest.approx(0.5)

    def test_bootstrap_cache_aware_ratio_is_computed_on_cache_aware_cost(self, monkeypatch):
        monkeypatch.setattr(
            kill_gate, "cache_prices", lambda models, shares=None: {"m": _cache_price("m")}
        )
        # t1 repeats the same model within the session in BOTH arms, so the discount
        # fires: control 15 (10 + 5), test 7.5 (5 + 2.5). The NAIVE ratio would be
        # 10/10 = 1.0 — asserting 0.5 pins that the CI is computed on cache-aware cost.
        control = [_d("t1", "m", True, 10.0), _d("t1", "m", True, 10.0)]
        test = [_d("t1", "m", True, 5.0), _d("t1", "m", True, 5.0)]
        res = kill_gate.bootstrap_cache_aware_ratio(control, test, {"m": {}}, n_iterations=200)
        assert res["n_eq"] == 1
        assert res["mean"] == pytest.approx(7.5 / 15.0)


class TestCacheCostScopingProperty:
    """The property the kill-gate comment asserts — a task's cache cost is invariant
    to the other tasks in the sample — is pinned by a test, so the comment and the test
    agree or the test fails."""

    def test_a_tasks_cache_cost_is_invariant_to_the_other_tasks(self):
        # t1 holds the only within-task repeat — the only place a discount may fire.
        decisions = [
            _d("t1", "kimi-k3", True, 10.0),
            _d("t1", "kimi-k3", True, 10.0),
            _d("t2", "kimi-k3", True, 10.0),
            _d("t3", "gpt-5-mini", True, 5.0),
        ]
        pricing = {"kimi-k3": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0}}
        prices = kill_gate.cache_prices(sorted({d[1] for d in decisions} | set(pricing)))
        full = kill_gate._per_task_cache_costs(decisions, prices)
        t1_cost = full["t1"]
        # Removing the other tasks, or reordering them, must not move t1's cost.
        assert kill_gate._per_task_cache_costs(decisions[:2], prices)["t1"] == t1_cost
        assert kill_gate._per_task_cache_costs(list(reversed(decisions)), prices)["t1"] == t1_cost
        # ... and the shared predicate agrees the sample is task-scoped.
        attempts = kill_gate._attempts_by_task(decisions)
        assert cache_cost.cache_cost_is_scoped_to_tasks(sorted(full), attempts, prices) is True


def _att(model: str, cost: float) -> BilledAttempt:
    """A billing record with no tokens — all the cache-aware cost model reads."""
    return BilledAttempt(model=model, cost=cost)


class TestMetricsAndKillGateShareScopingPredicate:
    """The two copies of the ruling resolve the SAME predicate — metrics.py's
    ``_assert_cache_cost_scoping`` and the kill gate's bootstrap — so they cannot
    silently disagree again."""

    def test_both_accept_a_task_scoped_sample(self):
        tids = ["t1", "t2", "t3"]
        attempts = {"t1": [_att("m", 10.0)], "t2": [_att("m", 10.0)], "t3": [_att("m", 10.0)]}
        prices = {"m": _cache_price("m")}
        metrics_mod._assert_cache_cost_scoping(tids, attempts, prices)  # must not raise
        assert cache_cost.cache_cost_is_scoped_to_tasks(tids, attempts, prices) is True

    def test_both_refuse_a_cross_task_leak(self, monkeypatch):
        # A flattened cross-run cache model: task B inherits task A's warm prefix through
        # hidden state. Both the metrics guard and the shared predicate must reject it —
        # this is the re-arming the comment promises if a future re-flattening returns.
        state: dict[str, str | None] = {"prev": None}

        def flat_cache_cost(attempts, prices):
            total = sum(a.cost for a in attempts)
            model = attempts[-1].model if attempts else None
            if state["prev"] is not None and model is not None and state["prev"] == model:
                total -= 1.0
            state["prev"] = model
            return total

        monkeypatch.setattr(cache_cost, "cache_aware_total", flat_cache_cost)
        monkeypatch.setattr(metrics_mod, "cache_aware_total", flat_cache_cost)

        tids = ["t1", "t2", "t3"]
        attempts = {"t1": [_att("m", 10.0)], "t2": [_att("m", 10.0)], "t3": [_att("m", 10.0)]}
        prices = {"m": _cache_price("m")}
        with pytest.raises(RuntimeError, match="not scoped per task"):
            metrics_mod._assert_cache_cost_scoping(tids, attempts, prices)
        state["prev"] = None  # the leak is stateful; re-arm it before the next check
        assert cache_cost.cache_cost_is_scoped_to_tasks(tids, attempts, prices) is False

    def test_kill_gate_bootstrap_refuses_the_leak_too(self, monkeypatch):
        # The kill-gate bootstrap resolves the same predicate: on a leaked fixture it
        # must raise rather than emit an interval the statistic does not support.
        state: dict[str, str | None] = {"prev": None}

        def flat_cache_cost(attempts, prices):
            total = sum(a.cost for a in attempts)
            model = attempts[-1].model if attempts else None
            if state["prev"] is not None and model is not None and state["prev"] == model:
                total -= 1.0
            state["prev"] = model
            return total

        monkeypatch.setattr(cache_cost, "cache_aware_total", flat_cache_cost)

        control = [_d("t1", "m", True, 10.0), _d("t2", "m", True, 10.0), _d("t3", "m", True, 10.0)]
        test = [_d("t1", "m", True, 5.0), _d("t2", "m", True, 5.0), _d("t3", "m", True, 5.0)]
        with pytest.raises(RuntimeError, match="scoped"):
            kill_gate.bootstrap_cache_aware_ratio(control, test, {"m": {}}, n_iterations=20)

    def test_no_second_hand_written_refusal_survives(self):
        # The refusal-to-bootstrap text no longer survives in kill_gate.py.
        src = Path(kill_gate.__file__).read_text()
        for phrase in (
            "do NOT bootstrap a CI on cache-aware",
            "deliberately do NOT bootstrap",
            "not bootstrappable",
            "destroying the very adjacency",
        ):
            assert phrase not in src, f"stale refusal phrase survives: {phrase!r}"
