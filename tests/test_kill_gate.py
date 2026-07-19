"""Tests for the kill-gate measurement (benchmark/runner/kill_gate.py): BUG 1
(unmeasured cells are UNSCORABLE, excluded from the equal-pass pairing) and BUG 3
(the verdict is driven by the real kNN-cascade router, and router errors surface)."""

from __future__ import annotations

import pytest

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

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
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

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", boom)

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

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
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

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
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

    def test_cache_guard_blocks_naive_cheaper_but_cache_costlier(self, monkeypatch):
        tasks = [f"t{i}" for i in range(10)]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        # Router is cheaper on NAIVE cost (0.93 vs 1.0) but switches model every
        # task, so it captures no cache discount; the fixed-model control does.
        # Once caching is priced the router is NOT cheaper -> guard must block PASS.
        def fake_router(m, task_ids, strategy=None):
            return [_d(t, "a" if i % 2 == 0 else "b", True, 0.93) for i, t in enumerate(task_ids)]

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        assert exit_code == 1
        assert "cache" in report.lower()

    def test_cache_guard_not_bypassed_by_phantom_zero_router_cell(self, monkeypatch):
        # Regression: the cache-aware guard must exclude unscorable ($0 coverage-gap)
        # cells from the cost basis, exactly like the bootstrap pairing does. A task
        # where the ROUTER is unmeasured ($0) but the CONTROL is measured ($1) would,
        # if unfiltered, inflate control_cache and zero-out router_cache -> ratio < 1
        # -> a spurious PASS on a router that is NOT cheaper once caching is priced.
        scorable = [f"t{i}" for i in range(10)]
        phantom = "tphantom"
        tasks = scorable + [phantom]
        matrix = _frontier_matrix(tasks)
        pricing = {"frontier": {"input": 5.0, "output": 5.0}}

        def fake_router(m, task_ids, strategy=None):
            out = []
            for i, t in enumerate(task_ids):
                if t == phantom:
                    out.append(_d(t, "a", False, 0.0, scorable=False))
                else:
                    out.append(_d(t, "a" if i % 2 == 0 else "b", True, 0.93))
            return out

        def fake_oracle(m, task_ids, pricing):
            return [_d(t, "cheap", True, 0.1) for t in task_ids]

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
        monkeypatch.setattr(kill_gate, "evaluate_test", fake_oracle)

        exit_code, report = kill_gate.run_kill_gate(
            matrix=matrix,
            pricing=pricing,
            task_ids=tasks,
            verifier_threshold=0.6,
            frontier_model="frontier",
            n_iterations=200,
        )
        # The phantom pair must not open the guard: same verdict as without it (FAIL).
        assert exit_code == 1
        assert "cache" in report.lower()

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

        monkeypatch.setattr(kill_gate, "evaluate_knn_cascade", fake_router)
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
