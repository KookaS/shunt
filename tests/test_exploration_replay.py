"""Offline matrix-replay evaluator tests — synthetic matrix, deterministic, no network."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.routing import exploration_replay as er
from shunt.router.policy import ExplorationPolicy, KnnPolicy

CHEAP = "deepseek-v4-flash"
MID = "qwen3.7-plus"
PRICEY = "kimi-k3"


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-embeddings — one seeded vector per text, no model download."""
    return np.array(
        [np.random.default_rng(abs(hash(t)) % (2**32)).random(8) for t in texts],
        dtype=np.float32,
    )


def _matrix(n_tasks: int = 8, models: tuple[str, ...] = (CHEAP, MID, PRICEY)) -> dict:
    """Dense synthetic matrix: the cheap model fails every 3rd task, the pricey one never."""
    results: dict[str, dict[str, dict]] = {}
    for i in range(n_tasks):
        tid = f"t{i}"
        results[tid] = {
            CHEAP: {"pass": i % 3 != 0, "cost": 0.01},
            MID: {"pass": i % 5 != 0, "cost": 0.05},
            PRICEY: {"pass": True, "cost": 0.20},
        }
        results[tid] = {m: cell for m, cell in results[tid].items() if m in models}
    return {
        "tasks": {tid: {"description": f"synthetic task {tid}"} for tid in results},
        "results": results,
        "models": {
            CHEAP: {"input_price": 0.3, "output_price": 0.9},
            MID: {"input_price": 1.0, "output_price": 3.0},
            PRICEY: {"input_price": 5.0, "output_price": 15.0},
        },
    }


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(er, "_embed_texts", _fake_embed)


class TestDenseSlice:
    def test_fully_dense_matrix_keeps_every_cell(self) -> None:
        s = er.dense_slice(_matrix()["results"])
        assert len(s.tasks) == 8
        assert s.models == sorted([CHEAP, MID, PRICEY])
        assert s.matrix_density == 1.0
        assert s.n_cells == 24

    def test_drops_the_model_that_costs_more_cells_than_it_adds(self) -> None:
        m = _matrix()
        for tid in ("t0", "t1", "t2", "t3", "t4", "t5"):
            del m["results"][tid][PRICEY]  # PRICEY survives on only 2 of 8 tasks
        s = er.dense_slice(m["results"])
        assert PRICEY not in s.models
        assert len(s.tasks) == 8
        assert s.matrix_density == pytest.approx(18 / 24)

    def test_empty_matrix_is_not_a_crash(self) -> None:
        s = er.dense_slice({})
        assert s.tasks == [] and s.models == [] and s.matrix_density == 0.0


class TestRestrictMatrix:
    def test_confines_results_to_the_slice(self) -> None:
        m = _matrix()
        sub = er.restrict_matrix(m, ["t0", "t1"], [CHEAP])
        assert sorted(sub["results"]) == ["t0", "t1"]
        assert list(sub["results"]["t0"]) == [CHEAP]
        assert sub["models"] is m["models"]  # untouched sections are shared, not rebuilt


class TestBootstrapCi:
    def test_ci_brackets_the_mean_and_is_seed_reproducible(self) -> None:
        values = [0.0, 1.0] * 20
        a = er.bootstrap_ci(values, seed=7)
        b = er.bootstrap_ci(values, seed=7)
        assert a == b
        assert a.value == pytest.approx(0.5)
        assert a.lo < a.value < a.hi

    def test_empty_sample_yields_nan_not_zero(self) -> None:
        est = er.bootstrap_ci([], seed=1)
        assert np.isnan(est.value) and np.isnan(est.lo) and np.isnan(est.hi)


class TestReplay:
    def test_exploit_only_replay_is_deterministic_and_scores_every_task(self) -> None:
        m = _matrix()
        first = er.replay(m, KnnPolicy(), exploration=None, seed=3)
        second = er.replay(m, KnnPolicy(), exploration=None, seed=3)
        assert [d.model for d in first.decisions] == [d.model for d in second.decisions]
        assert first.missing == []
        assert len(first.decisions) == 8
        assert not any(d.is_exploratory for d in first.decisions)

    def test_replay_reads_measured_outcomes_never_imputed_ones(self) -> None:
        m = _matrix()
        run = er.replay(m, KnnPolicy(), exploration=None, seed=3)
        for d in run.decisions:
            cell = m["results"][d.task][d.model]
            assert d.passed is bool(cell["pass"])
            assert d.cost == pytest.approx(cell["cost"])

    def test_unscorable_cells_are_skipped_and_counted_not_imputed(self) -> None:
        m = _matrix()
        picks = {d.task: d.model for d in er.replay(m, KnnPolicy(), None, seed=3).decisions}
        # Punch a hole in exactly one cell the policy routes to. Leave-one-out means t0's
        # own row never feeds its own neighbourhood, so the decision for t0 is unchanged.
        del m["results"]["t0"][picks["t0"]]
        run = er.replay(m, KnnPolicy(), exploration=None, seed=3)
        assert run.missing == [("t0", picks["t0"])]
        assert len(run.decisions) + len(run.missing) == 8
        assert all(d.task != "t0" for d in run.decisions)

    def test_exploration_run_is_seed_reproducible(self) -> None:
        m = _matrix()
        policy = ExplorationPolicy(propensity_mc_samples=8)
        a = er.replay(m, KnnPolicy(), policy, seed=11)
        b = er.replay(m, KnnPolicy(), policy, seed=11)
        assert [(d.task, d.model, d.is_exploratory) for d in a.decisions] == [
            (d.task, d.model, d.is_exploratory) for d in b.decisions
        ]

    def test_disabled_exploration_never_marks_a_decision_exploratory(self) -> None:
        m = _matrix()
        policy = ExplorationPolicy(enabled=False, propensity_mc_samples=8)
        run = er.replay(m, KnnPolicy(), policy, seed=11)
        assert not any(d.is_exploratory for d in run.decisions)
        assert run.budget_explore_ratio == 0.0

    def test_zero_budget_frac_forbids_exploratory_spend(self) -> None:
        m = _matrix()
        policy = ExplorationPolicy(explore_budget_frac=0.0, propensity_mc_samples=8)
        run = er.replay(m, KnnPolicy(), policy, seed=5)
        # frac=0 means NEVER explore — not "explore during bootstrap then stop". Exploratory
        # spend never raises exploit_cost, so a bootstrap branch ahead of the frac check
        # would stay open forever; ExplorationBudget.can_explore short-circuits at frac<=0.
        assert run.explore_spend == 0.0
        assert not any(d.is_exploratory for d in run.decisions)

    def test_run_spend_accounting(self) -> None:
        run = er.RunOutcome(decisions=[], missing=[], explore_spend=2.0, exploit_spend=8.0)
        assert run.explore_ratio == pytest.approx(0.25)
        assert run.total_spend == pytest.approx(10.0)

    def test_no_exploit_spend_reports_a_zero_ratio(self) -> None:
        run = er.RunOutcome(decisions=[], missing=[], explore_spend=0.0, exploit_spend=0.0)
        assert run.explore_ratio == 0.0
        assert run.total_spend == 0.0


class TestEvaluate:
    def test_report_compares_both_policies_on_the_same_dense_slice(self) -> None:
        report = er.evaluate(
            _matrix(),
            exploration=ExplorationPolicy(propensity_mc_samples=8),
            n_seeds=3,
            n_resamples=200,
        )
        assert len(report.slice_.tasks) == 8
        assert report.n_seeds == 3
        assert report.baseline_missing == 0 and report.exploration_missing_per_seed == 0.0
        for est in (
            report.baseline_pass_rate,
            report.exploration_pass_rate,
            report.baseline_cost,
            report.exploration_cost,
        ):
            assert est.lo <= est.value <= est.hi
        assert 0.0 <= report.baseline_pass_rate.value <= 1.0
        assert report.cost_multiple >= 1.0
        assert set(report.per_task_baseline_pass) == set(report.slice_.tasks)

    def test_explore_share_by_round_has_one_point_per_decision(self) -> None:
        report = er.evaluate(
            _matrix(),
            exploration=ExplorationPolicy(propensity_mc_samples=8),
            n_seeds=2,
            n_resamples=100,
        )
        assert len(report.explore_share_by_round) == 8
        assert all(0.0 <= s <= 1.0 for s in report.explore_share_by_round)

    def test_direct_method_reproduces_a_hand_computed_average(self) -> None:
        # One model in the slice pins every choice, so the Direct-Method estimate must equal
        # the arithmetic mean of the recorded cells. CHEAP passes 7/8 here — comfortably over
        # success_rate_threshold, so the rule never escalates off-slice and every cell scores.
        m = _matrix(models=(CHEAP,))
        for i, tid in enumerate(sorted(m["results"])):
            m["results"][tid][CHEAP]["pass"] = i != 0
        report = er.evaluate(
            m,
            exploration=ExplorationPolicy(propensity_mc_samples=8),
            n_seeds=2,
            n_resamples=100,
        )
        assert report.slice_.models == [CHEAP]
        assert report.baseline_missing == 0
        assert report.baseline_pass_rate.value == pytest.approx(7 / 8)
        assert report.baseline_cost.value == pytest.approx(0.01)
        assert report.exploration_pass_rate.value == pytest.approx(7 / 8)
        assert report.pass_delta.value == pytest.approx(0.0)
        assert report.cost_delta.value == pytest.approx(0.0)
        assert report.cost_multiple == pytest.approx(1.0)

    def test_paired_delta_is_the_difference_of_the_two_arms(self) -> None:
        report = er.evaluate(
            _matrix(),
            exploration=ExplorationPolicy(propensity_mc_samples=8),
            n_seeds=3,
            n_resamples=200,
        )
        expected = report.exploration_pass_rate.value - report.baseline_pass_rate.value
        assert report.pass_delta.value == pytest.approx(expected)
        assert report.pass_delta.lo <= report.pass_delta.value <= report.pass_delta.hi
        assert report.cost_multiple_worst_seed >= report.cost_multiple

    def test_evaluate_is_seed_reproducible(self) -> None:
        kwargs = {
            "exploration": ExplorationPolicy(propensity_mc_samples=8),
            "n_seeds": 2,
            "n_resamples": 100,
        }
        a = er.evaluate(_matrix(), **kwargs)  # type: ignore[arg-type]
        b = er.evaluate(_matrix(), **kwargs)  # type: ignore[arg-type]
        assert a.exploration_pass_rate == b.exploration_pass_rate
        assert a.explore_ratio == pytest.approx(b.explore_ratio)
