"""Tests for routing strategies: Oracle, AlwaysCheap, AlwaysFrontier, Random."""

import pytest

from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.knn_cascade import compute_cascade_order
from benchmark.routing.strategies.oracle import Oracle


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
        },
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
        strategy = AlwaysCheap(model="qwen3.7-plus")
        assert strategy.select("any-task", {}, {}) == "qwen3.7-plus"

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


class TestExternalPrior:
    """Route cheap unless the external field-wide rate (p_solve) is low, then escalate."""

    def _matrix(self):
        # Real model names so tier lookup (the registry) resolves cheap vs mid.
        return {
            "models": {
                "deepseek-v4-flash": {"input_price": 0.14, "output_price": 0.28},
                "gpt-5-mini": {"input_price": 0.25, "output_price": 2.0},
            },
            "results": {
                "easy": {
                    "deepseek-v4-flash": {"pass": True, "cost": 0.01},
                    "gpt-5-mini": {"pass": True, "cost": 0.05},
                },
                "hard": {
                    "deepseek-v4-flash": {"pass": False, "cost": 0.01},
                    "gpt-5-mini": {"pass": True, "cost": 0.05},
                },
            },
        }

    def _prior(self, tmp_path):
        from benchmark import config

        config.load("benchmark/benchmark.yaml")  # so tier lookup resolves
        p = tmp_path / "ext.csv"
        # external_prior keys on p_solve (p_cheap is degenerate on the leaderboard).
        p.write_text("instance_id,p_solve\neasy,1.0\nhard,0.0\n")
        return p

    def test_stays_cheap_when_external_cheap_cohort_passes(self, tmp_path):
        from benchmark.routing.strategies.external_prior import ExternalPriorCascade

        s = ExternalPriorCascade(threshold=0.5, prior_path=self._prior(tmp_path))
        assert s.select("easy", {}, self._matrix()) == "deepseek-v4-flash"

    def test_escalates_when_external_cheap_cohort_fails(self, tmp_path):
        from benchmark.routing.strategies.external_prior import ExternalPriorCascade

        s = ExternalPriorCascade(threshold=0.5, prior_path=self._prior(tmp_path))
        assert s.select("hard", {}, self._matrix()) == "gpt-5-mini"

    def test_unknown_task_falls_back_to_cheapest(self, tmp_path):
        from benchmark.routing.strategies.external_prior import ExternalPriorCascade

        s = ExternalPriorCascade(threshold=0.5, prior_path=self._prior(tmp_path))
        assert s.select("not-in-prior", {}, self._matrix()) == "deepseek-v4-flash"


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


class TestComputeCascadeOrder:
    """Tests for the pure cascade-order algorithm (no embeddings needed)."""

    def test_ranks_by_weighted_success_rate(self):
        pricing = {"cheap": 0.2, "mid": 2.0, "frontier": 10.0}
        neighbor_results = {
            "cheap": [
                (0.1, True),
                (0.2, False),
                (0.1, True),
            ],
            "mid": [
                (0.1, True),
                (0.2, True),
                (0.3, True),
            ],
            "frontier": [
                (0.1, True),
                (0.2, True),
                (0.3, False),
            ],
        }
        kws = dict(max_tries=3, min_samples=1, success_rate_threshold=0.0)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        models = [m for m, _ in order]
        assert len(models) == 3
        assert models.index("mid") < models.index("cheap")

    def test_respects_min_samples(self):
        pricing = {"cheap": 0.2, "mid": 2.0}
        neighbor_results = {
            "cheap": [(0.1, True)],
            "mid": [(0.1, True), (0.2, True), (0.3, True)],
        }
        kws = dict(max_tries=3, min_samples=2, success_rate_threshold=0.0)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        models = [m for m, _ in order]
        assert "cheap" not in models
        assert "mid" in models

    def test_respects_success_rate_threshold(self):
        pricing = {"model-a": 1.0, "model-b": 1.0}
        neighbor_results = {
            "model-a": [(0.1, True), (0.2, True), (0.3, True)],
            "model-b": [(0.1, False), (0.2, False), (0.3, False)],
        }
        kws = dict(max_tries=3, min_samples=1, success_rate_threshold=0.5)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        models = [m for m, _ in order]
        assert "model-a" in models
        assert "model-b" not in models

    def test_empty_neighbors_returns_empty(self):
        order = compute_cascade_order({}, {"cheap": 0.2}, max_tries=3)
        assert order == []

    def test_cost_tiebreak(self):
        """When success rates are equal, cheaper model should rank higher."""
        pricing = {"expensive": 10.0, "cheap": 0.2}
        neighbor_results = {
            "expensive": [(0.1, True), (0.2, True)],
            "cheap": [(0.1, True), (0.2, True)],
        }
        kws = dict(max_tries=3, min_samples=1, success_rate_threshold=0.0)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        models = [m for m, _ in order]
        assert models[0] == "cheap"

    def test_max_tries_limit(self):
        pricing = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        neighbor_results = {
            "a": [(0.1, True)],
            "b": [(0.2, True)],
            "c": [(0.3, True)],
            "d": [(0.4, True)],
        }
        kws = dict(max_tries=2, min_samples=1, success_rate_threshold=0.0)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        assert len(order) == 2


class TestCascadeCheapFirstOrdering:
    """BUG 2: within an equal-quality tolerance the cheaper model must rank first,
    yet a genuine (beyond-tolerance) quality gap must still let the better win.
    """

    @staticmethod
    def _outcomes(n_pass: int, n_fail: int) -> list[tuple[float, bool]]:
        # All neighbours at distance 0 (conf 1.0) so weighted_rate == pass fraction.
        return [(0.0, True)] * n_pass + [(0.0, False)] * n_fail

    def test_within_tolerance_prefers_cheaper(self):
        # cheap rate 0.70, expensive rate 0.75 -> within tolerance (tol=0.1),
        # so the cheaper model must be tried first despite lower success rate.
        pricing = {"cheap": 1.0, "expensive": 5.0}
        neighbor_results = {
            "cheap": self._outcomes(7, 3),  # 0.70
            "expensive": self._outcomes(3, 1),  # 0.75
        }
        order = compute_cascade_order(
            neighbor_results,
            pricing,
            max_tries=2,
            min_samples=1,
            success_rate_threshold=0.0,
            success_tolerance=0.1,
        )
        assert [m for m, _ in order][0] == "cheap"

    def test_large_quality_gap_prefers_better(self):
        # cheap rate 0.50 vs expensive rate 0.90 -> gap far beyond tolerance,
        # so the genuinely better (expensive) model must win despite its price.
        pricing = {"cheap": 0.2, "expensive": 10.0}
        neighbor_results = {
            "cheap": self._outcomes(1, 1),  # 0.50
            "expensive": self._outcomes(9, 1),  # 0.90
        }
        order = compute_cascade_order(
            neighbor_results,
            pricing,
            max_tries=2,
            min_samples=1,
            success_rate_threshold=0.0,
            success_tolerance=0.05,
        )
        assert [m for m, _ in order][0] == "expensive"


class TestkNNCascadeStrategy:
    """Tests for kNNCascadeStrategy integration (uses mock/synthetic matrix)."""

    def test_name(self):
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        strategy = kNNCascadeStrategy()
        assert strategy.name == "kNN-cascade"

    def test_cascade_metadata_tracking(self):
        """select() should update cascade_total_cost and cascade_tried_models."""
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        strategy = kNNCascadeStrategy()
        assert hasattr(strategy, "cascade_total_cost")
        assert hasattr(strategy, "cascade_tried_models")


class TestEmptyMatrixGraceful:
    """Both kNN strategies must degrade (not crash) on an empty results matrix."""

    def test_knn_returns_fallback_without_results(self):
        from benchmark.routing.strategies.knn import kNNStrategy

        chosen = kNNStrategy().select("t1", {}, {"results": {}, "models": {}, "tasks": {}})
        assert chosen == "deepseek-v4-flash"

    def test_knn_cascade_returns_fallback_without_results(self):
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        strategy = kNNCascadeStrategy()
        chosen = strategy.select("t1", {}, {"results": {}, "models": {}, "tasks": {}})
        assert chosen == "deepseek-v4-flash"
        assert strategy.cascade_total_cost == 0.0


class TestLookupEmbedderOnDemand:
    """Regression: routing an uncached task must embed on demand, not raise KeyError."""

    def test_embed_computes_and_caches_on_miss(self, monkeypatch):
        import numpy as np

        from benchmark.routing.strategies import knn

        known = np.array([1.0, 2.0], dtype=np.float32)
        computed = np.array([7.0, 8.0], dtype=np.float32)
        monkeypatch.setattr(knn, "_embed_texts", lambda texts: np.array([computed]))

        embedder = knn._LookupEmbedder({"known task": known})
        # Cache hit returns the precomputed vector unchanged.
        assert embedder.embed("known task") is known
        # Cache miss computes on demand (no KeyError) and caches the result.
        out = embedder.embed("unseen task")
        assert np.array_equal(out, computed)
        assert np.array_equal(embedder.embed("unseen task"), computed)


class TestSplitMachineryRemoved:
    """The dead train/test split (never wired into scoring) stays deleted (YAGNI)."""

    def test_knn_module_has_no_split_helpers(self):
        from benchmark.routing.strategies import knn

        assert not hasattr(knn, "_deterministic_split")
        assert not hasattr(knn, "cv_evaluate")

    def test_knn_cascade_module_has_no_split_helper(self):
        from benchmark.routing.strategies import knn_cascade

        assert not hasattr(knn_cascade, "_deterministic_split")

    def test_strategies_expose_no_split_surface(self):
        from benchmark.routing.strategies.knn import kNNStrategy
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        for strategy in (kNNStrategy(), kNNCascadeStrategy()):
            assert not hasattr(strategy, "train_tasks")
            assert not hasattr(strategy, "test_tasks")

    def test_constructors_reject_dead_split_kwargs(self):
        from benchmark.routing.strategies.knn import kNNStrategy
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        with pytest.raises(TypeError):
            kNNStrategy(test_split=0.2)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            kNNCascadeStrategy(seed=1)  # type: ignore[call-arg]
