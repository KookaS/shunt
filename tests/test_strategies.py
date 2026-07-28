"""Tests for routing strategies: Oracle, AlwaysCheap, AlwaysFrontier, Random, PriceCascade."""

from pathlib import Path

import pytest

from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.knn_cascade import compute_cascade_order
from benchmark.routing.strategies.oracle import Oracle
from benchmark.routing.strategies.price_cascade import PriceCascade


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

    def test_orders_eligible_models_cheapest_first(self):
        # Every model clears the bar (threshold 0.0), so the order is pure price
        # ascending — the neighbour rate gates eligibility, it does not rank.
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
        assert order == ["cheap", "mid", "frontier"]

    def test_respects_min_samples(self):
        pricing = {"cheap": 0.2, "mid": 2.0}
        neighbor_results = {
            "cheap": [(0.1, True)],
            "mid": [(0.1, True), (0.2, True), (0.3, True)],
        }
        kws = dict(max_tries=3, min_samples=2, success_rate_threshold=0.0)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        assert "cheap" not in order
        assert "mid" in order

    def test_respects_success_rate_threshold(self):
        pricing = {"model-a": 1.0, "model-b": 1.0}
        neighbor_results = {
            "model-a": [(0.1, True), (0.2, True), (0.3, True)],
            "model-b": [(0.1, False), (0.2, False), (0.3, False)],
        }
        kws = dict(max_tries=3, min_samples=1, success_rate_threshold=0.5)
        order = compute_cascade_order(neighbor_results, pricing, **kws)
        assert "model-a" in order
        assert "model-b" not in order

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
        assert order[0] == "cheap"

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
    """The cascade tries the CHEAPEST model that clears the success bar first and
    escalates from there. Quality gates eligibility; price decides the order.
    """

    @staticmethod
    def _outcomes(n_pass: int, n_fail: int) -> list[tuple[float, bool]]:
        # All neighbours at distance 0 (conf 1.0) so weighted_rate == pass fraction.
        return [(0.0, True)] * n_pass + [(0.0, False)] * n_fail

    def test_cheap_model_clearing_the_bar_is_tried_before_the_frontier(self):
        # The shipped scoring made the frontier the FIRST pick on 79% of tasks: a
        # 43x price gap lost to 2pp of neighbour success rate. A cheap model that
        # clears the bar must be tried first — escalation is what the frontier is for.
        pricing = {"cheap": 0.42, "frontier": 18.0}
        neighbor_results = {
            "cheap": self._outcomes(90, 10),  # 0.90
            "frontier": self._outcomes(96, 4),  # 0.96
        }
        order = compute_cascade_order(
            neighbor_results,
            pricing,
            max_tries=2,
            min_samples=1,
            success_rate_threshold=0.7,
        )
        assert order[0] == "cheap"

    def test_a_model_below_the_bar_is_not_tried_even_when_cheapest(self):
        # Price only orders models that already cleared the quality bar; it never
        # promotes one that failed it.
        pricing = {"cheap": 0.2, "expensive": 10.0}
        neighbor_results = {
            "cheap": self._outcomes(1, 1),  # 0.50 — below the bar
            "expensive": self._outcomes(9, 1),  # 0.90
        }
        order = compute_cascade_order(
            neighbor_results,
            pricing,
            max_tries=2,
            min_samples=1,
            success_rate_threshold=0.7,
        )
        assert order == ["expensive"]


class TestkNNCascadeStrategy:
    """Tests for kNNCascadeStrategy integration (uses mock/synthetic matrix)."""

    def test_name(self):
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        strategy = kNNCascadeStrategy()
        assert strategy.name == "kNN-cascade"


class TestCascadeNeverRetriesAModel:
    """The frontier fallback must not re-bill a model the cascade already tried.

    The shipped fallback appended the frontier unconditionally: 28 tasks paid for
    kimi-k3 twice ($1.5402 of the published total) for a guaranteed-identical outcome.
    """

    @staticmethod
    def _strategy(order):
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

        class _FixedOrder(kNNCascadeStrategy):
            def _get_cascade_order(self, task_id, task_meta, matrix):
                return list(order)

        strategy = _FixedOrder()
        # Skip the HNSW/embedding build — this exercises select()'s fallback only.
        strategy._ready = True
        strategy._pricing = {"cheap": 1.0, "frontier": 10.0}
        return strategy

    @staticmethod
    def _matrix():
        return {
            "models": {
                "cheap": {"input_price": 0.5, "output_price": 0.5},
                "frontier": {"input_price": 5.0, "output_price": 5.0},
            },
            "tasks": {"t1": {"description": "d"}},
            "results": {
                "t1": {
                    "cheap": {"pass": False, "cost": 0.5},
                    "frontier": {"pass": False, "cost": 5.0},
                }
            },
        }

    def test_frontier_already_in_the_shortlist_is_not_tried_twice(self):
        strategy = self._strategy(["cheap", "frontier"])
        chosen = strategy.select("t1", {}, self._matrix())
        assert chosen == "frontier"
        assert strategy.cascade_tried_models == ["cheap", "frontier"]
        assert len(set(strategy.cascade_tried_models)) == len(strategy.cascade_tried_models)
        assert strategy.cascade_total_cost == pytest.approx(5.5)

    def test_frontier_is_still_appended_when_the_shortlist_omits_it(self):
        strategy = self._strategy(["cheap"])
        chosen = strategy.select("t1", {}, self._matrix())
        assert chosen == "frontier"
        assert strategy.cascade_tried_models == ["cheap", "frontier"]
        assert strategy.cascade_total_cost == pytest.approx(5.5)


class TestPriceCascade:
    """The zero-ML price-ascending cascade: cheapest measured model first, frontier last."""

    @staticmethod
    def _matrix(t1_cells):
        return {
            "models": {
                "cheap": {"input_price": 0.1, "output_price": 0.1},
                "mid": {"input_price": 1.0, "output_price": 1.0},
                "dear": {"input_price": 3.0, "output_price": 3.0},
                "frontier": {"input_price": 9.0, "output_price": 9.0},
                "unmeasured": {"input_price": 0.0, "output_price": 0.0},
            },
            "tasks": {"t1": {"description": "d"}},
            "results": {"t1": t1_cells},
        }

    _ALL_FAIL = {
        "cheap": {"pass": False, "cost": 0.1},
        "mid": {"pass": False, "cost": 1.0},
        "dear": {"pass": False, "cost": 3.0},
        "frontier": {"pass": False, "cost": 9.0},
    }

    def test_name(self):
        assert PriceCascade().name == "Price-Cascade"

    def test_stops_at_the_cheapest_model_that_passes(self):
        cells = {**self._ALL_FAIL, "mid": {"pass": True, "cost": 1.0}}
        strategy = PriceCascade()
        assert strategy.select("t1", {}, self._matrix(cells)) == "mid"
        assert strategy.cascade_tried_models == ["cheap", "mid"]
        assert strategy.cascade_total_cost == pytest.approx(1.1)

    def test_escalates_to_the_frontier_last_and_bills_each_model_once(self):
        strategy = PriceCascade()
        assert strategy.select("t1", {}, self._matrix(self._ALL_FAIL)) == "frontier"
        assert strategy.cascade_tried_models == ["cheap", "mid", "dear", "frontier"]
        assert strategy.cascade_total_cost == pytest.approx(13.1)

    def test_frontier_inside_max_tries_is_not_tried_twice(self):
        strategy = PriceCascade(max_tries=9)
        assert strategy.select("t1", {}, self._matrix(self._ALL_FAIL)) == "frontier"
        assert strategy.cascade_tried_models == ["cheap", "mid", "dear", "frontier"]
        assert strategy.cascade_total_cost == pytest.approx(13.1)

    def test_never_routes_to_a_model_the_matrix_never_measured(self):
        # `unmeasured` is the cheapest priced model but has no cell anywhere.
        strategy = PriceCascade()
        strategy.select("t1", {}, self._matrix(self._ALL_FAIL))
        assert "unmeasured" not in strategy.cascade_tried_models

    def test_empty_matrix_degrades_without_crashing(self):
        strategy = PriceCascade()
        assert strategy.select("t1", {}, {"models": {}, "tasks": {}, "results": {}}) == (
            "deepseek-v4-flash"
        )
        assert strategy.cascade_total_cost == 0.0
        assert strategy.cascade_scorable is False

    def test_imports_no_ml_dependency(self):
        # The whole point of this baseline: dependency-free, so it can never inherit the
        # kNN family's embedding cost or its signal problem.
        import ast

        from benchmark.routing.strategies import price_cascade

        tree = ast.parse(Path(price_cascade.__file__).read_text())
        imported = {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not {m for m in imported if m.split(".")[0] in {"hnswlib", "numpy", "fastembed"}}


def _knn_cascade_with_order(order, pricing):
    """A kNN-cascade whose shortlist is fixed, so select()'s escalation path is exercised
    without building an HNSW index or loading an embedder."""
    from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

    class _FixedOrder(kNNCascadeStrategy):
        def _get_cascade_order(self, task_id, task_meta, matrix):
            return list(order)

    strategy = _FixedOrder()
    strategy._ready = True
    strategy._pricing = dict(pricing)
    return strategy


class TestBothCascadesEscalateToTheSameModel:
    """Price-Cascade is the like-for-like zero-ML baseline for kNN-cascade, so an
    asymmetric escalation target would make the two rows incomparable."""

    # kNN-cascade used to take the priciest model in the whole registry while Price-Cascade
    # took the priciest MEASURED one; on a matrix carrying a priced-but-never-run model the
    # two escalated to different cells.

    _PRICING = {"cheap": 0.2, "frontier": 10.0, "never-run": 100.0}

    @staticmethod
    def _matrix():
        return {
            "models": {
                "cheap": {"input_price": 0.1, "output_price": 0.1},
                "frontier": {"input_price": 5.0, "output_price": 5.0},
                # Priced above everything, but the benchmark never ran it: escalating here
                # makes the task unscorable instead of answering it.
                "never-run": {"input_price": 50.0, "output_price": 50.0},
            },
            "tasks": {"t1": {"description": "d"}},
            "results": {
                "t1": {
                    "cheap": {"pass": False, "cost": 0.2},
                    "frontier": {"pass": False, "cost": 5.0},
                }
            },
        }

    def test_price_cascade_escalates_to_the_priciest_measured_model(self):
        strategy = PriceCascade()
        assert strategy.select("t1", {}, self._matrix()) == "frontier"
        assert "never-run" not in strategy.cascade_tried_models

    def test_knn_cascade_escalates_to_the_same_model(self):
        strategy = _knn_cascade_with_order(["cheap"], self._PRICING)
        assert strategy.select("t1", {}, self._matrix()) == "frontier"
        assert "never-run" not in strategy.cascade_tried_models


class TestUnpricedModelIsNeverTheCheapestPick:
    """An unknown price must sort LAST, not as $0.

    Price is the sole cascade sort key, so a missing price defaulting to 0.0 would make an
    unpriced model the first thing every cascade tries.
    """

    def test_compute_cascade_order_puts_an_unpriced_model_last(self):
        neighbor_results = {
            "ghost": [(0.0, True)] * 3,
            "cheap": [(0.0, True)] * 3,
            "dear": [(0.0, True)] * 3,
        }
        order = compute_cascade_order(
            neighbor_results,
            {"cheap": 1.0, "dear": 9.0},
            max_tries=3,
            min_samples=1,
            success_rate_threshold=0.0,
        )
        assert order == ["cheap", "dear", "ghost"]

    def test_price_cascade_skips_a_model_the_registry_does_not_price(self):
        matrix = {
            "models": {
                "ghost": {},  # measured, but no input_price/output_price anywhere
                "cheap": {"input_price": 0.5, "output_price": 0.5},
                "frontier": {"input_price": 5.0, "output_price": 5.0},
            },
            "tasks": {"t1": {"description": "d"}},
            "results": {
                "t1": {
                    "ghost": {"pass": False, "cost": 0.0},
                    "cheap": {"pass": False, "cost": 1.0},
                    "frontier": {"pass": True, "cost": 10.0},
                }
            },
        }
        strategy = PriceCascade()
        assert strategy.select("t1", {}, matrix) == "frontier"
        assert strategy.cascade_tried_models == ["cheap", "frontier"]


class TestCascadeFallbackReusesTheCheapestPricedModel:
    """No cascade hardcodes a model name while the matrix carries prices."""

    _MATRIX = {
        "models": {
            "dear": {"input_price": 5.0, "output_price": 5.0},
            "budget": {"input_price": 0.1, "output_price": 0.1},
        },
        "tasks": {},
        "results": {},
    }

    def test_price_cascade_falls_back_to_the_cheapest_priced_model(self):
        strategy = PriceCascade()
        assert strategy.select("t1", {}, self._MATRIX) == "budget"
        assert strategy.cascade_scorable is False

    def test_knn_cascade_falls_back_to_the_cheapest_priced_model(self):
        strategy = _knn_cascade_with_order([], {})
        assert strategy.select("t1", {}, self._MATRIX) == "budget"
        assert strategy.cascade_scorable is False


class TestkNNCandidateSetIsBenchmarkEnabled:
    """kNN may only pick models the benchmark actually measured."""

    # The pool was built straight from the shipped registry, so the engine could select
    # `gemini-3.1-pro` — disabled in benchmark.yaml, priced from research not measurement.
    # Those cells do not exist in the matrix, so the task landed in `unscorable` and was
    # silently dropped: kNN reported n=174 while every other strategy reported n=177.

    def test_pool_holds_exactly_the_enabled_models(self):
        from benchmark import config
        from benchmark.routing.strategies import knn

        assert set(knn._benchmark_model_pool().model_names()) == set(config.enabled_models())

    def test_a_disabled_registry_model_is_not_a_candidate(self):
        from benchmark.routing.strategies import knn
        from shunt.models.config import ModelPool, default_registry_path

        unfiltered = set(ModelPool(str(default_registry_path())).model_names())
        candidates = set(knn._benchmark_model_pool().model_names())
        # The registry legitimately carries models the benchmark never ran; none may route.
        assert unfiltered - candidates
        assert "gemini-3.1-pro" not in candidates

    def test_every_scored_task_measures_every_enabled_model(self):
        # The other half of the n=177 guarantee: restricting the pool only removes the
        # dropped tasks if EVERY enabled model has a cell on every scored task. With both
        # halves, no kNN pick can be unscorable, so kNN scores the same 177 tasks as the
        # rest — it shipped at 174 while every other strategy shipped at 177.
        from benchmark import config
        from benchmark.routing import summary

        results = summary.load_scored_matrix()["results"]
        enabled = set(config.enabled_models())
        gaps = {tid: sorted(enabled - set(cells)) for tid, cells in results.items()}
        assert not {t: g for t, g in gaps.items() if g}

    def test_the_built_engine_routes_over_the_restricted_pool(self, monkeypatch):
        import numpy as np

        from benchmark import config
        from benchmark.routing.strategies import knn

        monkeypatch.setattr(
            knn, "_embed_texts", lambda texts: np.eye(len(texts), 4, dtype=np.float32)
        )
        strategy = knn.kNNStrategy()
        strategy._build(
            {
                "models": {"deepseek-v4-flash": {"input_price": 0.1, "output_price": 0.1}},
                "tasks": {"t1": {"description": "a"}, "t2": {"description": "b"}},
                "results": {
                    "t1": {"deepseek-v4-flash": {"pass": True, "cost": 0.1}},
                    "t2": {"deepseek-v4-flash": {"pass": True, "cost": 0.1}},
                },
            }
        )
        pool = strategy._engine._model_pool
        assert set(pool.model_names()) == set(config.enabled_models())


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
