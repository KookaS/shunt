"""TierClassifier single-shot rank-position prediction."""

from __future__ import annotations

from typing import Final

from benchmark.config import CapabilityRank, RankedModel
from benchmark.routing.strategies.tier_classifier import TierClassifier, predict_model

# Capability rank, weakest -> strongest.
MODELS: Final[list[str]] = ["c0", "m0", "h0", "f0"]


def _rank() -> CapabilityRank:
    return CapabilityRank(
        ordered=[RankedModel(m, "a", i, "measured") for i, m in enumerate(MODELS)], evidence={}
    )


def _matrix(results: dict) -> dict:
    return {"tasks": {}, "results": results}


def test_predict_lowest_passing_model() -> None:
    # Neighbours: c0 fails, m0 passes -> route to m0.
    results = {
        "n1": {"c0": {"pass": False}, "m0": {"pass": True}},
        "n2": {"c0": {"pass": False}, "m0": {"pass": True}},
        "n3": {"c0": {"pass": False}, "m0": {"pass": True}},
    }
    model = predict_model(
        ["n1", "n2", "n3"], _matrix(results), MODELS, threshold=0.6, min_samples=3
    )
    assert model == "m0"


def test_predict_cheapest_when_neighbours_solve_it() -> None:
    results = {f"n{i}": {"c0": {"pass": True}} for i in range(3)}
    model = predict_model(list(results), _matrix(results), MODELS, threshold=0.6, min_samples=3)
    assert model == "c0"


def test_predict_strongest_when_none_clears_bar() -> None:
    # No model's neighbour pass-rate reaches the threshold -> predicted hard, route strongest.
    results = {f"n{i}": {"c0": {"pass": False}, "m0": {"pass": False}} for i in range(3)}
    model = predict_model(list(results), _matrix(results), MODELS, threshold=0.6, min_samples=3)
    assert model == "f0"


def test_predict_skips_model_below_min_samples() -> None:
    # Only 1 neighbour observed c0 (passing) but min_samples=3 -> c0 skipped;
    # m0 has 3 passing observations -> m0 wins.
    results = {
        "n1": {"c0": {"pass": True}, "m0": {"pass": True}},
        "n2": {"m0": {"pass": True}},
        "n3": {"m0": {"pass": True}},
    }
    model = predict_model(
        ["n1", "n2", "n3"], _matrix(results), MODELS, threshold=0.6, min_samples=3
    )
    assert model == "m0"


def test_empty_rank_falls_back() -> None:
    got = predict_model(["n1"], _matrix({}), [], threshold=0.6, min_samples=1)
    assert got == "deepseek-v4-flash"


def test_imputed_pass_is_not_a_vote() -> None:
    # c0 has ONE measured fail; its other five neighbour "passes" are IMPUTED monotone-ladder
    # fills. Counted, they lift the mean to 5/6 >= 0.6 and c0 wins; excluded, its measured vote
    # is 0/1 and m0 (measured pass) wins. Isolates the pass-VOTE half of the exclusion.
    results = {
        "n1": {"c0": {"pass": False, "imputed": False}, "m0": {"pass": True, "imputed": False}},
        "n2": {"c0": {"pass": True, "imputed": True}, "m0": {"pass": True, "imputed": False}},
        "n3": {"c0": {"pass": True, "imputed": True}, "m0": {"pass": True, "imputed": False}},
        "n4": {"c0": {"pass": True, "imputed": True}},
        "n5": {"c0": {"pass": True, "imputed": True}},
        "n6": {"c0": {"pass": True, "imputed": True}},
    }
    model = predict_model(
        ["n1", "n2", "n3", "n4", "n5", "n6"],
        _matrix(results),
        MODELS,
        threshold=0.6,
        min_samples=1,
    )
    assert model == "m0"


def test_imputed_pass_is_not_a_sample() -> None:
    # c0's neighbours ALL pass, but every pass is imputed -> ZERO measured samples, so c0 is
    # below min_samples and skipped. Isolates the min_samples half of the exclusion.
    results = {f"n{i}": {"c0": {"pass": True, "imputed": True}} for i in range(1, 4)}
    model = predict_model(
        ["n1", "n2", "n3"], _matrix(results), MODELS, threshold=0.6, min_samples=3
    )
    # No model clears the bar with measured evidence -> strongest is the fallback.
    assert model == "f0"


def test_measured_pass_still_counts_and_wins() -> None:
    # The control: the SAME three c0 passes, flagged measured, make c0 the weakest eligible.
    results = {f"n{i}": {"c0": {"pass": True, "imputed": False}} for i in range(1, 4)}
    model = predict_model(
        ["n1", "n2", "n3"], _matrix(results), MODELS, threshold=0.6, min_samples=3
    )
    assert model == "c0"


def test_select_empty_matrix_falls_back_to_weakest(monkeypatch) -> None:
    from benchmark import config

    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: _rank())
    strat = TierClassifier()
    assert strat.select("t", {}, {"results": {}}) == "c0"  # weakest model, no embedding
