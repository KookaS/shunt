"""Unit tests for the derived per-model capability rank (config.capability_rank)."""

from __future__ import annotations

from typing import Final

import pytest

from benchmark import config
from benchmark.config import CapabilityRank, derive_capability_rank

# Default knobs; individual tests tighten K/W to exercise the price-prior fallback.
KNOBS: Final[dict] = {"K": 3, "W": 0.9, "min_pairs": 2, "baseline": None}
PRICES: Final[dict[str, float]] = {"cheap": 1.0, "mid": 3.0, "hi": 6.0, "front": 12.0}
ARMS: Final[dict[str, str]] = {m: "a" for m in PRICES}


def cell(passed: bool) -> dict:
    return {"pass": passed, "real_cost": 0.1, "cost": 0.1}


def _derive(matrix: dict, models: list[str], knobs: dict | None = None, **kw) -> CapabilityRank:
    return derive_capability_rank(matrix, models, PRICES, ARMS, knobs or KNOBS, **kw)


# ----------------------------------------------------------------- order_and_source


def test_order_and_source() -> None:
    # Clean monotone dominance: front > hi > mid > cheap on co-measured tasks.
    matrix = {
        f"t{i}": {
            "cheap": cell(i < 1),
            "mid": cell(i < 3),
            "hi": cell(i < 5),
            "front": cell(i < 7),
        }
        for i in range(7)
    }
    rank = _derive(matrix, ["cheap", "mid", "hi", "front"])
    assert [r.model for r in rank.ordered] == ["cheap", "mid", "hi", "front"]
    assert rank.strongest() == "front"
    assert all(r.source == "measured" for r in rank.ordered)
    assert rank.rank_of("cheap") == 0 and rank.rank_of("front") == 3


# --------------------------------------------------------------------- coverage_trap


def test_coverage_trap_frontier_ranks_above_cheap() -> None:
    # The frontier ran ONLY on the hard subset (t_hard*), where cheap fails and it passes.
    # Cheap also ran on many EASY tasks it passes -> its raw marginal pass-rate is HIGH.
    # Raw marginal rank would invert (cheap above frontier); Copeland on co-measured must not.
    matrix: dict = {}
    for i in range(10):  # easy tasks: only cheap ran, all pass -> inflate cheap's raw rate
        matrix[f"easy{i}"] = {"cheap": cell(True)}
    for i in range(6):  # hard co-measured subset: cheap fails, frontier passes
        matrix[f"hard{i}"] = {"cheap": cell(False), "front": cell(True)}
    rank = _derive(matrix, ["cheap", "front"])
    # Raw marginal: cheap = 10/16 ≈ 0.63, front = 6/6 = 1.0 (front already higher here) —
    # but the point is dominance: on co-measured tasks front beats cheap 6-0.
    front_rank, cheap_rank = rank.rank_of("front"), rank.rank_of("cheap")
    assert front_rank is not None and cheap_rank is not None and front_rank > cheap_rank
    assert rank.strongest() == "front"


# ------------------------------------------------------------------------ price_prior


def test_price_prior_low_data_takes_price_slot() -> None:
    # 'mid' has only 1 cell (< K=3) -> price-implied, slotted by price between measured.
    matrix = {
        f"t{i}": {"cheap": cell(i < 2), "hi": cell(True), "front": cell(True)} for i in range(4)
    }
    matrix["only"] = {"mid": cell(True)}  # single cell for mid (K defaults to 3)
    rank = _derive(matrix, ["cheap", "mid", "hi", "front"])
    assert rank.evidence["mid"].source == "price-prior"
    order = [r.model for r in rank.ordered]
    # mid (price 3.0) slots between cheap (1.0) and hi (6.0) by price.
    assert order.index("cheap") < order.index("mid") < order.index("hi")


def test_fully_cold_equals_price_order() -> None:
    # No model clears the gate (all n < K) -> the order is pure price ascending.
    matrix = {"t0": {"cheap": cell(True), "front": cell(True)}}
    knobs = {**KNOBS, "K": 50}
    rank = _derive(matrix, ["front", "cheap", "mid", "hi"], knobs)
    assert [r.model for r in rank.ordered] == ["cheap", "mid", "hi", "front"]
    assert all(r.source == "price-prior" for r in rank.ordered)


def test_measured_wins_price_tie() -> None:
    # A measured model and a price-only model at the SAME price: measured is placed first.
    # Three co-measured models (cheap/anchor/m_meas) so each clears the >=2-peer gate.
    prices = {"cheap": 1.0, "anchor": 3.0, "m_meas": 5.0, "p_only": 5.0}
    arms = {m: "a" for m in prices}
    matrix = {
        f"t{i}": {"cheap": cell(i < 1), "anchor": cell(i < 3), "m_meas": cell(i < 5)}
        for i in range(5)
    }
    matrix["one"] = {"p_only": cell(True)}  # single isolated cell -> price-prior
    knobs = {**KNOBS, "K": 3}
    rank = derive_capability_rank(
        matrix, ["cheap", "anchor", "m_meas", "p_only"], prices, arms, knobs
    )
    assert rank.evidence["m_meas"].source == "measured"
    assert rank.evidence["p_only"].source == "price-prior"
    order = [r.model for r in rank.ordered]
    assert order.index("m_meas") < order.index("p_only")  # measured wins the tie


# ------------------------------------------------------------------------ deterministic


def test_deterministic_and_tiebreak() -> None:
    matrix = {
        f"t{i}": {"cheap": cell(i < 1), "mid": cell(i < 3), "front": cell(i < 5)} for i in range(5)
    }
    first = _derive(matrix, ["cheap", "mid", "front"])
    second = _derive(matrix, ["front", "mid", "cheap"])  # input order must not matter
    assert first == second


def test_tiebreak_by_price_then_name() -> None:
    # Two models with identical Copeland score and pass-rate: break by price ascending.
    prices = {"x": 2.0, "y": 1.0}  # y cheaper -> y weaker on the price tie-break
    arms = {"x": "a", "y": "a"}
    matrix = {f"t{i}": {"x": cell(True), "y": cell(True)} for i in range(4)}  # never disagree
    knobs = {**KNOBS, "K": 3, "min_pairs": 2}
    rank = derive_capability_rank(matrix, ["x", "y"], prices, arms, knobs)
    assert [r.model for r in rank.ordered] == ["y", "x"]  # equal score/p̂ -> cheaper first


# --------------------------------------------------------------------------- hysteresis


# Three fully co-measured models so each clears the >=2-peer gate and every rank is measured.
_HYS_PRICES: Final[dict[str, float]] = {"p": 1.0, "q": 2.0, "r": 3.0}
_HYS_ARMS: Final[dict[str, str]] = {m: "x" for m in _HYS_PRICES}
_HYS_KNOBS: Final[dict] = {**KNOBS, "K": 3, "min_pairs": 2}


def _nested_matrix(n: int, passes: dict[str, int]) -> dict:
    """n co-measured tasks; model m passes the first ``passes[m]`` (nested -> clean dominance)."""
    return {f"t{i}": {m: cell(i < passes[m]) for m in passes} for i in range(n)}


def test_hysteresis_disjoint_ci_accepts_reversal() -> None:
    # Data dominance is clean: r(20/20) > q(10/20) > p(2/20); r and q CIs are DISJOINT.
    # Baseline wrongly ranks q strongest ([p, r, q]); the disjoint evidence reverses it.
    matrix = _nested_matrix(20, {"p": 2, "q": 10, "r": 20})
    rank = derive_capability_rank(
        matrix, ["p", "q", "r"], _HYS_PRICES, _HYS_ARMS, _HYS_KNOBS, baseline=["p", "r", "q"]
    )
    order = [r.model for r in rank.ordered]
    assert order.index("r") > order.index("q")  # disjoint CIs -> data (r strongest) wins


def test_hysteresis_overlapping_ci_keeps_baseline() -> None:
    # r(12/20) barely edges q(10/20) in the data, but their CIs OVERLAP -> not supported.
    # Baseline ranks q strongest ([p, r, q]); the weak signal must not flip it.
    matrix = _nested_matrix(20, {"p": 2, "q": 10, "r": 12})
    rank = derive_capability_rank(
        matrix, ["p", "q", "r"], _HYS_PRICES, _HYS_ARMS, _HYS_KNOBS, baseline=["p", "r", "q"]
    )
    order = [r.model for r in rank.ordered]
    assert order.index("r") < order.index("q")  # overlap -> baseline (q strongest) retained


# ------------------------------------------------------ real data (kill-gate 1 sanity)


def test_real_capability_rank_frontier_strongest() -> None:
    path = config.results_csv_path()
    if not path.exists():
        pytest.skip("no real results.csv in this checkout (CI)")
    config.load("benchmark/benchmark.yaml")
    rank = config.capability_rank()
    control = config.frontier_model()
    # The kill-gate sanity: Copeland puts the enabled frontier ON TOP, not inverted.
    assert rank.strongest() == control
    order = [r.model for r in rank.ordered]
    # gpt-5-mini out-ranked by deepseek-v4-flash FROM THE DATA (no manual re-tier hack).
    assert order.index("deepseek-v4-flash") > order.index("gpt-5-mini")
    # At current scale every enabled model clears the gate.
    assert all(r.source == "measured" for r in rank.ordered)
