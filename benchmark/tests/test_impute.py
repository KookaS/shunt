"""Unit tests for the pure monotone-ladder imputation."""

from __future__ import annotations

import hashlib
from typing import Final

import pytest

from benchmark import config
from benchmark.config import CapabilityRank, RankedModel
from benchmark.routing import censoring, impute
from benchmark.routing.impute import Violation, complete_matrix

# A synthetic 6-model capability rank (weakest -> strongest). c0<c1 sit below m0<m1 below
# h0 below f0, so "extend to every ranked model" (not just observed ones) is exercised and
# the models within a former tier (c0/c1, m0/m1) get DISTINCT ranks, not one collapsed rung.
_ORDER: Final[list[str]] = ["c0", "c1", "m0", "m1", "h0", "f0"]
RANK: Final[CapabilityRank] = CapabilityRank(
    ordered=[RankedModel(m, "a", i, "measured") for i, m in enumerate(_ORDER)],
    evidence={},
)
ALL_MODELS: Final[frozenset[str]] = frozenset(_ORDER)


def cell(passed: bool, cost: float) -> dict:
    return {"pass": passed, "real_cost": cost, "cost": cost}


def censored_cell(cost: float, stop_reason: str = "step_limit") -> dict:
    """A CENSORED cell (resource-limit stop): pass=False on disk, true outcome unknown."""
    return {"pass": False, "real_cost": cost, "cost": cost, "stop_reason": stop_reason}


# --------------------------------------------------------- s*/f* + extend-to-all-models


def test_ground_truth_regime_a() -> None:
    # Ladder collection for a task solvable at m0: every cheaper model fails, m0 passes, stop.
    # Regime-A observes densely up to the crossover, so completion has ZERO UNKNOWN band.
    matrix = {"t1": {"c0": cell(False, 0.01), "c1": cell(False, 0.02), "m0": cell(True, 0.20)}}
    im = complete_matrix(matrix, RANK)
    out = im.matrix["t1"]

    assert set(out) == ALL_MODELS  # every ranked model scorable -> equal coverage
    assert im.n_unknown == 0
    assert im.tau["t1"] == "m0"  # crossover = lowest observed PASS model

    assert out["c0"]["pass"] is False and out["c0"]["imputed"] is False  # real
    assert out["c1"]["pass"] is False and out["c1"]["imputed"] is False  # real
    assert out["m0"]["pass"] is True and out["m0"]["imputed"] is False  # real
    for m in ("m1", "h0", "f0"):  # rank >= s* imputed to pass, sourced from m0
        assert out[m]["pass"] is True and out[m]["imputed"] is True
        assert out[m]["source_model"] == "m0"


def test_extend_to_all_models_equal_coverage() -> None:
    # Dense (ladder-collected) observations per task -> zero UNKNOWN, every model covered.
    matrix = {
        "a": {"c0": cell(True, 0.01)},  # tau = c0 (weakest)
        "b": {"c0": cell(False, 0.01), "c1": cell(False, 0.01), "m0": cell(True, 0.2)},  # tau = m0
        "c": {  # tau = h0
            "c0": cell(False, 0.01),
            "c1": cell(False, 0.01),
            "m0": cell(False, 0.2),
            "m1": cell(False, 0.2),
            "h0": cell(True, 0.4),
        },
    }
    im = complete_matrix(matrix, RANK)
    assert im.n_unknown == 0
    # Every strategy would score every model on every task: coverage is equal.
    for model in ALL_MODELS:
        covered = sum(1 for task in im.matrix.values() if model in task)
        assert covered == len(matrix)


# ------------------------------------------------------- censored cells are non-observations


def test_censored_top_tier_leaves_task_incomplete() -> None:
    # Every cheaper model GENUINELY fails, but the top tier merely ran out of steps (censored).
    # Old semantics counted the censored top as an observed fail → complete/unsolvable. Now the
    # censored cell is NOT an observation, so the top rung stays UNKNOWN and the task is INCOMPLETE.
    matrix = {
        "t": {
            "c0": cell(False, 0.01),
            "c1": cell(False, 0.02),
            "m0": cell(False, 0.10),
            "m1": cell(False, 0.12),
            "h0": cell(False, 0.30),
            "f0": censored_cell(0.40, "step_limit"),
        }
    }
    im = complete_matrix(matrix, RANK)
    assert "t" not in im.complete  # INCOMPLETE — the top rung's true outcome is unknown
    assert im.n_unknown == 1
    assert im.tau["t"] is None  # no observed pass, and no complete all-fail either
    assert "f0" not in im.matrix["t"]  # the censored cell was neither kept-real nor imputed


def test_uncensored_all_fail_is_complete_unsolvable() -> None:
    # The same shape but the top tier GENUINELY fails (uncensored): a real all-fail is COMPLETE.
    matrix = {
        "t": {
            "c0": cell(False, 0.01),
            "c1": cell(False, 0.02),
            "m0": cell(False, 0.10),
            "m1": cell(False, 0.12),
            "h0": cell(False, 0.30),
            "f0": cell(False, 0.40),
        }
    }
    im = complete_matrix(matrix, RANK)
    assert "t" in im.complete  # COMPLETE — every tier observed and all genuinely failed
    assert im.n_unknown == 0
    assert im.tau["t"] is None  # unsolvable, but its crossover (none) is established


def test_censored_cell_imputed_when_lower_model_passed() -> None:
    # A weaker model passed, so monotonicity DETERMINES the stronger censored cell as a pass —
    # the censored observation is replaced by an imputed pass and the task stays COMPLETE.
    matrix = {"t": {"c0": cell(True, 0.02), "m0": censored_cell(0.10, "wall_limit")}}
    im = complete_matrix(matrix, RANK)
    out = im.matrix["t"]
    assert "t" in im.complete
    assert im.tau["t"] == "c0"
    assert out["m0"]["pass"] is True and out["m0"]["imputed"] is True


# ------------------------------------------------------------- observed truth wins


def test_observed_truth_wins_violation() -> None:
    # c0 passes but the higher-ranked m0 fails: a real higher-rank fail below a real pass.
    matrix = {"v": {"c0": cell(True, 0.02), "m0": cell(False, 0.50)}}
    im = complete_matrix(matrix, RANK)
    out = im.matrix["v"]

    assert im.violations == [Violation(task_id="v", pass_model="c0", fail_model="m0")]
    assert out["c0"]["pass"] is True and out["c0"]["imputed"] is False  # real kept
    assert out["m0"]["pass"] is False and out["m0"]["imputed"] is False  # real kept
    # The contradicted region (ranks in [s*=0, f*=2]) is NOT imputed -> c1 (rank 1) UNKNOWN.
    assert "c1" not in out
    assert im.n_unknown == 1
    for m in ("m1", "h0", "f0"):  # above the contradiction: imputed pass from s* (c0)
        assert out[m]["pass"] is True and out[m]["imputed"] is True
        assert out[m]["source_model"] == "c0"


# --------------------------------------------------------------- sparse UNKNOWN gap


def test_unknown_gap_random_partial() -> None:
    # Sparse (non-ladder) data: only c0 (fail) and h0 (pass) observed. Every rank strictly
    # between f*(c0=0) and s*(h0=4) — c1, m0, m1 — is an UNKNOWN coverage gap.
    matrix = {"g": {"c0": cell(False, 0.01), "h0": cell(True, 0.30)}}
    im = complete_matrix(matrix, RANK)
    out = im.matrix["g"]

    assert im.tau["g"] == "h0"  # s* model
    assert "c1" not in out and "m0" not in out and "m1" not in out  # strictly between f* and s*
    assert im.n_unknown == 3
    assert out["f0"]["pass"] is True and out["f0"]["imputed"] is True  # >= s*
    assert out["f0"]["source_model"] == "h0"


# ------------------------------------------------------------------------- tau


def test_tau_recovered_per_task() -> None:
    matrix = {
        "cheap_task": {"c0": cell(True, 0.01)},
        "high_task": {"c0": cell(False, 0.01), "m0": cell(False, 0.2), "h0": cell(True, 0.4)},
        "dead_task": {  # no model solves it
            "c0": cell(False, 0.01),
            "m0": cell(False, 0.2),
            "h0": cell(False, 0.4),
            "f0": cell(False, 0.6),
        },
    }
    im = complete_matrix(matrix, RANK)
    assert im.tau == {"cheap_task": "c0", "high_task": "h0", "dead_task": None}


# -------------------------------------------------------- unsolvable = universal fail


def test_unsolvable_universal_fail() -> None:
    matrix = {
        "u": {
            "c0": cell(False, 0.01),
            "m0": cell(False, 0.02),
            "h0": cell(False, 0.03),
            "f0": cell(False, 0.04),
        }
    }
    im = complete_matrix(matrix, RANK)
    out = im.matrix["u"]
    assert im.tau["u"] is None
    assert im.n_unknown == 0
    # Every ranked model (real or imputed) fails -> a fail@its-cost for every strategy.
    assert set(out) == ALL_MODELS
    assert all(c["pass"] is False for c in out.values())
    assert out["c1"]["imputed"] is True and out["m1"]["imputed"] is True


# ---------------------------------------------------------------- cost imputation


def test_cost_imputation_flagged_and_median() -> None:
    # m0 measured passing three times (median 0.2); on a c0-only task it is imputed.
    matrix = {
        "p1": {"c0": cell(False, 0.01), "c1": cell(False, 0.01), "m0": cell(True, 0.10)},
        "p2": {"c0": cell(False, 0.01), "c1": cell(False, 0.01), "m0": cell(True, 0.20)},
        "p3": {"c0": cell(False, 0.01), "c1": cell(False, 0.01), "m0": cell(True, 0.30)},
        "q": {"c0": cell(True, 0.05)},  # tau = c0 -> every higher model imputed pass
    }
    im = complete_matrix(matrix, RANK)
    q = im.matrix["q"]

    assert q["m0"]["imputed"] is True
    assert q["m0"]["cost"] == pytest.approx(0.20)  # per-MODEL median measured real_cost
    assert q["m1"]["cost"] == pytest.approx(0.20)  # rank-neighbour median fallback (m0 passes)
    # Every imputed cell is flagged; every real cell is not.
    for task in im.matrix.values():
        for c in task.values():
            assert c["imputed"] in (True, False)
    assert im.matrix["p1"]["m0"]["imputed"] is False  # real, untouched flag


# ----------------------------------------------------------------- determinism


def test_deterministic() -> None:
    matrix = {
        "a": {"c0": cell(True, 0.01)},
        "b": {"c0": cell(False, 0.01), "m0": cell(True, 0.2)},
        "v": {"c0": cell(True, 0.02), "m0": cell(False, 0.5)},
    }
    first = complete_matrix(matrix, RANK)
    second = complete_matrix(matrix, RANK)
    assert first == second  # frozen dataclasses -> structural equality


# ---------------------------------------------------- results.csv never persisted


def test_results_csv_byte_identical() -> None:
    path = config.results_csv_path()
    if not path.exists():
        pytest.skip("no results.csv to protect in this checkout")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    matrix = config.load_matrix()["results"]
    complete_matrix(matrix, config.capability_rank(matrix))
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after  # the pure completer writes nothing


def test_real_results_csv_completes_sanely() -> None:
    # Locks the flat-input contract against real data: the raw arm-nested cache would
    # collapse every tau to None; the default-arm-flattened view completes correctly.
    path = config.results_csv_path()
    if not path.exists():
        pytest.skip("no real results.csv in this checkout (CI)")
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    flat = config.flatten_default_arm(config.load_results())
    rank = config.capability_rank(flat)
    ranked = {r.model for r in rank.ordered}
    im = complete_matrix(flat, rank)

    n_tasks = len(flat)
    assert n_tasks > 0
    # (a) tau non-None for a clear majority (all-None is the regression this guards).
    n_tau = sum(1 for t in im.tau.values() if t is not None)
    assert n_tau > n_tasks // 2
    # (b) coverage strictly improves: every real UNCENSORED enabled cell is kept, plus imputed
    # ones. A CENSORED cell (resource-limit stop, unknown outcome) is not an observation, so it is
    # excluded from n_real — hence the raw enabled count is reduced by the censored cells.
    raw_enabled = sum(1 for cells in flat.values() for m in cells if m in ranked)
    raw_censored = sum(
        1
        for cells in flat.values()
        for m, c in cells.items()
        if m in ranked and censoring.is_censored(c)
    )
    assert im.n_real == raw_enabled - raw_censored
    assert im.n_real + im.n_imputed > raw_enabled - raw_censored
    # (d) violations are measured on real data (some tasks break monotonicity).
    assert len(im.violations) > 0
    # (c) results.csv byte-identical before/after.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# ------------------------------------------ FIX D: non-observations excluded from cost model


def test_is_non_observation_classifies_rows() -> None:
    # A timed-out cell OR a zero-call/$0 cell is a NON-observation; a real $0 call is kept.
    assert impute._is_non_observation({"timeout_flag": True, "calls": 5, "real_cost": 1.0}) is True
    assert impute._is_non_observation({"calls": 0, "real_cost": 0.0}) is True
    assert impute._is_non_observation({"pass": True, "calls": 2, "real_cost": 0.0}) is False
    assert impute._is_non_observation({"pass": False, "calls": 3, "real_cost": 0.30}) is False


def test_cost_model_excludes_non_observations_from_median() -> None:
    # m0 has two real fail observations at $0.30 plus three non-observations ($0 timeout/zero-work).
    # WITH exclusion the median is $0.30; WITHOUT it, median([.3,.3,0,0,0]) = $0 — the poison FIX D
    # removes (a $0 timeout row is not a $0 observation, it is no observation at all).
    matrix = {
        "t1": {"m0": {"pass": False, "real_cost": 0.30, "calls": 3}},
        "t2": {"m0": {"pass": False, "real_cost": 0.30, "calls": 3}},
        "t3": {"m0": {"pass": False, "real_cost": 0.0, "calls": 0, "timeout_flag": True}},
        "t4": {"m0": {"pass": False, "real_cost": 0.0, "calls": 0, "timeout_flag": True}},
        "t5": {"m0": {"pass": False, "real_cost": 0.0, "calls": 0}},
    }
    cm = impute._build_cost_model(matrix, tuple(_ORDER))
    assert cm.cost("m0", False) == pytest.approx(0.30)
