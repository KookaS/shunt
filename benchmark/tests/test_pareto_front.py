"""`pareto_front` generalises the frontier to N axes without moving the published 2-D one."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark.routing.metrics import (
    _LEGACY_AXES,
    Axis,
    _dominates,
    compute_pareto,
    pareto_front,
)

# A bounded finite draw OR a non-finite one — `st.floats` refuses `allow_nan` alongside
# bounds, so the two are composed rather than widened in place.
_AXIS_VALUE = st.one_of(
    st.floats(min_value=0, max_value=100, allow_nan=False),
    st.sampled_from([float("nan"), float("inf"), float("-inf")]),
)

SUMMARY_CSV = Path(__file__).resolve().parents[1] / "routing" / "reports" / "strategy_summary.csv"


def _legacy_compute_pareto(strategies_metrics: dict[str, dict]) -> dict[str, bool]:
    """The PRE-refactor 2-D algorithm, verbatim, as the oracle for the published columns."""
    names = list(strategies_metrics.keys())
    pareto = {name: True for name in names}

    for i, name_i in enumerate(names):
        mi = strategies_metrics[name_i]
        for j, name_j in enumerate(names):
            if i == j:
                continue
            mj = strategies_metrics[name_j]
            if (
                mj["AvgPerf%"] >= mi["AvgPerf%"]
                and mj["TotalCost"] <= mi["TotalCost"]
                and (mj["AvgPerf%"] > mi["AvgPerf%"] or mj["TotalCost"] < mi["TotalCost"])
            ):
                pareto[name_i] = False
                break

    return pareto


def _committed_rows(cost_field: str) -> dict[str, dict]:
    """Committed summary rows remapped exactly as `summary._apply_pareto` remaps them."""
    with SUMMARY_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        r["strategy"]: {"AvgPerf%": float(r["AvgPerf%"]), "TotalCost": float(r[cost_field])}
        for r in rows
        if int(r.get("n_tasks", 0) or 0) > 0
    }


class TestPublishedNumbersDoNotMove:
    @pytest.mark.parametrize("cost_field", ["TotalCost_cacheaware", "TotalCost"])
    def test_committed_summary_frontier_matches_the_legacy_algorithm(self, cost_field: str) -> None:
        rows = _committed_rows(cost_field)
        assert rows, "committed summary carries no scorable row — the oracle would be vacuous"
        assert pareto_front(rows, _LEGACY_AXES).members == _legacy_compute_pareto(rows)

    @pytest.mark.parametrize("cost_field", ["TotalCost_cacheaware", "TotalCost"])
    def test_committed_frontier_matches_the_published_column(self, cost_field: str) -> None:
        column = {"TotalCost_cacheaware": "Pareto", "TotalCost": "Pareto_naive"}[cost_field]
        with SUMMARY_CSV.open(newline="") as fh:
            published = {r["strategy"]: r[column] == "True" for r in csv.DictReader(fh)}
        members = compute_pareto(_committed_rows(cost_field))
        assert {n: members.get(n, False) for n in published} == published

    # NON-FINITE IS ADMITTED ON PURPOSE. Bounding the draw to finite values excluded exactly
    # the inputs where the two implementations disagree: a 30k-trial differential found 0
    # mismatches on finite rows and 14,121 once NaN was let in, because `nan < w` is False, so
    # a NaN row passed the legacy "at least as good" guard on every axis and evicted a real
    # strategy. The oracle is therefore the legacy algorithm on the FINITE SUBSET, plus the
    # exclusion contract on the rest — not the legacy algorithm applied to NaN, which is the
    # buggy behaviour itself.
    @settings(max_examples=300)
    @given(
        st.lists(
            st.tuples(_AXIS_VALUE, _AXIS_VALUE),
            min_size=1,
            max_size=8,
        )
    )
    def test_matches_the_legacy_algorithm_on_arbitrary_rows(
        self, pairs: list[tuple[float, float]]
    ) -> None:
        rows = {
            f"s{i}": {"AvgPerf%": perf, "TotalCost": cost} for i, (perf, cost) in enumerate(pairs)
        }
        result = pareto_front(rows, _LEGACY_AXES)

        finite = {
            name: row
            for name, row in rows.items()
            if all(math.isfinite(row[ax.column]) for ax in _LEGACY_AXES)
        }
        # Every non-finite row is excluded: absent from `members`, and reported in `excluded`.
        assert set(result.excluded) == set(rows) - set(finite)
        assert set(result.members) == set(finite)
        # And on what remains, the published algorithm has not moved.
        assert result.members == _legacy_compute_pareto(finite)


class TestGeneralBehaviour:
    def test_a_missing_value_on_any_axis_excludes_the_row_entirely(self) -> None:
        rows: dict[str, dict[str, float | None]] = {
            "good": {"AvgPerf%": 90.0, "TotalCost": 1.0},
            "bad": {"AvgPerf%": 10.0, "TotalCost": 9.0},
            "unmeasured": {"AvgPerf%": None, "TotalCost": 0.0},
        }
        result = pareto_front(rows, _LEGACY_AXES)
        assert result.excluded == ("unmeasured",)
        # Absent from members entirely — not False, and never on the frontier by way of a
        # coerced 0 cost, which would certify "measured nothing" as optimal.
        assert result.members == {"good": True, "bad": False}
        # And it did not dominate anything either: `bad` is off the frontier because of
        # `good`, and `good` stays on it.

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_value_is_excluded_exactly_as_a_missing_one_is(self, bad: float) -> None:
        # NaN is a non-observation wearing a float: `float("nan")` parses out of a CSV cell, and
        # every comparison against it is False, so an un-excluded NaN satisfies "at least as
        # good" on EVERY axis and evicts the rows it was never measured against.
        rows: dict[str, dict[str, float | None]] = {
            "good": {"AvgPerf%": 90.0, "TotalCost": 5.0},
            "poisoned": {"AvgPerf%": bad, "TotalCost": 1.0},
        }
        result = pareto_front(rows, _LEGACY_AXES)
        assert result.excluded == ("poisoned",)
        assert result.members == {"good": True}

    def test_a_non_finite_row_dominates_nothing(self) -> None:
        axes = (Axis("AvgPerf%", "max"), Axis("TotalCost", "min"))
        nan_row = {"AvgPerf%": float("nan"), "TotalCost": 1.0}
        real_row = {"AvgPerf%": 80.0, "TotalCost": 5.0}
        assert not _dominates(nan_row, real_row, axes)
        assert not _dominates(real_row, nan_row, axes)

    def test_an_exact_tie_leaves_both_rows_on_the_frontier(self) -> None:
        rows = {
            "a": {"AvgPerf%": 50.0, "TotalCost": 2.0},
            "b": {"AvgPerf%": 50.0, "TotalCost": 2.0},
        }
        assert pareto_front(rows, _LEGACY_AXES).members == {"a": True, "b": True}

    def test_equal_cost_is_decided_by_the_strictly_better_perf(self) -> None:
        rows = {
            "a": {"AvgPerf%": 60.0, "TotalCost": 2.0},
            "b": {"AvgPerf%": 50.0, "TotalCost": 2.0},
        }
        assert pareto_front(rows, _LEGACY_AXES).members == {"a": True, "b": False}

    def test_a_single_row_is_trivially_on_the_frontier(self) -> None:
        rows = {"only": {"AvgPerf%": 0.0, "TotalCost": 999.0}}
        assert pareto_front(rows, _LEGACY_AXES).members == {"only": True}

    def test_a_third_axis_can_rescue_a_row_the_2d_frontier_dominates(self) -> None:
        rows = {
            "fast": {"AvgPerf%": 90.0, "TotalCost": 1.0, "Latency": 9.0},
            "slow": {"AvgPerf%": 80.0, "TotalCost": 2.0, "Latency": 1.0},
        }
        assert pareto_front(rows, _LEGACY_AXES).members == {"fast": True, "slow": False}
        three = (*_LEGACY_AXES, Axis("Latency", "min"))
        assert pareto_front(rows, three).members == {"fast": True, "slow": True}

    def test_three_axes_still_dominate_a_row_beaten_everywhere(self) -> None:
        three = (*_LEGACY_AXES, Axis("Latency", "min"))
        rows = {
            "best": {"AvgPerf%": 90.0, "TotalCost": 1.0, "Latency": 1.0},
            "worst": {"AvgPerf%": 90.0, "TotalCost": 1.0, "Latency": 5.0},
        }
        assert pareto_front(rows, three).members == {"best": True, "worst": False}

    def test_result_records_the_axes_it_was_computed_over(self) -> None:
        result = pareto_front({"a": {"AvgPerf%": 1.0, "TotalCost": 1.0}}, _LEGACY_AXES)
        assert result.axes == _LEGACY_AXES
