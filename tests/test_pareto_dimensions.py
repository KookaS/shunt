"""The per-dimension frontier: membership is computed, missing values are excluded."""

# The figure's whole claim is that frontier MEMBERSHIP changes with the axis, so the thing
# worth pinning is the membership routine, not the drawing. Two failure modes are covered
# because both would publish a confident wrong answer: a row that is only optimal on one axis
# being credited on another, and a row with a MISSING value being read as zero — which makes
# it un-dominated by construction and certifies "measured nothing" as optimal.

from __future__ import annotations

from benchmark.routing.figures import pareto_dimensions as pdm

# Two live strategies, deliberately crossing: `always_cheap` wins on dollars and loses on
# calls, `always_frontier` the reverse. Display names must be ones `strategy_class` calls
# LIVE, because only live rows enter a frontier. The count axes are the PER-TASK columns
# (`CallsPerTask`/`OutTokPerTask`), not the corpus totals: an imputed cell carries no counts,
# so a total is a measured-subset sum whose coverage differs per row and cannot be ranked.
_CHEAP = "Always-Cheap"
_FRONTIER = "Always-Frontier"


def _rows(**overrides: dict[str, str]) -> list[dict[str, str]]:
    base = {
        _CHEAP: {
            "strategy": _CHEAP,
            "n_tasks": "10",
            "AvgPerf%": "70",
            "TotalCost_cacheaware": "1",
            "CallsPerTask": "90",
            "sessions_p95": "1",
            "OutTokPerTask": "90",
            "cost_cv": "1",
        },
        _FRONTIER: {
            "strategy": _FRONTIER,
            "n_tasks": "10",
            "AvgPerf%": "90",
            "TotalCost_cacheaware": "50",
            "CallsPerTask": "10",
            "sessions_p95": "1",
            "OutTokPerTask": "10",
            "cost_cv": "1",
        },
    }
    for name, patch in overrides.items():
        base[name] = {**base[name], **patch}
    return list(base.values())


class TestMembershipIsPerDimension:
    def test_a_row_optimal_on_one_axis_is_not_credited_on_another(self) -> None:
        front = pdm.membership(_rows())
        assert front["TotalCost_cacheaware"] == {_CHEAP: True, _FRONTIER: True}
        # On calls the cheap row is beaten on BOTH axes, so it leaves the frontier — which is
        # the figure's entire claim, reduced to two rows.
        assert front["CallsPerTask"] == {_CHEAP: False, _FRONTIER: True}

    def test_a_missing_value_is_excluded_not_read_as_zero(self) -> None:
        rows = _rows(**{_CHEAP: {"CallsPerTask": ""}})
        front = pdm.membership(rows)
        # Zero calls would be un-dominated by construction; excluded is the correct answer.
        assert front["CallsPerTask"][_CHEAP] is False
        assert pdm.excluded_counts(rows)["CallsPerTask"] == (_CHEAP,)
        assert pdm.excluded_counts(rows)["TotalCost_cacheaware"] == ()

    def test_a_nan_cell_is_excluded_and_counted_rather_than_evicting_a_real_row(self) -> None:
        # `float("nan")` parses straight out of a CSV cell, and every comparison against NaN is
        # False — so before this was excluded, one `nan` cell satisfied "at least as good" on
        # BOTH axes at once and took a real strategy off a published panel while the excluded
        # count still read 0. A non-finite cell is the same non-observation as an empty one.
        rows = _rows(**{_CHEAP: {"cost_cv": "nan"}})
        front = pdm.membership(rows)
        assert front["cost_cv"] == {_CHEAP: False, _FRONTIER: True}
        assert pdm.excluded_counts(rows)["cost_cv"] == (_CHEAP,)
