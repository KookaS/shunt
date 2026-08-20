"""Window semantics and census nesting — the two places this family disagreed with itself."""

# F2 (cost) and F6 (escalation) both draw a bar labelled `7d`. They used to compute it from two
# different definitions, so one corpus printed `7d n=1` on one figure and `7d n=15` on the other.
# The tests below fail if the two call paths ever diverge again, whichever way they drift.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shunt.db.loop_health import StratumCensus, StratumStages
from shunt.inspect.inference import data as idata

_WINDOWS = (7, 30, None)


def _row(session_id: str, *, when: datetime | None, rung: str | None = None) -> idata.SessionRow:
    return idata.SessionRow(
        session_id=session_id,
        timestamp=when,
        model_chosen="cheap",
        cost=1.0,
        cost_known=True,
        stratum=idata.LIVE,
        selection_rule_used="knn",
        selection_propensity=None,
        hold_reason=None,
        rung=rung,
        undeliverable=False,
        tier2_success=None,
    )


def _stale_burst() -> list[idata.SessionRow]:
    """15 rows, all a year old, clustered inside one week — the corpus that split the two."""
    # Newest-row-relative, every row sits inside 7 days of the newest and `7d` reads 15. Wall
    # clock, none of them arrived this week and `7d` reads 0. This is the real rig's disagreement.
    now = datetime.now(UTC)
    return [
        _row(f"live-old-{i}", when=now - timedelta(days=400 + i), rung="raise_rank")
        for i in range(15)
    ]


def _mixed_corpus() -> list[idata.SessionRow]:
    now = datetime.now(UTC)
    fresh = [
        _row(f"live-new-{i}", when=now - timedelta(days=1 + i), rung="raise_rank") for i in (0, 1)
    ]
    return [*_stale_burst(), *fresh]


def _cost_totals(rows: list[idata.SessionRow]) -> dict[str, int]:
    windows = idata.cost(rows, _WINDOWS).windows
    return {label: agg.n_cost_known + agg.n_cost_unknown for label, agg in windows}


def _rate_totals(rows: list[idata.SessionRow]) -> dict[str, int]:
    return {label: n for label, _escalated, n in idata.escalation(rows, _WINDOWS).rates}


class TestOneWindowDefinition:
    @pytest.mark.parametrize("rows", [_stale_burst(), _mixed_corpus()])
    def test_cost_and_escalation_agree_on_every_window_over_one_corpus(
        self, rows: list[idata.SessionRow]
    ) -> None:
        rates = _rate_totals(rows)
        assert _cost_totals(rows) == rates

    def test_a_year_old_burst_reports_an_empty_week_on_both_figures(self) -> None:
        rows = _stale_burst()
        rates = _rate_totals(rows)
        assert rates["7d"] == 0
        assert rates["all"] == 15

    def test_a_stale_burst_is_not_relabelled_as_the_last_seven_days(self) -> None:
        # The newest-row-relative definition reported the 400-day-old burst under `7d`; wall clock
        # reports only the two rows that really did arrive this week.
        rows = _mixed_corpus()
        rates = _rate_totals(rows)
        assert rates["7d"] == 2
        assert rates["all"] == len(rows)

    def test_an_unstamped_row_is_counted_by_all_but_never_by_a_window(self) -> None:
        rows = [*_mixed_corpus(), _row("live-unstamped", when=None)]
        rates = _rate_totals(rows)
        assert rates["7d"] == 2
        assert rates["all"] == len(rows)
        assert _cost_totals(rows) == rates


def _census(**live: int) -> StratumCensus:
    empty = {"stored": 0, "embedded": 0, "labeled": 0, "tier2": 0, "indexed": 0}
    return StratumCensus(
        seeded=StratumStages(stratum="seeded", **empty),
        live=StratumStages(stratum="live", **{**empty, **live}),
    )


class TestCensusNesting:
    def test_a_nested_census_reports_no_break(self) -> None:
        census = _census(stored=40, embedded=39, labeled=14, tier2=13, indexed=13)
        assert idata._nesting_breaks(census) == ()

    def test_a_tier2_without_its_outcome_event_is_named_not_drawn_as_a_funnel(self) -> None:
        # The real rig: `labeled` counts non-tombstoned `outcome_events`, `tier2` counts a
        # materialized `outcomes.tier2_outcome`. Two tables, so `tier2` can exceed `labeled`.
        census = _census(stored=40, embedded=39, labeled=13, tier2=14, indexed=13)
        assert idata._nesting_breaks(census) == ("live: tier2 (14) > labeled (13)",)

    def test_every_adjacent_stage_pair_is_checked(self) -> None:
        census = _census(stored=1, embedded=2, labeled=3, tier2=4, indexed=5)
        breaks = idata._nesting_breaks(census)
        assert len(breaks) == len(idata.CENSUS_STAGES) - 1
        assert all(name.startswith("live: ") for name in breaks)
