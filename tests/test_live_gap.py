"""live_gap.png must not price an empty class — honest words instead of a $0.00.

`_cheapest` returning 0.0 for an absent class printed "cheapest live $0.00" for the
empty set and silently dropped the figure's own headroom caveat.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.figures import live_gap  # noqa: E402
from benchmark.routing.strategy_class import StrategyClass  # noqa: E402

# The committed n=180 corpus: 8 strategies, 4 inside the 1pp band (all bound/blocked),
# and NOT ONE live row in it. Always-Frontier fell to 95.00% and dropped out; the live
# rows (Always-Frontier, kNN, Always-Cheap) are all below 95.67%.
_N180_ROWS = (
    ("Oracle", "15.18", "96.67"),
    ("Price-Cascade", "22.07", "96.67"),
    ("kNN-cascade", "24.95", "96.67"),
    ("Session-Cascade", "27.68", "96.67"),
    ("Always-Frontier", "91.15", "95.00"),
    ("kNN", "11.61", "78.89"),
    ("Always-Cheap", "1.39", "77.22"),
    ("Tier-Classifier", "9.46", "65.56"),
)


def _n180_rows() -> list[dict[str, str]]:
    """The committed n=180 corpus as summary-row dicts, freshly built per test."""
    return [
        {"strategy": name, "TotalCost": cost, "AvgPerf%": perf, "n_tasks": "180"}
        for name, cost, perf in _N180_ROWS
    ]


def _ctx(tmp_path: Path, rows: list[dict[str, str]]) -> ctxmod.RoutingContext:
    """The minimal RoutingContext live_gap.render actually reads (rows + write paths)."""
    return ctxmod.RoutingContext(
        out_dir=tmp_path,
        manifest=tmp_path / "figures.json",
        matrix={},
        completed={},
        imputed=None,
        tasks=[],
        rows=rows,
        raw=None,
        models_by_price=[],
        banner=None,
        by_strategy={},
        digest="stub-digest-for-tests",
    )


def _rendered_row(tmp_path: Path) -> dict:
    """Render live_gap on the committed n=180 fixture and return its figures.json row."""
    live_gap.render(_ctx(tmp_path, _n180_rows()))
    payload = json.loads((tmp_path / "figures.json").read_text())
    return payload["figures"]["live_gap.png"]


class TestCheapestPricesOnlyAClassThatExists:
    """The absent-class sentinel is gone; every caller gets None and handles it."""

    def test_empty_band_returns_none_not_zero(self) -> None:
        assert live_gap._cheapest([], StrategyClass.LIVE) is None

    def test_absent_class_returns_none_not_zero(self) -> None:
        band = [("Oracle", StrategyClass.BOUND, 15.18, 96.67)]
        assert live_gap._cheapest(band, StrategyClass.LIVE) is None
        assert live_gap._cheapest(band, StrategyClass.BLOCKED) is None

    def test_present_class_still_returns_its_cheapest_price(self) -> None:
        band = [
            ("Oracle", StrategyClass.BOUND, 15.18, 96.67),
            ("kNN-cascade", StrategyClass.BLOCKED, 24.95, 96.67),
            ("Price-Cascade", StrategyClass.BLOCKED, 22.07, 96.67),
        ]
        assert live_gap._cheapest(band, StrategyClass.BOUND) == 15.18
        assert live_gap._cheapest(band, StrategyClass.BLOCKED) == 22.07

    def test_no_zero_sentinel_remains_in_the_module(self) -> None:
        import inspect

        src = inspect.getsource(live_gap)
        assert "if costs else 0.0" not in src


class TestEmptyLiveIsStatedInWords:
    """With zero in-band live rows the figure says so and prints no live price."""

    def test_subtitle_has_no_zero_price_for_live(self, tmp_path: Path) -> None:
        row = _rendered_row(tmp_path)
        assert "$0.00" not in row["subtitle"]
        assert "cheapest live $0.00" not in row["subtitle"]

    def test_subtitle_states_the_empty_live_band_explicitly(self, tmp_path: Path) -> None:
        row = _rendered_row(tmp_path)
        assert "no live strategy reaches this band" in row["subtitle"]


class TestCaveatIsNeverDroppedByAMissingClass:
    """The caveat survives a missing class — the current code drops it."""

    def test_caveat_present_on_the_zero_live_fixture(self, tmp_path: Path) -> None:
        # The committed figure ships caveat=null. The honest statement — that NO strategy
        # is buyable at this quality — must appear as the red line, never be silently
        # dropped because live happened to be absent.
        row = _rendered_row(tmp_path)
        assert row["caveat"] is not None
        assert "no live strategy reaches this band" in row["caveat"]


class TestBandMembershipComesFromTheData:
    """The figure's band count is computed from the rows, not hardcoded."""

    def test_reports_4_of_8_on_the_committed_n180_corpus(self, tmp_path: Path) -> None:
        row = _rendered_row(tmp_path)
        assert row["n"] == {"in_band": 4, "strategies": 8}

    def test_the_count_is_derived_not_a_literal(self, tmp_path: Path) -> None:
        # Re-derive the band the figure would compute, so the assertion is against the
        # data's own shape rather than a hardcoded "4 of 8" string.
        rows = live_gap._rows(_ctx(tmp_path, _n180_rows()))
        _ceiling, band = live_gap._at_bound_quality(rows)
        in_band = [r for r in rows if r[3] >= 96.67 - 1.0]
        assert len(band) == len(in_band) == 4
        assert any(cls is StrategyClass.LIVE for _n, cls, _c, _p in rows)
        assert all(cls is not StrategyClass.LIVE for _n, cls, _c, _p in band)
        row = _rendered_row(tmp_path)
        assert f"{len(band)} of {len(rows)}" in row["subtitle"]
