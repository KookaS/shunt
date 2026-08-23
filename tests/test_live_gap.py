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

# The committed n=180 corpus: 8 strategies, 4 inside the 1pp band. Session-Cascade is one
# of the four and is now LIVE (`router.strategy: session_cascade`), so this corpus no longer
# demonstrates an empty live class — see `_ZERO_LIVE_ROWS` below for the fixture that still
# does. Always-Frontier fell to 95.00% and dropped out; kNN and Always-Cheap sit far below.
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


# The empty-live-band case, constructed deliberately rather than borrowed from whichever
# corpus happened to have no live row in the band. The invariant under test — an absent class
# is stated in words, never priced at $0.00 — is about how the figure HANDLES an empty class,
# so it must not evaporate the moment a strategy is promoted to live. Session-Cascade is
# dropped because promoting it is exactly what emptied this case out of the real corpus.
_ZERO_LIVE_ROWS = tuple(r for r in _N180_ROWS if r[0] != "Session-Cascade")


def _rows_of(spec: tuple[tuple[str, str, str], ...]) -> list[dict[str, str]]:
    """Summary-row dicts for a fixture spec, freshly built per test."""
    return [
        {"strategy": name, "TotalCost": cost, "AvgPerf%": perf, "n_tasks": "180"}
        for name, cost, perf in spec
    ]


def _n180_rows() -> list[dict[str, str]]:
    """The committed n=180 corpus as summary-row dicts, freshly built per test."""
    return _rows_of(_N180_ROWS)


def _zero_live_rows() -> list[dict[str, str]]:
    """The same corpus with no live row inside the quality band."""
    return _rows_of(_ZERO_LIVE_ROWS)


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


def _rendered_row(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> dict:
    """Render live_gap on a fixture (default: the committed n=180 corpus) and return its row."""
    live_gap.render(_ctx(tmp_path, _n180_rows() if rows is None else rows))
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
        row = _rendered_row(tmp_path, _zero_live_rows())
        assert "$0.00" not in row["subtitle"]
        assert "cheapest live $0.00" not in row["subtitle"]

    def test_subtitle_states_the_empty_live_band_explicitly(self, tmp_path: Path) -> None:
        row = _rendered_row(tmp_path, _zero_live_rows())
        assert "no live strategy reaches this band" in row["subtitle"]

    def test_a_live_row_in_the_band_is_priced_instead(self, tmp_path: Path) -> None:
        # The other half of the same invariant, and the one the real corpus now exercises:
        # when the band DOES hold a live row the figure must price it rather than keep
        # printing the unbuyable-headroom line, which would be false.
        row = _rendered_row(tmp_path)
        assert "no live strategy reaches this band" not in row["subtitle"]


class TestCaveatIsNeverDroppedByAMissingClass:
    """The caveat survives a missing class — the current code drops it."""

    def test_caveat_present_on_the_zero_live_fixture(self, tmp_path: Path) -> None:
        # The honest statement — that NO strategy is buyable at this quality — must appear
        # as the red line, never be silently dropped because live happened to be absent.
        row = _rendered_row(tmp_path, _zero_live_rows())
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
        # Session-Cascade is in the band AND live, so the headroom at this quality is now
        # buyable. This assertion is the inverse of the one it replaces, deliberately: it is
        # the figure's whole claim, and it must be re-derived from the data, never restated.
        assert any(cls is StrategyClass.LIVE for _n, cls, _c, _p in band)
        row = _rendered_row(tmp_path)
        assert f"{len(band)} of {len(rows)}" in row["subtitle"]
