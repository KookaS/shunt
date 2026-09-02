"""The repriced-cost contract: cheapest-today, naive-only, and MISSING stays missing."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from benchmark.routing import repricing
from benchmark.routing.scripts import refresh_price_sheet


def _quote(channel: str, listing: str, inp: float, out: float) -> dict[str, Any]:
    return {
        "channel": channel,
        "id": listing,
        "input_cost_per_1m": inp,
        "output_cost_per_1m": out,
        "source": f"https://{channel}.example/models",
        "as_of": "2026-08-28",
    }


@pytest.fixture
def sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A two-model sheet: one priced, one hosted nowhere."""
    path = tmp_path / "price_sheet.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "as_of": "2026-08-28",
                "models": {
                    "cheap": {
                        "canonical": _quote("openrouter", "v/cheap", 1.0, 2.0),
                        "quotes": [_quote("openrouter", "v/cheap", 1.0, 2.0)],
                        "unresolved": [],
                    },
                    "nowhere": {"canonical": None, "quotes": [], "unresolved": []},
                },
            }
        )
    )
    monkeypatch.setattr(repricing, "SHEET_PATH", path)
    repricing.load_sheet.cache_clear()
    yield path
    repricing.load_sheet.cache_clear()


class TestRepricedCost:
    def test_priced_model_reprices_naively(self, sheet: Path) -> None:
        assert repricing.naive_cost("cheap", 2_000_000, 1_000_000) == pytest.approx(4.0)

    def test_a_model_the_sheet_does_not_price_has_no_repriced_cost(self, sheet: Path) -> None:
        """MISSING, never 0.0 — a $0 cost is an affirmative claim this data cannot make."""
        assert repricing.naive_cost("nowhere", 1_000_000, 1_000_000) is None
        assert repricing.naive_cost("never-heard-of-it", 1_000_000, 1_000_000) is None

    def test_row_repricing_reads_the_rows_own_tokens(self, sheet: Path) -> None:
        row = {"model": "cheap", "in_tok": "1000000", "out_tok": "500000"}
        assert repricing.row_naive_cost(row) == pytest.approx(2.0)

    def test_a_total_is_all_or_nothing(self, sheet: Path) -> None:
        """One unpriced row refuses the whole total rather than quietly shrinking it."""
        priced = {"model": "cheap", "in_tok": 1_000_000, "out_tok": 0}
        unpriced = {"model": "nowhere", "in_tok": 1_000_000, "out_tok": 0}
        assert repricing.total_naive_cost([priced, priced], "test") == pytest.approx(2.0)
        assert repricing.total_naive_cost([priced, unpriced], "test") is None

    def test_an_absent_sheet_prices_nothing_and_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(repricing, "SHEET_PATH", tmp_path / "absent.json")
        repricing.load_sheet.cache_clear()
        try:
            assert repricing.naive_cost("cheap", 1, 1) is None
            assert repricing.provenance_stamp() is None
            assert repricing.axis_basis(repriced=False).startswith("cost as billed")
        finally:
            repricing.load_sheet.cache_clear()

    def test_the_axis_basis_dates_the_prices_that_drew_it(self, sheet: Path) -> None:
        assert repricing.axis_basis(repriced=True).endswith("2026-08-28")
        assert repricing.axis_basis(repriced=False) == "cost as billed when each run happened"

    def test_provenance_stamps_the_sheets_identity(self, sheet: Path) -> None:
        stamp = repricing.provenance_stamp()
        assert stamp is not None
        assert stamp["as_of"] == "2026-08-28"
        assert stamp["digest"] == repricing.sheet_digest(sheet)


class TestSheetAssembly:
    """The refresh writes only what it fetched, and resolves ties deterministically."""

    _MAP = {"m": {"openrouter": ["a"], "requesty": ["b", "gone"]}}

    def test_canonical_is_the_cheapest_total_price(self) -> None:
        prices = {"openrouter": {"a": (5.0, 5.0)}, "requesty": {"b": (1.0, 2.0)}}
        sheet = refresh_price_sheet.build_sheet(self._MAP, prices, {}, "2026-08-28")
        canonical = sheet["models"]["m"]["canonical"]
        assert (canonical["channel"], canonical["id"]) == ("requesty", "b")

    def test_a_listing_the_catalogue_lost_is_unresolved_not_invented(self) -> None:
        prices = {"openrouter": {"a": (5.0, 5.0)}, "requesty": {"b": (1.0, 2.0)}}
        sheet = refresh_price_sheet.build_sheet(self._MAP, prices, {}, "2026-08-28")
        assert sheet["models"]["m"]["unresolved"] == ["requesty:gone"]
        assert [q["id"] for q in sheet["models"]["m"]["quotes"]] == ["b", "a"]

    def test_a_model_no_channel_prices_gets_no_canonical(self) -> None:
        sheet = refresh_price_sheet.build_sheet(self._MAP, {}, {}, "2026-08-28")
        assert sheet["models"]["m"]["canonical"] is None
        assert sheet["models"]["m"]["quotes"] == []

    def test_equal_totals_break_on_input_price_then_listing(self) -> None:
        prices = {"openrouter": {"a": (1.0, 5.0)}, "requesty": {"b": (2.0, 4.0)}}
        sheet = refresh_price_sheet.build_sheet(self._MAP, prices, {}, "2026-08-28")
        assert sheet["models"]["m"]["canonical"]["id"] == "a"


class TestCommittedSheet:
    def test_every_committed_quote_names_a_real_source_and_date(self) -> None:
        sheet = json.loads(repricing.SHEET_PATH.read_text())
        for model, entry in sheet["models"].items():
            for quote in entry["quotes"]:
                assert quote["source"].startswith("https://"), model
                assert quote["as_of"] == sheet["as_of"], model
                assert quote["input_cost_per_1m"] > 0, model

    def test_the_sheet_prices_only_models_the_channel_map_declares(self) -> None:
        import yaml

        declared = set(yaml.safe_load(repricing.CHANNELS_PATH.read_text())["models"])
        assert set(json.loads(repricing.SHEET_PATH.read_text())["models"]) == declared
