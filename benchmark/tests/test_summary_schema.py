# A header whose meaning changed without its name — `Pareto` carrying the old raw-cost
# semantic under an unchanged name — is the Hyrum failure this file pins shut.
"""The strategy_summary schema is an interface: declared columns and the emitted CSV
header must agree in BOTH directions, and a published column's semantic must equal a
re-derivation from the current cost model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark import config
from benchmark.routing.cache_cost import MEASURED, CachePrice
from benchmark.routing.metrics import compute_pareto
from benchmark.routing.strategies import Strategy

_REPORT_CSV = Path("benchmark/routing/reports/strategy_summary.csv")


def _price(model: str, share: float, discount: float) -> CachePrice:
    return CachePrice(
        model=model,
        input_share=share,
        discount=discount,
        hit_rate=1.0,
        provenance=MEASURED,
        share_provenance=MEASURED,
    )


class _Repeater(Strategy):
    """Re-serves the SAME model on consecutive attempts — the exact shape that makes
    cache-aware cost part company with the naive sum, so the frontier CAN move."""

    name = "Repeater"

    def __init__(self, model: str, cost: float):
        self._model = model
        self._cost = cost

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        self.cascade_total_cost = 2 * self._cost
        self.cascade_attempts = [(self._model, self._cost), (self._model, self._cost)]
        self.cascade_scorable = True
        return self._model


class _CheapShot(Strategy):
    """One single-shot attempt per task — naive and cache-aware costs are identical."""

    name = "CheapShot"

    def __init__(self, model: str, cost: float):
        self._model = model
        self._cost = cost

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        return self._model


def _matrix() -> dict:
    """Two tasks; every strategy passes every task, so the frontier is a pure cost race."""
    return {
        "models": {
            "m0": {"input_price": 1.0, "output_price": 1.0},
            "m1": {"input_price": 1.0, "output_price": 1.0},
        },
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {"m0": {"pass": True, "cost": 3.0}, "m1": {"pass": True, "cost": 5.0}},
            "t2": {"m0": {"pass": True, "cost": 3.0}, "m1": {"pass": True, "cost": 5.0}},
        },
    }


def _rows(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """The fixture rows, scored through the real compute_strategy_rows pipeline."""
    config.load("benchmark/benchmark.yaml")
    monkeypatch.setattr(config, "impute_config", lambda: {"enabled": False})
    # Deterministic cache economics: a repeat of m0 banks half its cost, so Repeater's
    # cache-aware total (9.0) drops BELOW CheapShot's (10.0) while its naive total (12.0)
    # stays above — the two frontiers are different sets on this fixture.
    monkeypatch.setattr(
        "benchmark.routing.summary.cache_prices",
        lambda models, shares=None: {"m0": _price("m0", 1.0, 0.5), "m1": _price("m1", 1.0, 0.5)},
    )
    from benchmark.routing.summary import compute_strategy_rows

    return compute_strategy_rows(
        _matrix(), ["t1", "t2"], [_Repeater("m0", 3.0), _CheapShot("m1", 5.0)], bootstrap=50, seed=1
    )


def _frontier(rows: list[dict], cost_field: str) -> set[str]:
    """Re-derive the Pareto frontier over ONE cost column with the same rules the writer
    uses — the independent check that pins the column's semantic to its declared model."""
    metrics = {
        r["strategy"]: {
            "AvgPerf%": float(r["AvgPerf%"]),
            "TotalCost": float(r[cost_field]),
        }
        for r in rows
        if int(r.get("n_tasks", 0) or 0) > 0
    }
    return {name for name, on_pareto in compute_pareto(metrics).items() if on_pareto}


class TestDeclaredAndEmittedSchemaAgree:
    """The writer's declared schema and the emitted header agree exactly, in both directions."""

    def test_emitted_header_matches_the_declared_fields_exactly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmark.admissibility import admissibility_verdict
        from benchmark.routing.summary import SUMMARY_FIELDS, StrategyTable, write_summary_csv

        rows = _rows(monkeypatch)
        table = StrategyTable(
            admissibility=admissibility_verdict(0.51, 0.50, chance_level=0.5, chance_band=0.1),
            rows=tuple(rows),
        )
        out = tmp_path / "strategy_summary.csv"
        write_summary_csv(table, out)
        readback = list(csv.reader(out.open(newline="")))
        # Order matters AND the sets must be exactly equal — a header with one extra
        # column, one missing, or one renamed all fail here.
        assert readback[0] == list(SUMMARY_FIELDS)
        assert set(readback[0]) == set(SUMMARY_FIELDS)
        # A short write would emit trailing columns with nothing behind them.
        assert all(len(row) == len(readback[0]) for row in readback[1:])

    def test_on_disk_summary_agrees_with_the_declared_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmark.routing.summary import SUMMARY_FIELDS

        if not _REPORT_CSV.exists():
            pytest.skip("regenerated report CSV is gitignored; run the routing report to create it")
        readback = list(csv.reader(_REPORT_CSV.open(newline="")))
        assert readback[0] == list(SUMMARY_FIELDS)
        assert all(len(row) == len(readback[0]) for row in readback[1:])


class TestParetoCarriesItsDeclaredSemantic:
    """`Pareto` is decided on CACHE-AWARE cost, not the old raw sum, and a test
    re-derives it from the CURRENT cost model to say so."""

    def test_pareto_matches_a_rederivation_from_the_current_cost_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _rows(monkeypatch)
        # `Pareto` must equal the frontier over the cache-aware column; `Pareto_naive`
        # over the raw column. If the writer regressed `Pareto` to the raw semantic, the
        # cache-aware re-derivation would no longer match.
        assert {r["strategy"] for r in rows if r["Pareto"]} == _frontier(
            rows, "TotalCost_cacheaware"
        )
        assert {r["strategy"] for r in rows if r["Pareto_naive"]} == _frontier(rows, "TotalCost")

    def test_the_fixture_discriminates_the_two_cost_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The two frontiers must DIFFER on this fixture, or the test above would pass even
        # with the old raw-cost semantic — a vacuous lock is no lock at all.
        rows = _rows(monkeypatch)
        cache_aware = _frontier(rows, "TotalCost_cacheaware")
        naive = _frontier(rows, "TotalCost")
        assert cache_aware != naive
        by_name = {r["strategy"]: r for r in rows}
        assert by_name["Repeater"]["Pareto"] is True
        assert by_name["Repeater"]["Pareto_naive"] is False
        assert by_name["CheapShot"]["Pareto"] is False
        assert by_name["CheapShot"]["Pareto_naive"] is True


class TestPerModelNoteRowsDeriveFromTheResultsTable:
    """The per-strategy note rows quote the summary, never a hand-written draft."""

    def test_note_rows_are_a_deterministic_function_of_the_summary_rows(self) -> None:
        # The derivation must be MECHANICAL: given the summary columns, the note row is
        # fully determined — cost (cache-aware and naive), pass rate, task count and the
        # strategy's class all flow from the row, so nothing hand-written can slip in.
        from benchmark.routing.figures.cost_quality_frontier import per_strategy_note_rows

        rows = [
            {
                "strategy": "kNN",
                "AvgPerf%": "78.89",
                "TotalCost_cacheaware": "11.6115",
                "TotalCost": "11.6115",
                "n_tasks": "180",
            },
            {
                "strategy": "Always-Frontier",
                "AvgPerf%": "95.00",
                "TotalCost_cacheaware": "91.1513",
                "TotalCost": "91.1513",
                "n_tasks": "180",
            },
            {
                "strategy": "Price-Cascade",
                "AvgPerf%": "96.67",
                "TotalCost_cacheaware": "22.0735",
                "TotalCost": "22.0735",
                "n_tasks": "180",
            },
        ]
        assert per_strategy_note_rows(rows) == [
            "kNN: \\$11.61 cache-aware / \\$11.61 naive, 78.89% (n=180, live, on the frontier)",
            "Always-Frontier: \\$91.15 cache-aware / \\$91.15 naive, 95.00% "
            "(n=180, live, on the frontier)",
            "Price-Cascade: \\$22.07 cache-aware / \\$22.07 naive, 96.67% "
            "(n=180, blocked — no router.strategy names it)",
        ]

    def _manifest_notes(self) -> list[str]:
        manifest = json.loads(Path("benchmark/routing/figures.json").read_text())
        return list(manifest["figures"]["cost_quality_frontier.png"].get("notes", []))

    def test_note_rows_derive_from_the_summary_rows(self) -> None:
        from benchmark.pipeline import stale_figures
        from benchmark.routing.figures.cost_quality_frontier import per_strategy_note_rows

        if not _REPORT_CSV.exists():
            pytest.skip("regenerated report CSV is gitignored; run the routing report to create it")
        # The manifest's note rows can only be re-derived from the data the figures were
        # DRAWN FROM. When the corpus or the strategy code has moved since the last report
        # run, the committed notes are stale by definition — the freshness gate
        # (`pipeline.stale_figures`) owns that verdict and this check defers to it,
        # rather than re-reporting every data change as a note-row defect.
        if "report" in stale_figures():
            pytest.skip(
                "routing figures are stale vs the current corpus; the report run re-derives "
                "their note rows, and this check re-asserts them after it"
            )
        rows = list(csv.DictReader(_REPORT_CSV.open(newline="")))
        derived = per_strategy_note_rows(rows)
        actual = self._manifest_notes()
        # Every derived row is quoted verbatim in the manifest...
        assert set(derived) <= set(actual)
        # ...and every per-strategy note the manifest carries IS derivable: a hand-written
        # row that cannot be re-derived from the results table must be DELETED, not
        # re-asserted, so the two sets are exactly equal.
        per_strategy_actual = {
            n
            for n in actual
            if "cache-aware /" in n and n.split(":", 1)[0] in {r["strategy"] for r in rows}
        }
        assert per_strategy_actual == set(derived)
