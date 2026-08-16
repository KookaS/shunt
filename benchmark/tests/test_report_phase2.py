"""The equal-coverage report outputs and the phantom-machinery removal."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.config import CapabilityRank, ModelEvidence, RankedModel
from benchmark.routing import report
from benchmark.routing.impute import ImputedMatrix, complete_matrix, violation_ci

_ORDER: Final[list[str]] = ["c0", "m0", "h0", "f0"]
# Pass-rates chosen so 1-D gap-splitting yields one model per display cluster (four clusters).
_PASS: Final[dict[str, float]] = {"c0": 0.2, "m0": 0.5, "h0": 0.75, "f0": 0.95}
RANK: Final[CapabilityRank] = CapabilityRank(
    ordered=[RankedModel(m, "a", i, "measured") for i, m in enumerate(_ORDER)],
    evidence={
        m: ModelEvidence(m, 40, _PASS[m], _PASS[m] - 0.05, _PASS[m] + 0.05, i, "measured", float(i))
        for i, m in enumerate(_ORDER)
    },
)
# The ordinal bands RANK produces (weakest -> strongest), for the coverage-row tests.
BANDS: Final[dict[str, int]] = report.capability_bands(RANK)
BAND_ORDER: Final[list[int]] = report._band_order(BANDS)


def _cell(passed: bool, cost: float) -> dict:
    return {"pass": passed, "cost": cost, "real_cost": cost}


def _im() -> ImputedMatrix:
    matrix = {
        "e": {"c0": _cell(True, 0.01)},  # tau = c0
        "m": {"c0": _cell(False, 0.01), "m0": _cell(True, 0.20)},  # tau = m0
        "v": {"c0": _cell(True, 0.02), "m0": _cell(False, 0.50)},  # violation
        "d": {  # unsolvable
            "c0": _cell(False, 0.01),
            "m0": _cell(False, 0.20),
            "h0": _cell(False, 0.30),
            "f0": _cell(False, 0.40),
        },
    }
    return complete_matrix(matrix, RANK)


# ---------------------------------------------------------------- violation rate


def test_violation_ci_and_n_multi_observed() -> None:
    im = _im()
    assert im.n_multi_observed == 3  # m, v, d each have >=2 observed rungs; e has 1
    assert len(im.violations) == 1
    v, lo, hi = violation_ci(len(im.violations), im.n_multi_observed)
    assert v == pytest.approx(1 / 3)
    assert 0.0 <= lo <= v <= hi <= 1.0


def test_violation_line_reports_rate() -> None:
    line = report._violation_line(_im())
    assert "0.333" in line and "multi-observed" in line


# ------------------------------------------------------------------ coverage table


def _use_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the live capability rank to the synthetic RANK for report-time cluster derivation."""
    monkeypatch.setattr(config, "capability_rank", lambda *a, **k: RANK)
    monkeypatch.setattr(config, "frontier_model", lambda: "f0")


def test_coverage_rows_split_real_imputed_unknown() -> None:
    rows = report.coverage_rows(_im(), BANDS, BAND_ORDER)
    by_band = {r["band"]: r for r in rows}
    # Four bands, weakest -> strongest, one model each at this disjoint-CI spread.
    assert [r["band"] for r in rows] == BAND_ORDER
    assert BAND_ORDER == [1, 2, 3, 4]  # ordinal, band 4 is the strongest
    # 4 tasks, one model per band -> every band's real+imputed+unknown sums to 4.
    for r in rows:
        assert r["real"] + r["imputed"] + r["unknown"] == 4
    assert by_band[4]["imputed"] > 0  # the strongest column rests on imputation


def test_write_coverage_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_rank(monkeypatch)
    path = report.write_coverage_table(_im(), tmp_path)
    assert path.exists()
    assert "band,real,imputed,unknown" in path.read_text().splitlines()[0]


def test_write_capability_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    _use_rank(monkeypatch)
    path = report.write_capability_evidence(tmp_path)
    assert path.name == "capability_evidence.json"
    payload = json.loads(path.read_text())
    assert payload["control_model"] == "f0"
    assert [m["model"] for m in payload["models"]] == _ORDER  # weakest -> strongest
    assert payload["models"][-1]["band"] == 4  # ordinal strongest band
    assert all(m["source"] == "measured" for m in payload["models"])
    assert payload["generated_from"].startswith("results.csv@")
    # Ordinal bands, described ONLY by metadata — no semantic names.
    bands = payload["bands"]
    assert [b["band"] for b in bands] == [1, 2, 3, 4]
    assert all(b["n_models"] == 1 for b in bands)
    assert bands[0]["models"] == ["c0"] and bands[-1]["models"] == ["f0"]
    assert bands[0]["price_range"] == [0.0, 0.0]
    assert "pass_rate_min" in bands[0]["capability_range"]
    for banned in ("budget", "standard", "premium", "frontier"):
        assert banned not in path.read_text()


# --------------------------------------------------------------- disclosure banner


def test_disclosure_banner_equal_coverage_wording(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_rank(monkeypatch)
    equal_rows = [{"strategy": "a", "n_tasks": 4}, {"strategy": "b", "n_tasks": 4}]
    banner = report._disclosure_banner(_im(), equal_rows)
    assert banner is not None
    assert "equal-coverage via monotone imputation" in banner
    # The banner must NOT assert a direction imputation was never shown to have.
    assert "widens the router" not in banner
    assert "NEARLY every imputed cell is filled pass=True" in banner


def test_disclosure_banner_downgrades_when_unequal(monkeypatch: pytest.MonkeyPatch) -> None:
    # FIX B: unequal per-strategy n_tasks -> 'coverage-completed', names the residual band.
    _use_rank(monkeypatch)
    unequal_rows = [
        {"strategy": "Always-Frontier", "n_tasks": 142},
        {"strategy": "b", "n_tasks": 185},
    ]
    banner = report._disclosure_banner(_im(), unequal_rows)
    assert banner is not None
    assert "coverage-completed via monotone imputation" in banner
    assert "equal-coverage via monotone imputation" not in banner
    assert "per-strategy n=142–185" in banner
    assert "ladder collection" in banner


def test_disclosure_banner_none_when_disabled() -> None:
    assert report._disclosure_banner(None, []) is None


# ------------------------------------------------------------------- ordinal bands


def test_capability_bands_data_driven_ordinal() -> None:
    bands = report.capability_bands(RANK)
    # Disjoint CIs -> one band per model, numbered weakest (1) to strongest (4).
    assert bands == {"c0": 1, "m0": 2, "h0": 3, "f0": 4}


def test_capability_bands_merge_overlapping_cis() -> None:
    # Adjacent models with overlapping capability CIs collapse into ONE band; the
    # count is data-driven (CI overlap), never a fixed cap.
    order = ["a", "b", "c"]
    rate = {"a": 0.30, "b": 0.55, "c": 0.60}
    ci = {"a": (0.20, 0.40), "b": (0.45, 0.65), "c": (0.50, 0.70)}  # b,c overlap
    rank = CapabilityRank(
        ordered=[RankedModel(m, "x", i, "measured") for i, m in enumerate(order)],
        evidence={
            m: ModelEvidence(m, 40, rate[m], ci[m][0], ci[m][1], i, "measured", float(i))
            for i, m in enumerate(order)
        },
    )
    bands = report.capability_bands(rank)
    assert bands == {"a": 1, "b": 2, "c": 2}  # a distinct; b,c merge on CI overlap
    meta = report.band_metadata(rank, bands)
    assert meta[2]["models"] == ["b", "c"] and meta[2]["n_models"] == 2
    assert meta[2]["price_range"] == [1.0, 2.0]
    assert meta[2]["capability_range"]["pass_rate_min"] == 0.55
    assert meta[2]["capability_range"]["pass_rate_max"] == 0.60


def test_capability_bands_do_not_chain_through_disjoint_cis() -> None:
    # a-b overlap and b-c overlap, but a and c are DISJOINT. The old chained rule put
    # all three in one band while the legend promised "CIs overlap"; pairwise cuts.
    order = ["a", "b", "c"]
    ci = {"a": (0.10, 0.40), "b": (0.35, 0.60), "c": (0.55, 0.90)}
    rank = CapabilityRank(
        ordered=[RankedModel(m, "x", i, "measured") for i, m in enumerate(order)],
        evidence={
            m: ModelEvidence(m, 40, sum(ci[m]) / 2, ci[m][0], ci[m][1], i, "measured", float(i))
            for i, m in enumerate(order)
        },
    )
    bands = report.capability_bands(rank)
    assert bands["a"] != bands["c"]
    lo = max(rank.evidence[m].ci_lo for m in order if bands[m] == bands["a"])
    hi = min(rank.evidence[m].ci_hi for m in order if bands[m] == bands["a"])
    assert lo <= hi  # every member of a band overlaps every other member


def test_band_metadata_pct_tasks_from_im() -> None:
    meta = report.band_metadata(RANK, report.capability_bands(RANK), _im())
    # pct_tasks_min_solved is populated from the imputed matrix's τ distribution.
    total = sum(b["pct_tasks_min_solved"] for b in meta.values())
    assert 0.0 < total <= 1.0
    assert meta[1]["pct_tasks_min_solved"] > 0.0  # band 1 (c0) is τ for the easy tasks


# ------------------------------------------------------- phantom machinery removed


def test_no_phantom_baseline_machinery_remains() -> None:
    src = inspect.getsource(report)
    assert "PHANTOM BASELINE" not in src
    assert "phantom baseline" not in src
    assert not hasattr(report, "_frontier_coverage")
