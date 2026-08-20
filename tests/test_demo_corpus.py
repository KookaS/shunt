"""The illustrative demo corpus: determinism, its demo fence, and the figures it renders."""

# Every assertion here is about the GENERATOR, never about the router: the corpus is a
# resampling of 40 measured live rows and is evidence of nothing. The F7 test is the one that
# matters most — a demo that depicted an identified OPE leg would advertise an unshipped
# randomization, so the refusal is pinned rather than tolerated.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from benchmark.routing import demo_corpus
from shunt.db.store import OutcomeStore
from shunt.inspect.inference import _committed_home, render
from shunt.inspect.inference.data import AMBIGUOUS, LIVE, read_sessions


def _dump_sessions(db: Path) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(line for line in con.iterdump() if '"sessions"' in line)
    finally:
        con.close()


def _rows(db: Path) -> list[dict[str, object]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM sessions")]
    finally:
        con.close()


@pytest.fixture(scope="session")
def demo_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One build of the default corpus, shared: a build fsyncs ~600 times and costs ~40 s."""
    return demo_corpus.build_demo_store(tmp_path_factory.mktemp("demo") / "outcomes.db")


@pytest.fixture(scope="session")
def demo_rows(demo_db: Path) -> list[dict[str, object]]:
    return _rows(demo_db)


def test_same_seed_is_byte_identical(demo_db: Path, tmp_path: Path) -> None:
    twin = demo_corpus.build_demo_store(tmp_path / "twin" / "outcomes.db")
    assert _dump_sessions(twin) == _dump_sessions(demo_db)


def test_a_different_seed_is_a_different_corpus(demo_db: Path, tmp_path: Path) -> None:
    other = demo_corpus.build_demo_store(
        tmp_path / "other" / "outcomes.db", seed=demo_corpus.DEMO_SEED + 1
    )
    assert _dump_sessions(other) != _dump_sessions(demo_db)


def test_every_row_carries_the_demo_fence(demo_rows: list[dict[str, object]]) -> None:
    assert demo_rows, "the generator wrote nothing"
    assert all(str(r["session_id"]).startswith(demo_corpus.DEMO_PREFIX) for r in demo_rows)
    assert not any(str(r["session_id"]).startswith("bench:") for r in demo_rows)
    assert all(
        json.loads(str(r["decision_provenance"]))["synthetic_demo"] is True for r in demo_rows
    )


def test_the_corpus_classifies_as_live(demo_db: Path) -> None:
    store = OutcomeStore(db_path=str(demo_db))
    try:
        strata = {row.stratum for row in read_sessions(store)}
    finally:
        store.close()
    assert strata == {LIVE}
    assert AMBIGUOUS not in strata


def test_both_ope_legs_refuse(demo_db: Path) -> None:
    from shunt.inspect.inference import data as idata
    from shunt.inspect.inference import estimators

    store = OutcomeStore(db_path=str(demo_db))
    try:
        legs = idata.ope(store, estimators.certify(store))
    finally:
        store.close()
    assert not legs.routing.identified
    assert not any(leg.identified for leg in legs.escalation)


def test_the_measured_sparsity_survives(demo_db: Path, demo_rows: list[dict[str, object]]) -> None:
    n = len(demo_rows)
    store = OutcomeStore(db_path=str(demo_db))
    try:
        labeled = sum(1 for row in read_sessions(store) if row.tier2_success is not None)
    finally:
        store.close()
    unknown_cost = [r for r in demo_rows if not r["cost_known"]]
    assert 0.25 <= labeled / n <= 0.45
    assert 0.60 <= sum(1 for r in demo_rows if r["selection_propensity"] is None) / n <= 0.80
    assert 0.40 <= sum(1 for r in demo_rows if r["model_fingerprint"] is None) / n <= 0.65
    assert unknown_cost, "cost_known=0 rows vanished"
    assert any(float(str(r["cost"])) > 0.0 for r in unknown_cost), "unknown cost collapsed to zero"


def test_the_seven_figures_render_to_a_scratch_dir(demo_db: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "figures" / "demo"
    store = OutcomeStore(db_path=str(demo_db))
    try:
        report = render(store, out_dir)
    finally:
        store.close()
    assert report.inadmissible is None, report.inadmissible
    assert len(report.figures) == 7
    assert all(path.stat().st_size > 10_000 for path in report.figures)
    assert _committed_home(out_dir) is False


def test_origin_mix_is_the_seeded_share_not_the_distance(demo_db: Path) -> None:
    # Regression: `_origin_probes` yields (sid, seeded_share, k_found, mean_distance) and the
    # caller once bound the LAST field, so F4 panel C plotted distances under a seeded-share
    # axis. The demo corpus holds no `bench:` rows, so every live decision's seeded share must
    # be exactly 0.0 — a distance would land near the neighbour-distance mode instead.
    from shunt.inspect.inference.data import neighbourhood

    store = OutcomeStore(db_path=str(demo_db))
    try:
        view = neighbourhood(store, read_sessions(store))
    finally:
        store.close()
    assert view.origin_mix, "no live decision was probed — the assertion would be vacuous"
    assert all(share == 0.0 for share in view.origin_mix), (
        "a corpus with zero seeded rows must report a seeded share of 0.0 for every live "
        f"decision; got {sorted(set(view.origin_mix))[:5]}"
    )
