"""The illustrative demo corpus: determinism, its demo fence, and the figures it renders."""

# Every assertion here is about the GENERATOR, never about the router: the corpus is a
# resampling of 40 measured live rows and is evidence of nothing. The F7 test is the one that
# matters most — a demo that depicted an identified OPE leg would advertise an unshipped
# randomization, so the refusal is pinned rather than tolerated.

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from benchmark.routing import demo_corpus
from shunt.db.store import OutcomeStore
from shunt.inspect.inference import INFERENCE, render
from shunt.inspect.inference.data import (
    AMBIGUOUS,
    LIVE,
    SEEDED,
    SessionRow,
    escalation,
    neighbourhood,
    policy,
    read_sessions,
)
from shunt.inspect.inference.specs import HOLD_TOKENS


def _dump_sessions(db: Path) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(line for line in con.iterdump() if '"sessions"' in line)
    finally:
        con.close()


def _dump_content(db: Path) -> str:
    """Every CONTENT row of the store, in dump order — `schema_version` deliberately excluded."""
    # `_dump_sessions` matched `"sessions"` only, so 668 `outcome_events` and 459 `outcomes`
    # rows were outside the reproducibility assertion entirely: nondeterminism in either would
    # have passed green. `schema_version` is excluded because it stamps wall-clock migration
    # times, which is why the db FILES are not byte-identical even when every content row is.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(
            line
            for line in con.iterdump()
            if line.startswith("INSERT INTO") and '"schema_version"' not in line
        )
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


@pytest.fixture(scope="session")
def adjudicated(demo_db: Path) -> list[SessionRow]:
    """The corpus as the figure family sees it — one read, shared: a build costs ~100 s."""
    store = OutcomeStore(db_path=str(demo_db))
    try:
        return read_sessions(store)
    finally:
        store.close()


def _drawn(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The 300 rows resampled from `_ATOMS` — the only half the measured marginals describe."""
    invented = demo_corpus.DEMO_INVENTED_PREFIX
    return [
        r
        for r in rows
        if str(r["session_id"]).startswith(demo_corpus.DEMO_PREFIX)
        and not str(r["session_id"]).startswith(invented)
    ]


def test_same_seed_reproduces_every_content_row(demo_db: Path, tmp_path: Path) -> None:
    """Two builds at one seed agree on sessions, outcome_events AND outcomes — not just one."""
    # Renamed off "byte_identical": the db FILES are not byte-identical (schema_version carries
    # wall-clock migration stamps), and the old name invited exactly the wrong conclusion from
    # a green run. What IS reproducible is every content row, which is the property the figures
    # depend on.
    twin = demo_corpus.build_demo_store(tmp_path / "twin" / "outcomes.db")
    assert _dump_content(twin) == _dump_content(demo_db)
    for table in ('"sessions"', '"outcome_events"', '"outcomes"'):
        n = sum(1 for line in _dump_content(demo_db).splitlines() if table in line)
        assert n > 0, f"{table} contributed no rows — the assertion would be vacuous for it"


def test_a_different_seed_is_a_different_corpus(demo_db: Path, tmp_path: Path) -> None:
    other = demo_corpus.build_demo_store(
        tmp_path / "other" / "outcomes.db", seed=demo_corpus.DEMO_SEED + 1
    )
    assert _dump_sessions(other) != _dump_sessions(demo_db)


def test_every_row_carries_the_demo_fence(demo_rows: list[dict[str, object]]) -> None:
    """The id fence, now that it is split: two prefixes, one greppable substring."""
    # The seeded half MUST start with `bench:` or `classify_stratum` cannot adjudicate it SEEDED,
    # so the old "never `bench:`" assertion is replaced rather than kept: what is invariant is
    # that every id this module writes contains `demo`, which is what makes a stray row findable
    # by one grep whichever stratum it belongs to.
    assert demo_rows, "the generator wrote nothing"
    prefixes = (demo_corpus.DEMO_PREFIX, demo_corpus.DEMO_SEED_PREFIX)
    assert all(str(r["session_id"]).startswith(prefixes) for r in demo_rows)
    assert all("demo" in str(r["session_id"]) for r in demo_rows)
    assert all(
        json.loads(str(r["decision_provenance"]))["synthetic_demo"] is True for r in demo_rows
    )


def test_the_corpus_carries_both_strata_and_nothing_ambiguous(
    adjudicated: list[SessionRow],
) -> None:
    """Both halves adjudicate cleanly — an ambiguous row would be a real seeding defect."""
    strata = Counter(row.stratum for row in adjudicated)
    assert strata[LIVE] > 0
    assert strata[SEEDED] == demo_corpus.DEMO_SEEDED_SESSIONS
    assert strata[AMBIGUOUS] == 0, "a row whose prefix and rule dissent is a generator defect"


def test_every_seeded_row_reaches_the_index(demo_db: Path) -> None:
    """A seeded row that never gets indexed is invisible to the live decisions F4 C measures."""
    store = OutcomeStore(db_path=str(demo_db))
    try:
        census = store.stratum_census()
    finally:
        store.close()
    assert census.seeded.stored == demo_corpus.DEMO_SEEDED_SESSIONS
    assert census.seeded.indexed == census.seeded.stored


def test_every_hold_token_but_the_unreachable_one_is_drawn(adjudicated: list[SessionRow]) -> None:
    """F6 panel C was fully empty; `disabled` stays empty because the router cannot emit it."""
    view = escalation(adjudicated, now=demo_corpus.DEMO_NOW)
    holds = dict(view.holds)
    assert set(holds) == set(HOLD_TOKENS)
    assert holds["disabled"] == 0, "`disabled` is unreachable from the serving path (specs.py)"
    drawn = {token: n for token, n in holds.items() if token != "disabled"}
    assert all(n > 0 for n in drawn.values()), drawn
    assert len(set(drawn.values())) == len(drawn), "four identical bars read as a placeholder"
    assert not view.unknown_holds, "a hold outside HOLD_TOKENS is vocabulary drift"


def test_exactly_one_undeliverable_hold_lights_the_derived_bar(
    adjudicated: list[SessionRow],
) -> None:
    """The derived hatched bar is one row: a raise directive the engine could not deliver."""
    assert escalation(adjudicated, now=demo_corpus.DEMO_NOW).n_undeliverable == 1


def test_the_escalated_cohort_is_non_empty_and_worse(adjudicated: list[SessionRow]) -> None:
    """F6 panel D read `escalated 0/35`; two non-zero cohorts, escalation the worse one."""
    # Escalation fires on the hard tail, so the escalated cohort succeeding LESS often is the
    # plausible shape — parity would be the tell that the numbers were picked to look tidy.
    cohorts = {name: (ok, n) for name, ok, n in escalation(adjudicated).outcomes}
    escalated_ok, escalated_n = cohorts["escalated"]
    plain_ok, plain_n = cohorts["not escalated"]
    assert escalated_n > 0 and plain_n > 0
    assert escalated_ok > 0 and plain_ok > 0
    assert escalated_ok / escalated_n < plain_ok / plain_n


def test_the_frontier_arm_is_drawn(adjudicated: list[SessionRow]) -> None:
    """F5 panel B legended a frontier share over a corpus that had no frontier model in it."""
    from shunt.models.config import ModelPool

    view = policy(adjudicated, ModelPool.load())
    assert view.frontier_share, "no rolling series at all"
    assert max(share for _index, share in view.frontier_share) > 0.0


def test_every_live_model_reports_a_propensity(adjudicated: list[SessionRow]) -> None:
    """F5 panel C drew one bar because 28 of the 40 measured atoms carry a NULL propensity."""
    from shunt.models.config import ModelPool

    view = policy(adjudicated, ModelPool.load())
    live_models = {row.model_chosen for row in adjudicated if row.stratum == LIVE}
    assert {model for model, _n, _mean, _low in view.propensities} == live_models
    # 1.0 exactly, never `0 < p < 1`: the shipped router is deterministic, and a fractional
    # propensity here is precisely what would flip F7 out of its refusal.
    assert all(mean == 1.0 and low == 1.0 for _m, _n, mean, low in view.propensities)


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


_SLOW = pytest.mark.skipif(
    os.environ.get("SHUNT_SLOW_TESTS") != "1",
    reason="one ~50 s corpus build per seed; set SHUNT_SLOW_TESTS=1 to run",
)


@_SLOW
@pytest.mark.parametrize("offset", [2, 3, 4])
def test_both_ope_legs_refuse_at_other_seeds(tmp_path: Path, offset: int) -> None:
    # `test_both_ope_legs_refuse` pins the refusal at ONE seed, which proves nothing about the
    # corpora this generator can produce. The property that matters is that NO draw is
    # identified: the shipped router writes a propensity of 1.0 or NULL and never randomizes, so
    # an identified leg here would advertise an epsilon-greedy exploration that does not exist.
    # Swept out-of-band over 8 seeds (DEMO_SEED..DEMO_SEED+7) and session counts 100/300/600 —
    # every leg refused. Three of those seeds are kept here as the standing guard, gated because
    # each costs a full build.
    from shunt.inspect.inference import data as idata
    from shunt.inspect.inference import estimators

    db = demo_corpus.build_demo_store(
        tmp_path / "seed" / "outcomes.db", seed=demo_corpus.DEMO_SEED + offset
    )
    store = OutcomeStore(db_path=str(db))
    try:
        legs = idata.ope(store, estimators.certify(store))
    finally:
        store.close()
    assert not legs.routing.identified
    assert legs.escalation, "an empty escalation leg list would make the assertion vacuous"
    assert not any(leg.identified for leg in legs.escalation)


def test_the_measured_sparsity_survives(
    adjudicated: list[SessionRow], demo_rows: list[dict[str, object]]
) -> None:
    """The measured marginals, over the DRAWN rows only — the half `_ATOMS` describes."""
    # Narrowed from the whole store: the invented and seeded rows are deliberately denser
    # (every seed is labelled, every invented escalation is), so pooling them would test the
    # invented counts rather than the resampler. The propensity COLUMN moved too — it is now
    # 1.0 on every live row by design — so its measured NULL share is asserted on the
    # provenance key `router_propensity`, which the resampler still carries through untouched.
    drawn = _drawn(demo_rows)
    n = len(drawn)
    assert n == demo_corpus.DEMO_SESSIONS
    ids = {str(r["session_id"]) for r in drawn}
    labeled = sum(
        1 for row in adjudicated if row.session_id in ids and row.tier2_success is not None
    )
    logged = [json.loads(str(r["decision_provenance"])).get("router_propensity") for r in drawn]
    unknown_cost = [r for r in drawn if not r["cost_known"]]
    assert 0.25 <= labeled / n <= 0.45
    assert 0.75 <= sum(1 for value in logged if value is not None) / n <= 0.90
    assert 0.40 <= sum(1 for r in drawn if r["model_fingerprint"] is None) / n <= 0.65
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
    assert INFERENCE.manifest_for(out_dir) == out_dir.parent / "figures.json"


def test_origin_mix_is_the_seeded_share_not_the_distance(
    demo_db: Path, adjudicated: list[SessionRow]
) -> None:
    """F4 panel C plots a SHARE, and a share is a count over k — never a distance."""
    # Regression, re-armed for a corpus that now HAS seeded rows: `_origin_probes` yields
    # (sid, seeded_share, k_found, mean_distance) and the caller once bound the LAST field. The
    # old guard was `share == 0.0` for every decision, which a seeded stratum makes vacuous;
    # the invariant that survives is that every value is a multiple of 1/k inside [0, 1], which
    # a mean neighbour distance is not.
    store = OutcomeStore(db_path=str(demo_db))
    try:
        view = neighbourhood(store, adjudicated)
    finally:
        store.close()
    assert view.origin_mix, "no live decision was probed — the assertion would be vacuous"
    assert all(0.0 <= share <= 1.0 for share in view.origin_mix)
    assert all(abs(share * view.k - round(share * view.k)) < 1e-9 for share in view.origin_mix), (
        f"a seeded share is a count over k={view.k}; got {sorted(set(view.origin_mix))[:5]}"
    )


def test_the_seeded_share_is_a_distribution_not_a_spike(
    demo_db: Path, adjudicated: list[SessionRow]
) -> None:
    """F4 panel C was a single spike at 0.0 because no live decision could retrieve a seed."""
    store = OutcomeStore(db_path=str(demo_db))
    try:
        view = neighbourhood(store, adjudicated)
    finally:
        store.close()
    assert len(set(view.origin_mix)) > 1, "a single degenerate value is the defect, not the shape"
    assert max(view.origin_mix) > 0.0, "no live decision retrieved a seeded neighbour"


def test_the_demo_family_stamps_every_canvas_and_diverts_its_manifest(
    demo_db: Path, tmp_path: Path
) -> None:
    """The two fences that keep an illustrative figure from passing as a measured one."""
    # Not "does draw_strata remember the watermark" — no draw function is asked. The family
    # carries the mark, `plot_frame.save` applies it, and SH007 makes `save` the only exit.
    pytest.importorskip("matplotlib")
    from benchmark.demo.render_demo_figures import DEMO
    from shunt.inspect import plot_frame

    ink = tuple(int(plot_frame.CAVEAT_RED[i : i + 2], 16) for i in (1, 3, 5))
    want = tuple(255.0 - plot_frame._WATERMARK_ALPHA * (255 - c) for c in ink)

    out_dir = tmp_path / "figures" / "demo"
    store = OutcomeStore(db_path=str(demo_db))
    try:
        report = render(store, out_dir, family=DEMO)
    finally:
        store.close()

    assert report.manifest != DEMO.manifest, "a scratch render dirtied the committed manifest"
    assert len(report.figures) == 7
    for path in report.figures:
        pixels = set(Image.open(path).convert("RGB").getdata())
        assert any(all(abs(px[i] - want[i]) <= 2 for i in range(3)) for px in pixels), (
            f"{path.name} carries no watermark"
        )


def test_the_frozen_clock_is_load_bearing(adjudicated: list[SessionRow]) -> None:
    """DEMO_NOW must change the windowed panels — otherwise the freeze is decorative."""
    # AUDIT FINDING (2026-08-21): `DEMO_NOW` survived a mutation that made `data._in_window`
    # discard `now` entirely — the whole suite stayed green. `escalation()` windows only its
    # `rates`, the assertions elsewhere read un-windowed fields, and both render tests call
    # `render(store, out_dir)` with no `now` at all. So the one property the constant exists to
    # protect — that the committed 7d/30d panels do not drift with the wall clock — had no test.
    # This pins the 7d window's contents against the frozen clock; if `now` is ignored, the
    # counts move with today's date and this goes red.
    from shunt.inspect.inference import data as idata

    frozen = idata.cost(adjudicated, now=demo_corpus.DEMO_NOW)
    by_label = {label: agg for label, agg in frozen.windows}
    seven, thirty, whole = by_label["7d"], by_label["30d"], by_label["all"]

    assert seven.n_cost_known == 46, "the frozen 7d window moved — is `now` still threaded?"
    assert thirty.n_cost_known == 421
    assert whole.n_cost_known == 425
    # `all` strictly exceeds `30d`: the corpus spans 28.086 d and the 30d cutoff lands at
    # `_ANCHOR - 28 d`, so its oldest ~2 h sits outside. Documented at `demo_corpus.DEMO_NOW`.
    assert whole.n_cost_known > thirty.n_cost_known > seven.n_cost_known

    # And the freeze must actually differ from the wall clock, or the assertion above would
    # hold for the wrong reason on the day the two happen to agree.
    live = idata.cost(adjudicated)
    assert dict((label, agg.n_cost_known) for label, agg in live.windows) != {
        label: agg.n_cost_known for label, agg in frozen.windows
    }
