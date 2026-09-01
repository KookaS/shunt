"""The inference family's transforms and renders, over a store built in the test."""

# Every figure here can be legitimately EMPTY, which makes "it rendered" a weak assertion: an
# empty panel and a broken panel look the same to a smoke test. So the transforms are asserted
# on their numbers, and the renders are asserted under SHUNT_PLOT_STRICT=1 — the layout contract
# is what proves the empty state was drawn rather than skipped.

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from shunt.analysis.admissibility import admissibility_verdict
from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance
from shunt.inspect.inference import data as idata
from shunt.inspect.inference import estimators, figures, specs
from shunt.inspect.plot_style import usd
from shunt.router.engine import _void_exploration
from shunt.router.escalation import EscalationAction, ExplorationRecord

_DIM = 8


def _voided_provenance() -> dict[str, Any]:
    """Exactly what `engine._void_exploration` writes when a rung cannot be delivered."""
    record = ExplorationRecord(
        checkpoint_id="tests::flaky",
        decision_index=0,
        action=EscalationAction.RAISE_RANK,
        policy_action=EscalationAction.RAISE_RANK,
        propensity=0.8,
        epsilon=0.2,
        seed=1,
        randomized=True,
    )
    voided = _void_exploration({"escalation_exploration": record.persistable()})
    # Through JSON exactly as the store column does — the enum survives in-process and does not
    # survive the round trip, which is where a naive detector breaks.
    return cast("dict[str, Any]", json.loads(json.dumps({"selection_rule_used": "knn", **voided})))


@pytest.fixture(autouse=True)
def _strict_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_PLOT_STRICT", "1")


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=_DIM).astype(np.float32)
    return vector / float(np.linalg.norm(vector))


def _store(tmp_path: Path) -> OutcomeStore:
    return OutcomeStore(
        db_path=str(tmp_path / "outcomes.db"), index_path=str(tmp_path / "index.hnsw2")
    )


def _seed_session(  # noqa: PLR0913 (a session row has one arg per column under test)
    store: OutcomeStore,
    session_id: str,
    *,
    model: str,
    when: datetime,
    cost: float = 0.5,
    provenance: dict[str, Any] | None = None,
    outcome: str | None = "success",
    source: str = "benchmark_seed",
    propensity: float | None = None,
    seed: int = 0,
    cost_known: bool = True,
) -> None:
    store.store_session(
        session_id=session_id,
        prompt_text=f"prompt {session_id}",
        embedding=_embedding(seed),
        model_chosen=model,
        cost=cost,
        cache_stats={},
        duration=1.0,
        timestamp=when.isoformat(),
        decision_provenance=provenance,
        provenance=SessionProvenance(selection_propensity=propensity, cost_known=cost_known),
    )
    if outcome is not None:
        store.append_outcome_event(
            OutcomeEvent(
                session_id=session_id,
                tier=2,
                source=source,
                outcome=outcome,
                confidence=0.9,
                run_signature=f"test:{session_id}",
                created_at=when.isoformat(),
            )
        )


# Module-scoped, like `identified_store` below and for the same reason: every test that takes
# it only READS it, and rebuilding the corpus per test was the file's single largest cost —
# store writes, not renders, dominate the setup column.
@pytest.fixture(scope="module")
def mixed_store(tmp_path_factory: pytest.TempPathFactory) -> OutcomeStore:
    """Two seeded rows, two live rows, one row whose origin signals disagree."""
    store = _store(tmp_path_factory.mktemp("mixed"))
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    seed_prov = {"selection_rule_used": "benchmark_seed"}
    _seed_session(store, "bench:a", model="cheap", when=epoch, provenance=seed_prov, seed=1)
    _seed_session(
        store, "bench:b", model="cheap", when=epoch, provenance=seed_prov, outcome="failure", seed=2
    )
    now = datetime.now(UTC)
    _seed_session(
        store,
        "live-1",
        model="frontier",
        when=now - timedelta(days=1),
        cost=0.25,
        provenance={"selection_rule_used": "knn"},
        source="human",
        propensity=0.7,
        seed=3,
    )
    _seed_session(
        store,
        "live-2",
        model="cheap",
        when=now,
        cost=0.05,
        provenance={"selection_rule_used": "auto_escalation", "escalated_reasoning_arm": "high"},
        source="human",
        propensity=0.3,
        seed=4,
    )
    # Prefix says seeded, provenance and outcome source say live: surfaced, never assigned.
    _seed_session(
        store,
        "bench:disagreeing",
        model="cheap",
        when=now,
        provenance={"selection_rule_used": "knn"},
        source="human",
        seed=5,
    )
    return store


@pytest.fixture(scope="module")
def seed_only_store(tmp_path_factory: pytest.TempPathFactory) -> OutcomeStore:
    """The shape a docs render sees: seeded rows only, no live stratum at all."""
    store = _store(tmp_path_factory.mktemp("seed_only"))
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    prov = {"selection_rule_used": "benchmark_seed"}
    for index in range(12):
        _seed_session(
            store,
            f"bench:{index}",
            model="cheap" if index % 2 else "frontier",
            when=epoch,
            cost=0.1 * (index + 1),
            provenance=prov,
            outcome="success" if index % 3 else "failure",
            seed=index + 10,
        )
    return store


def _drawn(draw: Any, out_dir: Path, view: Any) -> Any:
    """Render, and hand back the Figure so the strings it really carries can be asserted."""
    # Wrapping the frame's own `save` is what makes this possible: the frame closes the figure
    # on the way out, so nothing is inspectable afterwards. Asserting on the PNG's size instead
    # is what let four wrong on-canvas strings ship green.
    from shunt.inspect import plot_frame

    captured: list[Any] = []
    original = plot_frame.save

    def spy(fig: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(fig)
        return original(fig, *args, **kwargs)

    plot_frame.save = spy  # type: ignore[assignment]
    try:
        draw(out_dir, view, None)
    finally:
        plot_frame.save = original  # type: ignore[assignment]
    return captured[0]


def _texts_of(fig: Any) -> list[str]:
    """Every claim-bearing string on an already-drawn figure — split out so one render serves
    both a `_drawn` assertion and a text assertion in the same test."""
    # `loc="left"` on panel labels: `get_title()` with no loc returns the CENTRE title, which
    # is always empty here, so an assertion over it silently passes on anything.
    strings = [t.get_text() for ax in fig.axes for t in ax.texts]
    strings += [ax.get_title(loc="left") for ax in fig.axes]
    # `plot_frame.attach_band` draws title/subtitle/caveat with `fig.text`, so they live on the
    # FIGURE, not on any axes. Omitting them left every claim-bearing string unassertable.
    strings += [t.get_text() for t in fig.texts]
    return [" ".join(text.split()) for text in strings]  # `_empty` hard-wraps its message


def _canvas_texts(draw: Any, out_dir: Path, view: Any) -> list[str]:
    return _texts_of(_drawn(draw, out_dir, view))


def _panel_labels(draw: Any, out_dir: Path, view: Any) -> list[str]:
    return [ax.get_title(loc="left") for ax in _drawn(draw, out_dir, view).axes]


# ------------------------------------------------------------------ transforms


@pytest.mark.parametrize(
    ("session_id", "rule", "source", "expected"),
    [
        ("bench:x", "benchmark_seed", "benchmark_seed", idata.SEEDED),
        # A seeded row that was LATER genuinely verified: `_SOURCE_PRIORITY` gives
        # benchmark_seed priority 0, so the winning source becomes auto_tier2/human. This is
        # the normal path, and it must stay seeded rather than flip to ambiguous.
        ("bench:x", "benchmark_seed", "auto_tier2", idata.SEEDED),
        ("bench:x", "benchmark_seed", "human", idata.SEEDED),
        # Both decision-time witnesses say live and the label source says seeded: a real
        # conflict, surfaced rather than resolved in either direction.
        ("live-x", "knn", "benchmark_seed", idata.AMBIGUOUS),
        # Witnesses disagree, and the source settles it toward seeded.
        ("live-x", "benchmark_seed", "benchmark_seed", idata.SEEDED),
        ("live-x", "knn", "human", idata.LIVE),
        ("bench:x", "knn", "human", idata.AMBIGUOUS),
        ("bench:x", "knn", None, idata.AMBIGUOUS),
        ("live-x", "benchmark_seed", "human", idata.AMBIGUOUS),
        ("bench:x", "benchmark_seed", None, idata.SEEDED),
        ("live-x", None, None, idata.LIVE),
    ],
)
def test_stratum_needs_all_three_signals_to_agree(
    session_id: str, rule: str | None, source: str | None, expected: str
) -> None:
    assert idata.classify_stratum(session_id, rule, source) == expected


def test_read_sessions_surfaces_the_disagreeing_row(mixed_store: OutcomeStore) -> None:
    rows = idata.read_sessions(mixed_store)
    by_id = {row.session_id: row for row in rows}
    assert by_id["bench:a"].stratum == idata.SEEDED
    assert by_id["live-1"].stratum == idata.LIVE
    assert by_id["bench:disagreeing"].stratum == idata.AMBIGUOUS
    assert [row.timestamp for row in rows] == sorted(
        row.timestamp for row in rows if row.timestamp is not None
    )


def test_cost_excludes_seeded_rows_and_says_how_many(mixed_store: OutcomeStore) -> None:
    rows = idata.read_sessions(mixed_store)
    view = idata.cost(rows)
    whole = dict(view.windows)["all"]
    # 0.25 + 0.05 live only; the two `bench:` rows contribute 1.0 and must not appear.
    assert whole.total == pytest.approx(0.30)
    assert view.n_seeded_excluded == 2
    assert view.n_live == 2
    assert [total for _when, total in view.cumulative][-1] == pytest.approx(0.30)


def test_cost_on_a_seed_only_store_is_empty_not_zero_cost(seed_only_store: OutcomeStore) -> None:
    rows = idata.read_sessions(seed_only_store)
    view = idata.cost(rows)
    assert view.n_live == 0
    assert view.cumulative == []
    assert all(agg.by_model == [] for _label, agg in view.windows)
    assert view.n_seeded_excluded == 12


def test_unit_economics_keeps_the_strata_apart(mixed_store: OutcomeStore) -> None:
    view = idata.unit_economics(idata.read_sessions(mixed_store))
    seeded = {row.model: row for row in view.seeded}
    assert seeded["cheap"].n_labeled == 2
    assert seeded["cheap"].n_success == 1
    assert seeded["cheap"].rate == pytest.approx(0.5)
    # Two seeded rows at 0.5 each, one success: 1.0 / 1.
    assert seeded["cheap"].cost_per_success == pytest.approx(1.0)
    assert {row.model for row in view.live} == {"cheap", "frontier"}


def test_cost_per_success_is_none_rather_than_zero_without_a_success(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_session(
        store, "live-fail", model="cheap", when=datetime.now(UTC), outcome="failure", source="human"
    )
    view = idata.unit_economics(idata.read_sessions(store))
    assert view.live[0].n_success == 0
    assert view.live[0].cost_per_success is None


def test_neighbourhood_leaves_the_probe_out_and_finds_no_live_mix(
    seed_only_store: OutcomeStore,
) -> None:
    rows = idata.read_sessions(seed_only_store)
    view = idata.neighbourhood(seed_only_store, rows, k=3)
    assert view.n_probed > 0
    assert view.n_live_decisions == 0
    assert view.origin_mix == []
    assert all(0.0 <= b.observed <= 1.0 for b in view.bins)
    assert sum(b.n for b in view.bins) == view.n_probed


def test_policy_reports_the_seed_mix_without_a_live_series(seed_only_store: OutcomeStore) -> None:
    view = idata.policy(idata.read_sessions(seed_only_store))
    assert view.n_live == 0
    assert view.live_series == []
    assert view.entropy == []
    assert dict(view.seed_mix) == {"cheap": 6, "frontier": 6}


def test_escalation_reads_the_rung_from_provenance(mixed_store: OutcomeStore) -> None:
    view = idata.escalation(idata.read_sessions(mixed_store))
    assert dict(view.rungs)["raise_effort"] == 1
    assert dict(view.rungs)["raise_rank"] == 0
    assert view.n_live == 2


def test_undeliverable_hold_is_recovered_without_a_hold_token(tmp_path: Path) -> None:
    # N4: `engine._void_exploration` rewrites the record to HOLD and returns BEFORE the branch
    # that writes `escalation_hold_reason`. Without this recovery, panel C would silently omit
    # every "no healthy higher model" hold — which is exactly the case it most needs to show.
    store = _store(tmp_path)
    now = datetime.now(UTC)
    _seed_session(
        store,
        "live-void",
        model="cheap",
        when=now,
        # Built by the PRODUCTION void path, not by a literal: a hand-written record would pin
        # the detector to a serialization the engine never actually emits, and the detector would
        # keep passing after the engine's changed.
        provenance=_voided_provenance(),
        source="human",
    )
    _seed_session(
        store,
        "live-token",
        model="cheap",
        when=now,
        provenance={"selection_rule_used": "knn", "escalation_hold_reason": "escalation_ceiling"},
        source="human",
        seed=2,
    )
    view = idata.escalation(idata.read_sessions(store))
    assert view.n_undeliverable == 1
    assert dict(view.holds)["escalation_ceiling"] == 1
    # The tokenised hold must NOT also be counted as undeliverable — that would double-count.
    assert sum(n for _t, n in view.holds) + view.n_undeliverable == 2


def test_a_genuinely_explored_hold_is_not_counted_as_undeliverable(tmp_path: Path) -> None:
    # The near-miss the detector has to survive: a sampled HOLD arm also serialises action="hold".
    # It is separated by randomized=True, by propensity<1, and by carrying a token at all.
    store = _store(tmp_path)
    sampled = ExplorationRecord(
        checkpoint_id="tests::flaky",
        decision_index=0,
        action=EscalationAction.HOLD,
        policy_action=EscalationAction.RAISE_RANK,
        propensity=0.3,
        epsilon=0.3,
        seed=1,
        randomized=True,
    )
    _seed_session(
        store,
        "live-sampled-hold",
        model="cheap",
        when=datetime.now(UTC),
        provenance={
            "selection_rule_used": "knn",
            "escalation_hold_reason": "exploration_hold",
            "escalation_exploration": json.loads(json.dumps(sampled.persistable())),
        },
        source="human",
    )
    view = idata.escalation(idata.read_sessions(store))
    assert view.n_undeliverable == 0
    assert dict(view.holds)["exploration_hold"] == 1


def test_void_signature_matches_the_engine_and_survives_json(tmp_path: Path) -> None:
    # Pins the detector to the engine, not to a string: if `_void_exploration` stops writing
    # action=hold / propensity=1.0 / randomized=False, this fails here rather than silently
    # zeroing a bar on a published figure.
    record = _voided_provenance()["escalation_exploration"]
    assert record["action"] == "hold"
    assert record["propensity"] == 1.0
    assert record["randomized"] is False


def test_unknown_hold_token_is_surfaced_not_bucketed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_session(
        store,
        "live-novel",
        model="cheap",
        when=datetime.now(UTC),
        provenance={"escalation_hold_reason": "a_token_that_does_not_exist_yet"},
        source="human",
    )
    view = idata.escalation(idata.read_sessions(store))
    assert view.unknown_holds == [("a_token_that_does_not_exist_yet", 1)]


# --------------------------------------------------------------------- renders


def _render_all(store: OutcomeStore, out_dir: Path) -> list[Path]:
    rows = idata.read_sessions(store)
    return [
        figures.draw_strata(out_dir, idata.strata(store, rows), None),
        figures.draw_cost(out_dir, idata.cost(rows), None),
        figures.draw_unit_economics(out_dir, idata.unit_economics(rows), None),
        figures.draw_neighbourhood(out_dir, idata.neighbourhood(store, rows, k=3), None),
        figures.draw_policy(out_dir, idata.policy(rows), None),
        figures.draw_escalation(out_dir, idata.escalation(rows), None),
        figures.draw_model_grid(out_dir, idata.model_grid(rows), None),
        figures.draw_ope(out_dir, idata.ope(store, _certified(store)), None),
    ]


def test_every_figure_renders_on_a_mixed_corpus(mixed_store: OutcomeStore, tmp_path: Path) -> None:
    out = tmp_path / "mixed"
    paths = _render_all(mixed_store, out)
    assert [p.name for p in paths] == [text.filename for text in specs.FIGURES]
    assert all(p.stat().st_size > 0 for p in paths)


def test_every_figure_renders_on_a_seed_only_corpus(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # The docs case. Under SHUNT_PLOT_STRICT=1 a panel that drew nothing at all would still
    # pass; what this proves is that the honest-empty path completes for all eight.
    paths = _render_all(seed_only_store, tmp_path / "seed")
    assert [p.name for p in paths] == [text.filename for text in specs.FIGURES]
    assert all(p.stat().st_size > 0 for p in paths)


def test_every_figure_renders_on_an_empty_store(tmp_path: Path) -> None:
    paths = _render_all(_store(tmp_path / "db"), tmp_path / "empty")
    assert [p.name for p in paths] == [text.filename for text in specs.FIGURES]
    assert all(p.stat().st_size > 0 for p in paths)


def test_manifest_row_carries_the_counts_that_explain_an_empty_panel(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    manifest = tmp_path / "figures.json"
    from shunt.inspect.plot_frame import Provenance

    provenance = Provenance(
        generator="shunt.inspect.inference.figures", data_digest="test", manifest=manifest
    )
    rows = idata.read_sessions(seed_only_store)
    figures.draw_cost(tmp_path, idata.cost(rows), provenance)
    payload = json.loads(manifest.read_text())
    row = payload["figures"][specs.COST.filename]
    assert row["n"]["live_sessions"] == 0
    assert row["n"]["seeded_excluded"] == 12
    assert row["title"] == specs.COST.title
    assert "seeded rows excluded (n=12)" in row["subtitle"]
    assert row["caveat"] == "no live sessions in this corpus — every panel is empty, not zero"


def test_specs_convert_to_frame_specs_within_the_frame_limits() -> None:
    for text in specs.FIGURES:
        spec = figures._spec(text)
        assert spec.title == text.title
        assert spec.reading and spec.goal


def test_sizes_cover_every_figure() -> None:
    assert set(figures.sizes()) == {text.name for text in specs.FIGURES}
    assert tuple(text.name for text in specs.FIGURES) == figures.DRAWERS


@pytest.mark.filterwarnings("ignore:constrained_layout not applied:UserWarning")
def test_strict_layout_rejects_a_figure_that_overflows_its_canvas() -> None:
    # Proves the strict gate the other render tests rely on is real, rather than re-asserting
    # the fixture that sets it.
    import matplotlib.pyplot as plt

    from shunt.inspect import plot_contract, plot_frame

    fig = plot_frame.new_figure(plot_frame.SINGLE)
    ax = fig.subplots()
    ax.set_xticks([0.0])
    ax.set_xticklabels(["x" * 400])
    try:
        assert plot_contract.audit(fig, band_top_px=float(fig.bbox.y1)) != []
    finally:
        plt.close(fig)


# ------------------------------------------- the numbers the canvas actually prints


def test_verified_seeded_rows_stay_seeded_and_keep_the_exclusion_count_honest(
    tmp_path: Path,
) -> None:
    # The regression that would silently shrink F2's central honesty claim: a seeded row that
    # gets genuinely verified has its outcome source overwritten, and an origin rule that voted
    # on the winning source would drop it from `seeded rows excluded (n=...)`.
    store = _store(tmp_path)
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    prov = {"selection_rule_used": "benchmark_seed"}
    for index in range(4):
        _seed_session(
            store,
            f"bench:{index}",
            model="cheap",
            when=epoch,
            provenance=prov,
            source="benchmark_seed",
            seed=index,
        )
    # Two of them are later verified for real, exactly as the rig is meant to do.
    for index in (0, 1):
        store.append_outcome_event(
            OutcomeEvent(
                session_id=f"bench:{index}",
                tier=2,
                source="auto_tier2",
                outcome="success",
                confidence=0.95,
                run_signature="verify",
                created_at=epoch.isoformat(),
            )
        )
    rows = idata.read_sessions(store)
    assert [row.stratum for row in rows] == [idata.SEEDED] * 4
    view = idata.cost(rows)
    assert view.n_seeded_excluded == 4
    assert view.n_live == 0
    strata_view = idata.strata(store, rows)
    assert strata_view.ambiguous == []


def test_normalized_entropy_matches_the_shipped_collapse_alarm() -> None:
    # Pins the figure's metric to `loop_health`'s, so a router pinned to 2 of 6 arms cannot read
    # all-clear here while the shipped alarm fires. Dividing by OBSERVED arms returns 1.0.
    from shunt.db.loop_health import LoopHealthSnapshot, LoopHealthThresholds, compute_loop_health

    choices = ["a"] * 50 + ["b"] * 50
    candidates = {"a", "b", "c", "d", "e", "f"}
    shipped = compute_loop_health(
        LoopHealthSnapshot(
            total_sessions=100,
            eligible_sessions=100,
            verified_labeled=0,
            any_labeled=0,
            model_propensities=[],
            recent_choices=choices,
            cost_by_model=[],
        ),
        frontier_models=set(),
        candidate_models=candidates,
        thresholds=LoopHealthThresholds(),
    )
    ours = idata.normalized_entropy(choices, len(candidates))
    assert ours == pytest.approx(shipped.routing_collapse.choice_entropy)
    assert shipped.routing_collapse.entropy_collapse_alarm is True
    assert ours < LoopHealthThresholds().entropy_collapse


def test_normalized_entropy_refuses_rather_than_guessing_the_arm_count() -> None:
    assert idata.normalized_entropy(["a", "b"], None) == 1.0
    assert idata.normalized_entropy(["a"], 1) == 1.0


def test_share_series_is_a_trailing_window_not_a_cumulative_total(tmp_path: Path) -> None:
    # A router that has gone single-arm recently must show as single-arm, not be diluted by the
    # history before it. Window 4 over [a,a,b,b,b,b] ends at 100% b.
    store = _store(tmp_path)
    base = datetime.now(UTC) - timedelta(days=1)
    for index, model in enumerate(["a", "a", "b", "b", "b", "b"]):
        _seed_session(
            store,
            f"live-{index}",
            model=model,
            when=base + timedelta(minutes=index),
            provenance={"selection_rule_used": "knn"},
            source="human",
            seed=index,
        )
    rows = idata.read_sessions(store)
    series = idata._share_series([r for r in rows if r.stratum == idata.LIVE], 4)
    assert series[-1][1] == {"b": 1.0}
    assert sum(series[-1][1].values()) == pytest.approx(1.0)


def test_window_of_seven_days_is_seven_days_not_eight(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime.now(UTC)
    for index, age_hours in enumerate((0, 24 * 7 + 1)):  # newest, then 7d+1h behind it
        _seed_session(
            store,
            f"live-{index}",
            model="cheap",
            when=now - timedelta(hours=age_hours),
            provenance={"selection_rule_used": "auto_escalation"},
            source="human",
            seed=index,
        )
    view = idata.escalation(idata.read_sessions(store))
    assert dict((label, total) for label, _n, total in view.rates)["7d"] == 1


def test_cost_coverage_label_does_not_triple_count_nested_windows(tmp_path: Path) -> None:
    # 7d nests inside 30d nests inside all, so a summed unknown count reports one session as
    # three — on the cost figure, and contradicting the counts block on the same canvas.
    store = _store(tmp_path)
    _seed_session(
        store,
        "live-nocost",
        model="cheap",
        when=datetime.now(UTC),
        provenance={"selection_rule_used": "knn"},
        source="human",
        cost_known=False,
    )
    rows = idata.read_sessions(store)
    view = idata.cost(rows)
    assert [agg.n_cost_unknown for _label, agg in view.windows] == [1, 1, 1]
    labels = _panel_labels(figures.draw_cost, tmp_path / "f", view)
    assert any("unknown = 1" in label for label in labels)


def test_cost_figure_prints_no_cost_when_there_is_no_live_traffic(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    manifest = tmp_path / "m.json"
    from shunt.inspect.plot_frame import Provenance

    rows = idata.read_sessions(seed_only_store)
    figures.draw_cost(
        tmp_path,
        idata.cost(rows),
        Provenance(generator="t", data_digest="t", manifest=manifest),
    )
    subtitle = json.loads(manifest.read_text())["figures"][specs.COST.filename]["subtitle"]
    assert "empty, not zero" not in subtitle
    assert "live cost" not in subtitle  # the self-contradiction: "$0.0000" above "not zero"


def test_policy_panel_b_refuses_without_a_registry(seed_only_store: OutcomeStore) -> None:
    view = idata.policy(idata.read_sessions(seed_only_store))
    assert view.candidate_models is None
    assert view.entropy == []
    assert view.frontier_share == []


def test_arrival_annotation_fires_only_on_genuinely_identical_stamps(tmp_path: Path) -> None:
    # A busy day of live traffic is not a one-burst seeded import, and must not be captioned so.
    store = _store(tmp_path)
    base = datetime.now(UTC) - timedelta(hours=6)
    for index in range(5):
        _seed_session(
            store,
            f"live-{index}",
            model="cheap",
            when=base + timedelta(hours=index),
            provenance={"selection_rule_used": "knn"},
            source="human",
            seed=index,
        )
    rows = idata.read_sessions(store)
    texts = _canvas_texts(figures.draw_strata, tmp_path / "f", idata.strata(store, rows))
    assert not any("shares one stamp" in text for text in texts)


def test_arrival_annotation_fires_on_the_seeded_burst(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    rows = idata.read_sessions(seed_only_store)
    texts = _canvas_texts(figures.draw_strata, tmp_path / "f", idata.strata(seed_only_store, rows))
    assert any("shares one stamp" in text for text in texts)


def test_empty_panels_say_what_is_absent_and_how_much(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # The absence of this test is why four wrong on-canvas strings shipped green: asserting the
    # PNG is non-empty proves nothing about what it says.
    rows = idata.read_sessions(seed_only_store)
    texts = _canvas_texts(figures.draw_cost, tmp_path / "c", idata.cost(rows))
    assert any("no live sessions in this corpus (n=0)" in text for text in texts)
    esc = _canvas_texts(figures.draw_escalation, tmp_path / "e", idata.escalation(rows))
    assert any("no live sessions in this corpus (n=0)" in text for text in esc)


def test_unit_economics_says_its_band_is_replayed_and_its_live_claim_is_empty(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # The band is the whole figure on a seed-only corpus, and a reader who takes it for live
    # routing reads replayed benchmark spend as this router's unit economics.
    rows = idata.read_sessions(seed_only_store)
    view = idata.unit_economics(rows)
    texts = _canvas_texts(figures.draw_unit_economics, tmp_path / "u", view)
    assert specs.UNIT_ECONOMICS.title in texts
    assert specs.UNIT_ECONOMICS.caveat in texts
    assert any("live labeled sessions n=0" in text for text in texts)
    assert "B · cost per verified success (live models: 0)" in texts


def test_neighbourhood_names_its_k_and_says_panel_c_is_empty_not_zero(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # `caveat=None` in the spec, so the empty-state caveat is the ONLY red line this figure
    # ever carries: without it an empty origin panel reads as "no live decision was seeded".
    rows = idata.read_sessions(seed_only_store)
    view = idata.neighbourhood(seed_only_store, rows, k=3)
    texts = _canvas_texts(figures.draw_neighbourhood, tmp_path / "n", view)
    assert specs.NEIGHBOURHOOD.title in texts
    assert any(text.startswith(specs.NEIGHBOURHOOD.subtitle) and "k=3" in text for text in texts)
    assert "no live decisions in this corpus — panel C is empty, not zero" in texts


def test_policy_keeps_the_seed_band_labelled_as_composition_not_a_decision(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # Panel A falls back to the seeded mix when there is no live traffic, which is the one
    # place this family can be misread as a routing policy the router actually chose.
    rows = idata.read_sessions(seed_only_store)
    texts = _canvas_texts(figures.draw_policy, tmp_path / "p", idata.policy(rows))
    assert specs.POLICY.title in texts
    assert specs.POLICY.caveat in texts
    assert any("live sessions n=0" in text for text in texts)
    assert "A · corpus composition, NOT a routing decision" in texts
    assert "no policy decision carries a selection propensity in this corpus" in texts


@pytest.fixture(scope="module")
def mismatched_store(tmp_path_factory: pytest.TempPathFactory) -> OutcomeStore:
    """One live row, plus a seeded row that lost its `bench:` prefix but kept its rule."""
    # The two definitions of "live" disagree on exactly this row: `classify_stratum`
    # adjudicates it SEEDED from the rule and the label source, while a prefix-only clause
    # calls it live and bills its cost as this router's own spend.
    store = _store(tmp_path_factory.mktemp("mismatched"))
    now = datetime.now(UTC)
    _seed_session(
        store,
        "live-1",
        model="cheap",
        when=now,
        cost=0.10,
        provenance={"selection_rule_used": "knn"},
        source="human",
        seed=1,
    )
    _seed_session(
        store,
        "noprefix-seed",
        model="frontier",
        when=now,
        cost=9.99,
        provenance={"selection_rule_used": "benchmark_seed"},
        source="benchmark_seed",
        seed=2,
    )
    return store


def test_cost_bills_only_the_stratum_it_claims_to_have_excluded(
    mismatched_store: OutcomeStore, tmp_path: Path
) -> None:
    rows = idata.read_sessions(mismatched_store)
    assert {row.session_id: row.stratum for row in rows} == {
        "live-1": idata.LIVE,
        "noprefix-seed": idata.SEEDED,
    }
    view = idata.cost(rows)
    whole = dict(view.windows)["all"]
    assert view.n_seeded_excluded == 1
    assert view.n_live == 1
    # The subtitle asserts the seeded row was excluded; the window total and the cumulative
    # curve must agree with that claim rather than carry $9.99 of benchmark spend.
    assert whole.total == pytest.approx(0.10)
    assert [model for model, _n, _total in whole.by_model] == ["cheap"]
    assert [total for _when, total in view.cumulative][-1] == pytest.approx(whole.total)


def test_cost_subtitle_and_plotted_curve_cannot_disagree(
    mismatched_store: OutcomeStore, tmp_path: Path
) -> None:
    from shunt.inspect.plot_frame import Provenance

    manifest = tmp_path / "m.json"
    rows = idata.read_sessions(mismatched_store)
    view = idata.cost(rows)
    figures.draw_cost(tmp_path, view, Provenance(generator="t", data_digest="t", manifest=manifest))
    row = json.loads(manifest.read_text())["figures"][specs.COST.filename]
    assert "seeded rows excluded (n=1)" in row["subtitle"]
    assert f"live cost {usd(view.cumulative[-1][1], 4)}" in row["subtitle"]
    assert row["n"]["live_sessions"] == 1


def test_strata_totals_count_adjudicated_rows_not_the_prefix_census(
    mixed_store: OutcomeStore, tmp_path: Path
) -> None:
    from shunt.inspect.plot_frame import Provenance

    rows = idata.read_sessions(mixed_store)
    view = idata.strata(mixed_store, rows)
    assert view.n_sessions == len(rows) == 5
    assert (view.n_seeded, view.n_live, len(view.ambiguous)) == (2, 2, 1)
    manifest = tmp_path / "m.json"
    figures.draw_strata(
        tmp_path, view, Provenance(generator="t", data_digest="t", manifest=manifest)
    )
    row = json.loads(manifest.read_text())["figures"][specs.STRATA.filename]
    # The ambiguous row was counted twice before: once inside the prefix census's seeded
    # funnel and once as ambiguous, so the figure claimed 6 sessions over a corpus of 5.
    assert row["n"]["sessions"] == 5
    assert "seeded n=2" in row["subtitle"]
    assert "live n=2" in row["subtitle"]


def test_strata_caveat_discloses_a_funnel_that_disagrees_with_the_adjudication(
    mismatched_store: OutcomeStore, tmp_path: Path
) -> None:
    from shunt.inspect.plot_frame import Provenance

    manifest = tmp_path / "m.json"
    rows = idata.read_sessions(mismatched_store)
    figures.draw_strata(
        tmp_path,
        idata.strata(mismatched_store, rows),
        Provenance(generator="t", data_digest="t", manifest=manifest),
    )
    caveat = json.loads(manifest.read_text())["figures"][specs.STRATA.filename]["caveat"]
    assert caveat is not None
    assert "prefix" in caveat and "1" in caveat


# ---------------------------------------------------------------- F7 off-policy


def _certified(
    store: OutcomeStore, *, admissible: bool = True, failing: tuple[tuple[str, str], ...] = ()
) -> estimators.CertifiedEstimates:
    """A verdict without paying for the real 11-second control — the control has its own test."""
    # `certify()` is exercised end to end by `test_inference_ope_instrument.py`. What these tests
    # need is the FIGURE's behaviour under each verdict, so the verdict is supplied.
    good = admissibility_verdict(1.0, 0.0, chance_level=0.0, chance_band=0.2)
    bad = admissibility_verdict(0.0, 0.0, chance_level=0.0, chance_band=0.2)
    certificates = tuple(
        estimators.EstimatorCertificate(
            admissibility=bad if (leg, name) in failing else good,
            leg=leg,
            estimator=name,
            n_rows=400,
            n_shuffles=200,
            seed=1,
        )
        for leg in (estimators.ROUTING, estimators.ESCALATION)
        for name in estimators.ESTIMATORS
    )
    # `admissible` and `failing` are INDEPENDENT axes and the fixture keeps them so. Deriving the
    # aggregate from the certificates (`overall_admissibility`, a conjunction, covered by
    # `test_inference_ope_instrument.py`) would collapse them: every `failing` fixture would also
    # be aggregate-inadmissible, and the per-estimator gate `quotable` guards — draw, name the
    # failed estimator, omit its bar — could never be reached, because `draw_ope` refuses first.
    routing, escalation = estimators.live_estimates(store)
    return estimators.CertifiedEstimates(
        admissibility=good if admissible else bad,
        certificates=certificates,
        routing=routing,
        escalation=escalation,
    )


def _explored_session(  # noqa: PLR0913 (one arg per logged column, as `_seed_session` above)
    store: OutcomeStore,
    session_id: str,
    *,
    escalated: bool,
    reward: bool,
    checkpoint: str,
    epsilon: float = 0.3,
) -> None:
    """One escalation decision logged with a real propensity — the shape `ope.py` can weight."""
    _seed_session(
        store,
        session_id,
        model="cheap",
        when=datetime.now(UTC),
        source="human",
        provenance={
            "selection_rule_used": "knn",
            "escalation_exploration": {
                "checkpoint_id": checkpoint,
                "action": "raise_effort" if escalated else "hold",
                "propensity": epsilon if escalated else 1.0 - epsilon,
                "epsilon": epsilon,
                "features": {"same_failure_count": 3.0},
            },
        },
        outcome="success" if reward else "failure",
        seed=abs(hash(session_id)) % 500,
    )


def _routed_session(
    store: OutcomeStore, session_id: str, *, arm: str, target: str, reward: bool
) -> None:
    """One epsilon-greedy routing decision, with the propensity and scores the router logs."""
    arms = ("cheap", "mid", "frontier")
    epsilon = 0.3
    propensity = epsilon / len(arms) + (1.0 - epsilon if arm == "cheap" else 0.0)
    _seed_session(
        store,
        session_id,
        model=arm,
        when=datetime.now(UTC),
        source="human",
        propensity=propensity,
        provenance={
            "selection_rule_used": "knn",
            "candidate_model_scores": {a: 1.0 if a == target else 0.0 for a in arms},
            "epsilon": epsilon,
        },
        outcome="success" if reward else "failure",
        seed=abs(hash(session_id)) % 500,
    )


# Module-scoped: the six tests below only READ this store, and rebuilding its 160 sessions
# per test dominated the file's runtime.
@pytest.fixture(scope="module")
def identified_store(tmp_path_factory: pytest.TempPathFactory) -> OutcomeStore:
    """A corpus with real randomization on both legs, so the panels draw bars rather than refuse."""
    store = _store(tmp_path_factory.mktemp("ident"))
    rng = random.Random(11)
    arms = ("cheap", "mid", "frontier")
    for index in range(80):
        target = arms[rng.randrange(2)]
        arm = arms[rng.randrange(3)] if rng.random() < 0.3 else "cheap"
        _routed_session(
            store,
            f"live:r:{index}",
            arm=arm,
            target=target,
            reward=(arm == target) != (rng.random() < 0.1),
        )
        needs = rng.random() < 0.85
        escalated = rng.random() < 0.3
        _explored_session(
            store,
            f"live:e:{index}",
            escalated=escalated,
            reward=escalated == needs,
            checkpoint=f"chk-{index % 7}",
        )
    return store


def _ope_view(store: OutcomeStore, **kwargs: Any) -> idata.OpeData:
    return idata.ope(store, _certified(store, **kwargs))


def test_ope_carries_the_refusal_rather_than_dropping_the_leg(
    seed_only_store: OutcomeStore,
) -> None:
    view = _ope_view(seed_only_store)
    assert not view.routing.identified
    assert [leg.policy for leg in view.escalation] == [idata.ALWAYS_POLICY, idata.NEVER_POLICY]
    assert all(not leg.identified for leg in view.escalation)
    assert "overlap fails" in view.routing.estimate.reason
    assert [d.n_logged for d in view.diagnostics] == [0, 0]
    assert view.headline.endswith(
        "Scores are band-normalised worst cases over 6 (leg, estimator) controls."
    )


def test_ope_refusal_prints_the_estimators_own_reason_verbatim(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # The single most valuable thing F7 does on committed data. `st_size > 0` cannot see any of
    # it: what is asserted here is the exact string a reader will find on the canvas.
    view = _ope_view(seed_only_store)
    texts = _canvas_texts(figures.draw_ope, tmp_path / "ope", view)
    assert texts.count("NOT IDENTIFIED") == 2  # panels A and B, one refusal each
    reason = " ".join(view.routing.estimate.reason.split())
    assert texts.count(reason) == 2
    assert reason.startswith("no randomized decision with a verified outcome — overlap fails")
    counts = "0 decisions logged · 0 usable · 0 target-arm / 0 complement · 0 independent sessions"
    assert texts.count(counts) == 2
    assert any("no usable importance weight in this corpus" in text for text in texts)


def test_ope_panel_labels_name_all_four_panels(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # `get_title()` with no loc returns the always-empty CENTRE title, so an assertion over it
    # passes on anything; the labels are set with loc="left".
    assert _panel_labels(figures.draw_ope, tmp_path / "ope", _ope_view(seed_only_store)) == [
        "A · routing: value of serving the top-scored candidate",
        "B · escalation: always vs never, and the contrast that decides",
        "C · importance-weight ECDF, with ESS and clipping",
        "D · identification floors: what the logs measured ÷ the floor",
    ]


def test_ope_floor_ledger_prints_every_measurement_against_its_floor(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    texts = _canvas_texts(figures.draw_ope, tmp_path / "ope", _ope_view(seed_only_store))
    assert texts.count("0 / 5") == 6  # two legs × (target arm, other arm, independent sessions)
    assert texts.count("0 / 0.01") == 2  # two legs × the minimum-propensity floor
    assert idata.FLOOR_PER_ARM == 5
    assert idata.FLOOR_CLUSTERS == 5
    assert idata.FLOOR_PROPENSITY == 0.01


def test_ope_refuses_before_drawing_when_the_instrument_is_inadmissible(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # The PNG must NOT exist: SH009 then finds a manifest row with no file and pre-commit blocks.
    out = tmp_path / "gated"
    view = _ope_view(seed_only_store, admissible=False)
    with pytest.raises(estimators.InstrumentInadmissibleError):
        figures.draw_ope(out, view, None)
    assert not (out / specs.OPE.filename).exists()


def test_the_other_figures_still_render_when_the_ope_instrument_is_inadmissible(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # A bad estimator must not take the family down.
    out = tmp_path / "family"
    rows = idata.read_sessions(seed_only_store)
    assert figures.draw_strata(out, idata.strata(seed_only_store, rows), None).exists()
    assert figures.draw_escalation(out, idata.escalation(rows), None).exists()


def test_ope_draws_escalations_contrast_and_never_routings(
    identified_store: OutcomeStore, tmp_path: Path
) -> None:
    # W7's asymmetry: routing's contrast compares against "some other arm", not a deployable
    # policy. Escalation's is a real two-arm decision. Drawing both would invite the comparison.
    view = _ope_view(identified_store)
    assert view.routing.identified
    assert view.escalation[0].identified
    fig = _drawn(figures.draw_ope, tmp_path / "ope", view)
    routing_ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    escalation_ticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert routing_ticks == ["IPS", "SNIPS", "DR"]
    assert escalation_ticks == [
        "IPS\nalways",
        "SNIPS\nalways",
        "DR\nalways",
        "IPS\nnever",
        "SNIPS\nnever",
        "DR\nnever",
        "DR\nescalate − hold",
    ]
    assert specs.ROUTING_CONTRAST_NOTE in _texts_of(fig)


def test_ope_names_an_estimator_whose_certificate_failed_and_does_not_draw_it(
    identified_store: OutcomeStore, tmp_path: Path
) -> None:
    # ONE certificate failed while the instrument as a whole is ADMISSIBLE — the case `quotable`
    # exists for. The other case, an aggregate-inadmissible instrument, must raise before any
    # canvas exists and is asserted by
    # `test_ope_refuses_before_drawing_when_the_instrument_is_inadmissible`. Widening a band or
    # softening this assertion to make it pass would hollow out that gate.
    view = _ope_view(identified_store, failing=((estimators.ESCALATION, "snips"),))
    assert view.admissible
    assert view.escalation[0].quotable == ("ips", "dr")
    snips = view.escalation[0].snips
    assert snips is not None  # the value EXISTS; the figure declines to draw it
    fig = _drawn(figures.draw_ope, tmp_path / "ope", view)
    ticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert "SNIPS\nalways" not in ticks
    assert ticks == ["IPS\nalways", "DR\nalways", "IPS\nnever", "DR\nnever", "DR\nescalate − hold"]
    heights = [round(patch.get_height(), 12) for patch in fig.axes[1].patches]
    assert len(heights) == len(ticks)
    assert round(snips, 12) not in heights
    assert any(text.startswith("not quotable: always_escalate/SNIPS") for text in _texts_of(fig))


def test_ope_value_axis_never_calls_an_unbounded_estimate_a_rate(
    identified_store: OutcomeStore, tmp_path: Path
) -> None:
    # IPS is unnormalised, so its bar lands above 1 on a low-propensity log. An axis reading
    # "verified-success rate" under a bar at 1.29 states a 129% success rate — the same class of
    # wrong on-canvas string the F1-F6 round shipped four of, and invisible on committed data
    # because both legs refuse there.
    view = _ope_view(identified_store)
    ips = view.escalation[0].value("ips")
    assert ips is not None and ips > 1.0  # the condition the label has to survive
    fig = _drawn(figures.draw_ope, tmp_path / "ope", view)
    labels = [fig.axes[0].get_ylabel(), fig.axes[1].get_ylabel()]
    assert labels == [figures._VALUE_AXIS, figures._VALUE_AXIS]
    assert figures._VALUE_AXIS == "estimated verified-success rate"
    assert not any(label == "verified-success rate" for label in labels)
    # The canvas says "estimated"; the reason an estimate may exceed 1 is published with the
    # figure rather than crammed onto a half-canvas axis, so SH009 carries it into the docs.
    assert any(term == "a value above 1" for term, _body in specs.OPE.definitions)


def test_ope_prints_the_logged_policys_own_mean_beside_the_estimate(
    identified_store: OutcomeStore, tmp_path: Path
) -> None:
    # A value estimate with no on-policy baseline is a number a reader compares against zero.
    view = _ope_view(identified_store)
    mean = view.routing.on_policy_mean
    assert mean is not None
    texts = _canvas_texts(figures.draw_ope, tmp_path / "ope", view)
    assert f"logged policy paid {mean:.3f} (n={view.routing.n_rewarded})" in texts


def test_ope_diagnostics_report_the_shipped_estimators_own_weights(
    identified_store: OutcomeStore,
) -> None:
    routing, escalation = _ope_view(identified_store).diagnostics
    assert escalation.n_usable == escalation.n_escalated + escalation.n_held
    assert escalation.n_usable > 0
    assert len(escalation.weights) == escalation.n_usable
    assert escalation.weights == sorted(escalation.weights)
    assert escalation.min_propensity == pytest.approx(0.3)
    assert 0.0 < escalation.ess_fraction <= 1.0
    assert escalation.n_clipped == 0  # 1/0.3 is far below the clip
    assert routing.n_clusters == routing.n_usable  # one session per routing decision


def test_ope_manifest_row_carries_the_instrument_verdict_verbatim(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # SH009 carries `notes` byte-identically into docs/inference.md, so the verdict is PUBLISHED
    # rather than filed. The band-normalisation clause travels with it: without that sentence a
    # reader pastes a normalised score as if it were a reward rate.
    from shunt.inspect.plot_frame import Provenance

    manifest = tmp_path / "figures.json"
    view = _ope_view(seed_only_store)
    figures.draw_ope(tmp_path, view, Provenance(generator="t", data_digest="t", manifest=manifest))
    row = json.loads(manifest.read_text())["figures"][specs.OPE.filename]
    assert row["notes"][-1] == view.headline
    assert "band-normalised worst cases" in row["notes"][-1]
    assert row["caveat"] == (
        "off-policy value is NOT IDENTIFIED here — each panel prints the estimator’s own refusal"
    )
    assert row["n"] == {
        "routing_logged": 0,
        "routing_usable": 0,
        "escalation_logged": 0,
        "escalation_usable": 0,
    }
    assert "usable rows: routing 0/0, escalation 0/0" in row["subtitle"]


def test_ope_caveat_names_the_contrast_asymmetry_once_the_legs_are_identified(
    identified_store: OutcomeStore, tmp_path: Path
) -> None:
    from shunt.inspect.plot_frame import Provenance

    manifest = tmp_path / "figures.json"
    figures.draw_ope(
        tmp_path,
        _ope_view(identified_store),
        Provenance(generator="t", data_digest="t", manifest=manifest),
    )
    row = json.loads(manifest.read_text())["figures"][specs.OPE.filename]
    assert row["caveat"] == specs.ROUTING_CONTRAST_NOTE
    assert "routing identified" in row["subtitle"]


def test_policy_propensities_are_adjudicated_live_rows_only(tmp_path: Path) -> None:
    # F5 panel C reads the support floor. The store's snapshot filters on the id prefix, so a
    # seeded row that lost its prefix and carries a propensity would drag the floor down with
    # benchmark replay; the adjudicated stratum is the family's single definition of live.
    store = _store(tmp_path)
    now = datetime.now(UTC)
    _seed_session(
        store,
        "live-1",
        model="cheap",
        when=now,
        provenance={"selection_rule_used": "knn"},
        source="human",
        propensity=0.7,
        seed=1,
    )
    _seed_session(
        store,
        "noprefix-seed",
        model="frontier",
        when=now,
        provenance={"selection_rule_used": "benchmark_seed"},
        source="benchmark_seed",
        propensity=0.05,
        seed=2,
    )
    view = idata.policy(idata.read_sessions(store))
    assert view.propensities == [("cheap", 1, pytest.approx(0.7), pytest.approx(0.7))]


# ------------------------------------------- an empty stratum must never look like a result


def _legend_labels(fig: Any, panel: int) -> list[str]:
    legend = fig.axes[panel].get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test_strata_names_its_empty_live_stratum_in_red_not_only_in_the_grey_subtitle(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # THE DEFECT THIS PINS. `caveat` was None on a store with zero live sessions, so the whole
    # disclosure was `live n=0` in the grey subtitle — beside five grey bars and a full-strength
    # blue "live" key. That canvas is indistinguishable from one with a small live stratum, which
    # is the single failure mode this family exists to prevent.
    rows = idata.read_sessions(seed_only_store)
    view = idata.strata(seed_only_store, rows)
    assert view.n_live == 0
    fig = _drawn(figures.draw_strata, tmp_path / "s", view)
    texts = _texts_of(fig)
    assert any("live stratum EMPTY (n=0)" in text for text in texts)
    # And the legend says it too, in both bar panels, rather than keying a series with no bars.
    assert "live (none in this corpus)" in _legend_labels(fig, 0)
    assert "live (none in this corpus)" in _legend_labels(fig, 2)
    # One drawn series per panel, not two — a zero-height bar is a bar that rendered.
    assert len(fig.axes[0].containers) == 1
    assert len(fig.axes[2].containers) == 1


def test_strata_keeps_both_legend_keys_when_both_strata_are_populated(
    mixed_store: OutcomeStore, tmp_path: Path
) -> None:
    # The absence marker is keyed on the count, never on the figure: a corpus that HAS live rows
    # must get the plain key back, or the panel understates a real population.
    rows = idata.read_sessions(mixed_store)
    view = idata.strata(mixed_store, rows)
    assert view.n_live > 0
    fig = _drawn(figures.draw_strata, tmp_path / "s", view)
    assert "live" in _legend_labels(fig, 0)
    assert not any("none in this corpus" in text for text in _texts_of(fig))


def test_unit_economics_labels_the_absent_live_stratum_in_both_panels(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # THE DEFECT THIS PINS. Panel A's legend already read "live (none in this corpus)" while
    # panel B kept a plain "live" swatch beside zero-length bars — the SAME absence, labelled in
    # one panel and not the other, so the reader had to guess which panel was telling the truth.
    rows = idata.read_sessions(seed_only_store)
    fig = _drawn(figures.draw_unit_economics, tmp_path / "u", idata.unit_economics(rows))
    assert "live (none in this corpus)" in _legend_labels(fig, 0)
    assert "live (none in this corpus)" in _legend_labels(fig, 1)
    # Panel B draws the seeded series only; the live bars are absent, not zero-length.
    assert len(fig.axes[1].containers) == 1


def test_the_policy_seed_band_percentages_survive_the_hatch(
    seed_only_store: OutcomeStore, tmp_path: Path
) -> None:
    # THE DEFECT THIS PINS. The in-bar shares were white bold text over a white 45-degree hatch,
    # so the glyphs broke up into the hatch strokes and the percentages rendered garbled. A dark
    # stroke around each glyph is what makes them readable whatever colour the segment is.
    rows = idata.read_sessions(seed_only_store)
    fig = _drawn(figures.draw_policy, tmp_path / "p", idata.policy(rows))
    shares = [t for t in fig.axes[0].texts if t.get_text().endswith("%")]
    assert shares, "panel A drew no share labels"
    for text in shares:
        assert text.get_path_effects(), f"{text.get_text()} has no contrast stroke over the hatch"


def test_a_long_strata_caveat_is_capped_rather_than_raising_in_the_frame() -> None:
    # `plot_frame` raises above 120 characters, so a fourth clause could take the whole render
    # down. The clauses that fit are kept and the rest are COUNTED — never silently dropped.
    parts = [f"clause number {i} that is quite long indeed" for i in range(4)]
    capped = figures._fit_caveat(parts)
    assert capped is not None
    assert len(capped) <= 120
    assert "more)" in capped
    assert figures._fit_caveat([]) is None
