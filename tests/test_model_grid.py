"""The model grid's encodings: what they mean, and what they refuse to invent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shunt.inspect import model_grid as grid_drawer
from shunt.inspect.model_grid import (
    _LABEL_OFFSETS,
    CLUSTER_COLOR,
    CLUSTER_ORDER,
    GridData,
    GridRow,
    _near,
    _size_axis_label,
    grid_annotations,
    label_offset,
    marker_area,
)


def _row(**kwargs: object) -> GridRow:
    base: dict[str, object] = {
        "name": "m",
        "x": 1.0,
        "serving_mode": "hosted",
        "n": 100,
        "passes": 50,
        "total_params": 300_000_000_000,
        "active_params": 20_000_000_000,
    }
    base.update(kwargs)
    return GridRow(**base)  # type: ignore[arg-type]


class TestClusters:
    def test_every_band_has_a_colour(self) -> None:
        assert set(CLUSTER_ORDER) == set(CLUSTER_COLOR)

    @pytest.mark.parametrize(
        ("total", "expected"),
        [
            (2_800_000_000_000, "≥1T total"),
            (1_000_000_000_000, "≥1T total"),
            (320_000_000_000, "100B–1T total"),
            (100_000_000_000, "100B–1T total"),
            (8_000_000_000, "<100B total"),
            (None, "size UNDISCLOSED"),
        ],
    )
    def test_band_boundaries(self, total: int | None, expected: str) -> None:
        assert _row(total_params=total).cluster == expected


class TestMarkerArea:
    def test_undisclosed_sits_off_the_ramp(self) -> None:
        # The whole point of the fixed mark: a reader must not be able to read a parameter
        # count off a model whose vendor publishes none.
        span = (3_000_000_000, 100_000_000_000)
        off = marker_area(None, span)
        assert off != marker_area(span[0], span)
        assert off != marker_area(span[1], span)

    def test_area_is_monotone_in_active_params(self) -> None:
        span = (3_000_000_000, 100_000_000_000)
        areas = [marker_area(v, span) for v in (3_000_000_000, 20_000_000_000, 100_000_000_000)]
        assert areas == sorted(areas)

    def test_a_single_sized_model_does_not_divide_by_zero(self) -> None:
        assert marker_area(8_000_000_000, (8_000_000_000, 8_000_000_000)) > 0


class TestZeroIsNotASmallNumber:
    def test_a_local_row_carries_no_price(self) -> None:
        row = _row(serving_mode="local", x=None)
        assert row.is_local
        assert row.x is None

    def test_the_subtitle_counts_free_and_priced_separately(self) -> None:
        data = GridData(
            rows=(
                _row(name="hosted-one"),
                _row(name="local-one", serving_mode="local", x=None),
            ),
            x_label="blended $ per 1M tokens (log)",
            price_basis="blend = 50% input / 50% output",
            source="a corpus",
            x_limitation="x is a list price",
        )
        facts = grid_annotations(data, sized=2).subtitle_facts
        assert any("1 at $0 (local) · 1 priced" in fact for fact in facts)


class TestAnnotations:
    def test_a_missing_size_is_named_in_the_notes(self) -> None:
        data = GridData(
            rows=(_row(name="closed", total_params=None, active_params=None),),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        notes = grid_annotations(data, sized=0).notes
        assert any("size UNDISCLOSED" in note for note in notes)
        assert any("fixed reference marker" in note for note in notes)

    def test_the_n_spread_is_stated_because_the_rows_are_unpaired(self) -> None:
        data = GridData(
            rows=(_row(name="a", n=21, passes=10), _row(name="b", n=198, passes=100)),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        facts = grid_annotations(data, sized=2).subtitle_facts
        assert any("n per model 21–198, unpaired" in fact for fact in facts)


class TestLabelPlacement:
    """Two rungs at the same price and the same rate must not overprint each other."""

    def test_a_close_pair_is_near(self) -> None:
        # deepseek-v4-flash and the daggered 3B rung, as actually drawn: 7% apart in price
        # and half a point apart in rate.
        assert _near((0.1421, 68.9), (0.1531, 69.4))

    def test_a_price_decade_apart_is_not_near(self) -> None:
        assert not _near((0.14, 69.0), (1.40, 69.0))

    def test_a_rate_gap_is_not_near(self) -> None:
        assert not _near((0.14, 55.0), (0.14, 69.0))

    def test_the_zero_column_compares_on_rate_alone(self) -> None:
        # Every $0 row sits at x=0, where a ratio has no meaning.
        assert _near((0.0, 69.0), (0.0, 70.0))
        assert not _near((0.0, 50.0), (0.0, 70.0))

    def test_three_clustered_rows_get_three_different_offsets(self) -> None:
        # THE DEFECT THIS REPLACES A VACUOUS TEST WITH. An earlier rule flipped a colliding
        # label to the other side of its marker without checking what was already there, so
        # the third row in a cluster landed on top of the second. The ladder is growing, so a
        # cluster of three at one price and one rate is the expected case.
        cluster = [(0.150, 69.0), (0.152, 69.3), (0.154, 68.8)]
        placed: list[tuple[tuple[float, float], float]] = []
        for anchor in cluster:
            offset = label_offset(anchor, placed, forced_below=False)
            placed.append((anchor, offset))
        offsets = [o for _a, o in placed]
        assert len(set(offsets)) == 3, offsets

    def test_a_distant_row_reuses_the_default_offset(self) -> None:
        # The ladder must not walk away from the marker for rows that never collided.
        placed = [((0.150, 69.0), 11.0)]
        assert label_offset((3.0, 84.0), placed, forced_below=False) == 11.0

    def test_the_ceiling_case_only_ever_goes_below(self) -> None:
        placed: list[tuple[tuple[float, float], float]] = []
        for anchor in [(0.150, 96.0), (0.152, 96.2), (0.154, 95.8)]:
            offset = label_offset(anchor, placed, forced_below=True)
            assert offset is not None
            placed.append((anchor, offset))
        assert all(o < 0 for _a, o in placed)
        assert len({o for _a, o in placed}) == len(placed)

    def test_eight_identical_anchors_get_eight_distinct_offsets(self) -> None:
        # F10 regression. The ladder had six slots and returned its last one when exhausted,
        # so the seventh clustered label printed on top of the sixth. A realistic cluster is a
        # handful of rungs at one price and one rate, so eight must all land somewhere new.
        cluster = [(0.150, 69.0 + 0.01 * i) for i in range(8)]
        placed: list[tuple[tuple[float, float], float]] = []
        for anchor in cluster:
            offset = label_offset(anchor, placed, forced_below=False)
            assert offset is not None
            placed.append((anchor, offset))
        offsets = [o for _a, o in placed]
        assert len(set(offsets)) == 8, offsets

    def test_the_ceiling_ladder_also_holds_eight(self) -> None:
        # The ceiling case may only go DOWNWARDS, and it had three negative slots; a cluster of
        # ceiling-height rows would collapse the fourth onto the third. The downward ladder is
        # now as deep as the general one.
        cluster = [(0.150, 96.0 + 0.01 * i) for i in range(8)]
        placed: list[tuple[tuple[float, float], float]] = []
        for anchor in cluster:
            offset = label_offset(anchor, placed, forced_below=True)
            assert offset is not None and offset < 0
            placed.append((anchor, offset))
        assert len({o for _a, o in placed}) == 8

    def test_an_exhausted_ladder_returns_a_sentinel(self) -> None:
        # Past the ladder the caller must route the name to the notes, never reuse a slot:
        # a missing direct label is recoverable, an overprinted pair is not.
        anchor = (0.150, 69.0)
        placed: list[tuple[tuple[float, float], float]] = []
        for _ in range(len(_LABEL_OFFSETS)):
            offset = label_offset(anchor, placed, forced_below=False)
            assert offset is not None
            placed.append((anchor, offset))
        assert label_offset(anchor, placed, forced_below=False) is None


class TestSizeAxisLabel:
    def test_the_scale_matches_the_axis_actually_drawn(self) -> None:
        # F17: the label said "(log)" even for an all-UNDISCLOSED roster whose axis is linear.
        assert "(log)" in _size_axis_label(log=True)
        assert "(linear)" in _size_axis_label(log=False)
        assert "hollow: total" in _size_axis_label(log=False)


class TestOutOfCorpusRows:
    """A daggered row is drawn, and every surface says it is not a corpus row."""

    NOTE = "measured on some other harness"

    def test_a_plain_row_is_not_external(self) -> None:
        assert not _row().is_external
        assert _row().provenance_note is None

    def test_the_note_is_printed_against_the_daggered_name(self) -> None:
        data = GridData(
            rows=(_row(name="outsider", provenance_note=self.NOTE), _row(name="insider")),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        ann = grid_annotations(data, sized=2)
        assert any(note.startswith("† outsider: ") and self.NOTE in note for note in ann.notes)
        # The per-row stat line carries the dagger too, so the two cannot be read as
        # two different rows.
        assert any(note.startswith("outsider †: ") for note in ann.notes)
        assert any(note.startswith("insider: ") for note in ann.notes)

    def test_the_dagger_costs_a_limitation(self) -> None:
        data = GridData(
            rows=(_row(name="outsider", provenance_note=self.NOTE),),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        limits = grid_annotations(data, sized=1).limitations
        assert any("DAGGERED row (†)" in limit for limit in limits)

    def test_a_corpus_only_panel_says_nothing_about_daggers(self) -> None:
        # The limitation is a CLAIM about the panel; printing it when no row is daggered
        # would tell a reader to discount a comparison that is in fact paired-by-corpus.
        data = GridData(
            rows=(_row(),),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        ann = grid_annotations(data, sized=1)
        assert not any("DAGGERED" in limit for limit in ann.limitations)
        assert dict(ann.counts)["external"] == 0


class TestBenchmarkAdapter:
    def test_the_committed_grid_renders_without_an_annotation_collision(
        self, tmp_path: Path
    ) -> None:
        # F10/F11 end to end: the committed corpus puts the `-explabs` mirror of a model in
        # the $0 column beside its paid row, so two rows share the label. The placement ladder
        # must separate them, and the opted-in annotation audit must pass; before the fix this
        # render raised LayoutError('deepseek-v4-pro' overlaps 'kimi-k3').
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing import model_validity
        from benchmark.routing.figures import model_grid as adapter

        evidence = model_validity.gather_evidence()
        raw = model_validity.filter_valid(config.load_results(), evidence)
        data = adapter.build(
            SimpleNamespace(raw=raw, validity=model_validity.validity_census(evidence))  # type: ignore[arg-type]
        )
        assert data is not None
        out = grid_drawer.render(tmp_path / "model_grid.png", data, adapter.SPEC)
        assert out.exists()

    def test_the_grid_reads_measured_cells_and_the_registry(self) -> None:
        # The one end-to-end assertion that the two data sources meet correctly: the rate
        # comes from results.csv and the parameter counts from models.yaml.
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing.figures import model_grid as adapter

        ctx = SimpleNamespace(raw=config.load_results())
        data = adapter.build(ctx)  # type: ignore[arg-type]
        assert data is not None
        by_name = {r.name: r for r in data.rows}
        deepseek = by_name["deepseek-v4-flash"]
        assert deepseek.total_params == 284_000_000_000
        assert deepseek.active_params == 13_000_000_000
        assert deepseek.serving_mode == "hosted"
        assert 0.0 < deepseek.rate < 1.0
        # A closed model is drawn, but never with an invented count.
        assert by_name["gpt-5-mini"].total_params is None

    def test_the_source_line_delegates_excluded_models_to_the_validity_figure(self) -> None:
        # THE CAPTION MUST NOT CLAIM A SWEEP A FILTER NARROWED. The report hands this adapter
        # a cache already scoped to the inference-valid set (`model_validity.filter_valid`), so
        # a dominated, unmeasured or collection-only model never reaches the canvas. The
        # excluded roster is drawn and explained in the model-validity figure, so the source
        # line points there rather than restating a wall of names.
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing import model_validity
        from benchmark.routing.figures import model_grid as adapter

        census = model_validity.validity_census()
        valid = {r.model for r in census if r.valid}
        cache = config.load_results()
        dropped = sorted({m for per_model in cache.values() for m in per_model} - valid)
        assert dropped, "the guard is vacuous with no inference-invalid model in the cache"
        scoped = {
            cid: {m: arms for m, arms in per_model.items() if m in valid}
            for cid, per_model in cache.items()
        }
        data = adapter.build(SimpleNamespace(raw=scoped, validity=census))  # type: ignore[arg-type]
        assert data is not None
        assert {row.name for row in data.rows}.isdisjoint(dropped)
        assert "model-validity figure" in data.source
        assert "model_validity.png" not in data.source
        assert "not enabled in benchmark.yaml" not in data.source

    def test_the_price_basis_names_the_measured_mix(self) -> None:
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing.figures import model_grid as adapter

        data = adapter.build(SimpleNamespace(raw=config.load_results()))  # type: ignore[arg-type]
        assert data is not None
        assert "corpus's own" in data.price_basis

    def test_the_committed_external_rung_transcribes_its_source_exactly(self) -> None:
        # The one guard that a hand-transcribed measurement cannot silently drift from the
        # run that produced it: the file states the interval it was measured at, and the
        # drawer recomputes it from k and n. A typo in either moves them apart.
        from benchmark.routing.figures import model_grid as adapter

        rows = {r.name: r for r in adapter._external_rows(0.98)}
        assert rows, "the committed external-rung file drew no row"
        doc = yaml.safe_load(adapter.EXTERNAL_RUNGS_PATH.read_text(encoding="utf-8"))
        for name, row in rows.items():
            stated = doc["rungs"][name]["measurement"]
            assert (row.passes, row.n) == (stated["passes"], stated["n"])
            assert row.rate == pytest.approx(stated["rate"], abs=5e-5)
            lo, hi = row.wilson
            assert (lo, hi) == pytest.approx(tuple(stated["wilson95"]), abs=5e-5)
            # A rung measured on one or two seeds may never be published as anything
            # stronger than SIGNAL, and a rung whose sample censored past the point of
            # estimating a rate may not be published as a rate at all.
            assert stated["verdict_ceiling"].startswith(("SIGNAL", "PRELIMINARY", "UNPROVEN"))
            assert row.is_external and row.provenance_note

    def test_a_local_external_rung_carries_no_price_and_no_zero(self) -> None:
        # THE UNDEFINED-COST GUARD. A locally served rung has no per-token list price, and its
        # dollar cost per solved task is UNDEFINED — never 0.0, which would validate cleanly
        # and then trivially dominate every cost comparison. The row must therefore reach the
        # canvas with `x is None` (panel A's category column) and must emit no dollar figure.
        from benchmark.routing.figures import model_grid as adapter

        doc = yaml.safe_load(adapter.EXTERNAL_RUNGS_PATH.read_text(encoding="utf-8"))
        local = {n for n, e in doc["rungs"].items() if e["serving_mode"] == "local"}
        assert local, "the guard is vacuous with no locally served rung"
        for row in adapter._external_rows(0.98):
            if row.name not in local:
                continue
            assert row.x is None
            assert row.provenance_note and "$0." not in row.provenance_note
            stated = doc["rungs"][row.name]
            assert stated["pricing"]["cost_basis"] == "UNDEFINED"
            assert "input_cost_per_1m" not in stated["pricing"]
            assert stated["measurement"]["measured_cost_per_instance_usd"] is None

    def test_a_heavily_censored_external_rung_says_so_on_the_canvas(self) -> None:
        # A marker drawn from a run whose cells mostly never concluded is a lower bound. The
        # note is where the reader learns that, so a row censoring more than a quarter of its
        # cells must state the count and the assumption-free bounds rather than only the rate.
        from benchmark.routing.figures import model_grid as adapter

        doc = yaml.safe_load(adapter.EXTERNAL_RUNGS_PATH.read_text(encoding="utf-8"))
        for row in adapter._external_rows(0.98):
            measured = doc["rungs"][row.name]["measurement"]
            if int(measured["censored"]) * 4 <= int(measured["cells_run"]):
                continue
            note = row.provenance_note or ""
            assert "censored" in note and "LOWER BOUND" in note

    def test_an_external_rung_is_not_in_the_shipped_registry(self) -> None:
        # THE RANK-BASIS GUARD. Cascade order is `input + output` list price over the
        # registry, so an external rung that leaked into it would re-order every cascade
        # underneath an already-published measurement claim. This test is what makes that a
        # failure rather than a review miss.
        from benchmark.routing.figures import model_grid as adapter
        from shunt.models.config import load_registry

        registered = set(load_registry().models)
        doc = yaml.safe_load(adapter.EXTERNAL_RUNGS_PATH.read_text(encoding="utf-8"))
        assert not registered & set(doc["rungs"])

    def test_no_measured_cell_draws_nothing(self) -> None:
        from types import SimpleNamespace

        from benchmark.routing.figures import model_grid as adapter

        assert adapter.build(SimpleNamespace(raw={})) is None  # type: ignore[arg-type]

    def test_the_external_row_cannot_be_the_only_row_on_the_canvas(self) -> None:
        # `source`, the blend and the x label all describe results.csv. A canvas carrying the
        # external row alone would assert a provenance it does not have, so the corpus is the
        # precondition — not merely one contributor to a non-empty row list. A raw cache with
        # a model but no scorable cell reaches exactly that state.
        from types import SimpleNamespace

        from benchmark.routing.figures import model_grid as adapter

        assert adapter._external_rows(0.98), "the guard is vacuous with no external rung"
        empty_arm = {"challenge-1": {"deepseek-v4-flash": {}}}
        assert adapter.build(SimpleNamespace(raw=empty_arm)) is None  # type: ignore[arg-type]


class TestSourceLine:
    def test_the_dagger_rides_the_word_not_the_count(self) -> None:
        # The count and the dagger were separate space-delimited tokens, so a wrap landed
        # between them and orphaned the glyph ("plus 2 / t out-of-corpus").
        from benchmark.routing.figures import model_grid as adapter

        line = adapter._source_line(4, 2, None)
        assert "2 out-of-corpus rung(s) (†)" in line
        assert "2 † out-of-corpus" not in line

    def test_no_external_rows_means_no_dagger_clause(self) -> None:
        from benchmark.routing.figures import model_grid as adapter

        assert "out-of-corpus" not in adapter._source_line(4, 0, None)


class TestInferenceAdapter:
    """The live half prices its x from what was BILLED, never from a list price (SH005)."""

    @staticmethod
    def _session(model: str, sid: str, stratum: str) -> object:
        from datetime import UTC, datetime

        from shunt.inspect.inference import data as idata

        return idata.SessionRow(
            session_id=sid,
            timestamp=datetime(2026, 8, 29, tzinfo=UTC),
            model_chosen=model,
            cost=0.5,
            cost_known=True,
            stratum=stratum,
            selection_rule_used=None,
            selection_propensity=None,
            hold_reason=None,
            rung=None,
            undeliverable=False,
            tier2_success=True,
        )

    def test_each_half_states_its_own_panel_a_limitation(self) -> None:
        # The two halves plot DIFFERENT quantities on panel A — a list price here, a measured
        # bill there — so one hardcoded sentence is necessarily false on one canvas. This is
        # the guard that keeps each half's limitation its own.
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing.figures import model_grid as adapter
        from shunt.inspect.inference import data as idata

        bench = adapter.build(SimpleNamespace(raw=config.load_results()))  # type: ignore[arg-type]
        assert bench is not None
        assert "LIST PRICE" in bench.x_limitation and "measured bill" in bench.x_limitation

        live = idata.model_grid([self._session("kimi-k3", "a", "live")])  # type: ignore[list-item]
        assert "MEASURED BILL" in live.x_limitation
        assert "LIST PRICE" not in live.x_limitation

    def test_a_replayed_row_cannot_read_as_a_live_measurement(self) -> None:
        # THE STRATUM DISCLOSURE. A seed-only store draws replayed benchmark sessions; the
        # canvas must say so where it states its subject set, not only in a limitations
        # paragraph. With zero live sessions the subtitle must report zero.
        from shunt.inspect.inference import data as idata

        seeded_only = idata.model_grid(
            [
                self._session("kimi-k3", "bench:a", "seeded"),  # type: ignore[list-item]
                self._session("kimi-k3", "bench:b", "seeded"),  # type: ignore[list-item]
            ]
        )
        assert "0 live" in seeded_only.source
        assert "replayed from the benchmark corpus" in seeded_only.source
        assert "0 of 1 models drawn carry any live session" in seeded_only.source

        mixed = idata.model_grid(
            [
                self._session("kimi-k3", "bench:a", "seeded"),  # type: ignore[list-item]
                self._session("kimi-k3", "c", "live"),  # type: ignore[list-item]
            ]
        )
        assert "1 live and 1 replayed" in mixed.source

    def test_a_model_with_an_unpriced_session_is_left_off_the_axis(self) -> None:
        from datetime import UTC, datetime

        from shunt.inspect.inference import data as idata

        def session(model: str, sid: str, *, known: bool) -> idata.SessionRow:
            return idata.SessionRow(
                session_id=sid,
                timestamp=datetime(2026, 8, 29, tzinfo=UTC),
                model_chosen=model,
                cost=0.5,
                cost_known=known,
                stratum="live",
                selection_rule_used=None,
                selection_propensity=None,
                hold_reason=None,
                rung=None,
                undeliverable=False,
                tier2_success=True,
            )

        rows = [
            session("deepseek-v4-flash", "a", known=True),
            session("deepseek-v4-flash", "b", known=True),
            session("kimi-k3", "c", known=True),
            session("kimi-k3", "d", known=False),
        ]
        grid = idata.model_grid(rows)
        by_name = {r.name: r for r in grid.rows}
        # Fully priced: on the axis at its measured mean.
        assert by_name["deepseek-v4-flash"].x == pytest.approx(0.5)
        # Partly priced: drawn nowhere rather than at a partial total that reads as cheaper.
        assert "kimi-k3" not in by_name
        assert "measured" in grid.x_label
