"""The model grid's encodings: what they mean, and what they refuse to invent."""

from __future__ import annotations

import pytest
import yaml

from shunt.inspect.model_grid import (
    CLUSTER_COLOR,
    CLUSTER_ORDER,
    GridData,
    GridRow,
    _near,
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
        facts = grid_annotations(data, sized=2, hosted=0, local=0).subtitle_facts
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
        notes = grid_annotations(data, sized=0, hosted=0, local=0).notes
        assert any("size UNDISCLOSED" in note for note in notes)
        assert any("fixed reference marker" in note for note in notes)

    def test_an_all_empty_latency_panel_says_so_in_the_limits(self) -> None:
        data = GridData(
            rows=(_row(),),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        limits = grid_annotations(data, sized=1, hosted=0, local=0).limitations
        assert any("Panels C and D are empty" in limit for limit in limits)

    def test_latency_present_drops_the_empty_limit(self) -> None:
        data = GridData(
            rows=(_row(latency_s=(1.0, 2.0)),),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        limits = grid_annotations(data, sized=1, hosted=2, local=0).limitations
        assert not any("Panels C and D are empty" in limit for limit in limits)

    def test_the_n_spread_is_stated_because_the_rows_are_unpaired(self) -> None:
        data = GridData(
            rows=(_row(name="a", n=21, passes=10), _row(name="b", n=198, passes=100)),
            x_label="x",
            price_basis="b",
            source="s",
            x_limitation="x is a list price",
        )
        facts = grid_annotations(data, sized=2, hosted=0, local=0).subtitle_facts
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
            placed.append((anchor, offset))
        assert all(o < 0 for _a, o in placed)
        assert len({o for _a, o in placed}) == len(placed)


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
        ann = grid_annotations(data, sized=2, hosted=0, local=0)
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
        limits = grid_annotations(data, sized=1, hosted=0, local=0).limitations
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
        ann = grid_annotations(data, sized=1, hosted=0, local=0)
        assert not any("DAGGERED" in limit for limit in ann.limitations)
        assert dict(ann.counts)["external"] == 0


class TestBenchmarkAdapter:
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

    def test_the_source_line_names_a_cache_model_the_canvas_does_not_draw(self) -> None:
        # THE CAPTION MUST NOT CLAIM A SWEEP A FILTER NARROWED. The report hands this adapter
        # a cache already scoped to `benchmark.yaml`'s enabled set, so a probe-only collection
        # in `results.csv` never reaches the canvas. That drop is deliberate; a source line
        # reading "the measured default-arm cells of results.csv" while it happened is not.
        from types import SimpleNamespace

        from benchmark import config
        from benchmark.routing.figures import model_grid as adapter

        enabled = set(config.enabled_models())
        cache = config.load_results()
        dropped = sorted({m for per_model in cache.values() for m in per_model} - enabled)
        assert dropped, "the guard is vacuous with no unenabled model in the cache"
        scoped = {
            cid: {m: arms for m, arms in per_model.items() if m in enabled}
            for cid, per_model in cache.items()
        }
        data = adapter.build(SimpleNamespace(raw=scoped))  # type: ignore[arg-type]
        assert data is not None
        assert {row.name for row in data.rows}.isdisjoint(dropped)
        for name in dropped:
            assert name in data.source
        assert "not enabled in benchmark.yaml" in data.source

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
