"""Latency instrumentation: what a live run measures, labels, and refuses to invent."""

# The four latency-shaped columns (`wall_clock_s`, `ttft_s`, `latency_per_call_s`,
# `provider_latency_source`) were declared in the schema before anything wrote them. These tests
# hold the collection contract now that the write path exists:
#
#   * a measured cell round-trips through the writer with its numbers and BOTH labels;
#   * a blank stays blank forever — through the writer, through the reader, and through
#     aggregation, where it RAISES instead of averaging in as zero;
#   * `ttft_s` is never written, because the scaffold does not stream and time-to-full-response
#     is a different quantity;
#   * a timing that cannot be labelled with its serving mode is dropped, not written;
#   * the 1265-row legacy corpus, which carries none of these columns, still loads and
#     validates clean — the fail-closed hazard this migration had to avoid.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Final

import pytest

from benchmark import config
from benchmark.routing import integrity, validate
from benchmark.runner import infer, run_matrix, scaffold_model

# The header of the committed corpus before the optional columns existed: 22 columns, and the
# permanent shape of every row written before this instrumentation landed.
_LEGACY_HEADER: Final[tuple[str, ...]] = (
    "challenge_id",
    "model",
    "reasoning",
    "pass",
    "cost",
    "in_tok",
    "out_tok",
    "calls",
    "version_hash",
    "model_version",
    "arm_hash",
    "real_cost",
    "estimated_cost",
    "timeout_flag",
    "image_digest",
    "computed_at",
    "stop_reason",
    "step_limit",
    "cost_limit",
    "scaffold_version",
    "sampling_hash",
    "prompt_hash",
)

_LEGACY_ROW: Final[tuple[str, ...]] = (
    "astropy__astropy-12907",
    "deepseek-v4-flash",
    "high",
    "True",
    "0.0034",
    "154633",
    "4613",
    "15",
    "vh",
    "deepseek-v4-flash",
    "ah",
    "0.0034",
    "0.0229",
    "False",
    "sha256:" + "f" * 64,
    "2026-07-25T23:34:22.530471+00:00",
    "",
    "250",
    "3.0",
    "unknown",
    "",
    "",
)


def _row(**overrides: Any) -> dict[str, Any]:
    """A schema-complete results row that clears every write-time invariant."""
    row = {
        "challenge_id": "repo__task-1",
        "model": "m",
        "reasoning": "default",
        "pass": True,
        "cost": 0.01,
        "in_tok": 10,
        "out_tok": 5,
        "calls": 1,
        "real_cost": 0.01,
        "estimated_cost": 0.01,
        "timeout_flag": False,
        "computed_at": "2026-08-28T00:00:00+00:00",
        "stop_reason": "solved",
        "version_hash": "vh",
        "model_version": "mv",
        "arm_hash": "ah",
        "image_digest": "sha256:" + "a" * 64,
        "step_limit": "40",
        "cost_limit": "1.0",
        "scaffold_version": "1.0",
        "sampling_hash": "sh",
        "prompt_hash": "ph",
    }
    row.update(overrides)
    return row


def _patch(**overrides: Any) -> infer.AgentPatch:
    """An `AgentPatch` shaped like one a completed live cell returns."""
    fields: dict[str, Any] = {
        "patch": "diff",
        "in_tok": 10,
        "out_tok": 5,
        "calls": 2,
        "cost": 0.01,
        "wall_clock_s": 12.5,
        "call_latencies_s": (2.0, 4.0),
    }
    fields.update(overrides)
    return infer.AgentPatch(**fields)


@pytest.fixture
def hosted_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """A registry lookup that resolves one hosted model, so a timing can be labelled."""
    monkeypatch.setattr(
        config,
        "load_pricing",
        lambda: {"m": {"provider": "acme", "serving_mode": "hosted", "route": "openai/m"}},
    )
    return "m"


class TestWhatALiveCellMeasures:
    def test_a_measured_cell_carries_both_numbers_and_both_labels(self, hosted_model: str):
        out = infer._latency_provenance(hosted_model, _patch())
        assert out["wall_clock_s"] == "12.500000"
        # The mean of the two REAL per-call measurements, not wall_clock/calls (which would
        # charge in-container tool execution to the model as latency).
        assert out["latency_per_call_s"] == "3.000000"
        assert out["provider_latency_source"] == integrity.LATENCY_SOURCE_CLIENT
        assert out["serving_mode"] == "hosted"
        assert out["provider"] == "acme"

    def test_ttft_is_never_written_because_the_scaffold_does_not_stream(self, hosted_model: str):
        # `LitellmModel._query` calls `litellm.completion` without `stream=True`, so there is
        # no first-token event to time. Substituting time-to-full-response would publish a
        # different quantity under the TTFT name.
        assert "ttft_s" not in infer._latency_provenance(hosted_model, _patch())

    def test_a_resumed_cell_reports_per_call_latency_but_not_a_wall_clock(self, hosted_model: str):
        # Each round trip this process timed is a complete measurement; the CELL's total wall
        # clock is not, because the earlier process's seconds were never recorded.
        out = infer._latency_provenance(hosted_model, _patch(wall_clock_s=None))
        assert "wall_clock_s" not in out
        assert out["latency_per_call_s"] == "3.000000"
        assert out["provider_latency_source"] == integrity.LATENCY_SOURCE_CLIENT

    def test_an_unmeasured_cell_writes_no_number_and_no_latency_source(self, hosted_model: str):
        empty = _patch(wall_clock_s=None, call_latencies_s=())
        assert set(infer._latency_provenance(hosted_model, empty)) == {"provider", "serving_mode"}

    def test_an_unlabellable_timing_is_dropped_rather_than_written(self):
        # A latency whose serving mode is unknown could later be pooled with its opposite —
        # a local batch-1 second against a batched hosted second — and nothing in the data
        # would record the mistake. Refusing to write it is the only safe direction.
        #
        # Against the REAL registry, not an emptied one: `load_pricing() == {}` is a state the
        # live path cannot reach (an unloadable registry raises far earlier), so faking it
        # tested a branch under conditions that never occur. An unknown model name reaches the
        # same branch through the door the live path would actually use.
        registry = config.load_pricing()
        assert registry, "the real registry must be non-empty, or the drop below proves nothing"
        assert "unregistered" not in registry
        assert infer._latency_provenance("unregistered", _patch()) == {}
        # ...and the drop is conditional, not universal: a registered model IS labelled.
        known = next(iter(registry))
        assert infer._latency_provenance(known, _patch())["latency_per_call_s"] == "3.000000"


class TestScaffoldCallTiming:
    """The per-call seam: `EnvKeyLitellmModel._query`, the innermost call boundary there is."""

    @staticmethod
    def _model() -> scaffold_model.EnvKeyLitellmModel:
        """The REAL class through its real constructor — no `__init__` stand-in."""
        # The previous stub overrode `__init__` without `super()`, so the constructor that
        # creates `call_latencies_s` (and the docstring's whole contract about it) was never
        # executed: the tests asserted on a list the test itself had made. The real one needs
        # nothing but a model name — no credential and no network — so there is nothing to fake.
        return scaffold_model.EnvKeyLitellmModel(model_name="acme/local-test")

    def test_a_successful_call_is_timed(self, monkeypatch: pytest.MonkeyPatch):
        model = self._model()
        monkeypatch.setattr(
            scaffold_model.LitellmModel, "_query", lambda self, messages, **kw: "response"
        )
        assert model._query([{"role": "user", "content": "hi"}]) == "response"
        assert len(model.call_latencies_s) == 1
        assert model.call_latencies_s[0] >= 0.0

    def test_a_failed_call_records_no_latency(self, monkeypatch: pytest.MonkeyPatch):
        # A rate-limited or timed-out call has a duration, but it is the duration of a
        # failure. Averaging it into a per-call latency would silently shift the column.
        model = self._model()

        def boom(self: Any, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("429")

        monkeypatch.setattr(scaffold_model.LitellmModel, "_query", boom)
        with pytest.raises(RuntimeError):
            model._query([{"role": "user", "content": "hi"}])
        assert model.call_latencies_s == []


class TestRowAssembly:
    def test_build_row_carries_a_measured_outcome_through(self):
        outcome = {
            "pass": True,
            "in_tok": 10,
            "out_tok": 5,
            "calls": 2,
            "real_cost": 0.01,
            "stop_reason": "solved",
            "wall_clock_s": "12.500000",
            "latency_per_call_s": "3.000000",
            "provider": "acme",
            "serving_mode": "hosted",
            "provider_latency_source": integrity.LATENCY_SOURCE_CLIENT,
        }
        row = run_matrix._build_row(
            "repo__task-1", "m", outcome, {}, {}, {"m": {"input": 1.0, "output": 1.0}}
        )
        assert row["wall_clock_s"] == "12.500000"
        assert row["latency_per_call_s"] == "3.000000"
        assert row["serving_mode"] == "hosted"
        assert row["ttft_s"] == ""  # unmeasured stays blank, never 0

    def test_an_outcome_without_timings_leaves_every_optional_column_blank(self):
        outcome = {"pass": True, "in_tok": 10, "out_tok": 5, "calls": 1, "real_cost": 0.01}
        row = run_matrix._build_row(
            "repo__task-1", "m", outcome, {}, {}, {"m": {"input": 1.0, "output": 1.0}}
        )
        # No fallback of ANY kind — not a configured value, not zero, not a derivation.
        assert all(row[column] == "" for column in integrity.OPTIONAL_COLUMNS)


class TestLabelsAreEnforcedAtWriteTime:
    def test_an_unlabelled_timing_is_refused(self):
        found = validate._check_latency_labels(_row(wall_clock_s="12.5"))
        assert [v.code for v in found] == [validate.MALFORMED_OPTIONAL] * 2

    def test_a_timing_missing_only_its_serving_mode_is_refused(self):
        row = _row(wall_clock_s="12.5", provider_latency_source=integrity.LATENCY_SOURCE_CLIENT)
        found = validate._check_latency_labels(row)
        assert [v.code for v in found] == [validate.MALFORMED_OPTIONAL]
        assert "serving_mode" in found[0].message

    def test_an_out_of_vocabulary_label_is_refused(self):
        for field_name, bad in (("serving_mode", "on-prem"), ("provider_latency_source", "vibes")):
            found = validate._check_latency_labels(_row(**{field_name: bad}))
            assert [v.code for v in found] == [validate.MALFORMED_OPTIONAL], field_name

    def test_a_fully_labelled_timing_passes(self):
        row = _row(
            wall_clock_s="12.5",
            latency_per_call_s="3.0",
            serving_mode="local",
            provider_latency_source=integrity.LATENCY_SOURCE_CLIENT,
        )
        assert validate._check_latency_labels(row) == []
        validate.enforce_row(row, {})

    def test_a_row_with_no_timing_needs_no_labels(self):
        assert validate._check_latency_labels(_row()) == []

    def test_the_latency_columns_are_never_write_required(self):
        # The fail-closed hazard: `enforce_row` runs on EVERY write, so requiring one of these
        # would ERROR on all 1265 committed rows and abort every future write.
        latency = set(validate._LATENCY_COLUMNS) | set(integrity.PROVENANCE_OPTIONAL_COLUMNS)
        assert latency.isdisjoint(validate._NUMERIC_FIELDS)
        assert latency.isdisjoint(validate._REQUIRED_COLLECTION_FIELDS)


class TestBlankIsMissingForever:
    def test_a_measured_row_round_trips_through_the_writer(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows(
            [
                _row(
                    wall_clock_s="12.500000",
                    latency_per_call_s="3.000000",
                    provider="acme",
                    serving_mode="hosted",
                    provider_latency_source=integrity.LATENCY_SOURCE_CLIENT,
                )
            ],
            path,
        )
        written = integrity.all_rows(path)[0]
        assert config._optional_num(written, "wall_clock_s") == 12.5
        assert config._optional_num(written, "latency_per_call_s") == 3.0
        assert written["serving_mode"] == "hosted"
        assert written["provider_latency_source"] == integrity.LATENCY_SOURCE_CLIENT
        # Unmeasured on this cell and unmeasurable at this seam — and still blank on read.
        assert config._optional_num(written, "ttft_s") is config.MISSING

    def test_a_blank_column_raises_rather_than_aggregating_as_zero(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows(
            [
                _row(
                    wall_clock_s="12.5",
                    serving_mode="hosted",
                    provider_latency_source=integrity.LATENCY_SOURCE_CLIENT,
                ),
                _row(challenge_id="repo__task-2"),  # legacy-shaped: no timing
            ],
            path,
        )
        values = [config._optional_num(r, "wall_clock_s") for r in integrity.all_rows(path)]
        # The anti-pattern: mean([12.5, 0.0]) == 6.25, published as a measured mean latency.
        with pytest.raises(validate.DataIntegrityError) as excinfo:
            validate.require_measured(values, "wall_clock_s", "a-latency-consumer")
        assert excinfo.value.violations[0].code == validate.MISSING_MEASUREMENT


class TestLegacyCorpusStillLoads:
    def test_a_22_column_csv_loads_and_validates_clean(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_LEGACY_HEADER)
            writer.writerow(_LEGACY_ROW)
        rows = integrity.all_rows(path)
        assert len(rows) == 1
        assert set(rows[0]) == set(_LEGACY_HEADER)  # the file really is the 22-column shape
        pricing = {"deepseek-v4-flash": {"input": 0.28, "output": 0.42}}
        errors = [
            v
            for v in validate.validate_row(rows[0], pricing)
            if v.severity is validate.Severity.ERROR
        ]
        assert errors == []
        # Every optional column reads as MISSING, and `rep` tautologically as 0.
        assert all(
            config._optional_num(rows[0], column) is config.MISSING
            for column in integrity.MEASUREMENT_OPTIONAL_COLUMNS
        )
        assert integrity.rep_index(rows[0]) == 0

    def test_the_committed_corpus_validates_against_the_widened_schema(self):
        rows = integrity.all_rows(config.results_csv_path())
        assert rows, "the committed corpus must be readable"
        offenders = [r for r in rows if validate._check_latency_labels(r)]
        assert offenders == []
