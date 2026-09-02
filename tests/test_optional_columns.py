"""Optional results.csv columns, the replicate key, and the missing-is-never-zero contract."""

# The four invariants this file exists to hold, each of which would fail silently otherwise:
#   * two concurrent mode="replicate" writers get DISTINCT replicate indices;
#   * a same-anchors re-run under mode="supersede" REFUSES rather than clobbering a paid row;
#   * a blank optional column RAISES when aggregated, instead of averaging in as 0;
#   * a legacy blank `rep` and a freshly written "0" are ONE key, not a duplicate.

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any, Final

import pytest

from benchmark import config
from benchmark.routing import authenticity, integrity, validate
from benchmark.runner import run_matrix

_ANCHORS: Final[dict[str, str]] = {
    "version_hash": "vh",
    "model_version": "mv",
    "arm_hash": "ah",
    "image_digest": "sha256:" + "a" * 64,
    "step_limit": "40",
    "sampling_hash": "sh",
    "prompt_hash": "ph",
}


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
        "cost_limit": "1.0",
        "scaffold_version": "1.0",
        **_ANCHORS,
    }
    row.update(overrides)
    return row


def _written(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class TestSchema:
    def test_optional_columns_follow_prompt_hash(self):
        fields = integrity.RESULTS_FIELDS
        tail = fields[fields.index("prompt_hash") + 1 :]
        assert tail == (integrity.REPLICATE_COLUMN, *integrity.OPTIONAL_COLUMNS)

    def test_no_optional_column_is_a_cache_column(self):
        # Every member of CACHE_COLUMNS is READ as a cache/staleness field; an optional
        # column is not one, and adding it there would give a blank staleness meaning.
        overlap = set(integrity.OPTIONAL_COLUMNS) | {integrity.REPLICATE_COLUMN}
        assert overlap.isdisjoint(integrity.CACHE_COLUMNS)

    def test_no_optional_column_is_write_required(self):
        # The fail-closed guard: adding one of these to either tuple would ERROR on all 1265
        # committed rows and abort every future write.
        forbidden = set(integrity.OPTIONAL_COLUMNS) | {integrity.REPLICATE_COLUMN}
        assert forbidden.isdisjoint(validate._NUMERIC_FIELDS)
        assert forbidden.isdisjoint(validate._REQUIRED_COLLECTION_FIELDS)

    def test_staleness_anchors_are_the_declared_set(self):
        assert integrity.STALENESS_ANCHORS == (
            "version_hash",
            "model_version",
            "arm_hash",
            "image_digest",
            "step_limit",
            "sampling_hash",
            "prompt_hash",
        )
        assert set(run_matrix._ANCHOR_CHECKS) == set(integrity.STALENESS_ANCHORS)


class TestReplicateKey:
    def test_legacy_blank_and_fresh_zero_are_one_key(self):
        # THE MIGRATION BUG this normalisation exists to prevent: 1265 committed rows carry a
        # blank rep, a fresh first observation carries "0", and keying on the raw spelling
        # would make one cell look like two rows.
        legacy = _row()
        legacy.pop(integrity.REPLICATE_COLUMN, None)
        fresh = _row(**{integrity.REPLICATE_COLUMN: "0"})
        assert run_matrix._row_key(legacy) == run_matrix._row_key(fresh)
        assert integrity.rep_index(legacy) == integrity.rep_index(fresh) == 0

    def test_duplicate_key_check_does_not_flag_the_pair(self):
        legacy = {k: str(v) for k, v in _row().items()}
        legacy[integrity.REPLICATE_COLUMN] = ""
        fresh = dict(legacy, **{integrity.REPLICATE_COLUMN: "0"})
        # One CELL observed once, spelled two ways: dup-detection must see one key, and the
        # two rows must therefore collide as a normal duplicate rather than as two cells.
        assert authenticity._key(legacy, {}) == authenticity._key(fresh, {})

    def test_a_real_replicate_is_not_a_duplicate(self):
        base = {k: str(v) for k, v in _row().items()}
        rep0 = dict(base, **{integrity.REPLICATE_COLUMN: "0"})
        rep1 = dict(base, **{integrity.REPLICATE_COLUMN: "1"})
        assert authenticity.check_duplicate_keys([rep0, rep1], {}) == []


class TestMergeModes:
    def test_supersede_refuses_a_same_anchors_rerun(self, tmp_path: Path):
        # The refusal IS the mechanism protecting paid observations: intent cannot be inferred
        # from content, because a re-run always differs on computed_at/real_cost.
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        rerun = _row(computed_at="2026-08-29T00:00:00+00:00", real_cost=0.02, cost=0.02)
        with pytest.raises(validate.DataIntegrityError) as excinfo:
            run_matrix.merge_rows([rerun], path)
        assert excinfo.value.violations[0].code == validate.REPLICATE_MISKEYED
        assert "mode='replicate'" in str(excinfo.value)
        assert _written(path)[0]["real_cost"] == "0.01"  # the paid row survived untouched

    def test_supersede_still_overwrites_when_an_anchor_moved(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        run_matrix.merge_rows([_row(prompt_hash="ph2", real_cost=0.02, cost=0.02)], path)
        rows = _written(path)
        assert len(rows) == 1
        assert rows[0]["prompt_hash"] == "ph2"

    def test_supersede_still_overwrites_a_zero_work_row(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        zero = _row(calls=0, real_cost=0.0, cost=0.0)
        zero["pass"] = False
        zero["stop_reason"] = "abandoned"
        zero["timeout_flag"] = True
        run_matrix.merge_rows([zero], path)
        run_matrix.merge_rows([_row()], path)
        rows = _written(path)
        assert len(rows) == 1 and rows[0]["calls"] == "1"

    def test_anchors_differ_grandfathers_an_empty_stored_anchor(self):
        # Mirrors _anchor_stale: a legacy row that predates a column proves nothing about it,
        # so its blank must not license an overwrite.
        old = _row(prompt_hash="")
        assert not run_matrix._anchors_differ(old, _row(prompt_hash="ph"))
        assert run_matrix._anchors_differ(_row(prompt_hash="ph"), _row(prompt_hash="ph2"))

    def test_replicate_appends_instead_of_superseding(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        run_matrix.merge_rows([_row(real_cost=0.02, cost=0.02)], path, mode="replicate")
        rows = _written(path)
        assert sorted(r[integrity.REPLICATE_COLUMN] for r in rows) == ["0", "1"]

    def test_replicate_writes_no_history(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        run_matrix.merge_rows([_row(real_cost=0.02)], path, mode="replicate")
        assert not run_matrix._history_path(path).exists()

    def test_concurrent_replicate_writers_get_distinct_reps(self, tmp_path: Path):
        # Two writers sharing the caller's lock must never both read max(rep)==0 and both
        # claim rep 1 — the index is assigned INSIDE the locked read-modify-write.
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        lock = threading.Lock()
        errors: list[BaseException] = []

        def write(cost: float) -> None:
            try:
                run_matrix._checkpoint_row(
                    _row(real_cost=cost, cost=cost), path, lock, mode="replicate"
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(c,)) for c in (0.02, 0.03)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        reps = sorted(int(r[integrity.REPLICATE_COLUMN]) for r in _written(path))
        assert reps == [0, 1, 2]

    def test_write_converges_blank_rep_to_zero(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        assert _written(path)[0][integrity.REPLICATE_COLUMN] == "0"


class TestMissingIsNeverZero:
    def test_optional_num_returns_missing_for_a_blank(self):
        assert config._optional_num({"ttft_s": ""}, "ttft_s") is config.MISSING
        assert config._optional_num({}, "ttft_s") is config.MISSING
        assert config._optional_num({"ttft_s": "1.5"}, "ttft_s") == 1.5
        assert config._optional_num({"retry_count": "2"}, "retry_count", int) == 2

    def test_a_blank_optional_column_raises_rather_than_aggregating_as_zero(self):
        rows = [{"ttft_s": "1.0"}, {"ttft_s": ""}]
        values = [config._optional_num(r, "ttft_s") for r in rows]
        # The anti-pattern this replaces would have produced mean([1.0, 0.0]) == 0.5 and
        # published it as a measurement.
        with pytest.raises(validate.DataIntegrityError) as excinfo:
            validate.require_measured(values, "ttft_s", "a-consumer")
        assert excinfo.value.violations[0].code == validate.MISSING_MEASUREMENT

    def test_require_measured_passes_a_fully_populated_column(self):
        assert validate.require_measured([1.0, 2], "ttft_s", "a-consumer") == [1.0, 2.0]


class TestWriteSideGuard:
    def test_blank_optional_columns_are_legal(self):
        assert validate._check_optional_measurements(_row()) == []

    def test_a_present_but_malformed_optional_value_is_an_error(self):
        for bad in ("-1", "abc", "nan", "inf"):
            found = validate._check_optional_measurements(_row(ttft_s=bad))
            assert [v.code for v in found] == [validate.MALFORMED_OPTIONAL], bad

    def test_a_negative_or_non_integer_rep_is_an_error(self):
        for bad in ("-1", "1.5", "x"):
            found = validate._check_optional_measurements(_row(**{integrity.REPLICATE_COLUMN: bad}))
            assert [v.code for v in found] == [validate.MALFORMED_OPTIONAL], bad

    def test_enforce_row_still_accepts_a_legacy_shaped_row(self):
        # The whole point of presence-tolerance: 1265 committed rows carry none of these.
        validate.enforce_row(_row(), {})


class TestReplicateAggregation:
    def test_reduce_reps_is_the_identity_at_depth_one(self):
        row = {k: str(v) for k, v in _row().items()}
        canonical, n_reps, pass_rate = config.reduce_reps([row])
        assert canonical is row
        assert (n_reps, pass_rate) == (1, 1.0)

    def test_rep_zero_is_canonical_and_the_scorer_never_sees_a_replicate(self):
        base = {k: str(v) for k, v in _row().items()}
        rep0 = dict(base, **{integrity.REPLICATE_COLUMN: "0"})
        rep1 = dict(base, **{integrity.REPLICATE_COLUMN: "1"})
        rep1["pass"] = "False"
        canonical, n_reps, pass_rate = config.reduce_reps([rep1, rep0])
        assert canonical is rep0  # order-independent: rep 0 wins whatever the file order
        assert n_reps == 2
        assert pass_rate == 0.5  # audit-only — no metric reads it

    def test_reduce_reps_falls_back_to_row_precedence_without_a_rep_zero(self):
        base = {k: str(v) for k, v in _row().items()}
        rep1 = dict(base, **{integrity.REPLICATE_COLUMN: "1"})
        rep2 = dict(base, **{integrity.REPLICATE_COLUMN: "2"})
        rep2["computed_at"] = "2026-08-29T00:00:00+00:00"
        canonical, _, _ = config.reduce_reps([rep1, rep2])
        assert canonical is rep2


class TestRawReaderPolicies:
    def test_rep_zero_and_all_rows_differ_only_on_replicates(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        run_matrix.merge_rows([_row(real_cost=0.02, cost=0.02)], path, mode="replicate")
        assert len(integrity.all_rows(path)) == 2
        assert len(integrity.rep_zero_rows(path)) == 1

    def test_spend_readers_sum_every_rep(self, tmp_path: Path):
        from benchmark import cost_reconcile

        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row()], path)
        run_matrix.merge_rows([_row(real_cost=0.02, cost=0.02)], path, mode="replicate")
        # A replicate really was billed, so the invoice-facing reader must see both rows.
        assert len(cost_reconcile.load_rows(path)) == 2


class TestReplicateDepthConfig:
    def test_shipped_default_is_disabled_and_depth_one(self):
        config.load("benchmark/benchmark.yaml")
        assert config.replicate_enabled() is False
        assert config.replicate_depth("deepseek-v4-flash") == 1

    def test_per_model_override_wins_when_enabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            config,
            "replicate_config",
            lambda: {"enabled": True, "default_r": 1, "by_model": {"a": 3}},
        )
        assert config.replicate_depth("a") == 3
        assert config.replicate_depth("b") == 1

    def test_classify_cells_emits_no_replicates_at_depth_one(self):
        config.load("benchmark/benchmark.yaml")
        status = run_matrix.classify_cells(["c1"], ["m"], {}, {"c1": "h"}, {"m": "v"})
        assert status.missing == [("c1", "m", "default")]
        assert status.replicate == []

    def test_classify_cells_emits_r_minus_one_replicates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "replicate_depth", lambda model: 3)
        status = run_matrix.classify_cells(["c1"], ["m"], {}, {"c1": "h"}, {"m": "v"})
        assert status.replicate == [("c1", "m", "default")] * 2
        assert status.to_run == [("c1", "m", "default")]  # replicates are NOT in to_run


class TestColumnCoverageReport:
    def test_it_imports_the_column_list_rather_than_restating_it(self):
        from benchmark import column_coverage

        classes = column_coverage.column_classes()
        assert set(classes) == {integrity.REPLICATE_COLUMN, *integrity.OPTIONAL_COLUMNS}

    def test_it_reports_a_blank_optional_column_as_missing(self, tmp_path: Path):
        from benchmark import column_coverage

        path = tmp_path / "results.csv"
        run_matrix.merge_rows([_row(ttft_s="1.5")], path)
        run_matrix.merge_rows([_row(challenge_id="repo__task-2")], path)
        by_column = {r.column: r for r in column_coverage.run(path, tmp_path / "reports")}
        assert (by_column["ttft_s"].n_populated, by_column["ttft_s"].n_blank) == (1, 1)
        assert by_column["ttft_s"].populated_models == ("m",)
        assert by_column[integrity.REPLICATE_COLUMN].n_blank == 0  # rep always has a value
        assert (tmp_path / "reports" / column_coverage.JSON_NAME).exists()
