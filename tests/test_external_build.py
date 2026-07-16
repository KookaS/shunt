"""Tests for the reproducible external-prior extraction (build_external_prior).

Synthetic ``results.json`` fixtures — no clone, no network. Pins the honest
denominator (attempted-but-failed counts) and deterministic output.
"""

from __future__ import annotations

import json

from benchmark.runner import build_external_prior as bx


def _write_submission(root, subset, name, results):
    d = root / "evaluation" / subset / name / "results"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(results))


class TestSubmissionAttempts:
    def test_attempted_includes_failed_excludes_no_generation(self):
        resolved, attempted = bx._submission_attempts(
            {
                "resolved": ["a"],
                "applied": ["a", "b"],  # b = generated-but-failed
                "no_apply": ["c"],  # attempted, patch didn't apply = failure
                "no_generation": ["d"],  # NOT attempted
            }
        )
        assert resolved == {"a"}
        assert attempted == {"a", "b", "c"}  # d excluded
        assert "d" not in attempted

    def test_resolved_always_counts_as_attempted(self):
        resolved, attempted = bx._submission_attempts({"resolved": ["x"]})
        assert resolved == {"x"} and attempted == {"x"}


class TestBuildRows:
    def _table(self):
        # instance i1: cheap solves, frontier fails; i2: everyone fails
        return {
            "i1": {"deepseek_sub": True, "opus_sub": False, "agentX": True},
            "i2": {"deepseek_sub": False, "opus_sub": False},
        }

    def test_denominator_counts_failures(self):
        rows = {r["instance_id"]: r for r in bx.build_rows(self._table())}
        # i1: 3 attempts, 2 solved -> 0.6667 (the failed opus_sub IS counted)
        assert rows["i1"]["n_sub"] == 3
        assert rows["i1"]["n_solved"] == 2
        assert rows["i1"]["p_solve"] == 0.6667

    def test_cohort_split_and_gap(self):
        rows = {r["instance_id"]: r for r in bx.build_rows(self._table())}
        # deepseek -> cheap, opus -> frontier, agentX -> unknown (uncohorted)
        assert rows["i1"]["p_cheap"] == 1.0  # deepseek solved
        assert rows["i1"]["p_frontier"] == 0.0  # opus failed
        assert rows["i1"]["gap"] == -1.0  # p_frontier - p_cheap
        assert rows["i1"]["n_cheap"] == 1 and rows["i1"]["n_frontier"] == 1

    def test_uncohorted_blank_not_zero(self):
        # i2 has no cohort with data for... actually deepseek+opus present; make a
        # purely-agent instance to prove blanks stay blank (not 0.0).
        table = {"i3": {"agentA": False, "agentB": True}}
        rows = {r["instance_id"]: r for r in bx.build_rows(table)}
        assert rows["i3"]["p_cheap"] == "" and rows["i3"]["p_frontier"] == ""
        assert rows["i3"]["gap"] == ""
        assert rows["i3"]["p_solve"] == 0.5  # overall still computed

    def test_rows_sorted_deterministic(self, tmp_path):
        _write_submission(
            tmp_path, "verified", "deepseek_a", {"resolved": ["z1"], "applied": ["z1", "a1"]}
        )
        _write_submission(
            tmp_path, "verified", "opus_b", {"resolved": ["a1"], "applied": ["a1", "z1"]}
        )
        t1 = bx.load_resolves(tmp_path, "verified")
        r1 = [r["instance_id"] for r in bx.build_rows(t1)]
        r2 = [r["instance_id"] for r in bx.build_rows(bx.load_resolves(tmp_path, "verified"))]
        assert r1 == sorted(r1), "instances must be emitted in sorted order"
        assert r1 == r2, "extraction must be deterministic"


class TestWriteCsv:
    def test_roundtrip_schema(self, tmp_path):
        table = {"i1": {"deepseek_x": True, "opus_y": False}}
        rows = bx.build_rows(table)
        out = tmp_path / "ext.csv"
        bx.write_csv(rows, out)
        text = out.read_text()
        assert text.splitlines()[0] == ",".join(bx.FIELDS)
        # Two independent writes are byte-identical (determinism end to end).
        out2 = tmp_path / "ext2.csv"
        bx.write_csv(bx.build_rows(table), out2)
        assert out.read_text() == out2.read_text()
