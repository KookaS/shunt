"""Adversarial regression tests for the benchmark result cache — offline and deterministic."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.routing import integrity
from benchmark.runner import infer, run_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRICING: Final[dict[str, dict[str, float]]] = {"m1": {"input": 0.14, "output": 0.28}}


def _outcome(**over: object) -> dict:
    base = {
        "pass": True,
        "in_tok": 100,
        "out_tok": 50,
        "calls": 3,
        "real_cost": 0.0029780744,
        "timeout_flag": False,
        "image_digest": "sha256:abc",
        "computed_at": "2026-07-15T18:50:14+00:00",
    }
    base.update(over)
    return base


def _typed_row(cid: str = "c1", model: str = "m1", **over: object) -> dict:
    """A row exactly as ``_build_row`` emits it — typed bool/float/int values."""
    row = run_matrix._build_row(
        cid,
        model,
        _outcome(**over),
        {cid: "a" * 64},
        {model: "v1"},
        _PRICING,
        {cid: "sha256:abc"},
    )
    return row


# ---------------------------------------------------------------------------
# End-to-end round trip: build -> write -> read -> classify == PRESENT.
# This is the "run twice => 0 recompute" guarantee against the REAL schema.
# ---------------------------------------------------------------------------


class TestRoundTripPresent:
    def test_written_row_reads_back_as_present(self, tmp_path):
        results = tmp_path / "results.csv"
        hashes = {"c1": "a" * 64}
        versions = {"m1": "v1"}
        digests = {"c1": "sha256:abc"}
        run_matrix.merge_rows([_typed_row()], results, tmp_path / "hist.csv")
        cache = config.load_results(results)
        status = run_matrix.classify_cells(["c1"], ["m1"], cache, hashes, versions, digests)
        assert status.present == 1
        assert status.to_run == []

    def test_round_trip_present_with_image_axis_off(self, tmp_path):
        # Default simulated pass has digests=None; the written row must still be PRESENT.
        results = tmp_path / "results.csv"
        run_matrix.merge_rows([_typed_row()], results, tmp_path / "hist.csv")
        cache = config.load_results(results)
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "a" * 64}, {"m1": "v1"}, None
        )
        assert status.present == 1

    def test_float_real_cost_survives_round_trip_without_restaling(self, tmp_path):
        # A messy float ("0.0029780744") written by DictWriter must read back and
        # classify PRESENT — pass/version/digest are the only anchors, cost is not.
        results = tmp_path / "results.csv"
        run_matrix.merge_rows([_typed_row(real_cost=0.005347154400000002)], results, None)
        cache = config.load_results(results)
        cell = cache["c1"]["m1"]
        assert cell["real_cost"] == pytest.approx(0.005347154400000002)
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "a" * 64}, {"m1": "v1"}, {"c1": "sha256:abc"}
        )
        assert status.present == 1

    def test_monkeypatched_default_path_round_trips(self, tmp_path, monkeypatch):
        # Exercise the real default-path resolution used by the loop.
        results = tmp_path / "routing" / "results.csv"
        monkeypatch.setattr(config, "results_csv_path", lambda: results)
        run_matrix.merge_rows([_typed_row()], config.results_csv_path(), tmp_path / "h.csv")
        cache = config.load_results()
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "a" * 64}, {"m1": "v1"}, {"c1": "sha256:abc"}
        )
        assert status.present == 1


# ---------------------------------------------------------------------------
# The real committed cache: deepseek cached (0 recompute), qwen missing.
# This is the exact second-run scenario the task describes, pinned end to end.
# ---------------------------------------------------------------------------


class TestRealCacheSecondRun:
    def test_committed_cells_present_and_absent_model_missing(self):
        # Content-independent: whatever models the committed cache actually holds
        # must re-classify PRESENT (0 recompute), and a never-run model's cells are
        # all missing. Does NOT hardcode WHICH models are populated, so it survives
        # future live runs adding models to results.csv.
        config.load()
        hashes = integrity.all_hashes()
        versions = integrity.model_versions()
        cache = config.load_results()
        tasks = sorted(hashes.keys())
        assert tasks, "expected materialised swebench specs"

        present_models = sorted({m for cell in cache.values() for m in cell})
        assert present_models, "expected committed results in the cache"
        # The set of cells that ACTUALLY have a committed row (a model may be
        # partially covered by design — e.g. claude-opus-4-6 runs one challenge).
        existing = {(cid, m) for cid, cell in cache.items() for m in cell}
        st_present = run_matrix.classify_cells(tasks, present_models, cache, hashes, versions)
        # Run-twice-zero guarantee: no COMMITTED cell recomputes. Nothing is stale,
        # and every to_run cell is one that genuinely has no committed row (not a
        # re-run of an existing cell). This survives partial coverage.
        assert st_present.stale == [], "committed cells must not go stale (0 recompute)"
        assert all((cid, m) not in existing for cid, m in st_present.to_run), (
            "no committed cell may re-classify as missing/stale"
        )

        # A synthetic never-run model: every cell missing, none stale.
        absent = "no-such-model-xyz"
        st_absent = run_matrix.classify_cells(tasks, [absent], cache, hashes, {absent: "v"})
        assert {m for _, m in st_absent.missing} == {absent}
        assert st_absent.stale == []


# ---------------------------------------------------------------------------
# merge_rows history archival: typed rows (bool/float) vs CSV-string existing.
# ---------------------------------------------------------------------------


class TestMergeHistoryArchival:
    def test_identical_typed_remerge_does_not_archive(self, tmp_path):
        # First merge writes typed row -> CSV strings. Second identical typed merge
        # must NOT archive: str(True)=="True", str(0.00297..)==stored string.
        results = tmp_path / "results.csv"
        history = tmp_path / "hist.csv"
        run_matrix.merge_rows([_typed_row()], results, history)
        run_matrix.merge_rows([_typed_row()], results, history)
        assert not history.exists(), "identical re-run wrongly archived history"

    def test_changed_pass_bool_archives_old_row(self, tmp_path):
        results = tmp_path / "results.csv"
        history = tmp_path / "hist.csv"
        run_matrix.merge_rows([_typed_row(**{"pass": True})], results, history)
        run_matrix.merge_rows([_typed_row(**{"pass": False})], results, history)
        assert history.exists()
        assert "superseded_at" in history.read_text().splitlines()[0]
        assert config.load_results(results)["c1"]["m1"]["pass"] is False

    def test_changed_hash_archives_and_updates_current(self, tmp_path):
        results = tmp_path / "results.csv"
        history = tmp_path / "hist.csv"
        r1 = run_matrix._build_row("c1", "m1", _outcome(), {"c1": "h1"}, {"m1": "v1"}, _PRICING)
        r2 = run_matrix._build_row("c1", "m1", _outcome(), {"c1": "h2"}, {"m1": "v1"}, _PRICING)
        run_matrix.merge_rows([r1], results, history)
        run_matrix.merge_rows([r2], results, history)
        assert "h1" in history.read_text()
        assert config.load_results(results)["c1"]["m1"]["version_hash"] == "h2"

    def test_second_model_added_leaves_first_present_and_unarchived(self, tmp_path):
        # Adding qwen must not touch the deepseek row or write history.
        results = tmp_path / "results.csv"
        history = tmp_path / "hist.csv"
        run_matrix.merge_rows([_typed_row(model="deepseek-v4-flash")], results, history)
        run_matrix.merge_rows([_typed_row(model="qwen3.7-plus")], results, history)
        assert not history.exists()
        cache = config.load_results(results)
        assert set(cache["c1"].keys()) == {"deepseek-v4-flash", "qwen3.7-plus"}


# ---------------------------------------------------------------------------
# classify_cells edge cases: partial coverage, empty cache, unknown model.
# ---------------------------------------------------------------------------


class TestClassifyEdgeCases:
    def test_empty_cache_all_missing(self):
        st = run_matrix.classify_cells(
            ["c1", "c2"], ["m1"], {}, {"c1": "h", "c2": "h"}, {"m1": "v"}
        )
        assert len(st.missing) == 2
        assert st.present == 0

    def test_unknown_model_not_in_versions_marks_present_row_stale(self):
        # A cached row whose model has no declared version (versions.get -> None)
        # must NOT silently pass: stored "v1" != None => stale, forcing recompute.
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "v1", "image_digest": ""}}}
        st = run_matrix.classify_cells(["c1"], ["m1"], cache, {"c1": "h1"}, {})
        assert st.stale == [("c1", "m1")]

    def test_partial_coverage_mix(self):
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "v1", "image_digest": ""}}}
        st = run_matrix.classify_cells(
            ["c1", "c2"], ["m1", "m2"], cache, {"c1": "h1", "c2": "h2"}, {"m1": "v1", "m2": "v2"}
        )
        assert st.present == 1
        assert set(st.missing) == {("c1", "m2"), ("c2", "m1"), ("c2", "m2")}


# ---------------------------------------------------------------------------
# Live-inference wiring — pure logic only (no Docker, no keys, no network).
# ---------------------------------------------------------------------------


class TestLitellmRouting:
    def test_direct_returns_route_and_empty_kwargs(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "load_pricing",
            lambda *a, **k: {"m": {"access_via": "direct", "route": "deepseek/x"}},
        )
        assert infer.litellm_model_target("m") == ("deepseek/x", {})

    def test_requesty_carries_base_url_and_key(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "load_pricing",
            lambda *a, **k: {"m": {"access_via": "requesty", "route": "openai/p/x"}},
        )
        monkeypatch.setenv("REQUESTY_API_KEY", "secret")
        route, kwargs = infer.litellm_model_target("m")
        assert route == "openai/p/x"
        assert kwargs["api_base"] == infer._REQUESTY_BASE
        assert kwargs["api_key"] == "secret"

    def test_requesty_without_key_raises(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "load_pricing",
            lambda *a, **k: {"m": {"access_via": "requesty", "route": "openai/p/x"}},
        )
        monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
        with pytest.raises(infer.MissingApiKeysError):
            infer.litellm_model_target("m")

    def test_missing_model_raises_keyerror(self, monkeypatch):
        monkeypatch.setattr(config, "load_pricing", lambda *a, **k: {})
        with pytest.raises(KeyError):
            infer.litellm_model_target("nope")

    def test_unknown_access_via_raises_notimplemented(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "load_pricing",
            lambda *a, **k: {"m": {"access_via": "openrouter", "route": "foo/bar"}},
        )
        with pytest.raises(NotImplementedError):
            infer.litellm_model_target("m")

    def test_missing_route_raises(self, monkeypatch):
        # No explicit route => KeyError (no guessing: requesty serving-provider
        # prefixes don't match the origin provider, so a derived route would be wrong).
        monkeypatch.setattr(
            config,
            "load_pricing",
            lambda *a, **k: {
                "m": {"access_via": "direct", "provider": "deepseek", "version": "vv"}
            },
        )
        with pytest.raises(KeyError):
            infer.litellm_model_target("m")


def _usage_msg(prompt: object, completion: object, cost: float) -> dict:
    return {
        "extra": {
            "response": {"usage": {"prompt_tokens": prompt, "completion_tokens": completion}},
            "cost": cost,
        }
    }


class TestSumUsage:
    def test_empty(self):
        assert infer._sum_usage([]) == (0, 0, 0, 0.0)

    def test_message_without_extra(self):
        assert infer._sum_usage([{"role": "assistant"}]) == (0, 0, 0, 0.0)

    def test_extra_without_response_not_counted(self):
        # No response => not a model call: neither counted nor cost-charged.
        assert infer._sum_usage([{"extra": {"cost": 5.0}}]) == (0, 0, 0, 0.0)

    def test_empty_dict_response_is_skipped(self):
        # `response = extra.get("response")` then `if not response: continue`, so an
        # EMPTY-dict response is falsy and skipped (not counted as a call). Documents
        # the guard: only a truthy (populated) response counts.
        assert infer._sum_usage([{"extra": {"response": {}, "cost": 1.5}}]) == (0, 0, 0, 0.0)

    def test_truthy_response_without_usage_counts_call_and_cost(self):
        msg = {"extra": {"response": {"id": "x"}, "cost": 1.5}}
        assert infer._sum_usage([msg]) == (0, 0, 1, 1.5)

    def test_usage_with_none_token_fields(self):
        assert infer._sum_usage([_usage_msg(None, None, 0.0)]) == (0, 0, 1, 0.0)

    def test_accumulates_multiple_messages(self):
        msgs = [
            _usage_msg(10, 5, 0.02),
            {"role": "user"},
            _usage_msg(3, 1, 0.01),
        ]
        assert infer._sum_usage(msgs) == (13, 6, 2, pytest.approx(0.03))


# ---------------------------------------------------------------------------
# REGRESSION GUARD (was a confirmed bug, now fixed): the superseded-history file
# must land in routing/artifacts/ (gitignored), NOT benchmark/artifacts/ (which is
# not gitignored). `_history_path` previously used parent.parent and leaked the
# archived (paid) rows one `git add -A` from the public repo.
# ---------------------------------------------------------------------------


class TestHistoryPathHygiene:
    def test_history_lands_beside_results_in_gitignored_artifacts(self):
        results = Path("/repo/benchmark/routing/results.csv")
        hist = run_matrix._history_path(results)
        # Per the docstring: routing/artifacts/, i.e. results.parent/artifacts.
        assert hist.parent == results.parent / "artifacts"

    def test_history_path_is_named_results_history(self):
        # Non-xfail companion: pins the filename regardless of the dir bug.
        hist = run_matrix._history_path(Path("/repo/benchmark/routing/results.csv"))
        assert hist.name == "results_history.csv"
