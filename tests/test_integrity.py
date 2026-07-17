"""Tests for benchmark integrity: hashing determinism, stale/missing detection, checks."""

import json

from benchmark import config
from benchmark.routing import coverage, integrity
from benchmark.routing.strategies.oracle import Oracle
from benchmark.runner import check_integrity, image_version, run_matrix


class TestHashing:
    def test_key_order_independent(self):
        a = {"id": "x", "prompt": "hello", "language": "python"}
        b = {"language": "python", "prompt": "hello", "id": "x"}
        assert integrity.hash_content(a) == integrity.hash_content(b)

    def test_content_change_changes_hash(self):
        a = {"id": "x", "prompt": "hello"}
        b = {"id": "x", "prompt": "world"}
        assert integrity.hash_content(a) != integrity.hash_content(b)

    def test_hash_is_sha256_hex(self):
        h = integrity.hash_content({"id": "x"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_all_hashes_covers_store(self):
        # SWE-bench Verified is the sole source: the store covers the full manifest
        # (count-agnostic — the whole Verified set, >100), each a valid sha256.
        manifest_ids = set(json.loads(config.challenges_path().read_text())["tasks"])
        hashes = integrity.all_hashes()
        assert set(hashes) == manifest_ids
        assert len(hashes) > 100
        for cid, h in hashes.items():
            assert len(h) == 64
            assert integrity.challenge_hash(cid) == h

    def test_missing_challenge_returns_empty(self):
        assert integrity.challenge_hash("does-not-exist") == ""


class TestEstimatedCost:
    def test_matches_known_row(self):
        pricing = {"m": {"input": 0.34, "output": 1.38}}
        cost = integrity.estimated_cost("m", 65928, 1078, pricing)
        assert round(cost, 6) == 0.023903

    def test_unknown_model_is_zero(self):
        assert integrity.estimated_cost("unknown", 100, 100, {}) == 0.0


class TestModelVersions:
    def test_reads_versions_from_pricing(self):
        versions = integrity.model_versions()
        assert versions["kimi-k2.5"] == "kimi-k2.5"
        assert versions["deepseek-v4-flash"] == "deepseek-v4-flash"


def _cache(version_hash="h1", model_version="v1", image_digest="sha256:d1"):
    return {
        "c1": {
            "m1": {
                "version_hash": version_hash,
                "model_version": model_version,
                "image_digest": image_digest,
            }
        },
    }


class TestResultsSchema:
    def test_reasoning_column_follows_model(self):
        fields = integrity.RESULTS_FIELDS
        assert fields[:3] == ("challenge_id", "model", "reasoning")

    def test_header_matches_expected_order(self):
        assert ",".join(integrity.RESULTS_FIELDS) == (
            "challenge_id,model,reasoning,pass,cost,in_tok,out_tok,calls,"
            "version_hash,model_version,real_cost,estimated_cost,timeout_flag,"
            "image_digest,computed_at"
        )

    def test_build_row_defaults_reasoning(self):
        row = run_matrix._build_row("c1", "m1", {}, {"c1": "h"}, {"m1": "v"}, {})
        assert row["reasoning"] == integrity.DEFAULT_REASONING

    def test_cost_falls_back_to_estimate_when_real_cost_unpriceable(self):
        # litellm can't price requesty routes ⇒ real_cost=0. The `cost` column (which
        # every metric/kill-gate reads) must NOT be 0 — it falls back to the listing
        # estimate so requesty models don't score as free.
        pricing = {"m1": {"input": 1.0, "output": 2.0}}  # per 1M
        outcome = {"pass": True, "in_tok": 1_000_000, "out_tok": 1_000_000, "real_cost": 0.0}
        row = run_matrix._build_row("c1", "m1", outcome, {"c1": "h"}, {"m1": "v"}, pricing)
        assert row["real_cost"] == 0.0
        assert row["estimated_cost"] == 3.0
        assert row["cost"] == 3.0  # fell back to the estimate, not 0

    def test_cost_prefers_measured_real_cost_when_present(self):
        pricing = {"m1": {"input": 1.0, "output": 2.0}}
        outcome = {"pass": True, "in_tok": 1_000_000, "out_tok": 1_000_000, "real_cost": 0.5}
        row = run_matrix._build_row("c1", "m1", outcome, {"c1": "h"}, {"m1": "v"}, pricing)
        assert row["cost"] == 0.5  # cache-aware measured cost wins when available


class TestLoadResultsEmpty:
    def test_missing_file_is_empty_dict(self, tmp_path):
        assert config.load_results(tmp_path / "absent.csv") == {}

    def test_header_only_file_is_empty_dict(self, tmp_path):
        p = tmp_path / "results.csv"
        p.write_text(",".join(integrity.RESULTS_FIELDS) + "\n")
        assert config.load_results(p) == {}

    def test_reasoning_reconstructed_with_default(self, tmp_path):
        p = tmp_path / "results.csv"
        header = ",".join(integrity.RESULTS_FIELDS)
        # A row that omits the reasoning value must reconstruct as "default".
        p.write_text(header + "\n" + "c1,m1,,True,0.1,10,2,1,h,v,0.1,0.1,False\n")
        cell = config.load_results(p)["c1"]["m1"]
        assert cell["reasoning"] == "default"
        assert cell["pass"] is True


class TestClassifyCells:
    def test_present_when_hash_and_version_match(self):
        status = run_matrix.classify_cells(["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "v1"})
        assert status.present == 1
        assert not status.to_run

    def test_missing_when_no_row(self):
        status = run_matrix.classify_cells(["c1"], ["m2"], _cache(), {"c1": "h1"}, {"m2": "v1"})
        assert status.missing == [("c1", "m2")]

    def test_stale_on_hash_mismatch(self):
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "DIFFERENT"}, {"m1": "v1"}
        )
        assert status.stale == [("c1", "m1")]

    def test_stale_on_model_version_mismatch(self):
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "DIFFERENT"}
        )
        assert status.stale == [("c1", "m1")]


class TestStalenessAnchors:
    """Image-digest anchoring + the offline-never-invalidates guarantee."""

    def _digests(self, value="sha256:d1"):
        return {"c1": value}

    def test_run_twice_computes_zero(self):
        # THE regression test: correct anchors + no changes ⇒ 0 missing-or-stale.
        # Proves stored == resolved for unchanged content across spec/image/model.
        cache = _cache()
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "h1"}, {"m1": "v1"}, self._digests()
        )
        assert status.present == 1
        assert status.to_run == []

    def test_changed_spec_hash_is_stale(self):
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "DIFFERENT"}, {"m1": "v1"}, self._digests()
        )
        assert status.stale == [("c1", "m1")]

    def test_changed_image_digest_is_stale(self):
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "v1"}, self._digests("sha256:OTHER")
        )
        assert status.stale == [("c1", "m1")]

    def test_bumped_model_version_stales_only_that_model(self):
        # A model bump invalidates only that model's cells, not the challenge's others.
        cache = {
            "c1": {
                "m1": {"version_hash": "h1", "model_version": "v1", "image_digest": "sha256:d1"},
                "m2": {"version_hash": "h1", "model_version": "v1", "image_digest": "sha256:d1"},
            }
        }
        status = run_matrix.classify_cells(
            ["c1"], ["m1", "m2"], cache, {"c1": "h1"}, {"m1": "BUMPED", "m2": "v1"}, self._digests()
        )
        assert status.stale == [("c1", "m1")]
        assert status.present == 1

    def test_empty_stored_digest_is_not_stale(self):
        # First-live cell: stored digest "" + a resolving registry digest must NOT
        # stale — else the paid cell recomputes forever (mirrors check_image_digests).
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "v1", "image_digest": ""}}}
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "h1"}, {"m1": "v1"}, self._digests("sha256:x")
        )
        assert status.present == 1
        assert status.to_run == []

    def test_nonempty_stored_digest_mismatch_is_stale(self):
        # A non-empty stored digest that differs from the resolved one IS stale.
        cell = {"version_hash": "h1", "model_version": "v1", "image_digest": "sha256:a"}
        cache = {"c1": {"m1": cell}}
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], cache, {"c1": "h1"}, {"m1": "v1"}, self._digests("sha256:b")
        )
        assert status.stale == [("c1", "m1")]

    def test_resolution_failure_is_not_stale(self):
        # Digest resolves to None (offline/yanked) ⇒ image axis skipped, NOT stale.
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "v1"}, {"c1": None}
        )
        assert status.present == 1
        assert status.to_run == []

    def test_image_rebuild_invalidates_all_models_for_challenge(self):
        cache = {
            "c1": {
                "m1": {"version_hash": "h1", "model_version": "v1", "image_digest": "sha256:old"},
                "m2": {"version_hash": "h1", "model_version": "v1", "image_digest": "sha256:old"},
            }
        }
        digests = {"c1": "sha256:new"}
        status = run_matrix.classify_cells(
            ["c1"], ["m1", "m2"], cache, {"c1": "h1"}, {"m1": "v1", "m2": "v1"}, digests
        )
        assert set(status.stale) == {("c1", "m1"), ("c1", "m2")}


class TestCacheKeyIsolation:
    """The cache key is (spec-hash, model_version, image_digest) — nothing else."""

    def test_price_and_date_metadata_do_not_change_model_version(self, monkeypatch):
        # Bumping input/output price and the price_as_of date but keeping `version`
        # fixed must leave model_versions() identical — pricing is not a cache key.
        base = {
            "m1": {
                "version": "v1",
                "input_cost_per_1m": 0.14,
                "output_cost_per_1m": 0.28,
                "price_as_of": "2026-07-15",
            }
        }
        bumped = {
            "m1": {
                "version": "v1",
                "input_cost_per_1m": 9.99,
                "output_cost_per_1m": 99.99,
                "price_as_of": "2027-01-01",
            }
        }
        monkeypatch.setattr(config, "load_pricing", lambda *a, **k: base)
        before = integrity.model_versions()
        monkeypatch.setattr(config, "load_pricing", lambda *a, **k: bumped)
        after = integrity.model_versions()
        assert before == after == {"m1": "v1"}

    def test_price_bump_keeps_cell_present(self):
        # Same version despite a price change ⇒ the stored cell stays present (no recompute).
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "v1"}, {"c1": "sha256:d1"}
        )
        assert status.present == 1
        assert status.to_run == []

    def test_date_suffixed_version_bump_is_stale(self):
        # Encoding a release date INTO `version` is a real version change ⇒ stale.
        status = run_matrix.classify_cells(
            ["c1"], ["m1"], _cache(), {"c1": "h1"}, {"m1": "v1@2026-07-16"}
        )
        assert status.stale == [("c1", "m1")]

    def test_model_rename_orphans_old_row_and_new_name_is_missing(self):
        # results.csv still holds a row under the OLD name; the enabled model is the
        # NEW name. The old row must NOT satisfy the new cell ⇒ new name is missing.
        cache = {
            "c1": {"old-name": {"version_hash": "h1", "model_version": "v1", "image_digest": ""}}
        }
        status = run_matrix.classify_cells(
            ["c1"], ["new-name"], cache, {"c1": "h1"}, {"new-name": "v1"}
        )
        assert status.missing == [("c1", "new-name")]
        assert status.present == 0


class TestDigestResolver:
    """Manifest-digest resolution + canonicalization via an injected runner."""

    def test_canonical_strips_repo_prefix(self):
        raw = "swebench/sweb.eval.x86_64.psf_1776_requests-1142@sha256:abc123"
        assert image_version.canonical_digest(raw) == "sha256:abc123"

    def test_canonical_passthrough_bare_digest(self):
        assert image_version.canonical_digest("  sha256:deadbeef\n") == "sha256:deadbeef"

    def test_canonical_rejects_non_digest(self):
        assert image_version.canonical_digest("not-a-digest") == ""

    def test_resolve_returns_bare_digest(self):
        runner = lambda argv: (0, "sha256:9b0b13\n")  # noqa: E731
        assert image_version.resolve_manifest_digest("ref", runner) == "sha256:9b0b13"

    def test_resolve_failure_returns_none(self):
        runner = lambda argv: (1, "")  # noqa: E731 (docker missing / offline)
        assert image_version.resolve_manifest_digest("ref", runner) is None

    def test_resolve_uses_manifest_not_config(self):
        # The argv must request .Manifest.Digest (never the config digest).
        seen: dict = {}

        def runner(argv):
            seen["argv"] = argv
            return 0, "sha256:x"

        image_version.resolve_manifest_digest("ref", runner)
        assert "{{.Manifest.Digest}}" in seen["argv"]

    def test_used_digest_from_repodigests(self):
        runner = lambda argv: (0, "repo@sha256:used123\n")  # noqa: E731
        assert image_version.used_image_digest("ref", runner) == "sha256:used123"


class TestHistoryArchiving:
    def _row(self, cid, model, **over):
        base = {k: "" for k in integrity.RESULTS_FIELDS}
        base.update({"challenge_id": cid, "model": model, "pass": "True", "cost": "0.1"})
        base.update(over)
        return base

    def test_superseded_row_appended_to_history(self, tmp_path):
        results = tmp_path / "results.csv"
        history = tmp_path / "history.csv"
        run_matrix.merge_rows([self._row("c1", "m1", version_hash="old")], results, history)
        # Recompute the same cell with a new hash ⇒ old row archived, current updated.
        run_matrix.merge_rows([self._row("c1", "m1", version_hash="new")], results, history)
        assert history.exists()
        hist_text = history.read_text()
        assert "old" in hist_text
        assert "superseded_at" in hist_text.splitlines()[0]
        current = config.load_results(results)["c1"]["m1"]
        assert current["version_hash"] == "new"

    def test_identical_rewrite_does_not_archive(self, tmp_path):
        results = tmp_path / "results.csv"
        history = tmp_path / "history.csv"
        run_matrix.merge_rows([self._row("c1", "m1", version_hash="h")], results, history)
        run_matrix.merge_rows([self._row("c1", "m1", version_hash="h")], results, history)
        assert not history.exists()


class TestAtomicWrite:
    """results.csv is rewritten atomically (temp + os.replace) so a crash can't corrupt it."""

    def _rows(self):
        base = {k: "" for k in integrity.RESULTS_FIELDS}
        return {("c1", "m1"): {**base, "challenge_id": "c1", "model": "m1", "pass": "True"}}

    def test_content_matches_and_no_temp_left(self, tmp_path):
        path = tmp_path / "results.csv"
        run_matrix._write_raw_rows(self._rows(), path)
        loaded = config.load_results(path)
        assert loaded["c1"]["m1"]["pass"] is True
        assert not (tmp_path / "results.csv.tmp").exists()

    def test_uses_temp_then_replace(self, tmp_path, monkeypatch):
        path = tmp_path / "results.csv"
        seen: dict = {}

        real_replace = run_matrix.os.replace

        def spy(src, dst):
            seen["src"] = str(src)
            seen["dst"] = str(dst)
            return real_replace(src, dst)

        monkeypatch.setattr(run_matrix.os, "replace", spy)
        run_matrix._write_raw_rows(self._rows(), path)
        assert seen["src"].endswith("results.csv.tmp")
        assert seen["dst"] == str(path)


class TestCheckImageDigests:
    def test_clean_when_stored_matches_resolved(self):
        cache = {"c1": {"m1": {"image_digest": "sha256:d1"}}}
        assert check_integrity.check_image_digests(cache, {"c1": "sha256:d1"}) == []

    def test_reports_drift(self):
        cache = {"c1": {"m1": {"image_digest": "sha256:old"}}}
        drift = check_integrity.check_image_digests(cache, {"c1": "sha256:new"})
        assert drift == [("c1", "m1", "sha256:old", "sha256:new")]

    def test_unresolved_digest_is_not_drift(self):
        cache = {"c1": {"m1": {"image_digest": "sha256:old"}}}
        assert check_integrity.check_image_digests(cache, {"c1": None}) == []

    def test_empty_stored_is_not_drift(self):
        cache = {"c1": {"m1": {"image_digest": ""}}}
        assert check_integrity.check_image_digests(cache, {"c1": "sha256:new"}) == []


class TestCoverage:
    def _matrix(self):
        return {
            "tasks": {"t1": {}, "t2": {}},
            "models": {"cheap": {"input_price": 0.1, "output_price": 0.1}},
            "results": {
                "t1": {"cheap": {"pass": True, "cost": 1.0}},
                "t2": {"cheap": {"pass": True, "cost": 1.0}},
            },
        }

    def test_complete_when_all_cells_cached(self):
        cov = coverage.cell_coverage(Oracle(), self._matrix(), ["t1", "t2"])
        assert cov.complete
        assert len(cov.needed) == 2

    def test_flags_uncached_cell(self):
        matrix = self._matrix()
        del matrix["results"]["t2"]  # outcome for t2 gone
        cov = coverage.cell_coverage(Oracle(), matrix, ["t1", "t2"])
        assert not cov.complete
        assert ("t2", "") in cov.missing or any(t == "t2" for t, _ in cov.missing)


class TestCheckIntegrity:
    def test_clean_tree_reports_nothing(self):
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "v1"}}}
        removed, changed = check_integrity.check_hashes(cache, {"c1": "h1"})
        assert removed == []
        assert changed == []

    def test_detects_changed(self):
        cache = {"c1": {"m1": {"version_hash": "old", "model_version": "v1"}}}
        removed, changed = check_integrity.check_hashes(cache, {"c1": "new"})
        assert changed == ["c1"]
        assert removed == []

    def test_detects_removed(self):
        cache = {"gone": {"m1": {"version_hash": "h", "model_version": "v1"}}}
        removed, changed = check_integrity.check_hashes(cache, {})
        assert removed == ["gone"]

    def test_detects_stale_model_version(self):
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "old"}}}
        stale = check_integrity.check_model_versions(cache, {"m1": "new"})
        assert stale == [("m1", "old", "new")]

    def test_current_model_version_not_stale(self):
        cache = {"c1": {"m1": {"version_hash": "h1", "model_version": "v1"}}}
        assert check_integrity.check_model_versions(cache, {"m1": "v1"}) == []
