"""Tests for the SWE-bench Verified execution pipeline (specs, infer, harness).

No Docker or network: harness parsing uses a captured report; gold-patch pulling
is monkeypatched; live inference stays key-gated (simulated).
"""

import json
from typing import Final

import pytest

from benchmark.routing import integrity
from benchmark.runner import infer, run_matrix, select_swebench, swebench_harness, swebench_specs

SMOKE_IDS: Final = [
    "astropy__astropy-7166",
    "psf__requests-1142",
    "pytest-dev__pytest-5809",
    "pylint-dev__pylint-6903",
    "pallets__flask-5014",
]

# The full materialised set: 5 original + 5 diverse repos added in the ood176→
# SWE-bench migration (each verified to have a prebuilt swebench Docker image).
ALL_IDS: Final = [
    *SMOKE_IDS,
    "django__django-12419",
    "sympy__sympy-20916",
    "scikit-learn__scikit-learn-14141",
    "pydata__xarray-3677",
    "sphinx-doc__sphinx-8595",
]


class TestSpecs:
    def test_difficulty_stratum_mapping(self):
        assert swebench_specs.difficulty_stratum("<15 min fix") == "easy"
        assert swebench_specs.difficulty_stratum("15 min - 1 hour") == "medium"
        assert swebench_specs.difficulty_stratum("1-4 hours") == "hard"
        assert swebench_specs.difficulty_stratum(">4 hours") == "hard"
        assert swebench_specs.difficulty_stratum("unknown") == "medium"

    def test_image_ref_mirrors_swebench_key(self):
        ref = swebench_specs.image_ref("psf__requests-1142")
        assert ref == "swebench/sweb.eval.x86_64.psf_1776_requests-1142:latest"

    def test_spec_from_dataset_row_parses_json_strings(self):
        row = {
            "instance_id": "a__b-1",
            "repo": "a/b",
            "base_commit": "deadbeef",
            "version": "1.0",
            "difficulty": "<15 min fix",
            "FAIL_TO_PASS": json.dumps(["t::x"]),
            "PASS_TO_PASS": json.dumps(["t::y", "t::z"]),
        }
        spec = swebench_specs.spec_from_dataset_row(row)
        assert spec.fail_to_pass == ["t::x"]
        assert spec.pass_to_pass == ["t::y", "t::z"]
        assert spec.difficulty_stratum == "easy"
        assert spec.image_ref.endswith("a_1776_b-1:latest")

    def test_all_materialised_smoke_specs_load(self):
        ids = {s.instance_id for s in swebench_specs.all_specs()}
        for iid in SMOKE_IDS:
            assert iid in ids
            spec = swebench_specs.load_spec(iid)
            assert spec is not None
            assert spec.fail_to_pass  # F2P is non-empty for every Verified task

    def test_store_holds_exactly_ten_specs(self):
        ids = {s.instance_id for s in swebench_specs.all_specs()}
        assert ids == set(ALL_IDS)

    def test_every_spec_carries_dataset_revision(self):
        for spec in swebench_specs.all_specs():
            assert spec.dataset_revision == swebench_specs.DATASET_REVISION

    def test_difficulty_strata_are_spread(self):
        strata = {s.difficulty_stratum for s in swebench_specs.all_specs()}
        assert {"easy", "medium", "hard"} <= strata

    def test_dataset_revision_survives_roundtrip(self):
        spec = swebench_specs.load_spec("django__django-12419")
        assert spec is not None
        assert spec.dataset_revision == swebench_specs.DATASET_REVISION
        assert swebench_specs.spec_from_dict(spec.to_dict()) == spec

    def test_load_missing_spec_returns_none(self):
        assert swebench_specs.load_spec("does__not-exist-0") is None

    def test_to_dict_roundtrips_through_from_dict(self):
        spec = swebench_specs.load_spec("psf__requests-1142")
        assert spec is not None
        again = swebench_specs.spec_from_dict(spec.to_dict())
        assert again == spec


class TestSpecHashing:
    def test_spec_hash_is_sha256(self):
        h = integrity.swebench_spec_hash("psf__requests-1142")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_missing_spec_hash_empty(self):
        assert integrity.swebench_spec_hash("does__not-exist-0") == ""

    def test_all_spec_hashes_cover_store(self):
        hashes = integrity.swebench_spec_hashes()
        for iid in SMOKE_IDS:
            assert hashes[iid] == integrity.swebench_spec_hash(iid)

    def test_hash_is_order_independent(self):
        spec = swebench_specs.load_spec("psf__requests-1142")
        assert spec is not None
        d = spec.to_dict()
        reordered = dict(reversed(list(d.items())))
        assert integrity.hash_content(d) == integrity.hash_content(reordered)


class TestGoldPredictions:
    def test_prediction_line_shape(self):
        line = infer.prediction_line("a__b-1", "gold", "diff --git ...")
        assert line == {
            "instance_id": "a__b-1",
            "model_name_or_path": "gold",
            "model_patch": "diff --git ...",
        }

    def test_write_and_read_predictions_roundtrip(self, tmp_path):
        preds = [infer.prediction_line("a__b-1", "gold", "patchA")]
        path = infer.write_predictions(preds, tmp_path / "p.jsonl")
        assert swebench_harness.read_predictions(path) == preds

    def test_build_gold_predictions_uses_dataset_patch(self, monkeypatch):
        monkeypatch.setattr(infer, "gold_patches", lambda ids: {i: f"patch-{i}" for i in ids})
        preds = infer.build_gold_predictions(["a__b-1", "c__d-2"])
        assert preds[0]["model_name_or_path"] == infer.GOLD_MODEL_NAME
        assert preds[0]["model_patch"] == "patch-a__b-1"
        assert [p["instance_id"] for p in preds] == ["a__b-1", "c__d-2"]


class TestHarnessParsing:
    SAMPLE_REPORT = {
        "total_instances": 3,
        "submitted_ids": ["a__b-1", "c__d-2", "e__f-3"],
        "resolved_ids": ["a__b-1", "e__f-3"],
        "unresolved_ids": ["c__d-2"],
        "error_ids": [],
        "schema_version": 2,
    }

    def test_parse_report_maps_resolved(self):
        resolved = swebench_harness.parse_report(self.SAMPLE_REPORT, ["a__b-1", "c__d-2", "e__f-3"])
        assert resolved == {"a__b-1": True, "c__d-2": False, "e__f-3": True}

    def test_parse_report_missing_instance_is_unresolved(self):
        resolved = swebench_harness.parse_report(self.SAMPLE_REPORT, ["z__z-9"])
        assert resolved == {"z__z-9": False}

    def test_build_command_pulls_prebuilt_images(self):
        cmd = swebench_harness.build_command(
            predictions_path=swebench_harness.Path("p.jsonl"),
            run_id="rid",
            instance_ids=["a__b-1"],
            namespace="swebench",
            cache_level="env",
        )
        assert "--namespace" in cmd and "swebench" in cmd
        assert cmd[cmd.index("--cache_level") + 1] == "env"
        assert "--instance_ids" in cmd and "a__b-1" in cmd
        assert "run_evaluation" in " ".join(cmd)


class TestLiveGating:
    def test_has_api_keys_false_without_keys(self):
        assert infer.has_api_keys(env={}) is False

    def test_has_api_keys_true_with_one_key(self):
        assert infer.has_api_keys(env={"DEEPSEEK_API_KEY": "x"}) is True

    def test_generate_patch_live_gated_without_keys(self):
        spec = swebench_specs.load_spec("psf__requests-1142")
        assert spec is not None
        with pytest.raises(infer.MissingApiKeysError):
            infer.generate_patch_live(spec, "deepseek-v4-flash", env={})

    def test_run_live_cell_gated_without_keys(self, tmp_path, monkeypatch):
        # No provider keys → run_live_cell must gate before any harness call.
        for k in infer._KEY_ENV:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(infer.MissingApiKeysError):
            infer.run_live_cell(
                "psf__requests-1142", "deepseek-v4-flash", work_dir=tmp_path, run_id="t"
            )

    def test_run_live_cell_unknown_instance_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        with pytest.raises(KeyError):
            infer.run_live_cell("no__such-0", "m", work_dir=tmp_path, run_id="t")


class TestCostAggregation:
    """Real cost prefers litellm's number, falling back to the provider usage.cost."""

    def test_call_cost_uses_litellm_cost_when_priced(self):
        # A direct route (deepseek) that litellm CAN price: use its cost, ignore usage.
        assert infer._call_cost({"cost": 0.004}, {"cost": 999.0}) == 0.004

    def test_call_cost_falls_back_to_provider_usage_cost(self):
        # A requesty route litellm can't price (cost=0) ⇒ use the real usage.cost.
        assert infer._call_cost({"cost": 0.0}, {"cost": 1.59e-05}) == 1.59e-05

    def test_call_cost_zero_when_neither_present(self):
        assert infer._call_cost({}, {}) == 0.0

    def test_sum_usage_sums_provider_cost_across_calls(self):
        # Two requesty-style calls: litellm cost 0, real cost carried in usage.cost.
        messages = [
            {
                "extra": {
                    "cost": 0.0,
                    "response": {
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.02}
                    },
                }
            },
            {
                "extra": {
                    "cost": 0.0,
                    "response": {
                        "usage": {"prompt_tokens": 200, "completion_tokens": 20, "cost": 0.03}
                    },
                }
            },
            {"content": "no extra/response here"},  # skipped (no response)
        ]
        in_tok, out_tok, calls, cost = infer._sum_usage(messages)
        assert (in_tok, out_tok, calls) == (300, 30, 2)
        assert cost == pytest.approx(0.05)

    def test_sum_usage_prefers_litellm_cost_when_priced(self):
        # deepseek-style: litellm priced it; usage.cost absent ⇒ litellm cost wins.
        messages = [
            {
                "extra": {
                    "cost": 0.006,
                    "response": {"usage": {"prompt_tokens": 500, "completion_tokens": 12}},
                }
            },
        ]
        _, _, calls, cost = infer._sum_usage(messages)
        assert calls == 1
        assert cost == pytest.approx(0.006)


class TestInfraFailure:
    """A harness infra crash must NOT cache as a model failure — cell stays MISSING."""

    def _patch(self):
        return infer.AgentPatch(patch="diff", in_tok=1, out_tok=1, calls=1, cost=0.0)

    def test_run_live_cell_raises_when_report_missing(self, tmp_path, monkeypatch):
        # Docker/image unavailable ⇒ run_harness returns report_path=None.
        monkeypatch.setattr(infer, "generate_patch_live", lambda spec, model: self._patch())
        monkeypatch.setattr(
            infer.swebench_harness,
            "run_harness",
            lambda **kw: swebench_harness.HarnessResult({"psf__requests-1142": False}, None, {}, 0),
        )
        with pytest.raises(infer.HarnessInfraError):
            infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="t")

    def test_run_live_cell_raises_on_nonzero_returncode(self, tmp_path, monkeypatch):
        # Harness exited non-zero (timeout/error) ⇒ not a real pass=False result.
        report = tmp_path / "r.json"
        monkeypatch.setattr(infer, "generate_patch_live", lambda spec, model: self._patch())
        monkeypatch.setattr(
            infer.swebench_harness,
            "run_harness",
            lambda **kw: swebench_harness.HarnessResult(
                {"psf__requests-1142": False}, report, {}, 137
            ),
        )
        with pytest.raises(infer.HarnessInfraError):
            infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="t")

    def test_run_live_cells_skips_infra_failure_no_row(self, monkeypatch):
        # An infra-failing cell writes NO row; the good cell still computes.
        def fake(cid, model, **kw):
            if cid == "bad__cell-1":
                raise infer.HarnessInfraError("docker down")
            return {"pass": True, "in_tok": 1, "out_tok": 1, "calls": 1, "real_cost": 0.0}

        monkeypatch.setattr(infer, "run_live_cell", fake)
        # This test is about infra isolation, not the caching gate — neutralise the gate.
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        rows = run_matrix.run_live_cells(
            [("bad__cell-1", "m"), ("good__cell-1", "m")],
            {},
            {"bad__cell-1": "h", "good__cell-1": "h"},
            {"m": "v"},
            timeout=10,
            verbose=False,
        )
        assert [r["challenge_id"] for r in rows] == ["good__cell-1"]

    def test_run_live_cells_skips_unexpected_exception(self, monkeypatch):
        # Fix 4: ANY scaffold exception (not just KeyError/MissingApiKeys) skips one cell.
        def fake(cid, model, **kw):
            if cid == "bad__cell-1":
                raise ImportError("scaffold missing")
            return {"pass": False, "in_tok": 0, "out_tok": 0, "calls": 0, "real_cost": 0.0}

        monkeypatch.setattr(infer, "run_live_cell", fake)
        monkeypatch.setattr(run_matrix.config, "models_missing_cache", lambda *a, **k: [])
        rows = run_matrix.run_live_cells(
            [("bad__cell-1", "m"), ("good__cell-1", "m")],
            {},
            {"bad__cell-1": "h", "good__cell-1": "h"},
            {"m": "v"},
            timeout=10,
            verbose=False,
        )
        assert [r["challenge_id"] for r in rows] == ["good__cell-1"]


class TestSelect:
    def test_discrimination_gap(self):
        row = {"opus": True, "gpt-5": True, "deepseek": False, "qwen": False}
        cheap, frontier = select_swebench.classify_submissions(row.keys())
        assert select_swebench.discrimination_gap(row, cheap, frontier) == 1.0

    def test_gap_zero_when_cohort_absent(self):
        row = {"opus": True}
        cheap, frontier = select_swebench.classify_submissions(row.keys())
        assert select_swebench.discrimination_gap(row, cheap, frontier) == 0.0

    def test_select_is_stratified_and_gap_ranked(self):
        # Two easy instances; only the higher-gap one is picked when target=1.
        table = {
            "i_easy_hi": {"opus": True, "deepseek": False},
            "i_easy_lo": {"opus": True, "deepseek": True},
            "i_med": {"opus": True, "deepseek": False},
        }
        difficulty = {"i_easy_hi": "easy", "i_easy_lo": "easy", "i_med": "medium"}
        cheap, frontier = select_swebench.classify_submissions(
            {s for r in table.values() for s in r}
        )
        selected = select_swebench.select(
            table, difficulty, cheap, frontier, strata={"easy": 1, "medium": 1, "hard": 0}
        )
        assert "i_easy_hi" in selected
        assert "i_easy_lo" not in selected
        assert "i_med" in selected

    def test_load_experiments_resolves(self, tmp_path):
        sub = tmp_path / "evaluation" / "verified" / "opus-run" / "results"
        sub.mkdir(parents=True)
        (sub / "results.json").write_text(
            json.dumps({"resolved": ["a__b-1"], "no_generation": ["c__d-2"]})
        )
        table = select_swebench.load_experiments_resolves(tmp_path)
        assert table["a__b-1"]["opus-run"] is True
        assert table["c__d-2"]["opus-run"] is False


class TestRoutingHeadroom:
    def test_zero_at_ceiling_and_floor(self):
        # Everyone solves (ceiling) and nobody solves (floor) → no routing value.
        assert select_swebench.routing_headroom(p_solve=0.98, p_frontier=0.99) < 0.05
        assert select_swebench.routing_headroom(p_solve=0.0, p_frontier=0.0) == 0.0

    def test_peaks_in_routable_band(self):
        # Field mostly fails but a frontier model solves → high headroom.
        assert select_swebench.routing_headroom(p_solve=0.2, p_frontier=0.7) == pytest.approx(0.5)

    def test_never_negative(self):
        # A task where the frontier cohort underperforms the field mean clamps to 0.
        assert select_swebench.routing_headroom(p_solve=0.9, p_frontier=0.6) == 0.0

    def test_rank_orders_by_headroom_then_id(self):
        rates = {
            "z__z-1": (0.9, 0.95),  # headroom 0.05
            "a__a-1": (0.2, 0.7),  # headroom 0.50
            "m__m-1": (0.2, 0.7),  # headroom 0.50 — tie, id-sorted after a__a-1
        }
        ranked = select_swebench.rank_by_headroom(rates)
        assert [r[0] for r in ranked] == ["a__a-1", "m__m-1", "z__z-1"]
        assert ranked[0][1] == pytest.approx(0.5)

    def test_load_external_rates_missing_file_is_empty(self, tmp_path):
        assert select_swebench.load_external_rates(tmp_path / "nope.csv") == {}

    def test_enrich_challenges_adds_derived_rates(self, tmp_path, monkeypatch):
        manifest = tmp_path / "challenges.json"
        manifest.write_text(json.dumps({"tasks": {"a__b-1": {"language": "python"}}}))
        monkeypatch.setattr(
            select_swebench,
            "load_external_rates",
            lambda *a, **k: {"a__b-1": (0.3, 0.8)},
        )
        select_swebench.enrich_challenges_with_rates(manifest)
        task = json.loads(manifest.read_text())["tasks"]["a__b-1"]
        assert task["p_solve"] == 0.3
        assert task["p_frontier"] == 0.8
        assert task["routing_headroom"] == pytest.approx(0.5)
