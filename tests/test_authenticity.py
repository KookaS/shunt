"""Edge-case coverage for the Layer-1 results authenticity gate.

Builds a valid row from real registry/spec data, mutates one field at a time to pin
each check, and tests the boundary Layer 1 cannot enforce (motivating Layer 2).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from benchmark import config
from benchmark.routing import authenticity as auth
from benchmark.routing import integrity

_NOW = datetime(2026, 7, 18, tzinfo=UTC)
_GOOD_DIGEST = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def valid_row() -> dict[str, str]:
    specs = integrity.swebench_spec_hashes()
    assert specs, "need at least one materialised spec to build a valid row"
    cid = sorted(specs)[0]
    model = "deepseek-v4-flash"
    mc = config.resolved_models()[model]
    assert mc.reasoning is not None
    arm = mc.reasoning.default_arm
    in_tok, out_tok = 1000, 200
    est = integrity.estimated_cost(model, in_tok, out_tok, config._pricing_dict())
    return {
        "challenge_id": cid,
        "model": model,
        "reasoning": arm,
        "pass": "True",
        "cost": repr(est),
        "in_tok": str(in_tok),
        "out_tok": str(out_tok),
        "calls": "5",
        "version_hash": specs[cid],
        "model_version": integrity.model_versions()[model],
        "arm_hash": integrity.arm_hash_value(mc, arm),
        "real_cost": "0.0",
        "estimated_cost": repr(est),
        "timeout_flag": "False",
        "image_digest": _GOOD_DIGEST,
        "computed_at": "2026-07-01T00:00:00+00:00",
    }


def _rules(findings: list[auth.Finding]) -> set[str]:
    return {f.rule for f in findings}


def _est(row: dict[str, str]) -> float:
    return integrity.estimated_cost(
        row["model"], int(row["in_tok"]), int(row["out_tok"]), config._pricing_dict()
    )


class TestValidRowPasses:
    def test_valid_row_has_no_findings(self, valid_row):
        assert auth.verify_rows([valid_row], now=_NOW) == []


class TestSchema:
    def test_pass_not_bool(self, valid_row):
        assert "schema.pass_not_bool" in _rules(
            auth.verify_rows([{**valid_row, "pass": "y"}], _NOW)
        )

    def test_negative_int_rejected(self, valid_row):
        assert "schema.bad_int" in _rules(auth.verify_rows([{**valid_row, "in_tok": "-5"}], _NOW))

    def test_underscore_int_rejected(self, valid_row):
        # Python int() accepts "1_000_000"; a CSV/pandas/Go reader would not — reject it.
        row = {**valid_row, "in_tok": "1_000_000"}
        assert "schema.bad_int" in _rules(auth.verify_rows([row], _NOW))

    def test_unicode_int_rejected(self, valid_row):
        row = {**valid_row, "out_tok": "١٢٣"}
        assert "schema.bad_int" in _rules(auth.verify_rows([row], _NOW))

    def test_nan_cost_rejected(self, valid_row):
        assert "schema.bad_float" in _rules(auth.verify_rows([{**valid_row, "cost": "nan"}], _NOW))

    def test_trailing_space_int_rejected(self, valid_row):
        assert "schema.bad_int" in _rules(auth.verify_rows([{**valid_row, "calls": "5 "}], _NOW))

    def test_timeout_flag_not_bool(self, valid_row):
        row = {**valid_row, "timeout_flag": "banana"}
        assert "schema.timeout_not_bool" in _rules(auth.verify_rows([row], _NOW))

    def test_missing_required_column(self, valid_row):
        row = {k: v for k, v in valid_row.items() if k != "calls"}
        assert "schema.missing_columns" in _rules(auth.verify_rows([row], _NOW))

    def test_missing_legacy_arm_hash_is_tolerated(self, valid_row):
        # A legacy default-arm row predating arm_hash is not a violation.
        legacy = {**valid_row, "reasoning": integrity.DEFAULT_REASONING}
        row = {k: v for k, v in legacy.items() if k != "arm_hash"}
        assert auth.errors(auth.verify_rows([row], _NOW)) == []

    def test_dropping_cost_column_is_caught(self, valid_row):
        # Evasion: removing estimated_cost entirely would otherwise disable the cost
        # checks — it's a required column, so its absence fails the gate.
        row = {k: v for k, v in valid_row.items() if k != "estimated_cost"}
        assert "schema.missing_columns" in _rules(auth.verify_rows([row], _NOW))


class TestRegistered:
    def test_unknown_challenge(self, valid_row):
        row = {**valid_row, "challenge_id": "fake__challenge-1", "version_hash": "0" * 64}
        assert "registered.unknown_challenge" in _rules(auth.verify_rows([row], _NOW))

    def test_unknown_model(self, valid_row):
        row = {**valid_row, "model": "gpt-9-ultra"}
        assert "registered.unknown_model" in _rules(auth.verify_rows([row], _NOW))

    def test_unknown_arm(self, valid_row):
        row = {**valid_row, "reasoning": "witchcraft", "arm_hash": ""}
        assert "registered.unknown_arm" in _rules(auth.verify_rows([row], _NOW))


class TestAnchors:
    def test_blank_version_hash_is_fabrication(self, valid_row):
        # Blanking an anchor no longer skips the check (the HIGH-3 evasion).
        row = {**valid_row, "version_hash": ""}
        assert "anchor.version_missing" in _rules(auth.verify_rows([row], _NOW))

    def test_version_hash_tampered(self, valid_row):
        row = {**valid_row, "version_hash": "deadbeef" * 8}
        assert "anchor.version_mismatch" in _rules(auth.verify_rows([row], _NOW))

    def test_model_version_mismatch(self, valid_row):
        # A stored model_version that differs from the registry's current version for the
        # model is a stale (re-run-signal) or fabricated row — a gate-failing ERROR.
        row = {**valid_row, "model_version": "some-old-version"}
        assert "anchor.model_version_mismatch" in _rules(auth.verify_rows([row], _NOW))

    def test_blank_model_version_is_fabrication(self, valid_row):
        # Blanking the anchor no longer skips it (mirrors version_hash).
        row = {**valid_row, "model_version": ""}
        assert "anchor.model_version_missing" in _rules(auth.verify_rows([row], _NOW))

    def test_arm_hash_tampered(self, valid_row):
        row = {**valid_row, "arm_hash": "deadbeef" * 8}
        assert "anchor.arm_mismatch" in _rules(auth.verify_rows([row], _NOW))

    def test_explicit_arm_missing_arm_hash(self, valid_row):
        # valid_row.reasoning is an explicit (non-default) arm id → arm_hash required.
        row = {**valid_row, "arm_hash": ""}
        assert "anchor.arm_missing" in _rules(auth.verify_rows([row], _NOW))

    def test_malformed_image_digest(self, valid_row):
        row = {**valid_row, "image_digest": "sha256:abc"}
        assert "anchor.bad_image_digest" in _rules(auth.verify_rows([row], _NOW))


class TestCostTamper:
    def test_cost_column_fiddled(self, valid_row):
        row = {**valid_row, "cost": "999.0"}
        assert "cost.derivation_mismatch" in _rules(auth.verify_rows([row], _NOW))

    def test_expensive_run_billed_as_free(self, valid_row):
        # HIGH-1: forge real_cost far below the token estimate — the kill-gate attack.
        rc = repr(_est(valid_row) * 1e-4)
        row = {**valid_row, "real_cost": rc, "cost": rc}
        assert "cost.real_cost_implausible" in _rules(auth.verify_rows([row], _NOW))

    def test_cheap_run_billed_as_expensive_warns_not_errors(self, valid_row):
        # Upward direction is WARN, not a hard ERROR: a high-reasoning arm (no data yet)
        # or reasoning-token-billing could push a LEGIT ratio high — don't red CI on it.
        rc = repr(_est(valid_row) * 100)
        row = {**valid_row, "real_cost": rc, "cost": rc}
        findings = auth.verify_rows([row], _NOW)
        assert auth.errors(findings) == []
        assert "cost.real_cost_unusual" in _rules(auth.warnings(findings))

    def test_high_reasoning_arm_above_estimate_does_not_error(self, valid_row):
        # A real cost 2.5x the token estimate (plausible if reasoning tokens are billed
        # outside completion_tokens) must NOT fail the gate — only warn.
        rc = repr(_est(valid_row) * 2.5)
        row = {**valid_row, "real_cost": rc, "cost": rc}
        assert auth.errors(auth.verify_rows([row], _NOW)) == []

    def test_plausible_real_cost_is_clean(self, valid_row):
        rc = repr(_est(valid_row) * 0.25)  # mid-band, matches observed real ratios
        row = {**valid_row, "real_cost": rc, "cost": rc}
        assert auth.verify_rows([row], _NOW) == []

    def test_unusual_real_cost_warns_not_errors(self, valid_row):
        rc = repr(_est(valid_row) * 0.01)  # in [ERR_LOW=0.005, WARN_LOW=0.02) → WARN
        row = {**valid_row, "real_cost": rc, "cost": rc}
        findings = auth.verify_rows([row], _NOW)
        assert auth.errors(findings) == []
        assert "cost.real_cost_unusual" in _rules(auth.warnings(findings))


class TestPassPlausibility:
    def test_pass_with_zero_output_tokens(self, valid_row):
        rc = "0.0"
        row = {**valid_row, "out_tok": "0", "cost": rc, "real_cost": rc, "estimated_cost": rc}
        assert "pass.no_output" in _rules(auth.verify_rows([row], _NOW))

    def test_pass_with_timeout_is_contradiction(self, valid_row):
        row = {**valid_row, "timeout_flag": "True"}
        assert "pass.timeout_contradiction" in _rules(auth.verify_rows([row], _NOW))

    def test_fail_with_zero_output_is_fine(self, valid_row):
        rc = "0.0"
        row = {**valid_row, "pass": "False", "out_tok": "0", "cost": rc, "estimated_cost": rc}
        assert "pass.no_output" not in _rules(auth.verify_rows([row], _NOW))


class TestBounds:
    def test_huge_token_count_warns(self, valid_row):
        row = {**valid_row, "in_tok": "999999999999"}
        findings = auth.verify_rows([row], _NOW)
        assert "bounds.tokens_huge" in _rules(auth.warnings(findings))


class TestTimestamp:
    def test_future_timestamp(self, valid_row):
        row = {**valid_row, "computed_at": (_NOW + timedelta(days=1)).isoformat()}
        assert "time.future" in _rules(auth.verify_rows([row], _NOW))

    def test_unparseable_timestamp(self, valid_row):
        row = {**valid_row, "computed_at": "last thursday"}
        assert "time.unparseable" in _rules(auth.verify_rows([row], _NOW))


class TestDuplicateKeys:
    def test_exact_duplicate(self, valid_row):
        findings = auth.verify_rows([dict(valid_row), dict(valid_row)], _NOW)
        assert "file.duplicate_key" in _rules(findings)

    def test_alias_and_explicit_default_arm_collide(self, valid_row):
        # MED-4: "default" resolves to the model's real default_arm — the two spellings
        # name the same cell and must be caught as a duplicate, not slip as two keys.
        alias = {**valid_row, "reasoning": integrity.DEFAULT_REASONING}
        findings = auth.verify_rows([alias, dict(valid_row)], _NOW)
        assert "file.duplicate_key" in _rules(findings)


class TestCommittedResultsCsv:
    def test_committed_results_yield_zero_errors(self):
        # The real 397-row committed results.csv must pass every Layer-1 anchor
        # (including the new model_version anchor) with zero gate-failing findings.
        import csv

        with config.results_csv_path().open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows, "committed results.csv should have rows"
        assert auth.errors(auth.verify_rows(rows)) == []


class TestLayer1Boundary:
    """Honest boundary: Layer 1 is internal-consistency, not proof-of-execution."""

    def test_flipping_pass_in_either_direction_is_not_caught(self, valid_row):
        # A row that could be a genuine FAILED run (plausible tokens/cost) is clean; a
        # fabricator flipping ONLY `pass` — in EITHER direction — leaves every derivable
        # invariant intact, so Layer 1 STILL reports nothing. Only a provenance signature
        # (Layer 2) or re-execution (Layer 3) closes this. Pinned so Layer 1 isn't
        # mistaken for tamper-proof.
        failed = {**valid_row, "pass": "False"}
        assert auth.errors(auth.verify_rows([failed], _NOW)) == []
        assert auth.errors(auth.verify_rows([{**failed, "pass": "True"}], _NOW)) == []
        assert auth.errors(auth.verify_rows([{**valid_row, "pass": "False"}], _NOW)) == []
