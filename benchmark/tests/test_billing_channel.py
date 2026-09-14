"""The listing `billing` entitlement, the observed `channel` rule, and their two walls.

Covers: `benchmark.routing.channel.observed_channel` (the backfill rule), `run_matrix._is_free_lane`
reading the DECLARATION rather than the `-explabs` suffix, and `validate`'s ACCOUNTING_HOLE /
FREE_LANE_BILLED behaviour under both billing values. Hermetic: no model call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import config
from benchmark.routing import channel, model_validity, validate
from benchmark.routing.scripts import backfill_channel
from benchmark.runner import run_matrix

REPO: Path = Path(__file__).resolve().parents[2]
OVERLAY: Path = REPO / "configs" / "free-tier" / "overlay.yaml"


@pytest.fixture(autouse=True)
def _overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh caches with the real overlay configured."""
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))


# ── the backfill rule (channel.observed_channel) ──────────────────────────────────────


def test_real_cost_positive_is_paid() -> None:
    assert channel.observed_channel("glm-5.3-explabs", 1.68, 9, "2026-09-09T00:00:00+00:00") == (
        "paid",
        channel.SOURCE_REAL_COST,
    )


def test_zero_calls_is_unobserved_before_any_free_claim() -> None:
    # calls==0 outranks a free listing: a censored row is unobserved, not a $0 measurement.
    assert channel.observed_channel("glm-5.3-explabs", 0.0, 0, "2026-09-09T00:00:00+00:00") == (
        "",
        channel.SOURCE_UNOBSERVED,
    )


def test_declared_free_window_beats_listing_billing() -> None:
    assert channel.observed_channel("glm-5.3-flash", 0.0, 40, "2026-08-25T00:00:00+00:00") == (
        "free",
        channel.SOURCE_FREE_WINDOW,
    )


def test_free_file_origin_is_free() -> None:
    assert channel.observed_channel(
        "deepseek-v4-flash-explabs", 0.0, 20, "2026-09-11T00:00:00+00:00", in_free_file=True
    ) == ("free", channel.SOURCE_FREE_FILE)


def test_listing_billing_free_is_free() -> None:
    # Resolved from the configured overlay, not the corpus file and not the suffix.
    assert channel.observed_channel("glm-5.3-explabs", 0.0, 30, "2026-09-10T00:00:00+00:00") == (
        "free",
        channel.SOURCE_BILLING_FREE,
    )


def test_no_evidence_is_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(channel, "listing_billing", lambda *_a, **_k: None)
    assert channel.observed_channel("some-lane", 0.0, 5, "2026-09-10T00:00:00+00:00") == (
        "",
        channel.SOURCE_NONE,
    )


def test_backfill_rows_recomputes_only_the_channel_columns() -> None:
    row = {
        "lane": "glm-5.3-explabs",
        "real_cost": "1.68",
        "calls": "9",
        "computed_at": "2026-09-09T00:00:00+00:00",
        "in_tok": "123",
    }
    out = backfill_channel.backfill_rows([row], in_free_file=False)[0]
    assert out["channel"] == "paid"
    assert out["channel_source"] == "real_cost"
    assert out["in_tok"] == "123"  # untouched


# ── _is_free_lane reads the DECLARATION, not the suffix ───────────────────────────────


def test_free_lane_true_only_for_billing_free_rows() -> None:
    assert run_matrix._is_free_lane("glm-5.3-explabs") is True
    assert run_matrix._is_free_lane("gpt-6-astra-explabs") is False  # billing: paid
    assert run_matrix._is_free_lane("kimi-k3") is False  # shipped paid row


def test_synthesized_collection_slug_declares_free() -> None:
    # A `-explabs` id with no overlay row is synthesized WITH billing: free, so the declaration
    # (not the suffix) is what admits it.
    assert run_matrix._billing_of("deepseek-v4.1-flash-explabs") == "free"
    assert run_matrix._is_free_lane("deepseek-v4.1-flash-explabs") is True


def test_overlay_free_row_on_a_provider_with_no_free_lane_is_not_a_lane() -> None:
    # The declaration alone is not enough: Together has no free lane, so the row stays
    # collection provenance and must not be admitted as a spendable $0 lane.
    assert run_matrix._billing_of("together-muse-glimmer-30b") == "free"
    assert run_matrix._is_free_lane("together-muse-glimmer-30b") is False


# ── validate: the two walls follow the declaration ────────────────────────────────────


def _row(lane: str, *, real_cost: str = "0", calls: str = "7", **over: object) -> dict:
    base: dict = {
        "challenge_id": "astropy__astropy-1",
        "model": lane,
        "lane": lane,
        "reasoning": "default",
        "pass": "False",
        "cost": real_cost,
        "in_tok": "100",
        "out_tok": "50",
        "calls": calls,
        "version_hash": "vh",
        "model_version": lane,
        "arm_hash": "",
        "real_cost": real_cost,
        "estimated_cost": real_cost,
        "timeout_flag": "False",
        "image_digest": "",
        "computed_at": "2026-09-10T00:00:00+00:00",
        "stop_reason": "unsolved",
        "step_limit": "",
        "cost_limit": "",
        "scaffold_version": "",
        "sampling_hash": "",
        "prompt_hash": "",
    }
    base.update(over)
    return base


def test_billing_paid_zero_cost_row_trips_accounting_hole() -> None:
    # A `billing: paid` lane recording real_cost==0 must be flagged, NOT silently exempted by a
    # `-explabs` suffix. The positive control the owner asked for.
    pricing = {
        "paid-explabs-lane": {
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 2.0,
            "billing": "paid",
        }
    }
    codes = {v.code for v in validate.validate_row(_row("paid-explabs-lane"), pricing)}
    assert validate.ACCOUNTING_HOLE in codes


def test_billing_free_zero_cost_row_is_not_an_accounting_hole() -> None:
    pricing = {
        "free-lane": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0, "billing": "free"}
    }
    codes = {v.code for v in validate.validate_row(_row("free-lane"), pricing)}
    assert validate.ACCOUNTING_HOLE not in codes


def test_billing_free_lane_billed_trips_free_lane_billed() -> None:
    # A `billing: free` lane with real_cost>0 must still trip FREE_LANE_BILLED when the run
    # admitted it as a free lane (run provenance).
    pricing = {
        "free-lane": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0, "billing": "free"}
    }
    row = _row("free-lane", real_cost="0.04")
    codes = {
        v.code for v in validate.validate_row(row, pricing, free_collection_models={"free-lane"})
    }
    assert validate.FREE_LANE_BILLED in codes


def test_paid_listing_promo_history_is_not_flagged_on_a_corpus_scan() -> None:
    # The real committed case: glm-5.3-explabs is billing:free, its billed rows carry
    # channel=paid. A plain corpus scan (no provenance) must not raise FREE_LANE_BILLED on the
    # measured history.
    row = _row("glm-5.3-explabs", real_cost="1.68", calls="9")
    report = validate.validate_results([row], dict(config.free_registry()))
    assert report.error_count == 0


# ── the declared-free provider guard still applies (regression) ───────────────────────


def test_require_zero_cost_still_refuses_overlay_rows_without_a_provider_free_lane() -> None:
    assert (
        run_matrix.require_zero_cost_refusal(
            live=True, enabled=[], extra=["together-muse-glimmer-30b"]
        )
        is not None
    )
    assert (
        run_matrix.require_zero_cost_refusal(
            live=True, enabled=[], extra=["opencode-deepseek-v4-flash-free"]
        )
        is not None
    )


# ── the census reads the entitlement, paid-wins ───────────────────────────────────────


def test_census_listing_billing_comes_from_the_listing_declaration() -> None:
    # The two committed explabs rows that billed standing are paid; the promo lanes stay free.
    assert model_validity._listing_billing("gpt-6-astra-explabs") == "paid"
    assert model_validity._listing_billing("gpt-5.6-luna-explabs") == "paid"
    assert model_validity._listing_billing("glm-5.3-explabs") == "free"
    # A shipped paid row and a synthesized collection slug, both declared at their source.
    assert model_validity._listing_billing("kimi-k3") == "paid"
    assert model_validity._listing_billing("deepseek-v4.1-flash-explabs") == "free"
