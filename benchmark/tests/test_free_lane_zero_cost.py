"""The $0 interlock: SH018, FREE_LANE_BILLED, the generalised free-lane refusal,
the per-run breaker, and the --require-zero-cost flag.

Hermetic: the real (untracked) overlay registry is configured, but no model call is made —
preflight and the runner are monkeypatched, so nothing can bill.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark import config
from benchmark.routing import validate
from benchmark.runner import free_tier_smoke, run_matrix

REPO: Path = Path(__file__).resolve().parents[2]
OVERLAY: Path = REPO / "configs" / "free-tier" / "overlay.yaml"
LANE: str = "requesty-gemma-4-31b-it"


@pytest.fixture(autouse=True)
def _overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh caches with the overlay configured (the collection path under test)."""
    # `_config` is reset too: another test file may leave the module's loaded config cached with
    # a free/collection-only model in its `models:` list, which makes `enabled_models()` refuse a
    # later unrelated test. Reset it so the collection path starts from the shipped default.
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))


def _row(model: str, **over: object) -> dict:
    """A minimally-valid row for *model*; override a field to construct a violation."""
    base: dict = {
        "challenge_id": "astropy__astropy-1",
        "model": model,
        "reasoning": "default",
        "pass": "False",
        "cost": "0",
        "in_tok": "100",
        "out_tok": "50",
        "calls": "7",
        "version_hash": "vh",
        "model_version": model,
        "arm_hash": "",
        "real_cost": "0",
        "estimated_cost": "0",
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


# ── layer 4: FREE_LANE_BILLED (the mirror of ACCOUNTING_HOLE) ─────────────────────────


def test_enforce_row_raises_when_an_overlay_lane_is_billed() -> None:
    pricing = dict(config.free_registry())
    row = _row(LANE, real_cost="0.02", cost="0.02")
    with pytest.raises(validate.DataIntegrityError) as excinfo:
        validate.enforce_row(row, pricing, free_collection_models={LANE})
    assert validate.FREE_LANE_BILLED in {v.code for v in excinfo.value.violations}


def test_overlay_zero_row_stays_clean_and_non_overlay_zero_still_flags_accounting() -> None:
    # The negative control: the two walls are symmetric, so neither can be satisfied
    # accidentally by a row that is not genuinely free.
    pricing = dict(config.free_registry())
    clean = {v.code for v in validate.validate_row(_row(LANE), pricing)}
    assert validate.FREE_LANE_BILLED not in clean
    assert validate.ACCOUNTING_HOLE not in clean

    pricing["not-an-overlay-model"] = {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}
    hole = {v.code for v in validate.validate_row(_row("not-an-overlay-model"), pricing)}
    assert validate.ACCOUNTING_HOLE in hole
    assert validate.FREE_LANE_BILLED not in hole


def test_build_row_aborts_the_run_when_an_overlay_lane_is_billed() -> None:
    # _build_row is the write-time wall; _run_one_cell re-raises its DataIntegrityError past
    # the broad per-cell handler, so a billed free lane aborts the whole run. The lane is
    # named as run provenance (the run admitted it), not inferred from a `-explabs` suffix.
    outcome: dict[str, Any] = {
        "pass": False,
        "real_cost": 0.02,
        "in_tok": 100,
        "out_tok": 50,
        "calls": 7,
        "stop_reason": "unsolved",
        "computed_at": "2026-09-10T00:00:00+00:00",
    }
    with pytest.raises(validate.DataIntegrityError):
        run_matrix._build_row(
            "astropy__astropy-1",
            LANE,
            outcome,
            {"astropy__astropy-1": "vh"},
            {LANE: LANE},
            {LANE: {"input": 1.0, "output": 2.0}},
            free_collection_models={LANE},
        )


def test_billed_synthesized_collection_slug_trips_free_lane_billed() -> None:
    # A synthesized `--extra-models` slug (S-explabs, no overlay row) that bills must trip the
    # SAME wall as an overlay row — driven by the run's admitted-free provenance, not by the
    # suffix. Regression: the suffix alone made PAID historical `-explabs` rows unreadable.
    slug = "qwen3.7-plus-explabs"
    row = _row(slug, real_cost="0.04", cost="0.04")
    # Without run provenance the suffix is not proof of a free lane (paid history).
    assert validate.FREE_LANE_BILLED not in {v.code for v in validate.validate_row(row, {})}
    # With the run's admitted-free provenance the wall fires.
    violations = validate.validate_row(row, {}, free_collection_models={slug})
    assert validate.FREE_LANE_BILLED in {v.code for v in violations}
    with pytest.raises(validate.DataIntegrityError) as excinfo:
        validate.enforce_row(row, {}, free_collection_models={slug})
    assert validate.FREE_LANE_BILLED in {v.code for v in excinfo.value.violations}


def test_paid_explabs_history_is_not_a_free_lane_without_provenance() -> None:
    # glm-5.3-explabs has 6 genuinely PAID rows in committed results.csv (~$4.60). A plain
    # corpus scan has no run provenance, so the `-explabs` suffix must not label it free and
    # must not raise FREE_LANE_BILLED.
    pricing = {"glm-5.3-explabs": {"input_cost_per_1m": 1.4, "output_cost_per_1m": 4.4}}
    row = _row("glm-5.3-explabs", real_cost="1.68", cost="1.68", calls="9")
    assert validate.FREE_LANE_BILLED not in {v.code for v in validate.validate_row(row, pricing)}
    report = validate.validate_results([row], pricing)
    assert report.error_count == 0


def test_free_lane_and_accounting_walls_are_symmetric_with_provenance() -> None:
    # One admitted-free set drives both walls: an admitted lane that billed is a
    # FREE_LANE_BILLED, while a non-overlay paid model that ran for $0 is still the
    # ACCOUNTING_HOLE the $35 miss depends on.
    free = {"qwen3.7-plus-explabs"}
    pricing = {
        "qwen3.7-plus-explabs": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0},
        "paid-direct": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0},
    }
    billed = _row("qwen3.7-plus-explabs", real_cost="0.04", cost="0.04", calls="7")
    billed_codes = {
        v.code for v in validate.validate_row(billed, pricing, free_collection_models=free)
    }
    assert validate.FREE_LANE_BILLED in billed_codes
    assert validate.ACCOUNTING_HOLE not in billed_codes

    hole = _row(
        "paid-direct",
        model_version="paid-direct",
        real_cost="0",
        calls="7",
        stop_reason="unsolved",
        **{"pass": "False"},
    )
    hole_codes = {v.code for v in validate.validate_row(hole, pricing, free_collection_models=free)}
    assert validate.ACCOUNTING_HOLE in hole_codes
    assert validate.FREE_LANE_BILLED not in hole_codes


def test_free_lane_billed_requires_run_provenance() -> None:
    pricing = {
        "brand-new-catalog-model-explabs": {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}
    }
    row = _row("brand-new-catalog-model-explabs", real_cost="0.04", cost="0.04", calls="7")
    assert validate.FREE_LANE_BILLED not in {v.code for v in validate.validate_row(row, pricing)}
    assert validate.FREE_LANE_BILLED in {
        v.code
        for v in validate.validate_row(
            row, pricing, free_collection_models={"brand-new-catalog-model-explabs"}
        )
    }


def test_build_row_aborts_on_a_billed_synthesized_collection_slug() -> None:
    # The write path the runner actually drives: a synthesized `-explabs` slug admitted as a
    # free lane and returning real_cost>0 must abort at _build_row via run provenance.
    slug = "brand-new-catalog-model-explabs"
    outcome: dict[str, Any] = {
        "pass": False,
        "real_cost": 0.04,
        "in_tok": 100,
        "out_tok": 50,
        "calls": 7,
        "stop_reason": "unsolved",
        "computed_at": "2026-09-10T00:00:00+00:00",
    }
    with pytest.raises(validate.DataIntegrityError) as excinfo:
        run_matrix._build_row(
            "astropy__astropy-1",
            slug,
            outcome,
            {"astropy__astropy-1": "vh"},
            {slug: "brand-new-catalog-model"},
            {slug: {"input": 1.0, "output": 2.0}},
            free_collection_models={slug},
        )
    assert validate.FREE_LANE_BILLED in {v.code for v in excinfo.value.violations}


def test_run_live_cells_threads_free_lane_provenance_to_the_write_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end: run_live_cells derives the admitted-free set from its own cells and passes
    # it through _LiveContext to _build_row, so a billed free lane aborts even though no
    # caller handed the set in directly.
    from benchmark.runner import infer

    slug = "brand-new-catalog-model-explabs"

    def fake_run_live_cell(cid: str, model: str, **kwargs: object) -> dict:
        return {
            "pass": False,
            "real_cost": 0.04,
            "in_tok": 100,
            "out_tok": 50,
            "calls": 7,
            "stop_reason": "unsolved",
            "computed_at": "2026-09-10T00:00:00+00:00",
        }

    monkeypatch.setattr(infer, "run_live_cell", fake_run_live_cell)
    with pytest.raises(validate.DataIntegrityError) as excinfo:
        run_matrix.run_live_cells(
            [("astropy__astropy-1", slug, "default")],
            {},
            {"astropy__astropy-1": "vh"},
            {slug: "brand-new-catalog-model"},
            timeout=10,
            verbose=False,
        )
    assert validate.FREE_LANE_BILLED in {v.code for v in excinfo.value.violations}


def _billed_poison() -> validate.DataIntegrityError:
    return validate.DataIntegrityError(
        [
            validate.Violation(
                validate.Severity.ERROR,
                validate.FREE_LANE_BILLED,
                "free-lane model 'billed' was billed real_cost=0.04",
            )
        ]
    )


def test_a_billed_lane_disables_that_lane_and_continues_the_batch(monkeypatch) -> None:
    # FLEET FATALITY (from the real crash F0-1): one billed free lane must not abort the whole
    # campaign. The poison row is never written, the lane is disabled by name, and the other
    # lane's cell still lands.
    from benchmark.runner import lane_scheduler as ls

    calls: list[tuple[str, str, str]] = []

    def fake_run(cell: tuple[str, str, str], _ctx: object) -> dict:
        calls.append(cell)
        if cell[1] == "billed":
            raise _billed_poison()
        return {
            "challenge_id": cell[0],
            "model": cell[1],
            "reasoning": cell[2],
            "real_cost": 0.0,
            "calls": 1,
            "in_tok": 1,
            "out_tok": 1,
        }

    monkeypatch.setattr(run_matrix, "_run_one_cell", fake_run)
    scheduler = ls.LaneScheduler(
        limits={
            "good": ls.LaneLimits(rpm=10, rpd=1000),
            "billed": ls.LaneLimits(rpm=10, rpd=1000),
        },
        reserves={"good": 1, "billed": 1},
        stall_timeout_s=0.01,
    )
    batch = [("c1", "good", "default"), ("c2", "billed", "default")]
    rows, _spent, stopped = run_matrix._run_scheduled_batch(
        batch,
        run_matrix._LiveContext.__new__(run_matrix._LiveContext),
        scheduler,
        run_matrix._FailureTracker(None, None),
        None,
        0.0,
        None,
        None,
        "",
    )
    assert [row["model"] for row in rows] == ["good"]
    assert stopped is False
    # The billed cell ran exactly once — never re-run — and the poison row was not written.
    assert calls.count(("c2", "billed", "default")) == 1
    assert [row["model"] for row in rows].count("billed") == 0
    # The lane is disabled by a named persisted reason, and its state round-trips through JSON.
    disabled = scheduler.lane_state("billed").disabled_reason
    assert disabled is not None and validate.FREE_LANE_BILLED in disabled
    assert validate.FREE_LANE_BILLED in ls.dump_lane_state(dict(scheduler.state))


def test_billing_the_last_lane_aborts_the_campaign(monkeypatch) -> None:
    # Only abort when zero lanes remain: with a single billed lane there is nothing left.
    from benchmark.runner import lane_scheduler as ls

    def _raise(cell: tuple[str, str, str], _ctx: object) -> dict:
        raise _billed_poison()

    monkeypatch.setattr(run_matrix, "_run_one_cell", _raise)
    scheduler = ls.LaneScheduler(
        limits={"billed": ls.LaneLimits(rpm=10, rpd=1000)},
        reserves={"billed": 1},
        stall_timeout_s=0.01,
    )
    with pytest.raises(run_matrix.RunAbortError):
        run_matrix._run_scheduled_batch(
            [("c1", "billed", "default")],
            run_matrix._LiveContext.__new__(run_matrix._LiveContext),
            scheduler,
            run_matrix._FailureTracker(None, None),
            None,
            0.0,
            None,
            None,
            "",
        )


# ── layer 2: generalised free-lane refusal + run-start pre-flight ────────────────────


def test_free_tier_refusal_accepts_an_overlay_lane() -> None:
    from shunt.models.config import resolve_models

    overlay = config.load_free_registry()
    assert overlay is not None
    model = resolve_models(overlay)[LANE]
    assert free_tier_smoke.free_tier_refusal(model) is None


def test_free_tier_refusal_still_rejects_a_paid_legacy_model() -> None:
    from shunt.models.config import ModelConfig, Pricing

    model = ModelConfig(
        name="not-an-overlay-model",
        model_id="openai/gpt-4o",
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        litellm_prefix="openai",
        pricing=Pricing(
            input_cost_per_1m=2.5,
            output_cost_per_1m=10.0,
            price_provider="openai",
            price_source="https://example.test",
            price_as_of="2026-09-10",
        ),
    )
    assert free_tier_smoke.free_tier_refusal(model) is not None


def test_preflight_free_lanes_probes_every_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark.runner import infer

    probed: list[str] = []

    def fake_probe(lane: str) -> infer.PreflightOutcome:
        probed.append(lane)
        return infer.PreflightOutcome("ok", "healthy")

    report = run_matrix.preflight_free_lanes(True, ["lane-a", "lane-b"], probe=fake_probe)
    assert not report.refused
    assert report.usable == ["lane-a", "lane-b"]
    assert probed == ["lane-a", "lane-b"]


def test_preflight_free_lanes_refuses_a_flagged_overlay_lane_before_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.runner import infer

    monkeypatch.setattr(
        free_tier_smoke, "free_tier_refusal", lambda model: "not a verifiable $0 lane"
    )
    probed: list[str] = []

    def fake_probe(lane: str) -> infer.PreflightOutcome:
        probed.append(lane)
        return infer.PreflightOutcome("ok", "healthy")

    report = run_matrix.preflight_free_lanes(True, [LANE], probe=fake_probe)
    # A static $0 refusal disables the lane (named reason) without spending a probe.
    assert report.refused
    assert LANE in report.disabled
    assert "not a verifiable $0 lane" in report.disabled[LANE]
    assert probed == []  # refused before the real completion


# ── --require-zero-cost: fail-closed on a non-free run ───────────────────────────────


def test_require_zero_cost_refuses_a_paid_model() -> None:
    reason = run_matrix.require_zero_cost_refusal(live=True, enabled=["kimi-k3"], extra=[LANE])
    assert reason is not None
    assert "kimi-k3" in reason


def test_require_zero_cost_refuses_when_not_live() -> None:
    reason = run_matrix.require_zero_cost_refusal(live=False, enabled=[], extra=[LANE])
    assert reason is not None
    assert "live" in reason


def test_require_zero_cost_refuses_a_run_with_no_lanes() -> None:
    reason = run_matrix.require_zero_cost_refusal(live=True, enabled=[], extra=[])
    assert reason is not None


def test_require_zero_cost_refuses_a_disabled_scaffold_cap() -> None:
    # `cost_limit: 0` disables the cap rather than capping at $0, so the interlock is not
    # engaged — refuse before the run.
    reason = run_matrix.require_zero_cost_refusal(
        live=True, enabled=[], extra=[LANE], cost_limit=0.0
    )
    assert reason is not None
    assert "cost_limit" in reason


def test_require_zero_cost_accepts_overlay_only_run() -> None:
    assert run_matrix.require_zero_cost_refusal(live=True, enabled=[], extra=[LANE]) is None


def test_require_zero_cost_refuses_an_overlay_lane_on_a_provider_with_no_free_lane() -> None:
    # An overlay row on Together is collection provenance only: Together has no free subset,
    # so the $0 interlock must refuse it before anything bills.
    reason = run_matrix.require_zero_cost_refusal(
        live=True, enabled=[], extra=["together-muse-glimmer-30b"]
    )
    assert reason is not None
    assert "together-muse-glimmer-30b" in reason


def test_require_zero_cost_refuses_an_app_gated_overlay_lane() -> None:
    reason = run_matrix.require_zero_cost_refusal(
        live=True, enabled=[], extra=["opencode-deepseek-v4-flash-free"]
    )
    assert reason is not None
    assert "opencode-deepseek-v4-flash-free" in reason


def test_require_zero_cost_refuses_a_non_full_strategy() -> None:
    # The default strategy is cost_optimal, which has no overlay lane to meter; the flag must
    # fail closed rather than silently do nothing.
    import argparse

    args = argparse.Namespace(require_zero_cost=True, strategy="cost_optimal")
    assert run_matrix._dispatch(args) == 2


# ── layer 3: the process-wide breaker, and the campaign config ─────────────────────────


def test_require_zero_cost_arms_the_global_cost_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minisweagent import models as mswea_models

    monkeypatch.delenv("MSWEA_GLOBAL_COST_LIMIT", raising=False)
    monkeypatch.setattr(mswea_models.GLOBAL_MODEL_STATS, "cost_limit", 0.0)
    run_matrix._arm_free_lane_cost_breaker()
    assert os.environ["MSWEA_GLOBAL_COST_LIMIT"] == run_matrix.FREE_LANE_GLOBAL_COST_LIMIT
    assert mswea_models.GLOBAL_MODEL_STATS.cost_limit == 0.05


def test_free_campaign_config_never_disables_the_scaffold_cap() -> None:
    # `cost_limit: 0` DISABLES the scaffold cap (mini-swe-agent gates on `0 < cost_limit`),
    # so the campaign must set a positive per-cell cap.
    path = REPO / "configs" / "free-tier" / "benchmark.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["live"]["cost_limit"] > 0
    assert document["paths"]["results_csv"] == "routing/results_free.csv"


# ── layer 1: SH018 is wired as a hook and refuses an overlay $0 price ────────────────
# (The pure gate's behaviour is covered in tests/lint/test_lint_checks.py; this proves the
# pre-commit registration exists and targets the real overlay.)


def test_sh018_hook_is_registered() -> None:
    text = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "sh018-free-registry-zero" in text
    assert "tools/lint/check_free_registry_zero.py" in text


def test_sh018_passes_on_the_real_overlay() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "lint" / "check_free_registry_zero.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
