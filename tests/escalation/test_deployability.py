"""The deployability gate: what production holds at a decision, and what an eval may claim off it.

Both directions are pinned — a feature set that must pass, and one that must fail — because a gate
that only ever fires one way is indistinguishable from a constant.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import pytest

from benchmark.escalation import deployability, features, run_eval
from shunt.router.escalation import EscalationContext, FailureEvent
from tests.escalation.factories import make_step

# The FEATURE_NAMES columns whose source fields DO reach the production decision context. With the
# 2026-08-02 removal of the two `failing_check_id`-derived columns (`distinct_check_id_rate`,
# `max_key_repeat_rate`) this is exactly one: `fail_rate` reads `success`, which production holds.
# Named here so the pass-direction cases are a real subset of what the eval scores, not an
# invented set chosen to make the gate green.
_PROJECTABLE: Final[tuple[str, ...]] = ("fail_rate",)


# ── the context is read off the shipped code, not restated ──────────────────────────


def test_context_fields_are_derived_from_the_shipped_decision_signature() -> None:
    # The whole point of the seam: `decide_escalation`'s inputs ARE the production context, so a
    # field added to either dataclass changes this gate without anyone editing it. A hand-copied
    # list would silently keep gating on the old signature.
    shipped = (*dataclasses.fields(FailureEvent), *dataclasses.fields(EscalationContext))
    assert {f.name for f in shipped} == deployability.CONTEXT_FIELDS


def test_offline_only_step_fields_are_not_in_the_production_context() -> None:
    # `action`/`tool` are never passed to the escalation decision, and `is_infra_failure` is folded
    # into `blocking` by `derive_blocking` before the FailureEvent exists. A feature reading any of
    # them is offline-only by construction, however well it scores.
    sources = set(deployability.OFFLINE_SOURCE.values())
    assert {"action", "tool", "is_infra_failure", "observation", "status"}.isdisjoint(sources)


def test_every_scored_feature_declares_its_sources() -> None:
    # Drift guard. A new column in FEATURE_NAMES with no entry here is treated as UNSUPPORTED by
    # `assess` (fail-closed), but silently shipping an unmapped column is still the drift this
    # catches at the point it happens.
    assert set(deployability.FEATURE_SOURCES) == set(features.FEATURE_NAMES)


# ── projection ──────────────────────────────────────────────────────────────────────


def test_a_fully_populated_record_still_cannot_fill_the_ladder_ceilings() -> None:
    # The ceilings are resolved from the LIVE model pool at decision time. No replay record has
    # ever carried them, so even a maximally-populated step reports them as the mismatch.
    projected = deployability.project_step(make_step(failing_check_id="pkg::t", success=False))
    assert projected.missing == frozenset({"max_rank_index", "max_effort_index"})
    assert projected.values["dedup_key"] == "pkg::t"
    assert projected.values["current_rank_index"] == 0


def test_falsy_values_are_filled_not_missing() -> None:
    # `success=False`, `decision_index=0` and `loop_signal=False` are answers, not absences.
    # A falsiness check here would manufacture a mismatch that is not there.
    projected = deployability.project_step(
        make_step(decision_index=0, success=False, failing_check_id="k")
    )
    assert projected.values["success"] is False
    assert projected.values["decision_index"] == 0
    assert projected.values["loop_health_alarm"] is False


def test_a_record_missing_fields_reports_them_rather_than_imputing() -> None:
    # The shipped corpus's shape: rank/effort never captured, and a passing step has no check id.
    bare = dataclasses.replace(
        make_step(success=True), rank_index=None, effort_index=None, exit_code=None
    )
    projected = deployability.project_step(bare)
    assert {
        "current_rank_index",
        "current_effort_index",
        "exit_code",
        "dedup_key",
    } <= projected.missing
    assert "current_rank_index" not in projected.values


def test_unfilled_is_measured_over_the_corpus_not_assumed_per_record() -> None:
    # A field one record misses is not a corpus mismatch; a field EVERY record misses is. The
    # ladder position is unfilled corpus-wide, while `dedup_key` is filled by the failing step.
    steps = [
        dataclasses.replace(make_step(success=True), rank_index=None, effort_index=None),
        dataclasses.replace(
            make_step(success=False, failing_check_id="k"), rank_index=None, effort_index=None
        ),
    ]
    unfilled = deployability.unfilled_context_fields(steps)
    assert {"current_rank_index", "current_effort_index"} <= unfilled
    assert "dedup_key" not in unfilled
    assert "success" not in unfilled


def test_an_empty_corpus_fills_nothing() -> None:
    assert deployability.unfilled_context_fields([]) == deployability.CONTEXT_FIELDS


# ── the gate, both directions ───────────────────────────────────────────────────────


def test_a_projectable_feature_set_at_the_production_cadence_is_deployable() -> None:
    verdict = deployability.assess(_PROJECTABLE, deployability.Cadence.SESSION)
    assert verdict.deployable
    assert verdict.label == "DEPLOYABLE ESTIMATE"
    assert verdict.supported == _PROJECTABLE
    assert verdict.unsupported == ()


def test_the_currently_evaluated_feature_set_is_not_deployable() -> None:
    # The result this gate exists to make unquotable-as-deployable. It fails on BOTH axes: two
    # columns read fields production never receives, and the cadence is a step boundary while
    # production decides once per session.
    verdict = deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP)
    assert not verdict.deployable
    assert verdict.label == "OFFLINE-ONLY UPPER BOUND"
    assert set(verdict.unsupported) == {"infra_rate", "max_action_repeat_rate"}
    assert set(verdict.supported) == set(_PROJECTABLE)
    assert "production decision context" in verdict.reason
    assert "cadence" in verdict.reason


def test_the_cadence_alone_can_sink_an_otherwise_projectable_set() -> None:
    # Same features that pass above; only the cadence changes. This isolates the cadence check
    # from the field check, so a gate that passed on fields alone cannot look correct.
    verdict = deployability.assess(_PROJECTABLE, deployability.Cadence.STEP)
    assert not verdict.deployable
    assert verdict.unsupported == ()
    assert verdict.starved == ()
    assert "decides once per 'session'" in verdict.reason


def test_a_feature_on_an_unfilled_production_field_is_starved_not_supported() -> None:
    # The class the "8 fields at 0.0%" census is about: production HAS the field, but if no corpus
    # record carries it, a model fitted on it was fitted on nothing. The 2026-08-02 feature
    # reduction removed the two `failing_check_id`-derived columns that USED to be the way this
    # class reached FEATURE_NAMES (the old fixture starved `distinct_check_id_rate` /
    # `max_key_repeat_rate` via `unfilled={"dedup_key"}`); `fail_rate` is the one remaining
    # projectable column, so `unfilled={"success"}` is the synthetic version of the same
    # mechanism — a field production holds that no record fills starves the feature that reads it.
    verdict = deployability.assess(
        _PROJECTABLE, deployability.Cadence.SESSION, unfilled=frozenset({"success"})
    )
    assert not verdict.deployable
    assert set(verdict.starved) == {"fail_rate"}
    assert verdict.supported == ()
    assert "no corpus record fills" in verdict.reason


def test_an_undeclared_feature_fails_closed() -> None:
    # A column nobody mapped is never assumed deployable — the safe default for a gate whose
    # whole job is to refuse an unearned claim.
    verdict = deployability.assess(("mystery_column",), deployability.Cadence.SESSION)
    assert not verdict.deployable
    assert verdict.unsupported == ("mystery_column",)


# ── the counting mode: the third mismatch, derived from the same context map ────────


def test_every_counting_mode_declares_its_sources() -> None:
    # Same drift guard as the features one. `assess` fails closed on an unmapped mode, but a mode
    # added to the enum without a source declaration is drift worth catching where it happens.
    assert set(deployability.COUNTING_SOURCES) == set(deployability.CountingMode)


def test_the_shipped_counter_reads_nothing_production_lacks() -> None:
    # The pass direction. `counts_as_failure` gates on the FailureEvent alone, so the as-shipped
    # counter adds no mismatch — a verdict that failed here would be gating on nothing real.
    verdict = deployability.assess(
        _PROJECTABLE,
        deployability.Cadence.SESSION,
        counting=deployability.CountingMode.AS_SHIPPED,
    )
    assert verdict.deployable
    assert verdict.counting_unsupported == ()


def test_the_edit_gated_counter_alone_sinks_an_otherwise_deployable_result() -> None:
    # THE DEFECT THIS GATE EXISTS FOR. Features projectable, cadence production's — and still not
    # deployable, because the edit gate decides on `StepView.action`, which no production decision
    # holds and no EscalationPolicy knob can ask for. Before this, the headline number's only
    # marker that it described a configuration the product cannot run was prose in seven files.
    verdict = deployability.assess(
        _PROJECTABLE,
        deployability.Cadence.SESSION,
        counting=deployability.CountingMode.EDIT_GATED,
    )
    assert not verdict.deployable
    assert verdict.label == "OFFLINE-ONLY UPPER BOUND"
    assert verdict.counting_unsupported == ("action",)
    assert verdict.unsupported == ()
    assert verdict.starved == ()
    assert "no such counting mode" in verdict.reason
    assert verdict.to_dict()["counting"] == "edit_gated"


def test_the_counting_verdict_is_derived_from_the_context_map_not_restated() -> None:
    # The classification is not a second opinion: `action` is unsupported precisely because it is
    # absent from OFFLINE_SOURCE's values, the same reference the feature check uses. If `action`
    # ever became part of the production context, this mode would clear on its own.
    assert "action" in deployability.COUNTING_SOURCES[deployability.CountingMode.EDIT_GATED]
    assert "action" not in set(deployability.OFFLINE_SOURCE.values())


def test_an_empty_feature_set_is_not_a_free_pass_at_the_wrong_cadence() -> None:
    # Degenerate input must not sneak through: no features to disqualify, but the cadence still is.
    assert not deployability.assess((), deployability.Cadence.STEP).deployable
    assert deployability.assess((), deployability.Cadence.SESSION).deployable


# ── the wiring: a verdict that is never read is not a gate ──────────────────────────


def test_a_report_cannot_be_built_without_stating_deployability() -> None:
    # Structural, not advisory: the field carries no default, so `EvalReport(...)` fails rather
    # than defaulting to a comfortable answer nobody chose.
    with pytest.raises(TypeError):
        run_eval.EvalReport(  # type: ignore[call-arg]
            status="OK",
            reason="hand-built",
            n_trajectories=0,
            n_stamped=0,
            n_multistep=0,
            authenticity_errors=0,
            policy_cells=[],
            depth_reports=[],
            coverage=[],
        )


def _report(verdict: deployability.Deployability) -> run_eval.EvalReport:
    return run_eval.EvalReport(
        status="OK",
        reason="hand-built",
        n_trajectories=1,
        n_stamped=1,
        n_multistep=1,
        authenticity_errors=0,
        policy_cells=[],
        depth_reports=[],
        coverage=[],
        deployability=verdict,
    )


def test_the_verdict_reaches_the_json_report() -> None:
    payload = _report(deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP))
    block = payload.to_dict()["deployability"]
    assert isinstance(block, dict)
    assert block["label"] == "OFFLINE-ONLY UPPER BOUND"
    assert block["deployable"] is False
    assert block["cadence"] == "step"
    assert block["production_cadence"] == "session"
    assert sorted(block["unsupported_features"]) == ["infra_rate", "max_action_repeat_rate"]  # type: ignore[arg-type]


def test_the_canonical_cells_own_verdict_reaches_the_json_report() -> None:
    # The shipped counter's verdict is not a stand-in for the canonical cell's: it names two
    # mismatches where the canonical cell has three. Both blocks are published so a reader
    # comparing the two families cannot take one scope statement for both.
    report = run_eval.EvalReport(
        status="OK",
        reason="hand-built",
        n_trajectories=1,
        n_stamped=1,
        n_multistep=1,
        authenticity_errors=0,
        policy_cells=[],
        depth_reports=[],
        coverage=[],
        deployability=deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP),
        canonical_deployability=deployability.assess(
            features.FEATURE_NAMES,
            deployability.Cadence.STEP,
            counting=deployability.CountingMode.EDIT_GATED,
        ),
    )
    block = report.to_dict()["canonical_deployability"]
    assert isinstance(block, dict)
    assert block["counting"] == "edit_gated"
    assert block["counting_unsupported_fields"] == ["action"]
    # And it lands on the canvases drawn from that cell, as a limitation, not only in the JSON.
    limits = run_eval._canonical_annotations(report).limitations
    assert any("EVAL-ONLY COUNTER" in lim for lim in limits)


def test_a_figure_drawn_from_the_shipped_counter_carries_no_eval_only_line() -> None:
    # The other direction. If this fired unconditionally it would be decoration, not a gate.
    report = _report(deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP))
    assert run_eval._canonical_annotations(report).limitations == ()


def test_the_verdict_reaches_every_figure_footer() -> None:
    notes = run_eval._run_annotations(
        _report(deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP))
    ).notes
    assert any("OFFLINE-ONLY UPPER BOUND" in note for note in notes)


def test_the_verdict_is_printed_beside_the_status(capsys: pytest.CaptureFixture[str]) -> None:
    run_eval._print_summary(
        _report(deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP))
    )
    out = capsys.readouterr().out
    assert "deployability: OFFLINE-ONLY UPPER BOUND" in out
    assert out.index("status:") < out.index("deployability:")
