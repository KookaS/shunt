"""run_eval end-to-end: the null gates the status, the shipped default is never silently moved,
and every figure is written through the annotated frame.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Final

import pytest

from benchmark.escalation import (
    deployability,
    features,
    metrics,
    policy_eval,
    run_eval,
    schema,
)
from tests.escalation.factories import make_depth_report, make_null, make_step, make_trajectory

pytest.importorskip("sklearn")

_PERMUTATIONS = 200
_DEPTHS: Final[tuple[int, ...]] = (5,)
_FIGURES: Final[frozenset[str]] = frozenset(
    {
        "pr_curve.png",
        "roc_curve.png",
        "recurrence_roc.png",
        "confusion_matrix.png",
        "permutation_null.png",
        "sweep_table.png",
        "trajectory_outcomes.png",
        "failure_capture_coverage.png",
        "edit_gated_sweep_table.png",
        "session_cadence.png",
    }
)


# A run must leave `features.MIN_WITHHELD` non-terminal steps unread after the prefix, so a
# depth-5 fixture needs 5 + MIN_WITHHELD scorable steps plus the label-stamped terminal one.
# Derived, never a literal: the margin is a measured property of the corpus and it moves.
_RUN_LENGTH: Final[int] = _DEPTHS[0] + features.MIN_WITHHELD + 2


def _run(tid: str, *, resolved: bool, rng: random.Random, fail_rate: float, n: int = _RUN_LENGTH):  # type: ignore[no-untyped-def]
    steps = [
        make_step(
            step_index=i,
            decision_index=i,
            action=f"cmd{rng.randrange(4)}",
            success=rng.random() >= fail_rate,
            failing_check_id="pkg::t" if rng.random() < fail_rate else None,
        )
        for i in range(n - 1)
    ]
    steps.append(make_step(step_index=n - 1, decision_index=n - 1, success=resolved))
    return make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)


def _null_corpus(n_pairs: int = 22):  # type: ignore[no-untyped-def]
    """Both classes behave identically: there is nothing to find, and the report must say so."""
    from dataclasses import replace

    rng = random.Random(4)
    out = []
    for i in range(n_pairs):
        for resolved, suffix in ((False, "a"), (True, "b")):
            traj = _run(f"c{i}__m__{suffix}", resolved=resolved, rng=rng, fail_rate=0.5)
            out.append(replace(traj, header=replace(traj.header, instance_id=f"c{i}")))
    return out


def _signal_corpus(n_pairs: int = 22):  # type: ignore[no-untyped-def]
    """Failed runs thrash, resolved runs do not — genuine prefix evidence at the same length."""
    from dataclasses import replace

    rng = random.Random(6)
    out = []
    for i in range(n_pairs):
        for resolved, suffix, rate in ((False, "a", 0.9), (True, "b", 0.1)):
            traj = _run(f"c{i}__m__{suffix}", resolved=resolved, rng=rng, fail_rate=rate)
            out.append(replace(traj, header=replace(traj.header, instance_id=f"c{i}")))
    return out


@pytest.fixture(scope="module")
def null_report() -> run_eval.EvalReport:
    """The pipeline is deterministic; refitting it per test only multiplies permutation cost."""
    return run_eval.evaluate(_null_corpus(), depths=_DEPTHS, n_permutations=_PERMUTATIONS)


@pytest.fixture(scope="module")
def signal_report() -> run_eval.EvalReport:
    return run_eval.evaluate(_signal_corpus(), depths=_DEPTHS, n_permutations=_PERMUTATIONS)


def test_a_pure_null_reports_no_skill_not_ok(null_report: run_eval.EvalReport) -> None:
    # The old gate was `auprc > prevalence` with no noise floor: a +0.0008 excess against a null sd
    # of 0.00055 printed `status: OK`. The gate is now the permutation band.
    report = null_report
    assert report.status == "NO_SKILL"
    assert "skill conditions" in report.reason
    assert not any(d.has_skill for d in report.depth_reports)


def test_offline_signal_reports_ok_offline_only(signal_report: run_eval.EvalReport) -> None:
    report = signal_report
    # The badge must not over-read: this corpus is scored at STEP cadence, so the cell that clears
    # the null is an OFFLINE-ONLY UPPER BOUND, not a shippable "OK". The status value says so.
    assert report.status == "OK_OFFLINE_ONLY"
    assert report.deployability.deployable is False
    # The null named in the OK sentence is the one the gate applied — family-wise across the
    # reported depths (here a family of one, which is why real signal still clears it).
    assert "clears its family-wise permutation null" in report.reason
    assert "OFFLINE-ONLY UPPER BOUND" in report.reason


def test_a_deployable_signal_reports_plain_ok() -> None:
    # A null-clearing cell on a DEPLOYABLE estimate — every scored feature projects onto the live
    # decision context AND the cadence is production's — keeps the plain "OK" badge. Only the
    # projecting feature subset is deployable at SESSION cadence: the full FEATURE_NAMES set is
    # non-deployable at ANY cadence, because infra_rate and max_action_repeat_rate read fields the
    # production decision never sees.
    from benchmark.escalation import policy_eval

    deployable = deployability.assess(("fail_rate",), deployability.Cadence.SESSION)
    assert deployable.deployable
    cell = policy_eval.PolicyCell(
        escalate_after_n=30,
        stale_window=1000,
        ladder="rank_only",
        n_trajectories=100,
        n_escalated=30,
        tp=20,
        fp=10,
        fn=10,
        tn=60,
        base_failure_rate=0.3,
        precision_ci=(0.5, 0.9),
        quiet_ci=(0.1, 0.3),
        null_auroc=make_null(0.66, 0.55, p_value=0.005),
    )
    assert cell.has_skill
    status, reason = run_eval._status([cell], [], 0, deployable)
    assert status == "OK"
    assert "OFFLINE-ONLY UPPER BOUND" not in reason
    report = run_eval.EvalReport(
        status=status,
        reason=reason,
        n_trajectories=100,
        n_stamped=100,
        n_multistep=100,
        authenticity_errors=0,
        policy_cells=[cell],
        depth_reports=[],
        coverage=[],
        deployability=deployable,
    )
    assert report.status == "OK"
    assert report.deployability.deployable is True


def test_no_skill_reason_carries_the_number_not_just_the_verdict(
    null_report: run_eval.EvalReport,
) -> None:
    assert "incremental AUROC" in null_report.reason
    report = null_report
    assert "p=" in report.reason


def test_no_skill_reason_names_the_condition_that_actually_failed() -> None:
    # Regression: the message hardcoded "no depth clears its permutation null" and then printed an
    # incremental AUROC sitting OUTSIDE that null's band — the sentence contradicted its own
    # numbers on two committed figure canvases. The failing condition here is the PREFIX-only null.
    depth = make_depth_report(
        auroc_prefix=0.428,
        incremental_auroc=0.144,
        null_prefix=make_null(0.428, 0.510, p_value=0.975),
        null_incremental=make_null(0.144, 0.135, p_value=0.015),
        ci_incremental=(0.003, 0.290),
    )
    assert depth.null_incremental.beats_null  # the increment DOES clear its null
    assert not depth.null_prefix.beats_null  # the prefix-only score does not
    assert not depth.has_skill  # ...so the gate still refuses, and must keep refusing

    status, reason = run_eval._status([], [depth])
    assert status == "NO_SKILL"
    # Behavioural, not a frozen sentence: every unmet condition is named, every met one is not.
    for met, clause in run_eval._skill_conditions(depth):
        assert (clause in reason) is (not met)
    # ...and the cleared increment is reported as cleared, not silently claimed to have failed.
    assert f"{0.144:+.3f}" in reason


def test_best_depth_never_headlines_a_rank_deficient_design() -> None:
    # Depth 5's design on the rebuilt corpus ranks 3 of 4 (412 of 414 rows carry one vector), so
    # its incremental is arithmetic, not evidence — and removing depth 20 from the ladder made it
    # the plain `max(incremental)` winner. The headline depth must come from a design the eval
    # could actually evaluate; the degenerate one is excluded, not preferred.
    degenerate = replace(make_depth_report(depth=5, incremental_auroc=0.1), design_full_rank=False)
    estimable = replace(make_depth_report(depth=10, incremental_auroc=0.03), design_full_rank=True)
    report = run_eval.EvalReport(
        status="NO_SKILL",
        reason="hand-built",
        n_trajectories=2,
        n_stamped=2,
        n_multistep=2,
        authenticity_errors=0,
        policy_cells=[],
        depth_reports=[degenerate, estimable],
        coverage=[],
        deployability=deployability.assess((), deployability.Cadence.SESSION),
    )
    assert report.best_depth is estimable
    # The no-skill sentence follows the same preference: name the depth that could be measured,
    # never the arithmetic one.
    status, reason = run_eval._status([], [degenerate, estimable])
    assert status == "NO_SKILL"
    assert "at depth 10" in reason
    assert "at depth 5" not in reason
    # ...and the fallback still reports SOMETHING when no depth ranks full.
    both_degenerate = [replace(d, design_full_rank=False) for d in (degenerate, estimable)]
    assert (
        run_eval.EvalReport(
            status="NO_SKILL",
            reason="hand-built",
            n_trajectories=2,
            n_stamped=2,
            n_multistep=2,
            authenticity_errors=0,
            policy_cells=[],
            depth_reports=both_degenerate,
            coverage=[],
            deployability=deployability.assess((), deployability.Cadence.SESSION),
        ).best_depth
        is both_degenerate[0]
    )  # the max-incremental one


def test_family_attribution_uses_identity_not_structural_equality() -> None:
    # `best in edit_gated_cells` used structural equality, so a report_cells cell that is
    # field-for-field equal to an edit-gated twin was credited to the edit-gated family — and
    # `family_size` quoted the wrong family's multiplicity. Attribution must follow identity.
    skilled = policy_eval.PolicyCell(
        escalate_after_n=2,
        stale_window=1000,
        ladder="effort_then_rank",
        n_trajectories=100,
        n_escalated=50,
        tp=40,
        fp=10,
        fn=10,
        tn=40,
        base_failure_rate=0.5,
        precision_ci=(0.6, 0.9),
        quiet_ci=(0.1, 0.4),
        null_auroc=make_null(0.7, 0.55, p_value=0.005),
        null_auroc_family=make_null(0.7, 0.55, p_value=0.005),
    )
    assert skilled.has_skill
    twin = replace(skilled)
    assert twin == skilled  # structurally equal...
    assert twin is not skilled  # ...but a different object
    status, reason = run_eval._status([skilled], [], 0, None, [twin])
    assert status == "OK"
    # The best cell is `skilled` (the report_cells copy, first in the pool); it must be named
    # as-shipped, not edit-gated — the twin in the edit-gated list is a different cell. (The
    # sentence still mentions the edit-gated FAMILY in its selection clause, so the assertion is
    # on the family label after "policy (", not on the bare word.)
    assert "escalation policy (as-shipped)" in reason
    assert "escalation policy (edit-gated" not in reason


def test_the_ok_reason_discloses_the_run_length_share() -> None:
    # A skilled cell's reason must carry `_length_disclosure`: how much of its AUROC a pure
    # length>=t predictor gets at the same flag count, and whether the recurrence excess clears
    # the length-stratified null. Without it an "OK" hides that the edge may be run-length
    # selection (the exact caveat that made the n=30 cell look better than it was).
    cell = policy_eval.PolicyCell(
        escalate_after_n=2,
        stale_window=1000,
        ladder="effort_then_rank",
        n_trajectories=100,
        n_escalated=50,
        tp=40,
        fp=10,
        fn=10,
        tn=40,
        base_failure_rate=0.5,
        precision_ci=(0.6, 0.9),
        quiet_ci=(0.1, 0.4),
        null_auroc=make_null(0.7, 0.55, p_value=0.005),
        null_auroc_family=make_null(0.7, 0.55, p_value=0.005),
        length_baseline_auroc=0.570,
        null_auroc_length=make_null(0.7, 0.62, p_value=0.01),
    )
    assert cell.has_skill
    status, reason = run_eval._status([cell], [], 0, None)
    assert status == "OK"
    assert "run length alone scores 0.570" in reason
    assert "the recurrence excess clears the length-stratified null" in reason


def test_best_skilled_cell_ranks_across_both_families_by_auroc() -> None:
    # `best_skilled_cell` is the detection headline: AUROC first, precision second, reading across
    # BOTH families. A higher-AUROC edit-gated cell must outrank an as-shipped one even when the
    # as-shipped cell has the higher precision (the old precision-only ranking would headline the
    # 71-run n=30 tail cell over the 456-run n=2 cell that actually separates runs).
    as_shipped = policy_eval.PolicyCell(
        escalate_after_n=2,
        stale_window=1000,
        ladder="effort_then_rank",
        n_trajectories=100,
        n_escalated=50,
        tp=40,
        fp=10,
        fn=10,
        tn=40,
        base_failure_rate=0.5,
        precision_ci=(0.6, 0.9),
        quiet_ci=(0.1, 0.4),
        null_auroc=make_null(0.6, 0.55, p_value=0.005),
        null_auroc_family=make_null(0.6, 0.55, p_value=0.005),
    )
    edit_gated = replace(
        as_shipped,
        null_auroc=make_null(0.8, 0.55, p_value=0.005),
        null_auroc_family=make_null(0.8, 0.55, p_value=0.005),
    )
    assert as_shipped.has_skill and edit_gated.has_skill
    assert edit_gated.precision == as_shipped.precision
    assert run_eval._best_skilled_cell([as_shipped, edit_gated]) is edit_gated
    report = run_eval.EvalReport(
        status="OK_OFFLINE_ONLY",
        reason="hand-built",
        n_trajectories=100,
        n_stamped=100,
        n_multistep=100,
        authenticity_errors=0,
        policy_cells=[as_shipped],
        policy_cells_edit_gated=[edit_gated],
        depth_reports=[],
        coverage=[],
        deployability=deployability.assess((), deployability.Cadence.STEP),
    )
    assert report.best_skilled_cell is edit_gated


def test_the_status_gate_reads_the_family_wise_null_not_the_per_depth_one() -> None:
    # THE DEFECT THIS PINS. `_status` says OK if ANY depth clears and `_no_skill_reason` reports the
    # max-incremental depth, so `DEFAULT_DEPTHS`'s depths are a max over that many tests — and
    # each was gated at its own nominal 2.5%, with no family-wise correction anywhere. This report
    # clears its per-depth nulls comfortably and MUST still be refused, because the max-statistic
    # band across the family sits above the observation.
    depth = replace(
        make_depth_report(
            auroc_prefix=0.62,
            incremental_auroc=0.05,
            null_prefix=make_null(0.62, 0.55),
            null_incremental=make_null(0.05, 0.02),
            ci_incremental=(0.01, 0.09),
        ),
        null_prefix_family=make_null(0.62, 0.70),
        null_incremental_family=make_null(0.05, 0.08),
    )
    assert depth.null_prefix.beats_null  # the UNCORRECTED nulls are both cleared...
    assert depth.null_incremental.beats_null
    assert not depth.has_skill  # ...and the corrected gate refuses anyway
    status, reason = run_eval._status([], [depth])
    assert status == "NO_SKILL"
    assert "family-wise" in reason
    # The numbers quoted are the family-wise band's, not the per-depth one's: a sentence citing the
    # uncorrected band beside a verdict taken on the corrected one is the contradiction this repo
    # already fixed once for a different pair of numbers.
    assert f"{0.70:.3f}" in reason
    assert f"{0.55:.3f}" not in reason


def test_headline_cell_is_the_shipped_default_not_an_argmax(
    signal_report: run_eval.EvalReport,
) -> None:
    # `best_config` used to be an argmax over a grid that held 2 distinct results, so the reported
    # configuration was a coin flip dressed as an optimisation. The headline is now the shipped
    # default by definition; a better cell is FLAGGED in the notes, never applied.
    report = signal_report
    assert report.headline_cell is not None
    assert report.headline_cell.escalate_after_n == run_eval.SHIPPED_ESCALATE_AFTER_N


def test_the_headline_matches_the_shipped_stale_window_too_not_just_n(
    signal_report: run_eval.EvalReport,
) -> None:
    # Matching `escalate_after_n` alone made the "shipped configuration" claim false: the swept
    # grid pins stale_window=5 where the product ships 10, and the two are NOT interchangeable
    # (`_in_window` admits at most `stale_window` events, so at n=20/sw=10 nothing can ever fire).
    from shunt.router.policy import load_router_policy, packaged_policy_path

    shipped = load_router_policy(packaged_policy_path()).escalation
    knobs = (shipped.escalate_after_n, shipped.stale_window, shipped.ladder)
    assert knobs == (
        run_eval.SHIPPED_ESCALATE_AFTER_N,
        run_eval.SHIPPED_STALE_WINDOW,
        run_eval.SHIPPED_LADDER,
    )
    cell = signal_report.headline_cell
    assert cell is not None
    assert (cell.escalate_after_n, cell.stale_window, cell.ladder) == knobs


def test_the_shipped_configuration_is_measured_even_when_the_grid_omits_it() -> None:
    # The registered grid pins a stale_window the product does not ship. Rather than silently
    # labelling `policy_cells[0]` "shipped", the shipped point is APPENDED and actually measured.
    from benchmark.escalation import replay

    grid = [replay.GridPoint(escalate_after_n=1, stale_window=999, ladder="rank_only")]
    # depths=() — this is a POLICY-half assertion; refitting the prefix model would only add cost.
    report = run_eval.evaluate(
        _signal_corpus(n_pairs=4), grid, depths=(), n_permutations=_PERMUTATIONS
    )
    assert len(report.policy_cells) == len(grid) + 1
    assert report.headline_cell is not None
    assert report.headline_cell.stale_window == run_eval.SHIPPED_STALE_WINDOW


def test_a_report_with_no_shipped_cell_has_no_headline_rather_than_a_mislabelled_one() -> None:
    # `policy_cells[0] if policy_cells else None` labelled an ARBITRARY cell "the shipped
    # configuration". Reporting none is honest; reporting the wrong one is not.
    from benchmark.escalation import deployability, features, policy_eval, replay

    corpus = _signal_corpus(n_pairs=3)
    off_grid = policy_eval.evaluate(
        corpus, [replay.GridPoint(9, 999, "rank_only")], n_permutations=_PERMUTATIONS
    )
    report = run_eval.EvalReport(
        # STEP-cadence deployability below: the corpus is not a deployable estimate, so a hand-built
        # report must not carry the shippable "OK" badge either — that is the over-read this pins.
        status="OK_OFFLINE_ONLY",
        reason="hand-built",
        n_trajectories=len(corpus),
        n_stamped=len(corpus),
        n_multistep=len(corpus),
        authenticity_errors=0,
        policy_cells=off_grid,
        depth_reports=[],
        coverage=[],
        # Required, so even a hand-built report must state whether its numbers are deployable.
        deployability=deployability.assess(features.FEATURE_NAMES, deployability.Cadence.STEP),
    )
    assert report.headline_cell is None
    assert report.to_dict()["headline_policy_cell"] is None


def test_a_better_configuration_is_flagged_rather_than_adopted(
    signal_report: run_eval.EvalReport,
) -> None:
    cells = signal_report.policy_cells
    scored = [c for c in cells if c.precision is not None]
    best = max(scored, key=lambda c: c.precision or 0.0)
    notes = run_eval._default_gap_note(cells)
    if run_eval._is_shipped(best):
        assert notes == []
    else:
        assert any("SHIPPED default" in n and "unchanged" in n for n in notes)


def test_unstamped_trajectories_are_excluded_and_counted() -> None:
    corpus = _signal_corpus()
    blind = [
        make_trajectory(
            [make_step(step_index=j, decision_index=j, confirmed=False) for j in range(9)]
            + [make_step(step_index=9, decision_index=9, success=False)],
            trajectory_id=f"z{i}__blind__x",
            terminal_resolved=False,
        )
        for i in range(20)
    ]
    report = run_eval.evaluate([*corpus, *blind], depths=_DEPTHS, n_permutations=_PERMUTATIONS)
    assert report.n_trajectories == len(corpus) + 20
    assert report.n_stamped == len(corpus)
    assert any("no per-step verified outcomes" in n for n in report.notes)
    # …and the exclusion reaches every figure's footer, not just the JSON.
    limits = run_eval._run_annotations(report).limitations
    assert any("excluded from this figure" in lim for lim in limits)


def test_report_dict_has_no_summed_escalation_count(
    signal_report: run_eval.EvalReport,
) -> None:
    payload = signal_report.to_dict()
    # The top-level `n_escalated: 4980` next to `n_trajectories: 799` is gone: the count lives on
    # each cell, where it means "trajectories this configuration escalated".
    assert "n_escalated" not in payload
    for cell in payload["policy_cells"]:  # type: ignore[union-attr]
        assert cell["n_escalated"] <= cell["n_trajectories"]


def test_capture_coverage_is_reported_over_the_whole_corpus(
    signal_report: run_eval.EvalReport,
) -> None:
    assert sum(c.n_trajectories for c in signal_report.coverage) == len(_signal_corpus())


def test_main_writes_every_figure_through_the_frame(tmp_path, capsys) -> None:
    live = tmp_path / "live"
    live.mkdir()
    for traj in _signal_corpus(
        n_pairs=25
    ):  # >= MIN_ROWS at depth 5, else the prefix figures are (correctly) skipped
        schema.dump_jsonl(traj, live / f"{traj.header.trajectory_id}.jsonl")
    plots_dir = tmp_path / "reports"
    rc = run_eval.main(
        [
            "--live-dir",
            str(live),
            "--plots-dir",
            str(plots_dir),
            "--permutations",
            str(_PERMUTATIONS),
            "--depths",
            "5",
        ]
    )
    assert rc == 0
    # `session_cadence.png` is absent because this synthetic corpus uses model "m", which is not
    # in the price registry — no task has both >=2 cheap sessions AND a frontier session, so the
    # session-cadence estimate is "not supported" (None) and the figure is skipped, exactly as
    # the run_eval contract says it should be.
    assert {p.name for p in plots_dir.glob("*.png")} == _FIGURES - {"session_cadence.png"}
    # dpi 150 at (9, 5.5) plus the footer: a bare dpi=80 figure never reaches this size.
    assert all(p.stat().st_size > 20_000 for p in plots_dir.glob("*.png"))
    assert '"prefix_model"' in capsys.readouterr().out


def test_authenticity_errors_gate_the_status_they_were_collected_for() -> None:
    # `authenticity_errors` was computed on every run and read by nothing: a corpus whose rows
    # fail their own integrity recompute still printed OK/NO_SKILL. It gates ahead of skill now.
    status, reason = run_eval._status([], [make_depth_report()], 3)
    assert status == "AUTHENTICITY_FAILED"
    assert "3 Layer-1 authenticity error(s)" in reason
    # ...and a clean corpus is unaffected — the gate must not fire on 0.
    clean, _ = run_eval._status([], [make_depth_report()], 0)
    assert clean != "AUTHENTICITY_FAILED"


def test_a_tampered_corpus_never_reports_a_skill_verdict() -> None:
    from dataclasses import replace

    corpus = _signal_corpus(n_pairs=3)
    forged = corpus[0]
    corpus[0] = replace(forged, header=replace(forged.header, content_sha256="0" * 64))
    report = run_eval.evaluate(corpus, depths=(), n_permutations=_PERMUTATIONS)
    assert report.authenticity_errors > 0
    assert report.status == "AUTHENTICITY_FAILED"
    limits = run_eval._run_annotations(report).limitations
    assert any("FAILED ITS INTEGRITY CHECK" in lim for lim in limits)


def test_permutations_below_the_floor_is_a_parser_error_not_a_traceback(capsys) -> None:
    # `--permutations 100` used to raise ValueError deep inside `permutation_null` — a CLI
    # contract enforced by a crash. The floor is checked where the argument is parsed.
    with pytest.raises(SystemExit) as exc:
        run_eval.main(["--permutations", str(metrics.MIN_PERMUTATIONS - 1)])
    assert exc.value.code == 2  # argparse usage error
    assert f">= {metrics.MIN_PERMUTATIONS}" in capsys.readouterr().err


def test_non_positive_depths_are_a_parser_error_not_a_traceback(capsys) -> None:
    # `--depths 0` / `--depths -3` are not valid absolute step counts; they used to fail deep
    # inside the prefix admission/refit path. Rejected at parse time, like `--permutations`.
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit) as exc:
            run_eval.main(["--depths", bad])
        assert exc.value.code == 2  # argparse usage error
        assert "must be > 0" in capsys.readouterr().err


def test_main_errors_when_the_live_directory_is_missing(tmp_path) -> None:
    assert run_eval.main(["--live-dir", str(tmp_path / "nope")]) == 1


def test_main_errors_on_an_empty_live_directory(tmp_path) -> None:
    empty = tmp_path / "live"
    empty.mkdir()
    assert run_eval.main(["--live-dir", str(empty)]) == 1


def test_status_caveats_reach_the_footer(
    null_report: run_eval.EvalReport, signal_report: run_eval.EvalReport
) -> None:
    annotations = run_eval._run_annotations(null_report)
    assert any("NO USABLE SIGNAL" in lim for lim in annotations.limitations)
    assert run_eval._run_annotations(signal_report).limitations == ()


def test_depths_are_absolute_decisions_not_fractions() -> None:
    # A fractional prefix needs the total length, which is future information. The CLI takes
    # integers, and a depth no run reaches yields no report rather than an imputed one.
    report = run_eval.evaluate(_signal_corpus(), depths=(5, 500), n_permutations=_PERMUTATIONS)
    assert [d.depth for d in report.depth_reports] == [5]
    # The `features.DEFAULT_DEPTHS == (5, 10, 20)` assertion that used to sit here is gone. It
    # pinned the VALUE of a constant inside a test about absolute-vs-fractional form, and justified
    # nothing about which depths: it would have passed just as happily on (30, 40, 45), where the
    # sweep shows this harness publishes a positive incremental at every depth purely because deep
    # admission pre-selects failures. The reason is pinned instead, on the real corpus, by
    # `tests/escalation/test_features.py::test_every_reported_depth_admits_a_population_close_to_
    # the_corpus_base_rate`.
    assert all(isinstance(d, int) for d in features.DEFAULT_DEPTHS)


class TestOffPolicyEstimateIsReachableFromTheEval:
    """`ope.py` had zero production consumers; the eval now reaches it.

    It is the identification guard for the escalation policy's value. Unreachable, it could
    never say the value is unmeasured — and that silence is why escalation looked measurable.
    """

    def _trajectories(self):
        return [make_trajectory([make_step(step_index=0), make_step(step_index=1)])]

    def test_omitting_an_exploration_log_leaves_the_estimate_absent_not_zero(self):
        # None means "not asked". It must never render as a numeric 0.0, which would read as
        # a measured no-effect.
        report = run_eval.evaluate(self._trajectories())
        assert report.policy_value is None
        assert report.to_dict()["escalation_policy_value"] is None

    def test_a_deterministic_exploration_log_is_reported_as_not_identified(self):
        # Today's shipped config is exploration_epsilon=0.0, so every logged decision has
        # propensity 1.0 and the estimator MUST refuse. Reporting that refusal honestly — in
        # the eval's own JSON — is the whole point of wiring this in.
        rows = [
            {
                "checkpoint_id": f"t::{i}",
                "action": "raise_rank",
                "propensity": 1.0,
                "epsilon": 0.0,
                "outcome": "success",
                "features": {},
            }
            for i in range(8)
        ]
        report = run_eval.evaluate(self._trajectories(), exploration_rows=rows)
        assert report.policy_value is not None
        assert report.policy_value.status == "not_identified"
        assert report.policy_value.dr_estimate is None
        assert report.to_dict()["escalation_policy_value"]["status"] == "not_identified"
