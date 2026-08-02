"""Plot code validated structurally: each figure populates its Axes and returns the caveats its
own data earned. The caveats that matter here are the ones that state a NULL result plainly — a
figure must never let a reader infer skill the numbers do not support.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmark.calibration.labeler_metrics import ConfusionMatrix  # noqa: E402
from benchmark.escalation import features, metrics, plots, policy_eval, replay  # noqa: E402
from tests.escalation.factories import make_step, make_trajectory  # noqa: E402

_PERMUTATIONS = 200


def _scores_labels() -> tuple[list[float], list[bool]]:
    scores = [0.9, 0.8, 0.6, 0.5, 0.4, 0.3]
    labels = [True, False, True, False, False, False]
    return scores, labels


def _null(*, real: bool, statistic: metrics.Statistic = metrics.auroc) -> metrics.NullResult:
    labels = [i < 20 for i in range(60)]
    scores = [1.0 if lab else 0.0 for lab in labels] if real else [float(i % 7) for i in range(60)]
    return metrics.permute_statistic(
        scores, labels, statistic, n_permutations=_PERMUTATIONS, seed=2
    )


def _auprc_null(*, real: bool) -> metrics.NullResult:
    return _null(real=real, statistic=metrics.auprc)


def test_pr_curve_draws_the_measured_null_not_prevalence() -> None:
    # THE DEFECT. The baseline was `prevalence`, the no-skill average precision for EXCHANGEABLE
    # rows. These rows cluster by challenge and the pipeline is refit per shuffle, so the
    # no-information AUPRC is measured, not arithmetic — and on the committed corpus the measured
    # null sat BELOW prevalence, i.e. the drawn bar was on the side that hides a detector.
    scores, labels = _scores_labels()
    null = _auprc_null(real=False)
    fig, ax = plt.subplots()
    plots.pr_curve(scores, labels, null, ax)
    labelled = [str(line.get_label()) for line in ax.lines]
    assert any("permutation null" in name for name in labelled)
    assert not any("prevalence" in name for name in labelled)
    assert any(all(y == pytest.approx(null.mean) for y in line.get_ydata()) for line in ax.lines)
    assert ax.patches or ax.collections, "the null's 95% interval was shaded as a band"
    plt.close(fig)


def test_pr_curve_states_both_the_measured_null_and_the_prevalence_it_replaced() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, _auprc_null(real=False), ax)
    assert any("6 trajectories, 2 of them failed" in n for n in ann.notes)
    # Both numbers on the canvas: the gap between them IS the assumption that was dropped.
    assert any("MEASURED permutation null" in n and "prevalence" in n for n in ann.notes)
    plt.close(fig)


def test_pr_curve_says_no_usable_signal_when_the_auprc_is_inside_its_null() -> None:
    scores, labels = _scores_labels()
    null = _auprc_null(real=False)
    assert not null.beats_null, "fixture precondition: this null is not cleared"
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, null, ax)
    assert any("NO USABLE SIGNAL" in lim for lim in ann.limitations)
    plt.close(fig)


def test_pr_curve_stays_quiet_when_the_auprc_clears_its_null() -> None:
    scores, labels = _scores_labels()
    null = _auprc_null(real=True)
    assert null.beats_null, "fixture precondition: this null IS cleared"
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, null, ax)
    assert not any("NO USABLE SIGNAL" in lim for lim in ann.limitations)
    plt.close(fig)


def test_pr_curve_draws_no_baseline_at_all_rather_than_falling_back_to_prevalence() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, None, ax)
    assert not any("prevalence" in str(line.get_label()) for line in ax.lines)
    assert any("NO BASELINE DRAWN" in lim for lim in ann.limitations)
    plt.close(fig)


def test_roc_curve_draws_the_measured_null_centre_and_de_emphasises_the_diagonal() -> None:
    # 0.5 is the chance level for a FIXED score vector against shuffled labels. This pipeline is
    # refit per shuffle, so its no-information AUROC is whatever the null measures — the diagonal
    # is orientation, and drawing it as boldly as the real reference invited the wrong comparison.
    scores, labels = _scores_labels()
    null = _null(real=True)
    fig, ax = plt.subplots()
    plots.roc_curve(scores, labels, null, ax)
    chance = next(line for line in ax.lines if "chance" in str(line.get_label()))
    centre = next(line for line in ax.lines if "measured null centre" in str(line.get_label()))
    assert "not the reference" in str(chance.get_label())
    assert chance.get_linewidth() < centre.get_linewidth()
    # The centre curve is the two-segment ROC (0,0)->(0.5,b)->(1,1) whose AREA is the null mean.
    assert centre.get_ydata()[1] == pytest.approx(max(0.0, 2 * (null.mean - 0.25)))
    assert ax.collections, "the permutation null band was shaded"
    assert "auxiliary" in ax.get_title().lower()
    plt.close(fig)


def test_roc_curve_states_the_measured_null_centre_in_its_footer() -> None:
    scores, labels = _scores_labels()
    null = _null(real=False)
    fig, ax = plt.subplots()
    ann = plots.roc_curve(scores, labels, null, ax)
    assert any(f"centres at {null.mean:.4f}" in n for n in ann.notes)
    plt.close(fig)


def test_roc_curve_says_no_usable_signal_when_inside_the_null() -> None:
    # The old ROC drew a bow above the diagonal for a below-chance detector and said nothing. A
    # figure whose observation sits inside its own null must state that in the footer.
    fig, ax = plt.subplots()
    ann = plots.roc_curve([0.5, 0.4, 0.6, 0.3], [True, False, False, True], _null(real=False), ax)
    assert any("NO USABLE SIGNAL" in lim for lim in ann.limitations)
    assert any("INSIDE the null band" in n for n in ann.notes)
    plt.close(fig)


def test_roc_curve_stays_quiet_when_the_signal_clears_its_null() -> None:
    fig, ax = plt.subplots()
    ann = plots.roc_curve([1.0, 1.0, 0.0, 0.0], [True, True, False, False], _null(real=True), ax)
    assert ann.limitations == ()
    plt.close(fig)


def test_confusion_matrix_prints_the_random_baseline_and_the_excess() -> None:
    fig, ax = plt.subplots()
    # 20 failed / 30 resolved, 25 flags. A random flagger at that rate catches 20*0.5 = 10.
    ann = plots.confusion_matrix_plot(
        ConfusionMatrix(tp=10, fp=15, fn=10, tn=15), ax, threshold=0.372, flag_budget=25
    )
    assert any(t.get_text() == "10\n[10]\n+0.0" for t in ax.texts), "count, random and excess"
    assert any("random flagger" in n for n in ann.notes)
    # The operating point is data-derived, so the reader must be able to see WHICH point it is.
    assert "0.372" in ax.get_title()
    assert any("0.372" in n for n in ann.notes)
    # tp equals the random expectation, so the figure must refuse to imply a catch.
    assert any("no more failures than random" in lim for lim in ann.limitations)
    plt.close(fig)


def test_confusion_matrix_reports_the_base_rate_from_the_data_not_a_literal() -> None:
    # THE DEFECT. The frozen caption carried "a corpus that fails ~38% of the time" while the
    # committed corpus failed 53% of the time. A base rate is data; it belongs in the runtime
    # annotations, where it cannot go stale, and nowhere in the frozen spec.
    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(
        ConfusionMatrix(tp=10, fp=15, fn=10, tn=15), ax, threshold=0.372, flag_budget=25
    )
    assert any("measured base failure rate is 0.4000" in n for n in ann.notes)
    frozen = (
        plots.CONFUSION_MATRIX_SPEC.reading,
        plots.CONFUSION_MATRIX_SPEC.goal,
        *plots.CONFUSION_MATRIX_SPEC.notes,
        *plots.CONFUSION_MATRIX_SPEC.limitations,
        *(meaning for _, meaning in plots.CONFUSION_MATRIX_SPEC.definitions),
    )
    assert not any("38%" in text or "0.38" in text for text in frozen)
    plt.close(fig)


def test_confusion_matrix_claims_one_degree_of_freedom_not_two_good_cells() -> None:
    # THE DEFECT. READ said the top-left cell was "the only good news this figure can carry" while
    # GOAL asked for top-left AND bottom-right — the caption contradicted itself. With the flag
    # count fixed the two excesses are algebraically identical (tn - E[tn] == tp - E[tp]), so the
    # goal was double-counting one number, and the colour scale implied four independent ones.
    cm = ConfusionMatrix(tp=12, fp=13, fn=8, tn=17)
    flagged, total, positives, negatives = 25, 50, 20, 30
    rate = flagged / total
    assert cm.tn - negatives * (1 - rate) == pytest.approx(cm.tp - positives * rate)
    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(cm, ax, threshold=0.4, flag_budget=flagged)
    assert "top-left" in plots.CONFUSION_MATRIX_SPEC.goal
    assert "bottom-right" not in plots.CONFUSION_MATRIX_SPEC.goal
    assert any("ONE degree of freedom" in n for n in ann.notes)
    assert not ax.images, "no colour channel over four counts that carry one number"
    plt.close(fig)


def test_confusion_matrix_flags_a_tie_block_that_overspends_the_budget() -> None:
    # `operating_threshold` returns the base-rate quantile SCORE and consumers flag `score >= cut`,
    # so a block of runs tied with the cut is admitted whole. On the committed corpus that turned a
    # 198-flag budget into 240 flags (rate 0.532 -> 0.645) with the caption still claiming the
    # budget equalled the base rate. Every count in the grid moves with it.
    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(
        ConfusionMatrix(tp=13, fp=12, fn=7, tn=18), ax, threshold=0.5, flag_budget=20
    )
    overspend = [lim for lim in ann.limitations if "BUDGET OVERSPENT" in lim]
    assert len(overspend) == 1
    assert "20 flags" in overspend[0]
    assert "admits 25" in overspend[0]
    plt.close(fig)


def test_confusion_matrix_is_quiet_when_it_beats_random_on_budget() -> None:
    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(
        ConfusionMatrix(tp=19, fp=6, fn=1, tn=24), ax, threshold=0.4, flag_budget=25
    )
    assert ann.limitations == ()
    plt.close(fig)


def _cell(*, resolved_fire: bool):  # type: ignore[no-untyped-def]
    """A policy cell where firing tracks either success (resolved_fire) or failure."""
    corpus = []
    for i in range(8):
        thrash = [
            make_step(step_index=j, decision_index=j, success=False, failing_check_id="k")
            for j in range(6)
        ]
        quiet = [make_step(step_index=j, decision_index=j) for j in range(6)]
        corpus.append(
            make_trajectory(thrash, trajectory_id=f"a{i}", terminal_resolved=resolved_fire)
        )
        corpus.append(
            make_trajectory(quiet, trajectory_id=f"b{i}", terminal_resolved=not resolved_fire)
        )
    return policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)


def test_outcome_bars_call_out_an_inverted_policy() -> None:
    # The headline the six old figures could not show: escalation firing on the runs that SUCCEED.
    # _cell(resolved_fire=True) separates perfectly, so the intervals do not overlap.
    fig, ax = plt.subplots()
    ann = plots.outcome_bars(_cell(resolved_fire=True), ax)
    assert any("INVERTED" in lim for lim in ann.limitations)
    assert any("P(fail|fired)" in n for n in ann.notes)
    plt.close(fig)


def test_outcome_bars_are_quiet_when_the_policy_points_the_right_way() -> None:
    fig, ax = plt.subplots()
    ann = plots.outcome_bars(_cell(resolved_fire=False), ax)
    assert ann.limitations == ()
    plt.close(fig)


def test_outcome_bars_refuse_to_call_a_direction_off_overlapping_intervals() -> None:
    # A point estimate below the base rate is NOT an inverted policy while the Wilson intervals
    # overlap. Reading a sign off overlapping bars is the same error as reading skill off a point
    # estimate inside its null — the figure must say "no separation", not "INVERTED".
    corpus = [
        make_trajectory(
            [
                make_step(step_index=j, decision_index=j, success=False, failing_check_id="k")
                for j in range(6)
            ],
            trajectory_id=f"a{i}",
            terminal_resolved=i % 3 == 0,
        )
        for i in range(30)
    ] + [
        make_trajectory(
            [make_step(step_index=j, decision_index=j) for j in range(6)],
            trajectory_id=f"b{i}",
            terminal_resolved=i % 3 != 0,
        )
        for i in range(30)
    ]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    fig, ax = plt.subplots()
    ann = plots.outcome_bars(cell, ax)
    assert any("NO SEPARATION" in lim for lim in ann.limitations)
    assert not any("INVERTED" in lim for lim in ann.limitations)
    plt.close(fig)


def test_sweep_table_says_when_no_configuration_clears_the_base_rate() -> None:
    fig, ax = plt.subplots()
    ann = plots.sweep_table([_cell(resolved_fire=True)], ax)
    assert ax.tables, "the sweep rendered as a table, not a two-value colour grid"
    assert any("no setting with measured value" in lim for lim in ann.limitations)
    plt.close(fig)


def test_one_separating_cell_does_not_vouch_for_the_rest_of_the_sweep() -> None:
    # `any(...)` used to silence the warning for EVERY cell as soon as one separated — including
    # for the shipped configuration, the one a reader is actually deciding on.
    separating = _counts_cell(2, 1, precision_ci=(0.9, 1.0))
    quiet = _counts_cell(2, 1, precision_ci=(0.1, 0.4))
    fig, ax = plt.subplots()
    ann = plots.sweep_table([separating, quiet], ax)
    assert any("Only 1 of 2 configurations clear" in lim for lim in ann.limitations)
    plt.close(fig)


def test_a_never_firing_cell_is_undefined_rather_than_ranked_at_zero() -> None:
    # precision returns None, not 0.0, for a cell that never fired: 0.0 would enter the argmax
    # and lose only by luck, and the bar would read as a measured "escalating predicts success".
    empty = _counts_cell(0, 0, precision_ci=_UNDEFINED_CI)
    assert empty.precision is None
    assert empty.lift is None
    fig, ax = plt.subplots()
    ann = plots.outcome_bars(empty, ax)
    assert any("UNDEFINED" in lim for lim in ann.limitations)
    assert not any("INVERTED" in lim for lim in ann.limitations)
    plt.close(fig)
    fig, ax = plt.subplots()
    table = plots.sweep_table([empty], ax)
    assert any("none of which fired" in n for n in table.notes)
    plt.close(fig)


def test_an_absent_bar_is_nan_high_not_zero_high() -> None:
    # A zero-height bar is pixel-identical to a genuine precision of 0.000 — a MEASURED claim that
    # every flagged run resolved — so "the configuration made no prediction" was distinguishable
    # only by reading the footer. Matplotlib omits a NaN-height bar from the axes; zero draws one.
    fig, ax = plt.subplots()
    cell = _counts_cell(0, 0, precision_ci=_UNDEFINED_CI, quiet=(2, 2), quiet_ci=(0.2, 0.8))
    plots.outcome_bars(cell, ax)
    absent, present = ax.patches[0], ax.patches[1]
    assert np.isnan(absent.get_height()), "the never-fired bar is absent, not a measured zero"
    assert present.get_height() == 0.5
    plt.close(fig)


def test_outcome_direction_reads_the_clustered_quiet_interval_not_wilson() -> None:
    # The fired arm is a challenge bootstrap and the quiet arm used to be Wilson over ROWS, and the
    # footer compared the two. A row-level interval is too narrow when runs cluster by challenge,
    # and too narrow makes BOTH directional branches easier to trip. Each case below is a cell
    # whose verdict flips depending on which estimator the quiet arm uses.
    fig, ax = plt.subplots()
    # wilson_interval(50, 100) = [0.404, 0.596], entirely below fired_lo=0.85 → the footer would
    # fall silent and endorse the policy. The clustered arm overlaps, so there is no separation.
    endorsed = _counts_cell(9, 1, precision_ci=(0.85, 0.95), quiet=(50, 50), quiet_ci=(0.30, 0.90))
    ann = plots.outcome_bars(endorsed, ax)
    assert any("NO SEPARATION" in lim for lim in ann.limitations)
    plt.close(fig)
    fig, ax = plt.subplots()
    # Mirror image: under Wilson, fired_hi=0.15 sits below quiet_lo=0.404 and the footer shouts
    # INVERTED. The honest quiet interval reaches down to 0.10, so it does not.
    inverted = _counts_cell(1, 9, precision_ci=(0.05, 0.15), quiet=(50, 50), quiet_ci=(0.10, 0.90))
    ann = plots.outcome_bars(inverted, ax)
    assert not any("INVERTED" in lim for lim in ann.limitations)
    assert any("NO SEPARATION" in lim for lim in ann.limitations)
    plt.close(fig)


def test_a_cell_that_fires_on_everything_has_no_quiet_arm_to_compare_against() -> None:
    # fn + tn == 0, so P(fail | not fired) is undefined and its interval is (nan, nan). Wilson
    # returned (0.0, 0.0) there, and 0.0 < fired_lo silently took the "policy points the right
    # way" branch — a clean bill of health read off an arm with no rows in it.
    always = _counts_cell(9, 1, precision_ci=(0.85, 0.95))
    assert always.p_fail_given_quiet is None
    fig, ax = plt.subplots()
    ann = plots.outcome_bars(always, ax)
    assert any("fired on EVERY run" in lim for lim in ann.limitations)
    assert np.isnan(ax.patches[1].get_height())
    plt.close(fig)


def test_permutation_null_plot_states_the_verdict_both_ways() -> None:
    fig, ax = plt.subplots()
    ann = plots.permutation_null_plot(_null(real=False), ax, label="AUROC")
    assert any("NO USABLE SIGNAL" in lim for lim in ann.limitations)
    plt.close(fig)
    fig, ax = plt.subplots()
    ann = plots.permutation_null_plot(_null(real=True), ax, label="AUROC")
    assert ann.limitations == ()
    assert any("clears the null band" in n for n in ann.notes)
    plt.close(fig)


def test_capture_coverage_names_the_models_with_no_per_step_outcomes() -> None:
    covered = make_trajectory(
        [
            make_step(step_index=i, decision_index=i, success=False, failing_check_id="k")
            for i in range(4)
        ],
        trajectory_id="inst__seeing-model__high",
        terminal_resolved=False,
    )
    blind = make_trajectory(
        [make_step(step_index=i, decision_index=i, confirmed=False) for i in range(4)],
        trajectory_id="inst__blind-model__high",
        terminal_resolved=False,
    )
    fig, ax = plt.subplots()
    ann = plots.capture_coverage(features.model_coverage([covered, blind]), ax)
    assert any("blind-model" in lim for lim in ann.limitations)
    assert any("NO per-step outcomes" in lim for lim in ann.limitations)
    plt.close(fig)


def test_capture_coverage_never_plots_the_tautological_capture_rate() -> None:
    # `capture_rate` is 1.000 for every model BY CONSTRUCTION (the normalizer writes `success`,
    # `failing_check_id` and `blocking` in one assignment), so plotting it would draw an identity
    # as a measurement. The bars are stamping coverage, which does vary, and the spec says so.
    covered = make_trajectory(
        [
            make_step(step_index=i, decision_index=i, success=False, failing_check_id="k")
            for i in range(4)
        ],
        trajectory_id="inst__seeing-model__high",
        terminal_resolved=False,
    )
    coverages = features.model_coverage([covered])
    assert coverages[0].capture_rate == 1.0
    fig, ax = plt.subplots()
    ann = plots.capture_coverage(coverages, ax)
    assert "per-step verified outcomes" in ax.get_xlabel()
    assert any("stamped" in t.get_text() for t in ax.texts)
    assert any("BY CONSTRUCTION" in lim for lim in plots.CAPTURE_COVERAGE_SPEC.limitations)
    assert any("terminal failure rate spans" in n for n in ann.notes)
    plt.close(fig)


_UNDEFINED_CI = (float("nan"), float("nan"))


def _counts_cell(
    tp: int,
    fp: int,
    *,
    precision_ci: tuple[float, float] = (0.0, 1.0),
    quiet: tuple[int, int] = (0, 0),
    quiet_ci: tuple[float, float] = _UNDEFINED_CI,
) -> policy_eval.PolicyCell:
    """A PolicyCell built straight from its 2x2, for the figures that read only counts."""
    # `quiet` is (fn, tn) and defaults to an EMPTY quiet arm, which is why `quiet_ci` defaults to
    # undefined: a builder that fabricated a finite interval for rows it never created would let a
    # test pass against an interval no estimator produced.
    return policy_eval.PolicyCell(
        escalate_after_n=2,
        stale_window=5,
        ladder="default",
        n_trajectories=tp + fp + sum(quiet),
        n_escalated=tp + fp,
        tp=tp,
        fp=fp,
        fn=quiet[0],
        tn=quiet[1],
        base_failure_rate=0.5,
        precision_ci=precision_ci,
        quiet_ci=quiet_ci,
        null_auroc=_null(real=False),
    )


def test_every_figure_spec_carries_read_and_goal() -> None:
    specs = [
        plots.PR_CURVE_SPEC,
        plots.ROC_CURVE_SPEC,
        plots.CONFUSION_MATRIX_SPEC,
        plots.SWEEP_TABLE_SPEC,
        plots.PERMUTATION_NULL_SPEC,
        plots.OUTCOME_BARS_SPEC,
        plots.CAPTURE_COVERAGE_SPEC,
    ]
    for spec in specs:
        assert spec.reading.strip()
        assert spec.goal.strip()


@pytest.mark.parametrize(
    "removed",
    [
        "steps_to_detection_hist",
        "sweep_heatmap",
        "cost_quality_frontier",
        "lead_time_by_outcome",
        "LEAD_TIME_SPEC",
    ],
)
def test_replaced_figures_are_gone(removed: str) -> None:
    # steps_to_detection drew 415 events as 4980; sweep_heatmap drew 6 cells of 2 distinct
    # results; cost_quality auto-scaled 12 dots that rendered as 2 around a null; lead_time could
    # not support a timing claim at all, because the policy fires in the first two decisions of
    # nearly every run it flags, so a lead time is the run length minus a constant. All were
    # removed, and leaving the old entry points around invites their reuse.
    assert not hasattr(plots, removed)
