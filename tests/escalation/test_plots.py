"""Plot code validated structurally: each figure populates its Axes and returns the caveats its
own data earned. The caveats that matter here are the ones that state a NULL result plainly — a
figure must never let a reader infer skill the numbers do not support.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display in CI
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from benchmark.calibration.labeler_metrics import ConfusionMatrix  # noqa: E402
from benchmark.escalation import features, metrics, plots, policy_eval, replay  # noqa: E402
from tests.escalation.factories import make_step, make_trajectory  # noqa: E402

_PERMUTATIONS = 200


def _scores_labels() -> tuple[list[float], list[bool]]:
    scores = [0.9, 0.8, 0.6, 0.5, 0.4, 0.3]
    labels = [True, False, True, False, False, False]
    return scores, labels


def _null(*, real: bool) -> metrics.NullResult:
    labels = [i < 20 for i in range(60)]
    scores = [1.0 if lab else 0.0 for lab in labels] if real else [float(i % 7) for i in range(60)]
    return metrics.permute_statistic(
        scores, labels, metrics.auroc, n_permutations=_PERMUTATIONS, seed=2
    )


def test_pr_curve_draws_the_prevalence_baseline() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    plots.pr_curve(scores, labels, ax)
    prevalence = metrics.prevalence(labels)
    assert any(all(y == prevalence for y in line.get_ydata()) for line in ax.lines)
    plt.close(fig)


def test_pr_curve_reports_sample_counts_at_runtime() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, ax)
    assert any("6 trajectories, 2 of them failed" in n for n in ann.notes)
    plt.close(fig)


def test_roc_curve_draws_chance_and_the_null_band() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    plots.roc_curve(scores, labels, _null(real=True), ax)
    assert any(line.get_label() == "chance" for line in ax.lines)
    assert ax.collections, "the permutation null band was shaded"
    assert "auxiliary" in ax.get_title().lower()
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


def test_confusion_matrix_prints_the_random_baseline() -> None:
    fig, ax = plt.subplots()
    # 20 failed / 30 resolved, 25 flags. A random flagger at that rate catches 20*0.5 = 10.
    ann = plots.confusion_matrix_plot(ConfusionMatrix(tp=10, fp=15, fn=10, tn=15), ax)
    assert any(t.get_text() == "10\n[10]" for t in ax.texts), "observed and random both rendered"
    assert any("random flagger" in n for n in ann.notes)
    # tp equals the random expectation, so the figure must refuse to imply a catch.
    assert any("no more failures than random" in lim for lim in ann.limitations)
    plt.close(fig)


def test_confusion_matrix_is_quiet_when_it_beats_random() -> None:
    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(ConfusionMatrix(tp=19, fp=6, fn=1, tn=24), ax)
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


def test_lead_time_reports_a_median_per_outcome_class() -> None:
    fig, ax = plt.subplots()
    ann = plots.lead_time_by_outcome(_cell(resolved_fire=False), ax)
    assert any("median lead time" in n for n in ann.notes)
    assert ax.patches, "the histogram drew bars"
    plt.close(fig)


def test_lead_time_states_an_empty_figure_rather_than_drawing_one() -> None:
    corpus = [
        make_trajectory(
            [make_step(step_index=j, decision_index=j) for j in range(5)],
            trajectory_id=f"q{i}",
            terminal_resolved=False,
        )
        for i in range(4)
    ]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    fig, ax = plt.subplots()
    ann = plots.lead_time_by_outcome(cell, ax)
    assert any("No escalation fired" in lim for lim in ann.limitations)
    plt.close(fig)


def test_sweep_table_says_when_no_configuration_clears_the_base_rate() -> None:
    fig, ax = plt.subplots()
    ann = plots.sweep_table([_cell(resolved_fire=True)], ax)
    assert ax.tables, "the sweep rendered as a table, not a two-value colour grid"
    assert any("no setting with measured value" in lim for lim in ann.limitations)
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


def test_every_figure_spec_carries_read_and_goal() -> None:
    specs = [
        plots.PR_CURVE_SPEC,
        plots.ROC_CURVE_SPEC,
        plots.CONFUSION_MATRIX_SPEC,
        plots.LEAD_TIME_SPEC,
        plots.SWEEP_TABLE_SPEC,
        plots.PERMUTATION_NULL_SPEC,
        plots.OUTCOME_BARS_SPEC,
        plots.CAPTURE_COVERAGE_SPEC,
    ]
    for spec in specs:
        assert spec.reading.strip()
        assert spec.goal.strip()


@pytest.mark.parametrize(
    "removed", ["steps_to_detection_hist", "sweep_heatmap", "cost_quality_frontier"]
)
def test_replaced_figures_are_gone(removed: str) -> None:
    # steps_to_detection drew 415 events as 4980; sweep_heatmap drew 6 cells of 2 distinct
    # results; cost_quality auto-scaled 12 dots that rendered as 2 around a null. All three were
    # replaced, and leaving the old entry points around invites their reuse.
    assert not hasattr(plots, removed)
