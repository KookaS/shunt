"""Plot CODE validated structurally on fixtures (real PNGs arrive with collected data): each
figure populates its Axes and returns the runtime Annotations its own data earned — the caveats
that used to be red boxes drawn over the data now travel to the frame's footer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display in CI
import matplotlib.pyplot as plt  # noqa: E402

from benchmark.escalation import plots, replay  # noqa: E402
from benchmark.escalation.replay import GridPoint  # noqa: E402
from tests.escalation.factories import make_step, make_trajectory  # noqa: E402


def _scores_labels() -> tuple[list[float], list[bool]]:
    scores = [0.9, 0.8, 0.6, 0.5, 0.4, 0.3]
    labels = [True, False, True, False, False, False]
    return scores, labels


def test_pr_curve_populates_axes_and_draws_prevalence() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    plots.pr_curve(scores, labels, ax)
    assert ax.lines, "PR curve drew at least one line"
    # the prevalence baseline is a horizontal line at the positive rate
    from benchmark.escalation import metrics

    prevalence = metrics.prevalence(labels)
    ydata = [line.get_ydata() for line in ax.lines]
    assert any(all(y == prevalence for y in yd) for yd in ydata), "prevalence baseline present"
    plt.close(fig)


def test_steps_to_detection_hist_runs() -> None:
    traj = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a"),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="t::a"),
        ]
    )
    sweep = replay.sweep([traj], [GridPoint(2, 10)])
    fig, ax = plt.subplots()
    plots.steps_to_detection_hist(sweep, ax)
    assert ax.patches, "the histogram drew bars for the escalated trajectory"
    plt.close(fig)


def test_cost_quality_frontier_runs() -> None:
    perf = {"a": 80.0, "b": 90.0, "c": 70.0}
    cost = {"a": 1.0, "b": 2.0, "c": 3.0}
    fig, ax = plt.subplots()
    plots.cost_quality_frontier(perf, cost, ax)
    assert ax.collections, "the scatter drew the config points"
    plt.close(fig)


def test_steps_to_detection_hist_discrete_bars() -> None:
    # Two trajectories escalate at distinct integer steps (1 and 2) → two separate bars, not
    # one solid float-binned block.
    traj_a = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="k"),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="k"),
        ],
        trajectory_id="a",
    )
    traj_b = make_trajectory(
        [
            make_step(step_index=0, decision_index=0),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="k"),
            make_step(step_index=2, decision_index=2, success=False, failing_check_id="k"),
        ],
        trajectory_id="b",
    )
    sweep = replay.sweep([traj_a, traj_b], [GridPoint(2, 10)])
    fig, ax = plt.subplots()
    plots.steps_to_detection_hist(sweep, ax)
    populated = [p for p in ax.patches if p.get_height() > 0]
    assert len(populated) >= 2, "distinct step counts render as separate bars, not one block"
    # rwidth<1 leaves a visible gap between adjacent bars so equal-height bars don't fuse.
    assert all(0 < p.get_width() < 1.0 for p in ax.patches), "bars are gapped, not full-width"
    plt.close(fig)


def test_cost_quality_frontier_annotates_single_cost() -> None:
    # All configs share one cost → no frontier to trace; must be honestly labelled in the footer,
    # not as a red box drawn over the scatter.
    perf = {"a": 80.0, "b": 90.0, "c": 70.0}
    cost = {"a": 1.0, "b": 1.0, "c": 1.0}
    fig, ax = plt.subplots()
    ann = plots.cost_quality_frontier(perf, cost, ax)
    assert ax.collections, "the real scatter is kept"
    assert not ax.texts, "the caveat travels in Annotations, it does not occlude the data"
    assert any("no frontier to trace" in lim for lim in ann.limitations)
    plt.close(fig)


def test_cost_quality_y_label_names_the_plotted_quantity() -> None:
    # The caller passes F1 x 100; the label used to claim AUPRC%.
    fig, ax = plt.subplots()
    plots.cost_quality_frontier({"a": 80.0, "b": 90.0}, {"a": 1.0, "b": 2.0}, ax)
    assert "F1" in ax.get_ylabel()
    assert "AUPRC" not in ax.get_ylabel()
    plt.close(fig)


def test_pr_curve_draws_no_confidence_interval_footnote() -> None:
    # The PR figure drew no error bars, so the Wilson-CI footnote it carried was a lie.
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    plots.pr_curve(scores, labels, ax)
    assert not any("Wilson" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_steps_to_detection_annotates_when_empty() -> None:
    # a resolved trajectory never escalates → the histogram is empty and MUST be annotated.
    traj = make_trajectory([make_step(step_index=0, decision_index=0)], terminal_resolved=True)
    sweep = replay.sweep([traj], [GridPoint(2, 10)])
    fig, ax = plt.subplots()
    ann = plots.steps_to_detection_hist(sweep, ax)
    assert not ax.patches, "no bars on an unexercised trigger"
    assert any("empty, not flat" in lim for lim in ann.limitations), "empty plot annotated"
    plt.close(fig)


def test_pr_curve_reports_sample_counts_at_runtime() -> None:
    # Counts are data-derived, so they can never be a stale static string in the spec.
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    ann = plots.pr_curve(scores, labels, ax)
    assert any("6 prefixes, 2 of them positive" in n for n in ann.notes)
    plt.close(fig)


def test_pr_curve_flags_a_corpus_with_no_positives() -> None:
    # With zero positives the helper emits one meaningless dot; the footer must say so.
    fig, ax = plt.subplots()
    ann = plots.pr_curve([0.0, 1.0], [False, False], ax)
    assert any("No positive prefixes" in lim for lim in ann.limitations)
    plt.close(fig)


def test_roc_curve_flags_a_single_class_corpus() -> None:
    # auroc() returns 0.5 as a no-information code, not a measurement — say it on the figure.
    fig, ax = plt.subplots()
    ann = plots.roc_curve([0.0, 1.0], [True, True], ax)
    assert any("no-information code" in lim for lim in ann.limitations)
    plt.close(fig)


def test_roc_curve_runs_and_labels_auxiliary() -> None:
    scores, labels = _scores_labels()
    fig, ax = plt.subplots()
    plots.roc_curve(scores, labels, ax)
    assert ax.lines, "ROC drew the curve"
    assert "auxiliary" in ax.get_title().lower()
    plt.close(fig)


def _cells():  # type: ignore[no-untyped-def]
    from benchmark.escalation import datasets, run_eval

    return run_eval.evaluate([_failing_multi_key()], datasets.DEFAULT_GRID).cells


def _failing_multi_key():  # type: ignore[no-untyped-def]
    steps = [
        make_step(step_index=i, decision_index=i, success=False, failing_check_id="k")
        for i in range(5)
    ]
    return make_trajectory(steps, terminal_resolved=False)


def test_sweep_heatmap_runs_and_states_hidden_cells() -> None:
    fig, ax = plt.subplots()
    cells = _cells()
    ann = plots.sweep_heatmap(cells, ax, ladder="effort_then_rank")
    assert ax.images, "the heatmap drew a colour grid"
    shown = sum(c.ladder == "effort_then_rank" for c in cells)
    assert any(f"showing {shown} of {len(cells)} swept cells" in n for n in ann.notes)
    assert any("different ladder" in lim for lim in ann.limitations), "hidden half stated"
    plt.close(fig)


def test_confusion_matrix_plot_runs() -> None:
    from benchmark.calibration.labeler_metrics import ConfusionMatrix

    fig, ax = plt.subplots()
    ann = plots.confusion_matrix_plot(ConfusionMatrix(tp=3, fp=1, fn=0, tn=2), ax)
    assert ax.images, "the confusion heatmap drew"
    assert any(t.get_text() == "3" for t in ax.texts), "tp count rendered"
    assert any("6 prefixes scored" in n for n in ann.notes)
    plt.close(fig)
