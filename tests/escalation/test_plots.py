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
from matplotlib.container import BarContainer  # noqa: E402

from benchmark.escalation import features, metrics, plots, policy_eval, replay  # noqa: E402
from benchmark.escalation.session_eval import (  # noqa: E402
    ArmContrast,
    ArmCost,
    SessionCadenceReport,
)
from tests.escalation.factories import make_step, make_trajectory  # noqa: E402

_PERMUTATIONS = 200
_UNDEFINED_CI = (float("nan"), float("nan"))


def _null(*, real: bool, statistic: metrics.Statistic = metrics.auroc) -> metrics.NullResult:
    labels = [i < 20 for i in range(60)]
    scores = [1.0 if lab else 0.0 for lab in labels] if real else [float(i % 7) for i in range(60)]
    return metrics.permute_statistic(
        scores, labels, statistic, n_permutations=_PERMUTATIONS, seed=2
    )


def _cell(*, resolved_fire: bool):  # type: ignore[no-untyped-def]
    """A policy cell where firing tracks either success (resolved_fire) or failure."""
    corpus = []
    # 12 pairs, not 8: both arms must clear `policy_eval.MIN_ARM` or the figure correctly refuses
    # to read a direction off them at all, and these tests are about the direction.
    for i in range(12):
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


def _counts_cell(
    tp: int,
    fp: int,
    *,
    precision_ci: tuple[float, float] = (0.0, 1.0),
    quiet: tuple[int, int] = (0, 0),
    quiet_ci: tuple[float, float] = _UNDEFINED_CI,
    budget: policy_eval.BudgetAggregates | None = None,
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
        budget=budget or policy_eval.BudgetAggregates(),
    )


def _axes(n: int):  # type: ignore[no-untyped-def]
    fig, axes = plt.subplots(1, n)
    return fig, list(np.atleast_1d(axes))


# ------------------------------------------------------------------- 1. the decision


def test_the_decision_figure_reports_both_families_and_its_detection_floor() -> None:
    labels = [i % 3 == 0 for i in range(30)]
    plain = [float(i % 4) for i in range(30)]
    edit = [float(3 if lab else 0) for lab in labels]
    null = metrics.permutation_null(0.9, [0.45 + 0.005 * (i % 10) for i in range(_PERMUTATIONS)])
    fig, axes = _axes(4)
    ann = plots.escalation_decision(
        plain,
        edit,
        labels,
        axes,
        null=null,
        band=((0.0, 0.5, 1.0), (0.0, 0.2, 0.8), (0.0, 0.5, 1.0)),
        shipped_n=3,
        value_verdict="OK",
    )
    assert any("AUROC as-shipped" in fact for fact in ann.subtitle_facts)
    assert any("score null 95%" in note for note in ann.notes)
    assert any(p.get_label() == "null 95% band" for p in axes[0].collections)
    # The shipped threshold is marked on BOTH threshold panels, not just one.
    for ax in axes[1:3]:
        assert any(line.get_label() == "shipped n=3" for line in ax.lines)
    plt.close(fig)


def test_the_decision_figure_stays_quiet_without_a_null() -> None:
    # The band and the footnote are opt-in: a caller with no null (hand-built scores) must not be
    # handed a fabricated detection floor.
    fig, axes = _axes(4)
    ann = plots.escalation_decision(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 2.0, 3.0, 4.0],
        [False, True, False, True],
        axes,
        shipped_n=3,
        value_verdict="OK",
    )
    assert not any("null 95%" in note for note in ann.notes)
    plt.close(fig)


def test_the_value_box_is_derived_from_the_report_and_moves_with_it() -> None:
    # THE DEFECT THIS PINS. The VALUE verdict was a module-level literal "OK" drawn on three
    # canvases, so a trivial competitor beating the escalate arm would have left three green
    # boxes claiming a value the data no longer supported. It is now read off the report.
    beaten = _contrast("always_frontier", rate=0.73, diff=-0.108, ci=(-0.165, -0.056))
    won = _contrast("always_cheap", rate=0.44, diff=0.185, ci=(0.006, 0.375))
    assert plots.session_value_verdict(_session(low=0.24, high=0.58, comparisons=(won,))) == "OK"
    assert (
        plots.session_value_verdict(_session(low=0.24, high=0.58, comparisons=(beaten, won)))
        == "not beaten: always-frontier"
    )
    # An interval spanning zero is not a value claim either, whatever the trivial arms did.
    assert plots.session_value_verdict(_session(low=-0.1, high=0.58, comparisons=(won,))) == (
        "interval spans zero"
    )
    # No session-cadence contrast at all must never render the same as a supported one.
    assert plots.session_value_verdict(None) == plots.UNMEASURED_VALUE


def test_the_scope_strip_paints_a_derived_non_ok_value_verdict_red() -> None:
    fig, ax = plt.subplots()
    plots.scope_strip(ax, "not beaten: always-frontier")
    texts = [t.get_text() for t in ax.texts]
    # The claim and its verdict may wrap onto two lines when the box cannot hold one.
    wanted = [t for t in texts if "VALUE (session cadence" in t]
    assert wanted and "not beaten: always-frontier" in wanted[0]
    # Red, bold and heavy-edged is what the function already does for an unsupported claim; the
    # point of the test is that the VALUE box now reaches that branch at all.
    value_box = [t for t in ax.texts if "VALUE" in t.get_text()][0]
    assert value_box.get_fontweight() == "bold"
    plt.close(fig)


def test_every_figure_carries_the_scope_strip_including_the_unidentified_claim() -> None:
    # The causal claim is NOT identified — no logged trajectory escalated — so it is rendered as a
    # labelled box rather than quietly omitted or, worse, estimated.
    fig, ax = plt.subplots()
    plots.scope_strip(ax, "OK")
    texts = [t.get_text() for t in ax.texts]
    assert any("not identified: P(escalate)=0" in t for t in texts)
    assert any("DETECTS" in t for t in texts)
    assert any("VALUE" in t for t in texts)
    assert not ax.axison
    plt.close(fig)


# ------------------------------------------------------------- 2. the operating point


def test_the_operating_point_calls_out_an_inverted_policy() -> None:
    # The headline the old figure set could not show: escalation firing on the runs that SUCCEED.
    # _cell(resolved_fire=True) separates perfectly, so the intervals do not overlap.
    fig, axes = _axes(3)
    ann = plots.operating_point(None, _cell(resolved_fire=True), axes, "OK")
    assert any("INVERTED" in lim for lim in ann.limitations)
    assert any("P(fail|fired)" in note for note in ann.notes)
    plt.close(fig)


def test_the_operating_point_is_quiet_when_the_policy_points_the_right_way() -> None:
    fig, axes = _axes(3)
    ann = plots.operating_point(None, _cell(resolved_fire=False), axes, "OK")
    assert ann.limitations == ()
    plt.close(fig)


def test_the_operating_point_refuses_a_direction_off_overlapping_intervals() -> None:
    # A point estimate below the base rate is NOT an inverted policy while the intervals overlap.
    # Reading a sign off overlapping bars is the same error as reading skill off a point estimate
    # inside its null — the figure must say "no separation", not "INVERTED".
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
    fig, axes = _axes(3)
    ann = plots.operating_point(None, cell, axes, "OK")
    assert any("NO SEPARATION" in lim for lim in ann.limitations)
    assert not any("INVERTED" in lim for lim in ann.limitations)
    plt.close(fig)


def test_an_arm_below_the_reporting_floor_renders_as_undefined_not_as_a_rate() -> None:
    # THE DEFECT THIS PINS. The as-shipped cell at the shipped knobs fires on 727 of 727 runs, so
    # its quiet arm is EMPTY (n=0); the committed figure draws it as "undefined (n=0)" rather
    # than a measured rate. A one-row arm would be worse — its 0/1 would read as a measured
    # 0.000 — "escalating never predicts failure". The statistic is untouched (0/1 IS 0.0); the
    # FIGURE refuses to draw a bar for it.
    thin = _counts_cell(40, 20, precision_ci=(0.55, 0.78), quiet=(0, 1), quiet_ci=(0.0, 0.0))
    assert thin.p_fail_given_quiet == 0.0  # the statistic is NOT corrupted to fix the figure
    fig, axes = _axes(3)
    ann = plots.operating_point(None, thin, axes, "OK")
    assert ann.caveat is not None
    assert f"n={policy_eval.MIN_ARM}" in ann.caveat
    assert any("undefined" in t.get_text() for t in axes[0].texts)
    # One real bar, and a hatched placeholder patch instead of a second one. (`bar(yerr=...)`
    # registers an ErrorbarContainer beside each BarContainer, so count the bars themselves.)
    assert len([c for c in axes[0].containers if isinstance(c, BarContainer)]) == 1
    assert any(p.get_hatch() == "///" for p in axes[0].patches)
    assert any("no direction can be read" in lim for lim in ann.limitations)
    plt.close(fig)


def test_a_readable_pair_of_arms_draws_two_real_bars() -> None:
    fine = _counts_cell(40, 20, precision_ci=(0.55, 0.78), quiet=(10, 30), quiet_ci=(0.15, 0.40))
    fig, axes = _axes(3)
    ann = plots.operating_point(None, fine, axes, "OK")
    assert ann.caveat is None
    assert len([c for c in axes[0].containers if isinstance(c, BarContainer)]) == 2
    plt.close(fig)


def test_the_shipped_configuration_is_drawn_beside_the_canonical_one() -> None:
    # THE NEGATIVE FINDING THIS KEEPS ON THE CANVAS. The configuration the product actually ships
    # fires on every run (727/727), so its P(fail|fired) IS the base rate and its quiet arm is
    # EMPTY. Showing only the edit-gated cell would demote that to one row of a 30-row
    # table — the exact demotion this whole redesign exists to undo.
    as_shipped = _counts_cell(
        306, 421, precision_ci=(0.38, 0.46), quiet=(0, 0), quiet_ci=(0.0, 0.0)
    )
    canonical = _counts_cell(
        258, 177, precision_ci=(0.56, 0.71), quiet=(48, 244), quiet_ci=(0.16, 0.27)
    )
    fig, axes = _axes(3)
    ann = plots.operating_point(as_shipped, canonical, axes, "OK")
    # Three real bars and one hatched placeholder: the as-shipped quiet arm holds no runs.
    assert len([c for c in axes[0].containers if isinstance(c, BarContainer)]) == 3
    assert any(p.get_hatch() == "///" for p in axes[0].patches)
    assert any("undefined\n(n=0)" in t.get_text() for t in axes[0].texts)
    labels = [t.get_text() for t in axes[0].get_xticklabels()]
    assert any("as-shipped" in lab for lab in labels)
    assert any("edit-gated" in lab for lab in labels)
    # Both families' numbers reach the subtitle and the manifest notes.
    assert any("as-shipped fires 727/727" in f for f in ann.subtitle_facts)
    assert any("edit-gated fires 435/727" in f for f in ann.subtitle_facts)
    assert any(note.startswith("as-shipped at") for note in ann.notes)
    assert ann.caveat is not None and "as-shipped not-escalated arm n=0" in ann.caveat
    # ...and the null panel stays keyed to the canonical cell only — one null, not two.
    assert plots.OPERATING_POINT_SPEC.title.startswith("The shipped counter sits at the base rate")
    plt.close(fig)


def test_the_operating_point_draws_the_length_stratified_null_beside_the_family_wise_one() -> None:
    # Clearing the challenge-block null alone is not enough: that shuffle destroys the run-length
    # association too, so a cell whose firing is length selection can clear it. Both must be drawn.
    cell = _cell(resolved_fire=False)
    assert cell.null_auroc_length is not None
    fig, axes = _axes(3)
    plots.operating_point(None, cell, axes, "OK")
    labels = [artist.get_label() for artist in axes[1].get_children() if artist.get_label()]
    assert any("length-stratified null" in str(lab) for lab in labels)
    assert any("family-wise null" in str(lab) for lab in labels)
    plt.close(fig)


# -------------------------------------------------------------------- 3. the sweep


def test_the_sweep_keeps_the_interval_and_the_run_length_control_per_family() -> None:
    # Neither column is droppable to make two families fit. The interval is what the challenge
    # bootstrap exists to produce; `len-only` is the run-length confound control, and without it
    # nothing on the canvas separates "recurrence detects failure" from "recurrence is a proxy for
    # long runs". If they will not fit, the canvas widens — the column does not go.
    fig, ax = plt.subplots()
    plots.policy_sweep([_cell(resolved_fire=False)], [_cell(resolved_fire=True)], ax)
    header = [ax.tables[0].get_celld()[(0, col)].get_text().get_text() for col in range(12)]
    assert header == [
        "n",
        "stale",
        "A fired",
        "A P(fail)",
        "A 95% CI",
        "A AUROC",
        "A len-only",
        "B fired",
        "B P(fail)",
        "B 95% CI",
        "B AUROC",
        "B len-only",
    ]
    row = [ax.tables[0].get_celld()[(1, col)].get_text().get_text() for col in range(12)]
    assert row[4].startswith("[") and "," in row[4]
    assert row[6] not in ("", "n/a")  # the length baseline is a real number on a fired cell
    plt.close(fig)


def test_the_sweep_is_a_table_pinned_to_its_axes_never_a_scaled_one() -> None:
    # `Table.scale()` is the overflow mechanism that rendered the old sweep's title THROUGH its
    # own rows: a scaled table exceeds its axes and matplotlib does not clip it. `bbox` pins it.
    fig, ax = plt.subplots()
    plots.policy_sweep([_cell(resolved_fire=True)], [_cell(resolved_fire=False)], ax)
    assert ax.tables, "the sweep rendered as a table, not a two-value colour grid"
    assert ax.tables[0]._bbox is not None
    plt.close(fig)


def test_the_sweep_says_when_no_configuration_clears_the_base_rate() -> None:
    fig, ax = plt.subplots()
    ann = plots.policy_sweep([_cell(resolved_fire=True)], [], ax)
    assert any("no setting with measured value" in lim for lim in ann.limitations)
    plt.close(fig)


def test_one_separating_cell_does_not_vouch_for_the_rest_of_the_sweep() -> None:
    # `any(...)` used to silence the warning for EVERY cell as soon as one separated — including
    # for the shipped configuration, the one a reader is actually deciding on.
    separating = _counts_cell(2, 1, precision_ci=(0.9, 1.0))
    quiet = _counts_cell(2, 1, precision_ci=(0.1, 0.4))
    fig, ax = plt.subplots()
    ann = plots.policy_sweep([separating, quiet], [], ax)
    assert any("1 of 2 configurations" in lim for lim in ann.limitations)
    plt.close(fig)


def test_a_never_firing_cell_prints_n_a_not_a_literal_nan() -> None:
    # A never-firing cell's interval is (nan, nan), not None, so a bare-value formatter renders
    # the literal "nan" — contradicting the promise that an undefined quantity prints n/a.
    empty = _counts_cell(0, 0, precision_ci=_UNDEFINED_CI)
    assert empty.precision is None
    assert empty.lift is None
    fig, ax = plt.subplots()
    ann = plots.policy_sweep([empty], [empty], ax)
    texts = [cell.get_text().get_text() for cell in ax.tables[0].get_celld().values()]
    assert "n/a" in texts
    assert not any("nan" in t.lower() for t in texts)
    assert any("none of which fired" in note for note in ann.notes)
    plt.close(fig)


def test_the_shipped_row_is_located_but_never_colour_coded_by_result() -> None:
    shipped = _counts_cell(9, 1, precision_ci=(0.85, 0.95))
    other = _counts_cell(2, 1, precision_ci=(0.1, 0.4))
    fig, ax = plt.subplots()
    plots.policy_sweep([shipped, other], [], ax, shipped_index=0)
    cells = ax.tables[0].get_celld()
    assert cells[(1, 0)].get_facecolor()[:3] != cells[(2, 0)].get_facecolor()[:3]
    plt.close(fig)


# --------------------------------------------------------------- 4. the session value


def _contrast(name: str, *, rate: float, diff: float, ci: tuple[float, float]) -> ArmContrast:
    return ArmContrast(
        name=name,
        n=45,
        resolved=round(rate * 45),
        rate=rate,
        ci=(max(0.0, rate - 0.1), min(1.0, rate + 0.1)),
        diff_estimate=diff,
        diff_ci=ci,
        n_instances=45,
    )


def _session(
    *,
    low: float,
    high: float,
    comparisons: tuple[ArmContrast, ...] = (),
    costs: tuple[ArmCost, ...] = (),
) -> SessionCadenceReport:
    draws = tuple(low + (high - low) * i / 99.0 for i in range(100))
    return SessionCadenceReport(
        n_overlap_instances=45,
        n_escalated=37,
        n_escalated_resolved=21,
        escalate_rate=0.568,
        escalate_ci=(0.44, 0.71),
        n_retried=31,
        n_retried_resolved=7,
        retry_rate=0.226,
        retry_ci=(0.09, 0.41),
        n_frontier_after_fail=52,
        n_frontier_after_fail_resolved=27,
        frontier_after_fail_rate=0.519,
        cheap_base_rate=0.653,
        diff_estimate=(low + high) / 2,
        diff_ci=(low, high),
        diff_draws=draws,
        n_instances_resampled=45,
        comparisons=comparisons,
        random_fire_rate=0.62,
        costs=costs,
    )


def test_the_session_figure_draws_the_paired_difference_not_two_marginals_alone() -> None:
    # Two marginal intervals failing to overlap is a conservative test OF a difference, not a test
    # of the difference. The paired panel is the one the claim rests on.
    fig, axes = _axes(3)
    ann = plots.session_value(_session(low=0.13, high=0.55), axes)
    assert any("paired difference" in fact for fact in ann.subtitle_facts)
    assert any(line.get_label() == "no difference" for line in axes[1].lines)
    assert ann.caveat is not None and "Observational" in ann.caveat
    plt.close(fig)


def test_a_session_difference_spanning_zero_says_so_in_red() -> None:
    fig, axes = _axes(3)
    ann = plots.session_value(_session(low=-0.10, high=0.40), axes)
    assert ann.caveat is not None
    assert "spans zero" in ann.caveat
    plt.close(fig)


def test_a_baseline_the_escalate_arm_loses_to_is_stated_not_buried_in_panel_c() -> None:
    # THE ARM THAT CAN KILL THE CLAIM. An always-frontier arm ahead of the escalate arm must reach
    # the reader through the caveat and the subtitle, not only as a dot below zero in panel C.
    losing = _contrast("always_frontier", rate=0.73, diff=-0.108, ci=(-0.165, -0.056))
    fig, axes = _axes(3)
    ann = plots.session_value(_session(low=0.24, high=0.58, comparisons=(losing,)), axes)
    assert ann.caveat is not None and "does not beat always-frontier" in ann.caveat
    assert any("always_frontier" in fact for fact in ann.subtitle_facts)
    # Panel C draws one point per baseline arm, with zero marked so the sign is readable.
    assert axes[2].containers
    assert any(line.get_xdata()[0] == 0.0 for line in axes[2].lines)
    plt.close(fig)


def test_a_baseline_tied_within_its_interval_also_counts_as_not_beaten() -> None:
    # A positive point estimate whose paired interval spans zero is not a win over that arm, and
    # the figure must not let a reader infer one.
    tied = _contrast("random_escalate", rate=0.66, diff=0.04, ci=(-0.147, 0.078))
    beaten = _contrast("always_cheap", rate=0.44, diff=0.185, ci=(0.006, 0.375))
    fig, axes = _axes(3)
    ann = plots.session_value(_session(low=0.24, high=0.58, comparisons=(tied, beaten)), axes)
    assert ann.caveat is not None and "does not beat random-escalate" in ann.caveat
    assert any("random_escalate" in fact for fact in ann.subtitle_facts)
    assert not any("always_cheap" in fact for fact in ann.subtitle_facts)
    plt.close(fig)


def test_the_session_costs_name_their_currency_and_carry_no_mathtext_dollar() -> None:
    # A price with no currency beside it is unciteable: naive and cache-aware are different
    # quantities. And no "$" may appear — matplotlib reads a PAIR of them as mathtext delimiters
    # and renders the money between them as an equation, eating both signs (observed on canvas).
    costs = tuple(
        ArmCost(
            name=name,
            currency=currency,
            n_sessions=10,
            n_instances_covered=8,
            total_cost=1.0,
            cost_per_instance=0.5,
            cost_ci=(0.4, 0.6),
            marginal_cost_per_resolve=1.25,
            marginal_ci=(1.0, 1.5),
            marginal_undefined_share=0.0,
        )
        for currency in ("naive", "cache_aware")
        for name in ("escalate", "always_frontier")
    )
    fig, axes = _axes(3)
    ann = plots.session_value(_session(low=0.24, high=0.58, costs=costs), axes)
    facts = "\n".join(ann.subtitle_facts)
    assert "naive" in facts and "cache-aware" in facts
    assert "USD" in facts
    assert "$" not in facts
    assert "$" not in "\n".join(ann.notes)
    plt.close(fig)


def test_the_session_figure_draws_no_baseline_panel_when_there_are_no_baseline_arms() -> None:
    # A corpus with no baseline arms gets a labelled empty panel, never fabricated dots.
    fig, axes = _axes(3)
    plots.session_value(_session(low=0.24, high=0.58), axes)
    assert any("no baseline arms" in t.get_text() for t in axes[2].texts)
    plt.close(fig)


def test_the_session_figure_never_inherits_the_per_step_stamping_caveat() -> None:
    # It reads EVERY trajectory, stamped or not, so the run-level "72 runs excluded" line is false
    # here. A non-None caveat of its own is what stops `_merge` handing it the run-level one.
    for interval in ((0.13, 0.55), (-0.10, 0.40)):
        fig, axes = _axes(3)
        ann = plots.session_value(_session(low=interval[0], high=interval[1]), axes)
        assert ann.caveat is not None
        plt.close(fig)


# ---------------------------------------------------------- 5. the corpus and coverage


def test_the_corpus_figure_names_the_models_with_no_per_step_outcomes() -> None:
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
    fig, axes = _axes(4)
    ann = plots.corpus_and_coverage(
        features.model_coverage([covered, blind]),
        [plots.ModelArm("seeing-model", 1, 1.0, None)],
        plots.StratifiedAuroc(0.78, 0.75, 0.71),
        None,
        axes,
    )
    assert any("blind-model" in lim for lim in ann.limitations)
    assert any("NO per-step outcomes" in lim for lim in ann.limitations)
    plt.close(fig)


def test_the_corpus_figure_never_plots_the_tautological_capture_rate() -> None:
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
    fig, axes = _axes(4)
    plots.corpus_and_coverage(coverages, [], plots.StratifiedAuroc(0.78, None, None), None, axes)
    assert "per-step verified outcomes" in axes[0].get_xlabel()
    assert any("BY CONSTRUCTION" not in lim for lim in plots.CORPUS_COVERAGE_SPEC.limitations)
    plt.close(fig)


def test_the_feature_mismatch_caveat_names_the_panels_it_is_about() -> None:
    # THE CORRECTNESS BUG THIS PINS. The deployability line used to be appended to EVERY figure,
    # telling readers the recurrence ROC depends on features production lacks. The AS-SHIPPED
    # score does not — it reads `failing_check_id` and `success`, both of which production holds.
    #
    # THE SECOND BUG, which the first fix introduced. The caveat then read "Panel D only: … The
    # recurrence trigger does not", and that was false on this canvas: panels B and C are scored
    # on the EDIT-GATED counter, which reads `StepView.action` — a per-step field production
    # lacks. So the caveat must scope BOTH: the eval-only counter on B/C, the prefix fields on D.
    fig, axes = _axes(4)
    ann = plots.corpus_and_coverage(
        [],
        [],
        plots.StratifiedAuroc(0.78, 0.75, 0.71),
        plots.Admission(10, 727, 10, 373, 344, 0.503, 0.421),
        axes,
    )
    assert ann.caveat is not None
    assert "panel D" in ann.caveat
    assert "edit-gated" in ann.caveat
    fig2, axes2 = _axes(4)
    decision = plots.escalation_decision(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 2.0, 3.0, 4.0],
        [False, True, False, True],
        axes2,
        shipped_n=3,
        value_verdict="OK",
    )
    assert decision.caveat is None
    plt.close(fig)
    plt.close(fig2)


def test_the_admission_waterfall_states_both_base_rates() -> None:
    fig, axes = _axes(4)
    ann = plots.corpus_and_coverage(
        [],
        [],
        plots.StratifiedAuroc(0.78, 0.75, 0.71),
        plots.Admission(10, 727, 10, 373, 344, 0.503, 0.421),
        axes,
    )
    assert any(
        "344/727" in fact and "0.503" in fact and "0.421" in fact for fact in ann.subtitle_facts
    )
    assert any("0.503" in t.get_text() for t in axes[3].texts)
    plt.close(fig)


# ---------------------------------------------------------------- 6. the budget


def _budget() -> policy_eval.BudgetAggregates:
    return policy_eval.BudgetAggregates(
        n_fired_positioned=356,
        steps_after_fire_failed=5623,
        steps_after_fire_resolved=3047,
        fire_fraction_median_failed=0.47,
        fire_fraction_median_resolved=0.46,
        fire_fraction_deciles_failed=tuple(i / 10.0 for i in range(11)),
        fire_fraction_deciles_resolved=tuple(i / 10.0 for i in range(11)),
        fire_step_median=14.0,
        run_length_median=31.0,
    )


def test_the_budget_figure_reports_the_ledger_and_the_fire_position() -> None:
    fig, axes = _axes(3)
    ann = plots.escalation_budget(_counts_cell(40, 20, budget=_budget()), axes, "OK")
    assert any(
        "5623 steps pre-empted vs 3047 interrupted (1.85:1)" in f for f in ann.subtitle_facts
    )
    assert any("median fire at step 14 of 31" in f for f in ann.subtitle_facts)
    plt.close(fig)


def test_the_budget_figure_refuses_to_draw_a_cell_with_no_measured_positions() -> None:
    fig, axes = _axes(3)
    ann = plots.escalation_budget(_counts_cell(40, 20), axes, "OK")
    assert any("not computed" in lim for lim in ann.limitations)
    assert any("no positioned fire" in t.get_text() for t in axes[0].texts)
    plt.close(fig)


def test_the_budget_figure_discloses_that_the_ledger_ratio_is_mostly_arm_size() -> None:
    # A reader could take the 1.85:1 bar ratio as a timing finding. It is not: most of it is that
    # more of the fired runs failed, and the fire-position panel is where a timing claim lives.
    assert any("ARM SIZE" in lim for lim in plots.ESCALATION_BUDGET_SPEC.limitations)


# ------------------------------------------------------------------------ the set


_SPECS = (
    plots.ESCALATION_DECISION_SPEC,
    plots.OPERATING_POINT_SPEC,
    plots.POLICY_SWEEP_SPEC,
    plots.SESSION_VALUE_SPEC,
    plots.CORPUS_COVERAGE_SPEC,
    plots.ESCALATION_BUDGET_SPEC,
)


def test_every_figure_spec_carries_a_claim_a_reading_and_a_goal() -> None:
    assert len(_SPECS) == 6
    for spec in _SPECS:
        assert spec.title.strip()
        assert spec.reading.strip()
        assert spec.goal.strip()
        assert len(spec.caveat or "") <= 120


@pytest.mark.parametrize(
    "removed",
    [
        # The seven retired figures plus the helpers that existed only to draw them. Leaving the
        # old entry points around invites their reuse at an operating point nothing else shows.
        "pr_curve",
        "policy_pr_curve",
        "policy_roc_curve",
        "policy_confusion",
        "outcome_bars",
        "sweep_table",
        "permutation_null_plot",
        "recurrence_roc",
        "session_cadence_bars",
        "capture_coverage",
        "_draw_confusion_cells",
        "_attach_excess_colorbar",
        "_confusion_notes",
        "_annotate_sweep_points",
        "PR_CURVE_SPEC",
        "ROC_CURVE_SPEC",
        "CONFUSION_MATRIX_SPEC",
        "SWEEP_TABLE_SPEC",
        "PERMUTATION_NULL_SPEC",
        "OUTCOME_BARS_SPEC",
        "CAPTURE_COVERAGE_SPEC",
        "RECURRENCE_ROC_SPEC",
        "EDIT_GATED_SWEEP_TABLE_SPEC",
        "SESSION_CADENCE_SPEC",
        # Retired before this rework, and still gone.
        "steps_to_detection_hist",
        "sweep_heatmap",
        "cost_quality_frontier",
        "lead_time_by_outcome",
        "LEAD_TIME_SPEC",
    ],
)
def test_replaced_figures_are_gone(removed: str) -> None:
    assert not hasattr(plots, removed)
