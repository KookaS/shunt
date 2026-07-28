"""Detector figures. Each function draws into a caller's Axes and returns the Annotations its"""

# own data earned — no file I/O here. The figure's static READ/GOAL/TERMS/NOTE/LIMITS text lives
# next to each function as a frozen FigureSpec; anything data-dependent comes back through
# Annotations so it can never go stale as the corpus grows.
#
# EVERY curve figure here is drawn against its null. A shape without a null band invites the reader
# to see signal in noise, which is exactly what the previous ROC figure did: its polyline admitted
# one tied row at a time in corpus order, so the drawn area was 0.554 while the title said 0.450 —
# visually above chance for a below-chance detector. Ties are now collapsed to real operating
# points (metrics.roc_operating_points / pr_operating_points), so the drawn area equals the number.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmark.escalation import metrics
from benchmark.plot_frame import Annotations, FigureSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from benchmark.calibration.labeler_metrics import ConfusionMatrix
    from benchmark.escalation.features import ModelCoverage
    from benchmark.escalation.metrics import NullResult
    from benchmark.escalation.policy_eval import PolicyCell

_NULL_BAND = "#BDBDBD"
_OBSERVED = "#B71C1C"

PR_CURVE_SPEC = FigureSpec(
    reading=(
        "x is recall and y is precision, both 0-1, over the out-of-fold risk score of one "
        "trajectory per row. One vertex is one DISTINCT score threshold — ties are collapsed, so "
        "the area under the drawn line is the AUPRC in the title, not a row-ordering artifact. "
        "The dashed grey line is prevalence, the precision reached by flagging everything."
    ),
    goal=(
        "Look for the curve sitting well above the dashed prevalence line and staying high as "
        "recall grows. A curve hugging the prevalence line is no signal."
    ),
    definitions=(
        ("precision", "share of flagged runs that really failed"),
        ("recall", "share of failed runs the model flagged"),
        ("prevalence", "share of all runs that failed; the no-skill precision"),
        ("AUPRC", "average precision over the ranked runs; above prevalence means signal"),
    ),
    notes=(
        "The score is the grouped-cross-validated probability that this run ends in failure, "
        "predicted from a fixed-depth prefix — a continuous score, not the policy's fired flag.",
    ),
    limitations=(
        "One row is one run, so prevalence is the corpus failure rate rather than the tiny "
        "per-prefix rate the old framing used — the two AUPRC numbers are not comparable.",
        "AUPRC alone does not say whether the model beats the router's own t=0 task prior; read "
        "the incremental number on the permutation-null figure for that.",
    ),
)

ROC_CURVE_SPEC = FigureSpec(
    reading=(
        "x is the false-positive rate and y the true-positive rate over the same out-of-fold "
        "risk score, one vertex per DISTINCT threshold. The dashed diagonal is chance. The grey "
        "band is the label-permutation null: the region between the two ROC curves whose areas "
        "equal the null's 2.5th and 97.5th percentiles. A curve inside that band is noise."
    ),
    goal=(
        "Look for the curve leaving the grey null band toward the top-left. A curve inside the "
        "band means the model is indistinguishable from randomly shuffled labels."
    ),
    definitions=(
        ("true-positive rate", "share of failed runs the model flagged"),
        ("false-positive rate", "share of resolved runs the model wrongly flagged"),
        ("AUROC", "tie-averaged rank statistic behind this curve; 0.5 is chance"),
        ("null band", "where the curve lands when the outcome labels are shuffled at random"),
    ),
    notes=("Auxiliary: AUPRC against prevalence is the primary measure.",),
    limitations=(
        "The band is drawn as two two-segment curves with exactly the null's boundary areas; it "
        "is an area-equivalent envelope, not the pointwise spread of the permuted curves.",
    ),
)

CONFUSION_MATRIX_SPEC = FigureSpec(
    reading=(
        "A 2x2 count grid at the operating threshold. Rows are the truth (run failed / resolved), "
        "columns are what the detector said (flagged / not flagged). Each cell prints the observed "
        "count and, in brackets, the count a RANDOM flagger at the same flag rate would produce. "
        "Observed above random in the top-left cell is the only good news this figure can carry."
    ),
    goal=(
        "Want top-left and bottom-right well ABOVE their bracketed random counterparts. Observed "
        "at or below random means the flag carries no information."
    ),
    definitions=(
        ("flagged", "detector score above the operating threshold"),
        ("random baseline", "expected count if the same number of flags were scattered at random"),
    ),
    notes=("Counts are per trajectory, not per prefix.",),
    limitations=(
        "One arbitrary operating point, not a sweep — the curve figures show the rest.",
        "The random baseline is an expectation, not a sampled interval; a cell one or two counts "
        "above it is not evidence.",
    ),
)

LEAD_TIME_SPEC = FigureSpec(
    reading=(
        "Two overlaid histograms of lead time — decisions between the policy's first escalation "
        "and the end of the run — split by how the run actually ended. Orange is runs that failed, "
        "blue is runs that resolved. Counts are trajectories in ONE configuration, not pooled."
    ),
    goal=(
        "Want the orange (failed) distribution shifted RIGHT of the blue one: escalating earlier "
        "on the runs that go on to fail. Two overlapping distributions mean the policy fires at "
        "the same time whether or not the run is doomed — no timing signal."
    ),
    definitions=(
        ("lead time", "decisions remaining after the first escalation fired"),
        ("detection", "the first non-HOLD directive the replayed policy emitted"),
    ),
    notes=(
        "This replaces the old steps-to-detection histogram, which pooled every grid cell and so "
        "drew 415 real events as 4980 bars.",
    ),
    limitations=(
        "Only escalated trajectories appear; runs the policy never flagged have no lead time and "
        "are absent from both distributions.",
        "Lead time is measured backwards from the end, so it is an offline diagnostic — an online "
        "detector cannot know it.",
    ),
)

SWEEP_TABLE_SPEC = FigureSpec(
    reading=(
        "One row per swept configuration. Columns: escalations fired, P(fail | fired) with its "
        "95% Wilson interval, the base failure rate, lift, and the permutation-null verdict. The "
        "table is drawn rather than plotted because the sweep has too few distinct results to "
        "carry a colour channel honestly."
    ),
    goal=(
        "Read the P(fail | fired) interval against the base rate column. An interval containing "
        "the base rate is a configuration with no measured value; an interval BELOW it means "
        "firing predicts success."
    ),
    definitions=(
        ("P(fail | fired)", "share of escalated runs that ultimately failed"),
        ("lift", "P(fail | fired) divided by the base failure rate; 1.0 is worthless"),
    ),
    notes=(
        "stale_window and ladder are pinned, not swept: both were measured inert on this corpus "
        "(12 cells collapsed to 2 distinct score vectors).",
    ),
    limitations=(
        "Wilson intervals are per row and unadjusted for the multiple configurations compared.",
    ),
)

PERMUTATION_NULL_SPEC = FigureSpec(
    reading=(
        "The grey histogram is the statistic recomputed under randomly shuffled outcome labels, "
        "with the whole fitting pipeline re-run per shuffle. The dashed lines bound the null's "
        "central 95%. The red line is the real, unshuffled value. x is AUROC."
    ),
    goal=(
        "The red line must sit clearly to the RIGHT of the upper dashed line. Inside the dashed "
        "bounds means the result is indistinguishable from noise, whatever the point estimate."
    ),
    definitions=(
        ("null", "what this pipeline scores when the labels carry no information"),
        ("incremental", "AUROC of prior+prefix minus AUROC of the router's t=0 prior alone"),
    ),
    notes=("The null is the gate: a point estimate above 0.5 is not skill on its own.",),
    limitations=(
        "Shuffling labels globally destroys the challenge-level clustering of outcomes, so the "
        "null is slightly narrower than a fully group-preserving null would be.",
    ),
)

OUTCOME_BARS_SPEC = FigureSpec(
    reading=(
        "Two bars with 95% Wilson intervals: the share of runs that failed among those the policy "
        "escalated, and among those it left alone. The dashed line is the corpus base failure "
        "rate. n is printed on each bar."
    ),
    goal=(
        "Want the left bar (escalated) clearly ABOVE the dashed base rate and above the right "
        "bar. Left below right means the policy is INVERTED: it fires on the runs that succeed."
    ),
    definitions=(("base rate", "share of all runs in this corpus that ended unresolved"),),
    notes=("This is the question the product asks, at the unit the policy acts on.",),
    limitations=(
        "Association only: no stored trajectory contains an escalation that actually happened, so "
        "this cannot say what escalating would have CHANGED.",
    ),
)

CAPTURE_COVERAGE_SPEC = FigureSpec(
    reading=(
        "One bar per model: the share of that model's trajectories that went through the per-step "
        "verified-outcome stamping stage. A bar at zero means not one of that model's runs carries "
        "per-step outcomes, so the recurrence trigger is structurally dead on it. The number of "
        "trajectories is printed on each bar."
    ),
    goal=(
        "Want every bar at 1.0. Any bar at zero invalidates every per-model comparison on this "
        "corpus — that model contributes outcome labels but no detectable evidence."
    ),
    definitions=(
        ("stamped", "the offline container replay ran and wrote per-step verified outcomes"),
        ("capture rate", "share of failed steps carrying a failing_check_id"),
    ),
    notes=(
        "Unstamped runs are a pipeline coverage gap, not agent behaviour: their per-step fields "
        "are the parser defaults, so they look uniformly successful until the terminal grade.",
    ),
    limitations=(
        "Stamping coverage tracks capture DATE, and capture date correlates with model, so model "
        "and coverage are confounded on this corpus and cannot be separated from it.",
    ),
)


def _scored_note(labels: Sequence[bool]) -> str:
    """How much data is behind a curve — runtime, so it can never go stale as the corpus grows."""
    return f"scored on {len(labels)} trajectories, {sum(labels)} of them failed"


def pr_curve(scores: Sequence[float], labels: Sequence[bool], ax: Axes) -> Annotations:
    """Tie-collapsed precision-recall curve with the prevalence (no-skill) baseline."""
    points = metrics.pr_operating_points(scores, labels)
    ax.plot([r for r, _ in points], [p for _, p in points], marker="o", label="risk model")
    baseline = metrics.prevalence(labels)
    ax.axhline(baseline, linestyle="--", color="grey", label=f"prevalence={baseline:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AUPRC={metrics.auprc(scores, labels):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return Annotations(
        notes=(_scored_note(labels), f"{len(points)} distinct operating points"),
        limitations=() if any(labels) else ("No failed runs: the curve is meaningless.",),
    )


def roc_curve(
    scores: Sequence[float], labels: Sequence[bool], null: NullResult, ax: Axes
) -> Annotations:
    """Tie-collapsed ROC with the chance diagonal and the permutation-null area band."""
    points = metrics.roc_operating_points(scores, labels)
    _null_band(null, ax)
    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label="risk model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(f"ROC (AUROC={metrics.auroc(scores, labels):.3f}, auxiliary)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    verdict = "outside" if null.beats_null else "INSIDE"
    return Annotations(
        notes=(
            _scored_note(labels),
            f"{len(points)} distinct operating points",
            f"observed AUROC sits {verdict} the null band "
            f"[{null.ci_low:.3f}, {null.ci_high:.3f}], p={null.p_value:.3f}",
        ),
        limitations=(
            ()
            if null.beats_null
            else ("NO USABLE SIGNAL: the observed AUROC is inside its own permutation null.",)
        ),
    )


def _null_band(null: NullResult, ax: Axes) -> None:
    """Shade between the two two-segment ROC curves whose areas equal the null's 95% bounds."""
    # A polyline (0,0)->(0.5,b)->(1,1) has area 0.5*b + 0.25, so b = 2*(A - 0.25) draws a curve of
    # exactly area A. Two of them bound the region an AUROC in the null interval can occupy.
    x = [0.0, 0.5, 1.0]
    low = [0.0, max(0.0, 2 * (null.ci_low - 0.25)), 1.0]
    high = [0.0, min(1.0, 2 * (null.ci_high - 0.25)), 1.0]
    ax.fill_between(x, low, high, color=_NULL_BAND, alpha=0.55, label="permutation null 95%")


def confusion_matrix_plot(cm: ConfusionMatrix, ax: Axes) -> Annotations:
    """2x2 confusion counts with the expected-under-random count printed in each cell."""
    grid = np.array([[cm.tp, cm.fn], [cm.fp, cm.tn]], dtype=float)
    total = float(grid.sum())
    flagged = (cm.tp + cm.fp) / total if total else 0.0
    positives = cm.tp + cm.fn
    negatives = cm.fp + cm.tn
    expected = np.array(
        [
            [positives * flagged, positives * (1 - flagged)],
            [negatives * flagged, negatives * (1 - flagged)],
        ]
    )
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1], ["flagged", "not flagged"])
    ax.set_yticks([0, 1], ["failed", "resolved"])
    ax.set_title("confusion @ operating threshold (random baseline in brackets)")
    for r in range(2):
        for c in range(2):
            ax.text(
                c,
                r,
                f"{int(grid[r, c])}\n[{expected[r, c]:.0f}]",
                ha="center",
                va="center",
                color="black",
            )
    return Annotations(
        notes=(
            f"{int(total)} trajectories at threshold {metrics.DETECTION_THRESHOLD}",
            f"flag rate {flagged:.3f}; a random flagger at that rate catches "
            f"{expected[0, 0]:.0f} of {positives} failures against the detector's {cm.tp}",
        ),
        limitations=(
            ()
            if cm.tp > expected[0, 0]
            else ("The detector catches no more failures than random flagging at the same rate.",)
        ),
    )


def lead_time_by_outcome(cell: PolicyCell, ax: Axes) -> Annotations:
    """Overlaid lead-time distributions for escalated runs, split by terminal outcome."""
    failed = list(cell.lead_times_failed)
    resolved = list(cell.lead_times_resolved)
    both = failed + resolved
    if both:
        # Clip at p99 with an overflow bin: a handful of 80-step outliers otherwise stretch the
        # axis until 97% of the canvas is blank, which is what the old figure did.
        cap = float(np.percentile(both, 99))
        bins = np.linspace(0, max(cap, 1.0), 25).tolist()
        ax.hist(np.clip(failed, 0, cap), bins=bins, alpha=0.6, label=f"failed (n={len(failed)})")
        ax.hist(
            np.clip(resolved, 0, cap), bins=bins, alpha=0.6, label=f"resolved (n={len(resolved)})"
        )
        ax.legend()
    ax.set_xlabel("lead time (decisions between first escalation and the end of the run)")
    ax.set_ylabel("trajectories")
    ax.set_title(f"lead time by outcome — escalate_after_n={cell.escalate_after_n}")
    return Annotations(
        notes=(_lead_note(failed, resolved),),
        limitations=() if both else ("No escalation fired: both distributions are empty.",),
    )


def _lead_note(failed: Sequence[int], resolved: Sequence[int]) -> str:
    """Median lead time per outcome class, stated so the reader need not eyeball the overlap."""
    med_f = f"{np.median(failed):.0f}" if failed else "n/a"
    med_r = f"{np.median(resolved):.0f}" if resolved else "n/a"
    return f"median lead time: failed={med_f}, resolved={med_r}"


def sweep_table(cells: Sequence[PolicyCell], ax: Axes) -> Annotations:
    """The sweep as a table with intervals — the honest replacement for a 2-result heatmap."""
    ax.axis("off")
    header = ["n", "escalated", "P(fail|fired)", "95% CI", "base", "lift", "vs null"]
    rows = [
        [
            str(c.escalate_after_n),
            str(c.n_escalated),
            f"{c.precision:.3f}",
            f"[{c.precision_ci[0]:.3f}, {c.precision_ci[1]:.3f}]",
            f"{c.base_failure_rate:.3f}",
            f"{c.lift:.2f}x",
            "beats" if c.null_auroc.beats_null else "inside",
        ]
        for c in cells
    ]
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("policy sweep — escalate_after_n (stale_window and ladder pinned)")
    return Annotations(notes=(_sweep_note(cells),), limitations=_sweep_limits(cells))


def _sweep_note(cells: Sequence[PolicyCell]) -> str:
    if not cells:
        return "no configurations swept"
    best = max(cells, key=lambda c: c.precision)
    return (
        f"{len(cells)} configurations; highest P(fail|fired) is {best.precision:.3f} at "
        f"escalate_after_n={best.escalate_after_n} against a base rate of "
        f"{best.base_failure_rate:.3f}"
    )


def _sweep_limits(cells: Sequence[PolicyCell]) -> tuple[str, ...]:
    """Say plainly when no configuration's interval clears the base rate."""
    if not cells:
        return ()
    if any(c.precision_ci[0] > c.base_failure_rate for c in cells):
        return ()
    return (
        "No configuration's precision interval clears the base failure rate: on this corpus the "
        "sweep contains no setting with measured value.",
    )


def permutation_null_plot(null: NullResult, ax: Axes, *, label: str) -> Annotations:
    """The null distribution with its 95% bounds and the observed value drawn over it."""
    ax.hist(list(null.draws), bins=40, color=_NULL_BAND, label=f"null ({null.n_permutations} perm)")
    ax.axvline(null.ci_low, linestyle="--", color="grey")
    ax.axvline(null.ci_high, linestyle="--", color="grey", label="null 95%")
    ax.axvline(null.observed, color=_OBSERVED, linewidth=2, label=f"observed={null.observed:.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("permutations")
    ax.set_title(f"{label} against its permutation null")
    ax.legend()
    verdict = (
        "clears the null band"
        if null.beats_null
        else "sits INSIDE the null band — indistinguishable from chance"
    )
    return Annotations(
        notes=(f"observed {null.observed:.4f} {verdict}; p={null.p_value:.4f}",),
        limitations=()
        if null.beats_null
        else (f"NO USABLE SIGNAL on {label}: p={null.p_value:.3f} against shuffled labels.",),
    )


def outcome_bars(cell: PolicyCell, ax: Axes) -> Annotations:
    """P(fail | fired) vs P(fail | not fired) with Wilson intervals and the base-rate line."""
    fired_lo, fired_hi = cell.precision_ci
    quiet_lo, quiet_hi = metrics.wilson_interval(cell.fn, cell.fn + cell.tn)
    heights = [cell.precision, cell.p_fail_given_quiet]
    errors = np.array(
        [
            [cell.precision - fired_lo, cell.p_fail_given_quiet - quiet_lo],
            [fired_hi - cell.precision, quiet_hi - cell.p_fail_given_quiet],
        ]
    )
    bars = ax.bar(["escalated", "not escalated"], heights, yerr=errors, capsize=6)
    for bar, n in zip(bars, [cell.tp + cell.fp, cell.fn + cell.tn], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"n={n}", ha="center", color="white")
    ax.axhline(
        cell.base_failure_rate,
        linestyle="--",
        color="grey",
        label=f"base rate={cell.base_failure_rate:.3f}",
    )
    ax.set_ylabel("P(run ultimately failed)")
    ax.set_ylim(0, 1)
    ax.set_title(f"outcome by escalation — escalate_after_n={cell.escalate_after_n}")
    ax.legend()
    return Annotations(notes=(_outcome_note(cell),), limitations=_outcome_limits(cell))


def _outcome_note(cell: PolicyCell) -> str:
    return (
        f"P(fail|fired)={cell.precision:.3f} vs P(fail|not fired)="
        f"{cell.p_fail_given_quiet:.3f}, base rate {cell.base_failure_rate:.3f} "
        f"(lift {cell.lift:.2f}x)"
    )


def _outcome_limits(cell: PolicyCell) -> tuple[str, ...]:
    """Call the direction ONLY when the two Wilson intervals actually separate."""
    # A point estimate below the base rate is not evidence of an inverted policy while the intervals
    # overlap — reading a sign off overlapping bars is the same error as reading skill off a point
    # estimate inside its null, which is what this whole harness exists to stop.
    quiet_lo, quiet_hi = metrics.wilson_interval(cell.fn, cell.fn + cell.tn)
    fired_lo, fired_hi = cell.precision_ci
    if fired_hi < quiet_lo:
        return (
            "INVERTED: escalated runs failed significantly LESS often than the runs the policy "
            "left alone, so firing spends budget on the attempts least likely to need it.",
        )
    if quiet_hi < fired_lo:
        return ()
    return (
        "NO SEPARATION: the two intervals overlap and both contain the base rate, so on this "
        "corpus escalating carries no measured information about the outcome either way.",
    )


def capture_coverage(coverages: Sequence[ModelCoverage], ax: Axes) -> Annotations:
    """Per-model share of trajectories that carry per-step verified outcomes."""
    names = [c.model for c in coverages]
    shares = [c.n_stamped / c.n_trajectories if c.n_trajectories else 0.0 for c in coverages]
    bars = ax.barh(names, shares)
    for bar, cov in zip(bars, coverages, strict=True):
        ax.text(0.01, bar.get_y() + bar.get_height() / 2, f"n={cov.n_trajectories}", va="center")
    ax.set_xlabel("share of trajectories with per-step verified outcomes")
    ax.set_xlim(0, 1)
    ax.set_title("failure-capture coverage by model")
    dead = [c for c in coverages if c.n_stamped == 0]
    limits: tuple[str, ...] = ()
    if dead:
        lost = sum(c.n_trajectories for c in dead)
        total = sum(c.n_trajectories for c in coverages)
        limits = (
            f"{', '.join(c.model for c in dead)} ({lost} trajectories, "
            f"{lost / total:.0%} of the corpus) carry NO per-step outcomes at all: the recurrence "
            "trigger cannot fire on them and they are excluded from the risk model.",
        )
    return Annotations(
        notes=(
            f"{len(coverages)} models, {sum(c.n_trajectories for c in coverages)} trajectories",
        ),
        limitations=limits,
    )


__all__ = [
    "CAPTURE_COVERAGE_SPEC",
    "CONFUSION_MATRIX_SPEC",
    "LEAD_TIME_SPEC",
    "OUTCOME_BARS_SPEC",
    "PERMUTATION_NULL_SPEC",
    "PR_CURVE_SPEC",
    "ROC_CURVE_SPEC",
    "SWEEP_TABLE_SPEC",
    "capture_coverage",
    "confusion_matrix_plot",
    "lead_time_by_outcome",
    "outcome_bars",
    "permutation_null_plot",
    "pr_curve",
    "roc_curve",
    "sweep_table",
]
