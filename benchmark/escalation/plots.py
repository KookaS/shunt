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
# The theoretical 0.5 ROC diagonal, drawn faint. It is orientation, not the reference: the
# reference is the MEASURED null centre, which on a refit-per-shuffle pipeline is not 0.5.
_FAINT = "#CCCCCC"
_NULL_CENTRE = "#616161"

PR_CURVE_SPEC = FigureSpec(
    reading=(
        "x is recall and y is precision, both 0-1, over the out-of-fold risk score of one "
        "trajectory per row. One vertex is one DISTINCT score threshold — ties are collapsed, so "
        "the area under the drawn line is the AUPRC in the title, not a row-ordering artifact. "
        "The grey band is the MEASURED permutation null and the dashed line through it is that "
        "null's centre: what this pipeline's own AUPRC comes to when the outcome labels carry no "
        "information. That band, not prevalence, is the no-skill reference on this corpus."
    ),
    goal=(
        "Look for the curve sitting well above the grey null band and staying high as recall "
        "grows. A curve inside or hugging the band is no signal."
    ),
    definitions=(
        ("precision", "share of flagged runs that really failed"),
        ("recall", "share of failed runs the model flagged"),
        ("AUPRC", "average precision over the ranked runs"),
        (
            "permutation null",
            "the AUPRC this same fitted pipeline reaches on shuffled outcome labels — measured, "
            "not assumed",
        ),
    ),
    notes=(
        "The score is the grouped-cross-validated probability that this run ends in failure, "
        "predicted from a fixed-depth prefix — a continuous score, not the policy's fired flag.",
        "PREVALENCE IS NOT DRAWN AS THE BASELINE. It is the no-skill average precision for "
        "EXCHANGEABLE rows; these rows cluster by challenge and the whole pipeline is refit per "
        "shuffle, so the pipeline's no-information AUPRC is an empirical quantity. Both numbers "
        "are printed below — the gap between them is the size of the assumption that was dropped.",
    ),
    limitations=(
        "One row is one run, so the corpus failure rate quoted below is per-trajectory rather "
        "than the tiny per-prefix rate the old framing used — the two AUPRC numbers are not "
        "comparable.",
        "AUPRC alone does not say whether the model beats the router's own t=0 task prior; read "
        "the incremental number on the permutation-null figure for that.",
        "This null is this depth's own. The family-wise correction on the permutation-null figure "
        "applies to the AUROC statistics the skill gate reads, and AUPRC is not one of them.",
    ),
)

ROC_CURVE_SPEC = FigureSpec(
    reading=(
        "x is the false-positive rate and y the true-positive rate over the same out-of-fold "
        "risk score, one vertex per DISTINCT threshold. The grey band is the label-permutation "
        "null and the dashed line through it is that null's MEASURED centre — the reference here. "
        "The faint dotted diagonal is the theoretical 0.5 chance line, drawn for orientation "
        "only: this pipeline's no-information AUROC is the MEASURED number in the legend, which "
        "need not be 0.5. A curve inside the band is noise."
    ),
    goal=(
        "Look for the curve leaving the grey null band toward the top-left. A curve inside the "
        "band means the model is indistinguishable from randomly shuffled labels — read it "
        "against the band and its measured centre, never against the dotted diagonal."
    ),
    definitions=(
        ("true-positive rate", "share of failed runs the model flagged"),
        ("false-positive rate", "share of resolved runs the model wrongly flagged"),
        ("AUROC", "tie-averaged rank statistic behind this curve"),
        ("null band", "where the curve lands when the outcome labels are shuffled at random"),
        (
            "measured null centre",
            "the mean AUROC over those shuffles; the operative chance level for this pipeline",
        ),
    ),
    notes=("Auxiliary: AUPRC against its own measured null is the primary measure.",),
    limitations=(
        "The band and its centre are drawn as two-segment curves with exactly the null's areas; "
        "they are area-equivalent envelopes, not the pointwise spread of the permuted curves.",
    ),
)

CONFUSION_MATRIX_SPEC = FigureSpec(
    reading=(
        "A 2x2 count grid at the operating threshold. Rows are the truth (run failed / resolved), "
        "columns are what the detector said (flagged / not flagged). Each cell prints the observed "
        "count, in brackets the count a RANDOM flagger at the same flag rate would produce, and "
        "the difference between them. There is deliberately NO colour channel: with the number of "
        "flags fixed, all four differences are the SAME number up to sign, so shading four counts "
        "would dress one quantity up as four."
    ),
    goal=(
        "Want the top-left count well ABOVE its bracketed random counterpart. That one excess IS "
        "the figure — the other three cells restate it with a sign and add no evidence. At or "
        "below random means the flag carries no information."
    ),
    definitions=(
        ("flagged", "detector score at or above the operating threshold"),
        (
            "operating threshold",
            "the base-rate quantile of the score; rows tied with the cut are admitted whole, so "
            "the realised flag count can exceed the base-rate budget",
        ),
        ("random baseline", "expected count if the same number of flags were scattered at random"),
        ("excess", "observed minus that random baseline, in the cell's own units"),
    ),
    notes=(
        "Counts are per trajectory, not per prefix.",
        "The threshold is DERIVED from this corpus (the base-rate quantile of the score), never a "
        "fixed 0.5: the score is a calibrated probability, so its centre tracks the corpus failure "
        "rate and a cut pinned away from that rate degenerates to flagging almost nothing or "
        "almost everything. This corpus's measured base rate is stated below.",
    ),
    limitations=(
        "One operating point, not a sweep — the curve figures show the rest.",
        "Spending a flag budget equal to prevalence is a choice, not an optimum; a different "
        "budget moves every count in this grid.",
        "The random baseline is an expectation, not a sampled interval; a cell one or two counts "
        "above it is not evidence.",
    ),
)

SWEEP_TABLE_SPEC = FigureSpec(
    reading=(
        "One row per swept configuration. Columns: escalations fired, P(fail | fired) with its "
        "95% challenge-bootstrap interval, the base failure rate, lift, and the permutation-null "
        "verdict. The table is drawn rather than plotted because the sweep has too few distinct "
        "results to carry a colour channel honestly."
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
        "ladder is pinned, not swept, and stale_window is swept only incidentally: the shipped "
        "cell (stale_window=10) is appended to a grid pinned at 5, so the column carries two "
        "values rather than one. Both knobs were measured inert on this corpus (12 cells "
        "collapsed to 2 distinct score vectors) — which is why the two n=2 rows agree on every "
        "measured column rather than being a duplicated row.",
        "The interval is a CHALLENGE-level bootstrap, not a Wilson interval over rows: the corpus "
        "is 791 runs drawn from 160 challenges, so rows are not independent draws and a row-level "
        "interval is roughly 2x too narrow.",
    ),
    limitations=(
        "The intervals are per row of this table and unadjusted for the multiple configurations "
        "compared side by side.",
        "A configuration that never fires has no P(fail | fired) at all; it prints n/a rather "
        "than 0.000, and it is excluded from the highest-precision line below.",
    ),
)

PERMUTATION_NULL_SPEC = FigureSpec(
    reading=(
        "The grey histogram is the statistic recomputed under outcome labels shuffled WITHIN each "
        "challenge, with the whole fitting pipeline re-run per shuffle. It is the FAMILY-WISE "
        "null: each shuffle is scored at EVERY reported depth and only the largest of those "
        "values enters the histogram, so the band already prices the fact that the verdict takes "
        "the best depth. The dashed lines bound that max distribution's central 95%. The red line "
        "is the real, unshuffled value at this depth. x is the statistic named on the axis."
    ),
    goal=(
        "The red line must sit clearly to the RIGHT of the upper dashed line. Inside the dashed "
        "bounds means the result is indistinguishable from noise, whatever the point estimate."
    ),
    definitions=(
        ("null", "what this pipeline scores when the labels carry no information"),
        (
            "family-wise",
            "the max-statistic (maxT) null over the reported depths, so the p-value beside it is "
            "already the multiplicity-adjusted one — exact under any dependence between depths",
        ),
        (
            "incremental",
            "AUROC of prior+prefix minus the router's t=0 prior FLOORED at chance, "
            "max(prior, 0.5) — an anti-predictive prior must not be beatable",
        ),
    ),
    notes=(
        "The null is the gate: a point estimate above chance is not skill on its own.",
        "Permuting inside each challenge preserves every challenge's outcome multiset, so the "
        "deployable prior is IDENTICAL under the null and the observation and only the prefix's "
        "contribution is nulled. A global shuffle collapses the prior to chance and leaves the "
        "two arms in different headroom regimes, which is a gate with no power.",
        "The uncorrected per-depth null is reported in the JSON beside this one, so the size of "
        "the correction is readable rather than folded away.",
    ),
    limitations=(
        "Clearing this null is necessary, not sufficient: the paired grouped bootstrap must also "
        "exclude zero, or the estimate is a property of this particular set of challenges.",
        "A challenge whose runs all share an outcome cannot move under a within-challenge "
        "permutation, so this null is estimated off the heterogeneous challenges only — here "
        "roughly half the corpus.",
        "One shared randomization serves every depth, so a deeper depth's own outcome multiset is "
        "not preserved exactly within its subset. That widens the null at depth rather than "
        "narrowing it, which is the conservative direction.",
    ),
)

OUTCOME_BARS_SPEC = FigureSpec(
    reading=(
        "Two bars with 95% intervals: the share of runs that failed among those the policy "
        "escalated, and among those it left alone. BOTH intervals are bootstrapped over whole "
        "CHALLENGES from the same resamples, since runs inside a challenge are correlated and the "
        "footer compares the two. The dashed line is the corpus base failure rate. n is printed "
        "on each bar."
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
        "The direction in the footer is called from two marginal intervals failing to overlap. "
        "That is a conservative test of a difference — it can miss a real gap, but it cannot "
        "invent one; it is not a paired test of the difference itself.",
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
        ("terminal failure rate", "share of that model's runs that ended unresolved"),
    ),
    notes=(
        "Unstamped runs are a pipeline coverage gap, not agent behaviour: their per-step fields "
        "are the parser defaults, so they look uniformly successful until the terminal grade.",
    ),
    limitations=(
        "Stamping coverage tracks capture DATE, and capture date correlates with model, so model "
        "and coverage are confounded on this corpus and cannot be separated from it.",
        "This figure deliberately does NOT plot the `capture_rate` field that the JSON report "
        "carries. That field is the share of failed steps holding a failing_check_id, and the "
        "normalizer writes `success`, `failing_check_id` and `blocking` in one assignment — so it "
        "is 1.000 for every model BY CONSTRUCTION (measured 6644/6644, 5278/5278, 3260/3260, "
        "993/993, 2245/2245, 841/841). A bar chart of it would be six identical full bars "
        "presenting an identity as a measurement.",
    ),
)


def _scored_note(labels: Sequence[bool]) -> str:
    """How much data is behind a curve — runtime, so it can never go stale as the corpus grows."""
    return f"scored on {len(labels)} trajectories, {sum(labels)} of them failed"


def pr_curve(
    scores: Sequence[float], labels: Sequence[bool], null: NullResult | None, ax: Axes
) -> Annotations:
    """Tie-collapsed precision-recall curve against the MEASURED AUPRC null, not prevalence."""
    # PREVALENCE IS NOT THE NO-SKILL LINE HERE. It is the no-skill average precision for
    # exchangeable rows; these rows cluster by challenge and `prefix_eval` refits the whole
    # pipeline per shuffle, so the pipeline's own no-information AUPRC is something you MEASURE.
    # Drawing prevalence as "no skill" set the bar at a number no null ever produced, and on the
    # committed corpus it sat above the measured null — i.e. the wrong side, the one that hides a
    # detector rather than flattering it. The measured band comes in through `null`; it is never
    # reconstructed here from a literal.
    points = metrics.pr_operating_points(scores, labels)
    if null is not None:
        ax.axhspan(
            null.ci_low, null.ci_high, color=_NULL_BAND, alpha=0.55, label="permutation null 95%"
        )
        ax.axhline(
            null.mean,
            linestyle="--",
            color=_NULL_CENTRE,
            label=f"permutation null={null.mean:.3f}",
        )
    ax.plot([r for r, _ in points], [p for _, p in points], marker="o", label="risk model")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AUPRC={metrics.auprc(scores, labels):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return Annotations(
        notes=(
            _scored_note(labels),
            f"{len(points)} distinct operating points",
            *_pr_null_notes(null, labels),
        ),
        limitations=(
            *(() if any(labels) else ("No failed runs: the curve is meaningless.",)),
            *_pr_null_limits(null),
        ),
    )


def _pr_null_notes(null: NullResult | None, labels: Sequence[bool]) -> tuple[str, ...]:
    """State the measured null and the prevalence it replaced, so the gap is on the canvas."""
    prevalence = metrics.prevalence(labels)
    if null is None:
        return (f"corpus prevalence {prevalence:.4f} (printed, NOT drawn as a no-skill line)",)
    verdict = "outside" if null.beats_null else "INSIDE"
    return (
        f"no-skill reference is the MEASURED permutation null {null.mean:.4f} "
        f"(95% [{null.ci_low:.4f}, {null.ci_high:.4f}], sd {null.sd:.4f}, "
        f"{null.n_permutations} shuffles); corpus prevalence is {prevalence:.4f}",
        f"observed AUPRC {null.observed:.4f} sits {verdict} that null band, p={null.p_value:.3f}",
    )


def _pr_null_limits(null: NullResult | None) -> tuple[str, ...]:
    """Refuse to imply skill, and refuse to imply a baseline that was never supplied."""
    if null is None:
        return (
            "NO BASELINE DRAWN: no permutation null was supplied for this figure. Prevalence is "
            "not a substitute on a challenge-clustered corpus, so the curve is shown bare.",
        )
    if null.beats_null:
        return ()
    return ("NO USABLE SIGNAL: the observed AUPRC is inside its own permutation null.",)


def roc_curve(
    scores: Sequence[float], labels: Sequence[bool], null: NullResult, ax: Axes
) -> Annotations:
    """Tie-collapsed ROC against the MEASURED null band and centre; 0.5 is drawn faint."""
    points = metrics.roc_operating_points(scores, labels)
    _null_band(null, ax)
    # De-emphasised on purpose. 0.5 is the chance level for a fixed score vector against shuffled
    # labels; it is NOT this pipeline's no-information level, because the pipeline is refit per
    # shuffle. Drawing it as a bold dashed reference beside a measured null centre elsewhere on
    # the canvas invited the reader to judge the curve against the wrong line.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        color=_FAINT,
        linewidth=0.8,
        label="theoretical chance 0.5 (not the reference)",
    )
    centre_x, centre_y = _area_polyline(null.mean)
    ax.plot(
        centre_x,
        centre_y,
        linestyle="--",
        color=_NULL_CENTRE,
        linewidth=1.4,
        label=f"measured null centre (AUROC={null.mean:.3f})",
    )
    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label="risk model")
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
            f"the null is MEASURED, not assumed: it centres at {null.mean:.4f} "
            f"(sd {null.sd:.4f}, {null.n_permutations} shuffles), so that value and not the "
            f"dotted 0.5 diagonal is this pipeline's no-information level",
        ),
        limitations=(
            ()
            if null.beats_null
            else ("NO USABLE SIGNAL: the observed AUROC is inside its own permutation null.",)
        ),
    )


def _area_polyline(area: float) -> tuple[list[float], list[float]]:
    """The two-segment ROC (0,0)->(0.5,b)->(1,1) whose area is exactly `area`: b = 2*(area-0.25)."""
    return [0.0, 0.5, 1.0], [0.0, min(1.0, max(0.0, 2 * (area - 0.25))), 1.0]


def _null_band(null: NullResult, ax: Axes) -> None:
    """Shade between the two two-segment ROC curves whose areas equal the null's 95% bounds."""
    # A polyline (0,0)->(0.5,b)->(1,1) has area 0.5*b + 0.25, so b = 2*(A - 0.25) draws a curve of
    # exactly area A. Two of them bound the region an AUROC in the null interval can occupy.
    x, low = _area_polyline(null.ci_low)
    _, high = _area_polyline(null.ci_high)
    ax.fill_between(x, low, high, color=_NULL_BAND, alpha=0.55, label="permutation null 95%")


def confusion_matrix_plot(
    cm: ConfusionMatrix, ax: Axes, *, threshold: float, flag_budget: int
) -> Annotations:
    """2x2 counts, the expected-under-random count, and the one excess all four cells share."""
    counts = ((cm.tp, cm.fn), (cm.fp, cm.tn))
    positives = cm.tp + cm.fn
    negatives = cm.fp + cm.tn
    total = positives + negatives
    flagged = cm.tp + cm.fp
    rate = flagged / total if total else 0.0
    expected = (
        (positives * rate, positives * (1 - rate)),
        (negatives * rate, negatives * (1 - rate)),
    )
    _draw_confusion_cells(counts, expected, ax)
    ax.set_title(
        f"confusion @ derived operating threshold {threshold:.3f} "
        "(count, [random at the same flag rate], excess)"
    )
    return Annotations(
        notes=_confusion_notes(cm, expected[0][0], total=total, threshold=threshold, rate=rate),
        limitations=(
            *_budget_limits(flagged, flag_budget, total),
            *(
                ()
                if cm.tp > expected[0][0]
                else (
                    "The detector catches no more failures than random flagging at the same rate.",
                )
            ),
        ),
    )


def _draw_confusion_cells(
    counts: tuple[tuple[int, int], tuple[int, int]],
    expected: tuple[tuple[float, float], tuple[float, float]],
    ax: Axes,
) -> None:
    """A plain ruled 2x2 — NO colour channel, because the four cells carry ONE number."""
    # The previous version shaded the cells by raw count, which put the heaviest ink on the cell
    # that was furthest BELOW its random baseline: the eye read "biggest = best" off a deficit.
    # Worse, with the flag count fixed the four excesses are algebraically the same number up to
    # sign (tn - E[tn] == tp - E[tp], and both off-diagonals are its negation), so a four-valued
    # colour scale was encoding one degree of freedom as four. Same reasoning as `sweep_table`,
    # which drops colour because two distinct results cannot carry a colour channel honestly.
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1], ["flagged", "not flagged"])
    ax.set_yticks([0, 1], ["failed", "resolved"])
    ax.axhline(0.5, color=_NULL_CENTRE, linewidth=0.8)
    ax.axvline(0.5, color=_NULL_CENTRE, linewidth=0.8)
    for r in range(2):
        for c in range(2):
            ax.text(
                c,
                r,
                f"{counts[r][c]}\n[{expected[r][c]:.0f}]\n{counts[r][c] - expected[r][c]:+.1f}",
                ha="center",
                va="center",
                color="black",
            )


def _confusion_notes(
    cm: ConfusionMatrix, expected_tp: float, *, total: int, threshold: float, rate: float
) -> tuple[str, ...]:
    """The corpus's own numbers, so no caption ever restates a base rate as a literal."""
    positives = cm.tp + cm.fn
    base_rate = positives / total if total else 0.0
    excess = cm.tp - expected_tp
    return (
        f"{total} trajectories at the derived operating threshold {threshold:.4f}; this corpus's "
        f"measured base failure rate is {base_rate:.4f} ({positives} of {total} runs failed)",
        f"flag rate {rate:.3f} ({cm.tp + cm.fp} of {total}); a random flagger at that rate catches "
        f"{expected_tp:.0f} of {positives} failures against the detector's {cm.tp}",
        f"ONE degree of freedom: with the flag count fixed, the whole grid is the caught-failure "
        f"excess {excess:+.1f}. The resolved/not-flagged cell repeats it and the two off-diagonal "
        f"cells are {-excess:+.1f}; none of them is independent evidence.",
    )


def _budget_limits(flagged: int, budget: int, total: int) -> tuple[str, ...]:
    """Say when the cut spent more flags than the base-rate budget it was derived from."""
    # `operating_threshold` returns the base-rate quantile SCORE and every consumer flags
    # `score >= cut`, so a block of runs tied with the cut is admitted whole. When that block
    # straddles the budget the grid is drawn at a flag rate the caption's derivation never chose,
    # and every count in it moves. Stated here rather than silently absorbed.
    if budget <= 0 or flagged <= budget or total <= 0:
        return ()
    return (
        f"BUDGET OVERSPENT BY THE TIE BLOCK: the base-rate budget is {budget} flags "
        f"({budget / total:.3f} of the corpus) but the cut admits {flagged} "
        f"({flagged / total:.3f}), "
        f"because {flagged - budget} run(s) tie with the threshold score and `score >= cut` takes "
        "the whole tie block. Every count in this grid, and the random baseline beside it, is at "
        "the realised rate, not the intended one.",
    )


def sweep_table(cells: Sequence[PolicyCell], ax: Axes) -> Annotations:
    """The sweep as a table with intervals — the honest replacement for a 2-result heatmap."""
    ax.axis("off")
    # `stale_window` is a COLUMN, not a pinned constant: the shipped cell is appended to the swept
    # grid and differs from it on exactly that knob. Rendering `n` alone drew the shipped row and
    # the swept n=2 row as two identical lines — a duplicate to any reader, under a title that
    # claimed the knob was pinned.
    header = ["n", "stale", "escalated", "P(fail|fired)", "95% CI", "base", "lift", "vs null"]
    rows = [
        [
            str(c.escalate_after_n),
            str(c.stale_window),
            str(c.n_escalated),
            _rate(c.precision),
            f"[{_rate(c.precision_ci[0])}, {_rate(c.precision_ci[1])}]",
            f"{c.base_failure_rate:.3f}",
            "n/a" if c.lift is None else f"{c.lift:.2f}x",
            "beats" if c.null_auroc.beats_null else "inside",
        ]
        for c in cells
    ]
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    # 8 columns, not 7: at fontsize 9 the "95% CI" text ran over its own cell rules, so the table
    # is scaled wider and the font dropped a point to keep every cell inside its border.
    table.set_fontsize(8)
    table.scale(1.16, 1.6)
    ax.set_title("policy sweep — escalate_after_n × stale_window (ladder pinned)")
    return Annotations(notes=(_sweep_note(cells),), limitations=_sweep_limits(cells))


def _sweep_note(cells: Sequence[PolicyCell]) -> str:
    # A cell that never fired has NO precision (None), not a precision of zero. Ranking it as 0.0
    # let a never-firing configuration enter the argmax and lose only by luck; it is excluded here.
    scored = [c for c in cells if c.precision is not None]
    if not cells:
        return "no configurations swept"
    if not scored:
        return f"{len(cells)} configurations, none of which fired: P(fail|fired) is undefined"
    best = max(scored, key=lambda c: c.precision or 0.0)
    return (
        f"{len(cells)} configurations ({len(scored)} fired); highest P(fail|fired) is "
        f"{best.precision:.3f} at escalate_after_n={best.escalate_after_n} "
        f"(stale_window={best.stale_window}) against a base rate of {best.base_failure_rate:.3f}"
    )


def _sweep_limits(cells: Sequence[PolicyCell]) -> tuple[str, ...]:
    """Say plainly, per cell rather than globally, when an interval fails to clear the base rate."""
    # `any(...)` used to suppress this warning for EVERY cell as soon as ONE separated — including
    # for the shipped configuration, which is the one a reader is actually deciding on. The count
    # is reported instead, so a lone separating cell can never vouch for the rest of the sweep.
    if not cells:
        return ()
    separating = [c for c in cells if c.precision_ci[0] > c.base_failure_rate]
    if len(separating) == len(cells):
        return ()
    if not separating:
        return (
            "No configuration's precision interval clears the base failure rate: on this corpus "
            "the sweep contains no setting with measured value.",
        )
    cleared = ", ".join(f"n={c.escalate_after_n}/stale={c.stale_window}" for c in separating)
    return (
        f"Only {len(separating)} of {len(cells)} configurations clear the base failure rate "
        f"({cleared}); every other cell in this figure, including any not listed, does not.",
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
    """P(fail | fired) vs P(fail | not fired) on one challenge bootstrap, plus the base rate."""
    # A cell that never fired has an UNDEFINED precision, not one of zero. Drawing it as a
    # zero-height bar would read as "escalating never predicts failure" — a measured claim — when
    # the truth is that the configuration made no prediction at all. It is drawn as absent instead.
    #
    # Both arms read their interval off the SAME challenge resamples (`policy_eval._arm_intervals`).
    # The quiet bar used to carry `metrics.wilson_interval` over rows while the fired bar carried
    # the bootstrap, and the footer below compared the two — a directional verdict read off two
    # estimators that do not answer the same question.
    fired = _bar_geometry(cell.precision, cell.precision_ci)
    quiet = _bar_geometry(cell.p_fail_given_quiet, cell.quiet_ci)
    heights = [fired[0], quiet[0]]
    errors = np.array([[fired[1], quiet[1]], [fired[2], quiet[2]]])
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


def _bar_geometry(value: float | None, interval: tuple[float, float]) -> tuple[float, float, float]:
    """Bar height plus its two error-bar arms; an undefined proportion is drawn as absent."""
    # Absent is a NaN HEIGHT, not a height of zero. A zero-height bar is pixel-identical to a
    # genuine precision of 0.000 — "this policy fired and every flagged run resolved" — so the two
    # readings were separated only by the footer. Matplotlib omits a NaN-height bar from the axes
    # entirely, which is what "the configuration made no prediction" should look like.
    #
    # The arms stay finite at 0.0: `Axes.bar` draws a whisker of unbounded length from a NaN arm,
    # rendering maximum uncertainty as a full-height line — the opposite of absent — and a NaN
    # centre with 0.0 arms already yields nothing to draw.
    if value is None or np.isnan(interval[0]) or np.isnan(interval[1]):
        return (float("nan"), 0.0, 0.0)
    return (value, max(0.0, value - interval[0]), max(0.0, interval[1] - value))


def _rate(value: float | None, digits: int = 3) -> str:
    """Format a proportion for a caption, keeping an undefined one visibly undefined."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _outcome_note(cell: PolicyCell) -> str:
    lift = "n/a" if cell.lift is None else f"{cell.lift:.2f}x"
    return (
        f"P(fail|fired)={_rate(cell.precision)} vs P(fail|not fired)="
        f"{_rate(cell.p_fail_given_quiet)}, base rate {cell.base_failure_rate:.3f} "
        f"(lift {lift})"
    )


def _outcome_limits(cell: PolicyCell) -> tuple[str, ...]:
    """Call the direction ONLY when the two like-for-like intervals actually separate."""
    # A point estimate below the base rate is not evidence of an inverted policy while the intervals
    # overlap — reading a sign off overlapping bars is the same error as reading skill off a point
    # estimate inside its null, which is what this whole harness exists to stop.
    #
    # Both bounds come from `policy_eval`'s challenge bootstrap. Reading `quiet_*` off a Wilson
    # interval over rows, as this did, compared an interval to one built under an assumption the
    # corpus rejects: runs cluster by challenge, so a row-level interval is too narrow. Too narrow
    # raises `quiet_lo` and lowers `quiet_hi`, so it makes BOTH branches below easier to trip — a
    # mismatched pair can manufacture an "INVERTED" caption or a silent endorsement, and neither
    # would be a measurement. Whether any given cell's verdict actually moves is a property of the
    # corpus loaded, not of this fix; what the fix buys is that the two bars are now the same
    # estimator on the same resamples, so the comparison is defined at all.
    fired_lo, fired_hi = cell.precision_ci
    quiet_lo, quiet_hi = cell.quiet_ci
    if cell.precision is None or np.isnan(fired_lo) or np.isnan(fired_hi):
        return (
            "This configuration never fired on this corpus, so P(fail | fired) is UNDEFINED and "
            "the left bar is absent rather than zero. No direction can be read here at all.",
        )
    if cell.p_fail_given_quiet is None or np.isnan(quiet_lo) or np.isnan(quiet_hi):
        return (
            "This configuration fired on EVERY run, so P(fail | not fired) is UNDEFINED and the "
            "right bar is absent rather than zero. There is nothing to compare the left bar "
            "against, so no direction can be read here at all.",
        )
    if fired_hi < quiet_lo:
        return (
            "INVERTED: escalated runs failed significantly LESS often than the runs the policy "
            "left alone, so firing spends budget on the attempts least likely to need it.",
        )
    if quiet_hi < fired_lo:
        return ()
    return (
        "NO SEPARATION: the two challenge-bootstrap intervals overlap, so on this corpus "
        "escalating carries no measured information about the outcome either way.",
    )


def capture_coverage(coverages: Sequence[ModelCoverage], ax: Axes) -> Annotations:
    """Per-model share of trajectories that carry per-step verified outcomes."""
    # STAMPING coverage is plotted, never `ModelCoverage.capture_rate`. The latter is 1.000 for
    # every model on this corpus by construction (see the spec's limitation), so a bar of it would
    # be a tautology drawn six times. Stamping coverage is the real collection diagnostic and it
    # does vary — 262/268 for one model against 284/284 for another, 8 unstamped runs overall.
    names = [c.model for c in coverages]
    shares = [c.n_stamped / c.n_trajectories if c.n_trajectories else 0.0 for c in coverages]
    bars = ax.barh(names, shares)
    for bar, cov in zip(bars, coverages, strict=True):
        ax.text(
            0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cov.n_stamped}/{cov.n_trajectories} stamped · "
            f"terminal-fail {cov.terminal_failure_rate:.2f}",
            va="center",
        )
    ax.set_xlabel("share of trajectories with per-step verified outcomes")
    ax.set_xlim(0, 1)
    ax.set_title("failure-capture coverage by model")
    return Annotations(notes=_coverage_notes(coverages), limitations=_coverage_limits(coverages))


def _coverage_notes(coverages: Sequence[ModelCoverage]) -> tuple[str, ...]:
    """The counts behind the bars, plus the terminal-outcome spread the bars cannot show."""
    total = sum(c.n_trajectories for c in coverages)
    stamped = sum(c.n_stamped for c in coverages)
    if not coverages:
        return ("no models in this corpus",)
    rates = [c.terminal_failure_rate for c in coverages]
    return (
        f"{len(coverages)} models, {total} trajectories, {stamped} stamped ({total - stamped} not)",
        f"terminal failure rate spans {min(rates):.3f} to {max(rates):.3f} across these models — "
        "the outcome labels differ per model even where stamping coverage does not",
    )


def _coverage_limits(coverages: Sequence[ModelCoverage]) -> tuple[str, ...]:
    """Name the models the recurrence trigger is structurally dead on, if any."""
    dead = [c for c in coverages if c.n_stamped == 0]
    if not dead:
        return ()
    lost = sum(c.n_trajectories for c in dead)
    total = sum(c.n_trajectories for c in coverages)
    return (
        f"{', '.join(c.model for c in dead)} ({lost} trajectories, "
        f"{lost / total:.0%} of the corpus) carry NO per-step outcomes at all: the recurrence "
        "trigger cannot fire on them and they are excluded from the risk model.",
    )


__all__ = [
    "CAPTURE_COVERAGE_SPEC",
    "CONFUSION_MATRIX_SPEC",
    "OUTCOME_BARS_SPEC",
    "PERMUTATION_NULL_SPEC",
    "PR_CURVE_SPEC",
    "ROC_CURVE_SPEC",
    "SWEEP_TABLE_SPEC",
    "capture_coverage",
    "confusion_matrix_plot",
    "outcome_bars",
    "permutation_null_plot",
    "pr_curve",
    "roc_curve",
    "sweep_table",
]
