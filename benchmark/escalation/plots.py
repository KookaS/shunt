"""Detector figures. Each function draws into a caller's Axes and returns the Annotations its"""

# own data earned — no file I/O here. The figure's static READ/GOAL/TERMS/NOTE/LIMITS text lives
# next to each function as a frozen FigureSpec; anything data-dependent comes back through
# Annotations so it can never go stale as the corpus grows.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from benchmark.calibration.labeler_metrics import ConfusionMatrix
from benchmark.plot_frame import Annotations, FigureSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.table import Table

    from benchmark.escalation.features import ModelCoverage
    from benchmark.escalation.metrics import NullResult
    from benchmark.escalation.policy_eval import PolicyCell

_NULL_BAND = "#BDBDBD"
_OBSERVED = "#B71C1C"
# The theoretical 0.5 ROC diagonal, drawn faint. It is orientation, not the reference: the
# reference is the MEASURED null centre, which on a refit-per-shuffle pipeline is not 0.5.
_FAINT = "#CCCCCC"
_NULL_CENTRE = "#616161"
# The shipped row's locator shade — bold text plus a light amber fill. It is a POINTER, not a
# colour channel: the sweep table's reading already refuses a colour scale (two distinct results
# cannot carry one honestly), and shading the single known default locates it without encoding
# any result.
_SHIPPED_ROW = "#FFF3CD"

# Diverging fill for the confusion matrix: red where a cell exceeds its random-flagger baseline,
# blue where it falls short, white AT the baseline. Light enough that the black count/expected/
# excess text stays legible on top; intensity is proportional to |excess| (TwoSlopeNorm).
_EXCESS_POS = "#FFCDD2"
_EXCESS_NEG = "#BBDEFB"
_EXCESS_ZERO = "#FFFFFF"
# Floor for the colour scale when every cell sits exactly on its baseline (peak = |excess| max).
_DELTA_FLOOR = 1.0

PR_CURVE_SPEC = FigureSpec(
    reading=(
        "x is recall and y is precision, both 0-1, over the SWEPT escalation policy: one point per "
        "escalate_after_n value that fired. As the recurrence threshold rises, the policy flags "
        "fewer runs with higher precision. The dashed line is the corpus base failure rate — the "
        "no-skill precision a random flagger reaches. The prefix risk model's own PR curve is not "
        "drawn: its score is constant at the evaluated depths (early steps are all "
        "bug-reproduction), so it ranks nothing."
    ),
    goal=(
        "Look for points sitting clearly ABOVE the dashed base-rate line. A point above it with "
        "its interval excluding the base rate is a measured operating point for escalation."
    ),
    definitions=(
        ("precision", "share of flagged runs that really failed"),
        ("recall", "share of failed runs the policy flagged"),
        ("base failure rate", "the corpus's overall failure rate — a random flagger's precision"),
    ),
    notes=(
        "The sweep is ONE family: the point with the highest precision is measured, never adopted "
        "as the shipped default — the shipped configuration is reported separately on the sweep "
        "table.",
        "One point per swept cell, so the curve is a discrete operating characteristic, not a "
        "continuous score sweep.",
    ),
    limitations=(
        "A cell that never fires has no precision at all and is absent rather than plotted at 0.0.",
        "This shows the detector's operating points, NOT what escalating would have CHANGED — no "
        "stored trajectory contains an escalation that actually happened.",
    ),
)

ROC_CURVE_SPEC = FigureSpec(
    reading=(
        "x is the false-positive rate and y the true-positive rate over the SWEPT escalation "
        "policy, one point per escalate_after_n value that fired. The dotted diagonal is chance. "
        "As the recurrence threshold rises the policy moves up the curve: more precision, less "
        "recall. The prefix risk model's ROC is not drawn: its score is constant at the evaluated "
        "depths, so it ranks nothing."
    ),
    goal=(
        "Look for points leaving the diagonal toward the top-left. A point clearly above the "
        "diagonal means firing concentrates on failed runs."
    ),
    definitions=(
        ("true-positive rate", "share of failed runs the policy flagged"),
        ("false-positive rate", "share of resolved runs the policy wrongly flagged"),
        ("escalate_after_n", "the number of same-key verified failures that trigger escalation"),
    ),
    notes=("Auxiliary to the PR figure: precision vs recall is the primary view.",),
    limitations=(
        "This shows the detector's operating points, NOT what escalating would have CHANGED.",
        "A cell that never fires is absent rather than plotted at (0,0).",
    ),
)

CONFUSION_MATRIX_SPEC = FigureSpec(
    reading=(
        "A 2x2 count grid at one swept policy configuration — the cell whose precision is highest "
        "among those that fired. Rows are the truth (run failed / resolved), columns are what the "
        "escalation policy said (flagged / not flagged). Each cell prints the observed count, in "
        "brackets the count a RANDOM flagger at the same flag rate would produce, and the "
        "difference between them. The cell fill is a diverging heatmap on that excess — red where "
        "a cell exceeds its random baseline, blue where it falls short, white AT it. With the "
        "number of flags fixed, all four differences are the SAME number up to sign, so the two "
        "hues restate the one excess rather than carry four independent facts."
    ),
    goal=(
        "Want the top-left count well ABOVE its bracketed random counterpart. That one excess IS "
        "the figure — the other three cells restate it with a sign and add no evidence. At or "
        "below random means the flag carries no information."
    ),
    definitions=(
        ("flagged", "the policy fired on this run (same-key recurrence reached escalate_after_n)"),
        (
            "random baseline",
            "expected count if the same number of flags were scattered at random",
        ),
        ("excess", "observed minus that random baseline, in the cell's own units"),
    ),
    notes=(
        "Counts are per trajectory, not per prefix.",
        "The cell shown is the SWEPT cell with the highest measured precision, not the shipped "
        "default: the shipped default (escalate_after_n=2) fires on every trajectory on this "
        "corpus, so its 'not flagged' column is empty by construction. Read this figure together "
        "with the sweep table, which shows every cell.",
    ),
    limitations=(
        "One operating point, not a sweep — the PR/ROC figures show the rest.",
        "Spending a flag budget equal to the cell's fired count is a choice, not an optimum.",
        "The random baseline is an expectation, not a sampled interval; a cell one or two counts "
        "above it is not evidence.",
    ),
)

SWEEP_TABLE_SPEC = FigureSpec(
    reading=(
        "One row per swept configuration. Columns: escalations fired, P(fail | fired) with its "
        "95% challenge-bootstrap interval, the base failure rate, lift, and the permutation-null "
        "verdict. The table is drawn rather than plotted because the sweep has too few distinct "
        "results to carry a colour channel honestly. The row that IS the shipped default is "
        "highlighted — bold text on a shaded background — so the configuration the product "
        "actually ships can be spotted at a glance."
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
        "escalate_after_n AND stale_window are both swept (the two are coupled: reaching n "
        "recurrences needs a window at least that wide, so the stale=10 rows stop firing at "
        "n>=15). The shipped default (escalate_after_n=2, stale_window=10) is guaranteed a row "
        "and is the highlighted one.",
        "The interval is a CHALLENGE-level bootstrap, not a Wilson interval over rows: the corpus "
        "is 799 runs drawn from ~160 challenges, so rows are not independent draws and a row-level "
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
        "The grey histogram is the maximum AUROC the swept policy reaches under BLOCK "
        "permutation — whole challenge blocks are shuffled, so outcomes move between challenges "
        "while the global multiset is preserved — with one shared shuffle scored at every swept "
        "cell and only the largest kept — the FAMILY-WISE (maxT) null across the whole sweep. The "
        "dashed lines bound that max distribution's central 95%. The red line is the real, "
        "unshuffled AUROC at the cell that separates best. x is the statistic named on the axis."
    ),
    goal=(
        "The red line must sit clearly to the RIGHT of the upper dashed line. Inside the dashed "
        "bounds means the sweep's best cell is indistinguishable from noise."
    ),
    definitions=(
        ("null", "what this policy scores when the labels carry no information"),
        (
            "family-wise",
            "the max-statistic (maxT) null over the swept cells, so the p-value beside it is "
            "already the multiplicity-adjusted one — exact under any dependence between cells",
        ),
        (
            "AUROC",
            "rank statistic of the fired/not-fired flag against the run outcome, tie-averaged",
        ),
    ),
    notes=(
        "The null is the gate: a point estimate above chance is not skill on its own.",
        "Permuting whole challenge blocks preserves the between-challenge variance the "
        "observation carries; a global shuffle would destroy the clustering and make the band too "
        "narrow. One shared randomization serves every swept cell, and only the max enters the "
        "histogram — exactly the distribution the verdict is read off.",
    ),
    limitations=(
        "Clearing this null is necessary, not sufficient: the precision interval must also clear "
        "the base rate (see the confusion figure), or the AUROC reflects a rank ordering with no "
        "operating value.",
        "A cell that never fires has AUROC 0.5 by definition and contributes nothing to the max.",
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


def _draw_confusion_cells(
    counts: tuple[tuple[int, int], tuple[int, int]],
    expected: tuple[tuple[float, float], tuple[float, float]],
    ax: Axes,
) -> None:
    """A 2x2 heatmap whose fill is the cell's excess over its random-flagger baseline."""
    # The four cells carry ONE number: with the flag count fixed every excess is the same magnitude
    # up to sign (tn - E[tn] == tp - E[tp], and both off-diagonals are its negation), so the fill
    # is a diverging scale on that single excess — red above the baseline, blue below, white at it —
    # NOT a shade per cell. The previous version shaded by RAW count, which put the heaviest ink on
    # the cell furthest BELOW its baseline: the eye read "biggest = best" off a deficit, and a
    # four-valued scale was encoding one degree of freedom as four. Same reasoning as `sweep_table`,
    # which refuses a colour scale because two distinct results cannot carry one honestly; here the
    # single degree of freedom is what the colour encodes, and the footer says so.
    deltas = np.array([[counts[r][c] - expected[r][c] for c in range(2)] for r in range(2)])
    peak = max(float(np.abs(deltas).max()), _DELTA_FLOOR)
    cmap = LinearSegmentedColormap.from_list("excess", (_EXCESS_NEG, _EXCESS_ZERO, _EXCESS_POS))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-peak, vmax=peak)
    for r in range(2):
        for c in range(2):
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(deltas[r, c])),
                    edgecolor=_NULL_CENTRE,
                    linewidth=0.8,
                )
            )
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
    _attach_excess_colorbar(ax, cmap, norm, peak)


def _attach_excess_colorbar(ax: Axes, cmap, norm, peak: float) -> None:
    """The scale the fills live on; the footer's notes explain what it encodes."""
    # The grid uses only the two extremes of the scale, so the colorbar ticks at them rather than
    # at intermediate values the data never reaches.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.array([-peak, peak]))
    ax.figure.colorbar(
        mappable, ax=ax, label="excess over random flagging", ticks=[-peak, peak], format="%+.1f"
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


def sweep_table(
    cells: Sequence[PolicyCell], ax: Axes, *, shipped_index: int | None = None
) -> Annotations:
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
    title = "policy sweep — escalate_after_n × stale_window (ladder pinned)"
    if shipped_index is not None and 0 <= shipped_index < len(rows):
        _mark_shipped_row(table, shipped_index)
        title += " — shipped default is the highlighted row"
    ax.set_title(title)
    return Annotations(notes=(_sweep_note(cells),), limitations=_sweep_limits(cells))


def _mark_shipped_row(table: Table, index: int) -> None:
    """Bold the shipped row's text and shade its cells, so the default is spot-table."""
    # `get_celld()` keys are (row, col); the header occupies row 0, so data row `index` is cell
    # row `index + 1`. The shipped index is passed in (run_eval owns what "shipped" means) rather
    # than matched by value here, so plots.py never imports the eval's constants.
    for (row, _col), cell in table.get_celld().items():
        if row == index + 1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor(_SHIPPED_ROW)


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


def _annotate_sweep_points(
    ax: Axes,
    cells: Sequence[PolicyCell],
    xy_of: Callable[[PolicyCell], tuple[float, float | None]],
) -> None:
    """Label each fired operating point, deduplicating co-located cells and staggering overlaps."""
    # The two stale_window rows of one escalate_after_n can land on the SAME plotted point (the
    # window only gates how many events the trigger admits, not whether it fired), and the old
    # code annotated EVERY cell — two identical `n=2` labels printed on top of each other, then
    # the next threshold's label on top of both. Coincident points are annotated ONCE (their n
    # values joined); points that merely land close together on the curve are pushed apart
    # vertically so no two labels sit on the same spot.
    grouped: dict[tuple[float, float], list[PolicyCell]] = {}
    for c in cells:
        if c.precision is None:
            continue
        x, y = xy_of(c)
        if y is None:
            continue
        grouped.setdefault((round(x, 6), round(y, 6)), []).append(c)
    placed: list[tuple[float, float]] = []
    for (x, y), members in grouped.items():
        label = "n=" + ",".join(f"{c.escalate_after_n}" for c in members)
        # Drop the label until its anchor clears every earlier one: consecutive thresholds sit
        # within ~0.03-0.05 of each other on the curve, so a flat (6, 4) offset stacks every
        # label on its neighbour. `step` is roughly one text line in data units (fontsize 8).
        nudge = 0
        step, radius = 0.035, 0.022
        while any(
            abs(px - x) < 0.10 and abs(py - (y - step * nudge)) < radius for px, py in placed
        ):
            nudge += 1
        placed.append((x, y - step * nudge))
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 4 - 14 * nudge),
            fontsize=8,
        )


def policy_pr_curve(cells: Sequence[PolicyCell], ax: Axes) -> Annotations:
    """Precision vs recall across SWEPT thresholds — the escalation method's curve."""
    # The old PR figure was drawn from the prefix risk model, whose score is constant at the
    # evaluated depths (early steps are all bug-reproduction, so the model ranks nothing) — a
    # degenerate 1-2-point curve. The escalation METHOD is the policy, and its operating
    # characteristic is real: as escalate_after_n rises, precision climbs from the base rate to
    # 0.71 while recall falls. One point per swept cell; the base rate is the no-skill line.
    points = [(c.recall, c.precision) for c in cells if c.precision is not None]
    scored = [c for c in cells if c.precision is not None]
    if not points:
        return Annotations(
            limitations=(
                "NO CONFIGURATION FIRED: the sweep contains no measurable operating point.",
            )
        )
    base = next((c.base_failure_rate for c in cells if c.precision is not None), 0.0)
    ax.axhline(
        base,
        linestyle="--",
        color=_NULL_CENTRE,
        label=f"base failure rate={base:.3f}",
    )
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, marker="o", label="policy sweep (escalate_after_n rising)")
    _annotate_sweep_points(ax, cells, lambda c: (c.recall, c.precision))
    ax.set_xlabel("recall (share of failed runs flagged)")
    ax.set_ylabel("precision (P(fail | flagged))")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("escalation policy operating characteristic — precision vs recall")
    ax.legend()
    best = max(scored, key=lambda c: c.precision or 0.0)
    return Annotations(
        notes=(
            f"{len(scored)} of {len(cells)} swept cells fired; one point per escalate_after_n",
            f"highest precision {best.precision:.3f} at n={best.escalate_after_n} "
            f"(recall {best.recall:.3f}) against a base rate of {best.base_failure_rate:.3f}",
        ),
        limitations=(
            ()
            if best.precision_ci[0] > best.base_failure_rate
            else (
                "NO OPERATING POINT CLEARS THE BASE RATE: no swept configuration's precision "
                "interval excludes the base failure rate.",
            )
        ),
    )


def policy_roc_curve(cells: Sequence[PolicyCell], ax: Axes) -> Annotations:
    """TPR vs FPR across the SWEPT recurrence thresholds — the escalation method's ROC."""
    # Same re-pointing as `policy_pr_curve`: the method's curve, not the degenerate prefix score's.
    points = [
        (
            c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0,
            c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0,
        )
        for c in cells
        if c.precision is not None
    ]
    if not points:
        return Annotations(
            limitations=(
                "NO CONFIGURATION FIRED: the sweep contains no measurable operating point.",
            )
        )
    ax.plot([0, 1], [0, 1], linestyle=":", color=_FAINT, linewidth=0.8, label="chance")
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, marker="o", label="policy sweep (escalate_after_n rising)")
    rates = {
        c: (
            c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0,
            c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0,
        )
        for c in cells
        if c.precision is not None
    }
    _annotate_sweep_points(ax, cells, lambda c: rates[c])
    ax.set_xlabel("false-positive rate (share of resolved runs flagged)")
    ax.set_ylabel("true-positive rate (share of failed runs flagged)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("escalation policy ROC across recurrence thresholds")
    ax.legend()
    return Annotations(notes=(f"{len(points)} swept cells fired; one point per escalate_after_n",))


def policy_confusion(cell: PolicyCell, ax: Axes, *, flag_budget: int | None = None) -> Annotations:
    """The 2x2 of the SWEPT CELL that separates best — the escalation method's operating point."""
    # The old confusion figure came from the prefix score, which flags every row when the score is
    # constant, so the "not flagged" column was empty (0/0) — exactly the "no datapoints being
    # kept" the audit flagged. The policy cell at high n has all four cells populated.
    cm = ConfusionMatrix(tp=cell.tp, fp=cell.fp, fn=cell.fn, tn=cell.tn)
    flagged = cell.tp + cell.fp
    budget = flag_budget if flag_budget is not None else flagged
    counts = ((cm.tp, cm.fn), (cm.fp, cm.tn))
    positives = cm.tp + cm.fn
    negatives = cm.fp + cm.tn
    total = positives + negatives
    rate = flagged / total if total else 0.0
    expected = (
        (positives * rate, positives * (1 - rate)),
        (negatives * rate, negatives * (1 - rate)),
    )
    _draw_confusion_cells(counts, expected, ax)
    ax.set_title(
        f"escalation policy confusion @ escalate_after_n={cell.escalate_after_n}, "
        f"stale_window={cell.stale_window} "
        "(count, [random at the same flag rate], excess)"
    )
    return Annotations(
        notes=_confusion_notes(cm, expected[0][0], total=total, threshold=0.0, rate=rate),
        limitations=(
            *_budget_limits(flagged, budget, total),
            *(
                ()
                if cm.tp > expected[0][0]
                else (
                    "The detector catches no more failures than random flagging at the same rate.",
                )
            ),
        ),
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
    "outcome_bars",
    "permutation_null_plot",
    "policy_confusion",
    "policy_pr_curve",
    "policy_roc_curve",
    "sweep_table",
]
