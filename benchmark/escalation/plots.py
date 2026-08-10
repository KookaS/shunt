"""Six escalation figures. Each draws into a caller's Axes and returns the Annotations its data"""

# earned — no file I/O here. Every per-cell figure is pinned to the SHIPPED knobs, and the only
# thing that varies across the set is how the recurrence counter treats the reproduction phase.
# Ten figures used to show three different cells — the shipped default, an argmax over 60 swept
# cells, and a continuous score at a third window — so no two of them described the same policy.
# The canonical cell (`run_eval.canonical_cell`, edit-gated at those knobs) drives the null panel
# and figures 5 and 6; figure 2 draws BOTH counting modes at those same knobs, because "the
# configuration we actually ship reads the base rate" is the escalation half's negative finding
# and it has to be on a canvas, not one row of a 30-row table.
#
# The static FigureSpec beside each function carries the reading/goal/terms text that used to be
# rendered as a footer; the canvas gets a title, a subtitle and at most one red caveat, and the
# rest reaches the reader through figures.json and docs/escalation.md.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from benchmark.escalation import metrics, policy_eval
from benchmark.plot_frame import MUTED, Annotations, FigureSpec, panel_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.table import Table

    from benchmark.escalation.features import ModelCoverage
    from benchmark.escalation.metrics import NullResult
    from benchmark.escalation.policy_eval import PolicyCell
    from benchmark.escalation.session_eval import SessionCadenceReport

_NULL_BAND = "#BDBDBD"
_OBSERVED = "#B71C1C"
_SHIPPED = "#455A64"
_FAINT = "#CCCCCC"
_NULL_CENTRE = "#616161"
_LENGTH_NULL = "#1565C0"
# The shipped row's locator shade — a POINTER, not a colour channel: the sweep table's reading
# refuses a colour scale (two distinct results cannot carry one honestly), and shading the single
# known default locates it without encoding any result.
_SHIPPED_ROW = "#FFF3CD"
_GATED_COL = "#EEF4FA"
_UNDEFINED = "#9E9E9E"

_TICK_PT = 8.5
_LEGEND_PT = 8.0


# --------------------------------------------------------------------- the scope strip


_SCOPE_BOXES: tuple[tuple[str, str], ...] = (
    ("DETECTS (per-step, eval-only)", "OK"),
    ("VALUE (session cadence, observational)", "OK"),
    ("CAUSAL VALUE AT THE TRIGGER", "not identified: P(escalate)=0"),
)


def scope_strip(ax: Axes) -> None:
    """Three boxes saying which of the three escalation claims this figure can support."""
    # The third box is the one that matters. No logged trajectory ever escalated, so the
    # propensity of the escalate action is 0 everywhere and `ope.estimate_policy_value` returns
    # `not_identified` — correctly. Drawing an estimated policy value anyway would be fabrication,
    # so the absence is rendered as a labelled box instead of quietly left off the figure.
    ax.axis("off")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    for index, (claim, verdict) in enumerate(_SCOPE_BOXES):
        supported = verdict == "OK"
        ax.add_patch(
            Rectangle(
                (index + 0.01, 0.08),
                0.98,
                0.84,
                facecolor="#F1F8E9" if supported else "#FFEBEE",
                edgecolor=_NULL_CENTRE if supported else _OBSERVED,
                linewidth=0.8 if supported else 1.6,
            )
        )
        ax.text(
            index + 0.5,
            0.5,
            f"{claim} — {verdict}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#1a1a1a" if supported else _OBSERVED,
            fontweight="bold" if not supported else "normal",
        )


# ------------------------------------------------------------- 1. escalation_decision


ESCALATION_DECISION_SPEC = FigureSpec(
    title="Counting the reproduction phase is what decides the answer: AUROC 0.601 vs 0.781",
    reading=(
        "Left: the COMPLETE ROC of the recurrence score as a continuous statistic — each run is "
        "scored by the largest number of times one failing-check id recurred inside the shipped "
        "stale_window, and the curve sweeps every threshold, so it has a point at every possible "
        "escalate_after_n rather than only the swept grid. Two curves: as-shipped (every same-key "
        "failure counted) and edit-gated (failures before the agent's first edit-like action are "
        "not counted). The grey band is the score's own challenge-block permutation null. Middle: "
        "P(run failed | score >= t) against t for both families, with the corpus base rate as the "
        "no-skill line. Right: what share of the corpus each threshold fires on. The shipped "
        "escalate_after_n is marked on both right-hand panels."
    ),
    goal=(
        "The two curves must differ. If they do, the reproduction phase — not the recurrence "
        "mechanism — is what the as-shipped counter is measuring, and the gap between them is its "
        "size. Read the middle panel at the shipped threshold: the edit-gated precision there is "
        "the operating point every other figure in this set uses."
    ),
    definitions=(
        ("recurrence score", "max same-key verified-failure count a run reaches in the window"),
        ("edit-gated", "failures before the agent's first edit-like action are not counted"),
        ("null band", "central 95% of ROC curves under challenge-block label shuffles"),
    ),
    notes=(
        "stale_window is held FIXED at the shipped value for BOTH curves. It is a knob in its own "
        "right — the as-shipped score reaches 0.728 at stale_window=1000 — so letting it vary "
        "between the two curves would have credited the counting change with a window change.",
        "The AUROC of the score bounds what ANY single escalate_after_n can reach.",
    ),
    limitations=(
        "Per-step cadence and eval-only: the live router has no per-step action stream to gate on, "
        "so the edit-gated family measures what a per-step detector could do, not what ships.",
        "Association only — no stored trajectory contains an escalation that actually happened.",
    ),
)


def escalation_decision(
    scores_plain: Sequence[float],
    scores_edit: Sequence[float],
    labels: Sequence[bool],
    axes: Sequence[Axes],
    *,
    null: NullResult | None = None,
    band: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None,
    shipped_n: int,
) -> Annotations:
    """ROC + null band, precision-vs-threshold and fire rate — the detection claim, in one."""
    ax_roc, ax_precision, ax_rate, ax_scope = axes
    _draw_roc(ax_roc, scores_plain, scores_edit, labels, band=band)
    base = metrics.prevalence(labels)
    _draw_threshold_panels(ax_precision, ax_rate, scores_plain, scores_edit, labels, shipped_n)
    ax_precision.axhline(
        base, linestyle="--", color=_NULL_CENTRE, linewidth=1.0, label=f"base rate {base:.3f}"
    )
    ax_precision.legend(fontsize=_LEGEND_PT, loc="lower right")
    scope_strip(ax_scope)
    auroc_plain = metrics.auroc(scores_plain, labels)
    auroc_edit = metrics.auroc(scores_edit, labels)
    return Annotations(
        subtitle_facts=(
            f"base rate {base:.3f}",
            f"AUROC as-shipped {auroc_plain:.3f} · edit-gated {auroc_edit:.3f}",
        ),
        notes=(
            *(
                (
                    f"score null 95% [{null.ci_low:.3f}, {null.ci_high:.3f}], "
                    f"p={null.p_value:.4f} over {null.n_permutations} challenge-block shuffles",
                )
                if null is not None
                else ()
            ),
        ),
        counts=(("stamped_runs", len(labels)),),
    )


def _draw_roc(
    ax: Axes,
    scores_plain: Sequence[float],
    scores_edit: Sequence[float],
    labels: Sequence[bool],
    *,
    band: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None,
) -> None:
    ax.plot([0, 1], [0, 1], linestyle=":", color=_FAINT, linewidth=0.8, label="chance")
    if band is not None:
        grid, low, high = band
        ax.fill_between(grid, low, high, color=_NULL_BAND, alpha=0.45, label="null 95% band")
    for scores, name, colour, style in (
        (scores_plain, "as-shipped", _SHIPPED, "--"),
        (scores_edit, "edit-gated", _OBSERVED, "-"),
    ):
        points = metrics.roc_operating_points(scores, list(labels))
        xs, ys = zip(*points, strict=True)
        ax.plot(
            xs,
            ys,
            label=f"{name} ({metrics.auroc(scores, labels):.3f})",
            color=colour,
            linestyle=style,
            linewidth=1.8,
        )
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="lower right")
    panel_label(ax, "A · complete ROC of the recurrence score")


def _threshold_curve(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[list[float], list[float], list[float]]:
    """(threshold, P(fail | score >= t), share firing) at every distinct positive threshold."""
    thresholds = sorted({s for s in scores if s > 0})
    xs: list[float] = []
    precision: list[float] = []
    rate: list[float] = []
    for t in thresholds:
        flagged = [labels[i] for i, s in enumerate(scores) if s >= t]
        if not flagged:
            continue
        xs.append(t)
        precision.append(sum(flagged) / len(flagged))
        rate.append(len(flagged) / len(scores))
    return xs, precision, rate


def _draw_threshold_panels(
    ax_precision: Axes,
    ax_rate: Axes,
    scores_plain: Sequence[float],
    scores_edit: Sequence[float],
    labels: Sequence[bool],
    shipped_n: int,
) -> None:
    for scores, name, colour, style in (
        (scores_plain, "as-shipped", _SHIPPED, "--"),
        (scores_edit, "edit-gated", _OBSERVED, "-"),
    ):
        xs, precision, rate = _threshold_curve(scores, labels)
        if not xs:
            continue
        ax_precision.plot(xs, precision, color=colour, linestyle=style, linewidth=1.8, label=name)
        ax_rate.plot(xs, rate, color=colour, linestyle=style, linewidth=1.8, label=name)
    for ax in (ax_precision, ax_rate):
        ax.axvline(shipped_n, color="#F9A825", linewidth=1.4, label=f"shipped n={shipped_n}")
        # Linear, not log: the score is an integer recurrence count with a single-digit useful
        # range, and a log axis on 1-10 labels two decades and nothing between them.
        ax.set_xlabel("escalate_after_n threshold")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=_TICK_PT)
    ax_precision.set_ylabel("P(run failed | fired)")
    ax_rate.set_ylabel("share of corpus fired on")
    ax_rate.legend(fontsize=_LEGEND_PT, loc="lower left")
    panel_label(ax_precision, "B · precision at every threshold")
    panel_label(ax_rate, "C · fire rate at every threshold")


# ----------------------------------------------------------------- 2. operating_point


OPERATING_POINT_SPEC = FigureSpec(
    title="The shipped counter sits at the base rate; edit-gated counting separates outcomes",
    reading=(
        "Left: at the SAME shipped knobs, the share of runs that ultimately failed among those "
        "the policy escalated and among those it left alone — for BOTH counting modes. The left "
        "pair is the configuration the product actually ships, which fires on essentially every "
        "run: its escalated bar sits on the dashed base rate and its not-escalated arm holds so "
        "few runs that no rate can be read off it, so it is drawn as a hatched 'undefined' box "
        "rather than as a measured 0.000. The right pair is the same rule with the reproduction "
        "phase excluded. Intervals are the central 95% of the same challenge-bootstrap resamples, "
        "so the two arms of a pair are paired draw-for-draw. Right: the CANONICAL (edit-gated) "
        "cell's AUROC against TWO nulls — the family-wise max-over-cells challenge-block null "
        "(grey), which asks whether any cell in the sweep could reach this by chance, and the "
        "length-stratified null (blue), which shuffles failures inside equal-count run-length "
        "bins and so asks whether firing predicts failure BEYOND what the lengths of the fired "
        "runs already predict."
    ),
    goal=(
        "The left pair IS the negative result and it belongs on a canvas, not in a table row: a "
        "shipped configuration whose escalated bar sits on the base rate is a null detector. Then "
        "want the right pair's escalated bar clearly above both the dashed line and its own quiet "
        "bar, and the red observed line to the right of BOTH null distributions. Clearing the "
        "grey null alone is not enough: the challenge-block shuffle destroys the run-length "
        "association along with everything else, so a cell whose firing is really length "
        "selection can clear it and still sit inside the blue one."
    ),
    definitions=(
        ("as-shipped", "every same-key verified failure counts, which is what production runs"),
        ("canonical cell", "edit-gated counting at the shipped escalate_after_n/stale_window"),
        ("family-wise null", "max AUROC over the swept cells under one shared block shuffle"),
        ("length-stratified null", "failures shuffled within equal-count run-length bins"),
    ),
    notes=(
        "Both pairs are at the SAME knobs, so the only thing that differs between them is how the "
        "counter treats the reproduction phase.",
        "The intervals are a CHALLENGE-level bootstrap: the corpus is drawn from ~166 challenges, "
        "each attempted by several model/effort arms, so a row-level interval is roughly 2x too "
        "narrow.",
    ),
    limitations=(
        "Association, not causation — see the scope strip: no logged trajectory escalated.",
        "Two operating points; the sweep figure shows every other configuration.",
    ),
)


def operating_point(
    as_shipped: PolicyCell | None, canonical: PolicyCell, axes: Sequence[Axes]
) -> Annotations:
    """Both counting modes' outcome arms at the shipped knobs, and the canonical cell's nulls."""
    # The as-shipped pair is drawn even though — BECAUSE — it is the null result. Collapsing the
    # whole figure set onto the canonical cell would demote "the configuration we actually ship
    # fires on 727 of 727 runs and reads the base rate" to one row of a 30-row table, which is
    # the exact failure this redesign exists to remove.
    ax_bars, ax_null, ax_scope = axes
    _draw_arms(ax_bars, as_shipped, canonical)
    _draw_nulls(ax_null, canonical)
    scope_strip(ax_scope)
    return Annotations(
        subtitle_facts=(
            f"both at escalate_after_n={canonical.escalate_after_n}, "
            f"stale_window={canonical.stale_window} · base rate "
            f"{canonical.base_failure_rate:.3f}",
            *_family_facts(as_shipped, canonical),
        ),
        caveat=_arm_floor_caveat(as_shipped, canonical),
        notes=tuple(
            _cell_note(name, cell)
            for name, cell in (("as-shipped", as_shipped), ("edit-gated", canonical))
            if cell is not None
        ),
        counts=(
            ("fired", canonical.tp + canonical.fp),
            ("quiet", canonical.fn + canonical.tn),
        ),
        limitations=_direction_limits(canonical),
    )


def _family_facts(as_shipped: PolicyCell | None, canonical: PolicyCell) -> tuple[str, ...]:
    facts = []
    if as_shipped is not None:
        facts.append(
            f"as-shipped fires {as_shipped.n_escalated}/{as_shipped.n_trajectories} at "
            f"P(fail|fired)={_rate(as_shipped.precision)}"
        )
    facts.append(
        f"edit-gated fires {canonical.n_escalated}/{canonical.n_trajectories} at "
        f"{_rate(canonical.precision)} vs {_rate(canonical.p_fail_given_quiet)} quiet"
    )
    return tuple(facts)


def _arm_counts(cell: PolicyCell) -> tuple[int, int]:
    return cell.tp + cell.fp, cell.fn + cell.tn


def _arm_floor_caveat(as_shipped: PolicyCell | None, canonical: PolicyCell) -> str | None:
    """Name an arm too small to read a rate off, in the one line the canvas allows."""
    thin: list[str] = []
    for family, cell in (("as-shipped", as_shipped), ("edit-gated", canonical)):
        if cell is None:
            continue
        fired, quiet = _arm_counts(cell)
        thin += [
            f"{family} {name} arm n={n}"
            for name, n in (("escalated", fired), ("not-escalated", quiet))
            if n < policy_eval.MIN_ARM
        ]
    if not thin:
        return None
    return f"{' and '.join(thin)}: below the n={policy_eval.MIN_ARM} floor, drawn as undefined"


# Two pairs with a gap between them, so the eye groups each family's arms before comparing
# families. The gap is not decoration: adjacent bars read as comparable, and the comparison that
# matters is fired-vs-quiet WITHIN a family, then family against family.
_ARM_POSITIONS: tuple[float, ...] = (0.0, 1.0, 2.4, 3.4)
_ARM_LABELS: tuple[str, ...] = (
    "escalated\nas-shipped",
    "not escalated\nas-shipped",
    "escalated\nedit-gated",
    "not escalated\nedit-gated",
)
_ARM_COLOURS: tuple[str, ...] = ("#78909C", "#B0BEC5", _OBSERVED, "#EF9A9A")


def _draw_arms(ax: Axes, as_shipped: PolicyCell | None, canonical: PolicyCell) -> None:
    arms: list[tuple[float | None, tuple[float, float], int] | None] = []
    for cell in (as_shipped, canonical):
        if cell is None:
            arms += [None, None]
            continue
        fired_n, quiet_n = _arm_counts(cell)
        arms.append((cell.precision, cell.precision_ci, fired_n))
        arms.append((cell.p_fail_given_quiet, cell.quiet_ci, quiet_n))
    for position, arm in zip(_ARM_POSITIONS, arms, strict=True):
        if arm is None:
            continue
        value, interval, n = arm
        if n < policy_eval.MIN_ARM or value is None or np.isnan(interval[0]):
            _draw_undefined_arm(ax, position, n)
            continue
        ax.bar(
            [position],
            [value],
            width=0.7,
            color=_ARM_COLOURS[_ARM_POSITIONS.index(position)],
            yerr=[[max(0.0, value - interval[0])], [max(0.0, interval[1] - value)]],
            capsize=6,
        )
        ax.text(position, 0.03, f"n={n}", ha="center", color="white", fontsize=9)
        # Above the interval's UPPER arm, not above the bar: a label placed at value + a constant
        # lands inside the error bar whenever the interval is wider than that constant.
        ax.text(
            position,
            min(0.95, max(value, interval[1]) + 0.025),
            f"{value:.3f}",
            ha="center",
            fontsize=9.5,
        )
    ax.set_xticks(list(_ARM_POSITIONS), list(_ARM_LABELS))
    ax.axhline(
        canonical.base_failure_rate,
        linestyle="--",
        color=_NULL_CENTRE,
        label=f"base rate {canonical.base_failure_rate:.3f}",
    )
    ax.set_ylabel("P(run ultimately failed)")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 4.0)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="upper right")
    panel_label(ax, "A · outcome by escalation, both counting modes, 95% challenge bootstrap")


def _draw_undefined_arm(ax: Axes, position: float, n: int) -> None:
    """An arm below the reporting floor: a hatched box over the whole range, never a bar."""
    # A one-row arm rendered 0/1 = 0.000 as a full-looking measurement on the committed figure.
    # Height 1.0 spans every value the rate could take, which is exactly what "undefined" means
    # here, and the hatch plus the label make it unreadable as a point estimate.
    ax.add_patch(
        Rectangle(
            (position - 0.275, 0.0),
            0.55,
            1.0,
            facecolor="none",
            edgecolor=_UNDEFINED,
            hatch="///",
            linewidth=1.0,
        )
    )
    ax.text(
        position,
        0.5,
        f"undefined\n(n={n})",
        ha="center",
        va="center",
        fontsize=9,
        color=_UNDEFINED,
        fontweight="bold",
        # The label sits ON the hatch, which is the point — but hatch strokes through 9pt text
        # make it unreadable, so it gets its own opaque backing.
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
    )


def _draw_nulls(ax: Axes, cell: PolicyCell) -> None:
    ax.hist(
        list(cell.gate_null.draws),
        bins=40,
        color=_NULL_BAND,
        label=f"family-wise null ({cell.gate_null.n_permutations} perm)",
    )
    if cell.null_auroc_length is not None:
        ax.hist(
            list(cell.null_auroc_length.draws),
            bins=40,
            histtype="step",
            color=_LENGTH_NULL,
            linewidth=1.4,
            label="length-stratified null",
        )
    ax.axvline(
        cell.null_auroc.observed,
        color=_OBSERVED,
        linewidth=2,
        label=f"observed AUROC {cell.null_auroc.observed:.3f}",
    )
    if cell.length_baseline_auroc is not None:
        ax.axvline(
            cell.length_baseline_auroc,
            color=_LENGTH_NULL,
            linestyle=":",
            linewidth=1.4,
            label=f"run-length-only {cell.length_baseline_auroc:.3f}",
        )
    ax.set_xlabel("AUROC of the fired flag against the terminal outcome")
    ax.set_ylabel("permutations")
    # Room to the right of the observed line, so it reads as a line rather than as the frame.
    left, right = ax.get_xlim()
    ax.set_xlim(left, max(right, cell.null_auroc.observed) + 0.02)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="center right")
    panel_label(ax, "B · observed against both nulls")


def _cell_note(family: str, cell: PolicyCell) -> str:
    length = "n/a" if cell.length_baseline_auroc is None else f"{cell.length_baseline_auroc:.3f}"
    return (
        f"{family} at escalate_after_n={cell.escalate_after_n}, "
        f"stale_window={cell.stale_window}; fired on "
        f"{cell.n_escalated}/{cell.n_trajectories}; P(fail|fired)={_rate(cell.precision)} "
        f"[{_rate(cell.precision_ci[0])}, {_rate(cell.precision_ci[1])}] vs quiet "
        f"{_rate(cell.p_fail_given_quiet)}; AUROC {cell.null_auroc.observed:.3f} against a "
        f"run-length-only {length}"
    )


def _direction_limits(cell: PolicyCell) -> tuple[str, ...]:
    """Call the direction ONLY when both readable intervals actually separate."""
    fired_n, quiet_n = _arm_counts(cell)
    if fired_n < policy_eval.MIN_ARM or quiet_n < policy_eval.MIN_ARM:
        return (
            f"One arm holds fewer than {policy_eval.MIN_ARM} runs, so no direction can be read "
            "off this cell at all.",
        )
    fired_lo, fired_hi = cell.precision_ci
    quiet_lo, quiet_hi = cell.quiet_ci
    if np.isnan(fired_lo) or np.isnan(quiet_lo):
        return ("An arm has no estimable interval on this corpus.",)
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


# -------------------------------------------------------------------- 3. policy_sweep


POLICY_SWEEP_SPEC = FigureSpec(
    title="Every swept configuration, in both counting modes, against one base rate",
    reading=(
        "One row per configuration of the two coupled knobs — escalate_after_n and stale_window, "
        "with the escalation ladder pinned to the shipped one. The A columns are as-shipped "
        "counting (every same-key verified failure counts); the shaded B columns are the same "
        "configuration with the reproduction phase excluded, i.e. failures before the agent's "
        "first edit-like action are not escalation evidence. Per family: how many trajectories the "
        "cell fired on, P(run failed | fired), and the AUROC of the fired flag against the "
        "terminal outcome. The row that IS the shipped default is highlighted."
    ),
    goal=(
        "Compare the A and B columns row by row. A configuration whose P(fail|fired) sits at the "
        "base rate has no measured value at all; the gap between the A and B columns of the SAME "
        "row is the reproduction phase's contribution, isolated from every other knob. The two "
        "knobs are coupled — reaching n recurrences needs a window at least that wide — which is "
        "why the stale_window=10 rows stop firing above n=10."
    ),
    definitions=(
        ("A", "as-shipped counting"),
        ("B", "edit-gated counting (failures before the first edit excluded)"),
        ("P(fail)", "share of the runs this cell fired on that ultimately failed"),
        (
            "len-only",
            "the AUROC a pure 'run length >= t' predictor reaches at THIS cell's flag count — the "
            "ceiling run length alone can explain, so an AUROC no higher than it is length "
            "selection rather than recurrence",
        ),
    ),
    notes=(
        "The table is drawn rather than plotted because the sweep has too few distinct results to "
        "carry a colour channel honestly.",
        "The interval is the CHALLENGE-level bootstrap, not a Wilson interval over rows: the "
        "corpus is drawn from ~166 challenges, so rows are not independent draws and a row-level "
        "interval is roughly 2x too narrow.",
    ),
    limitations=(
        "Every number here is unadjusted for the 60 configurations compared side by side; the "
        "family-wise correction lives in each cell's null, not in this table.",
        "A configuration that never fires has no P(fail|fired) at all and prints n/a.",
    ),
)


def policy_sweep(
    as_shipped: Sequence[PolicyCell],
    edit_gated: Sequence[PolicyCell],
    ax: Axes,
    *,
    shipped_index: int | None = None,
) -> Annotations:
    """Both families as one table — no Table.scale(), the bbox pins it to the axes rect."""
    ax.axis("off")
    # NEITHER `95% CI` NOR `len-only` is droppable to make the two families fit. The interval is
    # what the whole challenge-level bootstrap exists to produce, and a precision quoted bare
    # invites exactly the over-reading this set is built to prevent.
    #
    # `len-only` is this table's built-in
    # run-length confound control: an AUROC no higher than the length-only figure at the same flag
    # count means the "recurrence" signal is run-length selection, and nothing else on the canvas
    # stands between those two readings. It is per family because the baseline is evaluated at
    # THAT family's flag count, which differs — one shared column would be wrong for one of them.
    header = [
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
    gated = {(c.escalate_after_n, c.stale_window): c for c in edit_gated}
    rows = [_sweep_row(c, gated.get((c.escalate_after_n, c.stale_window))) for c in as_shipped]
    table = ax.table(
        cellText=rows, colLabels=header, cellLoc="center", bbox=Bbox.from_bounds(0, 0, 1, 1)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # NO `table.scale()`. Scaling is what pushed the table past its own axes rect, which
    # matplotlib does not clip — the title was then drawn straight through the top rows. The bbox
    # above pins the table to the rect the frame reserved, and `plot_frame.table_size` gives the
    # rect enough height for the row count.
    _shade_columns(table, first=7, last=11, n_rows=len(rows))
    if shipped_index is not None and 0 <= shipped_index < len(rows):
        _mark_shipped_row(table, shipped_index)
    return Annotations(
        subtitle_facts=(
            f"{len(rows)} configurations x 2 counting modes",
            "A = as-shipped · B = edit-gated · shipped default highlighted",
            "len-only = the AUROC run length alone reaches at that cell's flag count",
        ),
        notes=(_sweep_note(as_shipped, edit_gated),),
        limitations=_sweep_limits(as_shipped, edit_gated),
        counts=(("configurations", len(rows)),),
    )


def _sweep_row(shipped: PolicyCell, gated: PolicyCell | None) -> list[str]:
    row = [
        str(shipped.escalate_after_n),
        str(shipped.stale_window),
        str(shipped.n_escalated),
        _rate(shipped.precision),
        _ci(shipped),
        f"{shipped.null_auroc.observed:.3f}",
        _rate(shipped.length_baseline_auroc),
    ]
    if gated is None:
        return [*row, "n/a", "n/a", "n/a", "n/a", "n/a"]
    return [
        *row,
        str(gated.n_escalated),
        _rate(gated.precision),
        _ci(gated),
        f"{gated.null_auroc.observed:.3f}",
        _rate(gated.length_baseline_auroc),
    ]


def _ci(cell: PolicyCell) -> str:
    """The precision's challenge-bootstrap interval. A point estimate without one over-reads."""
    return f"[{_rate(cell.precision_ci[0])}, {_rate(cell.precision_ci[1])}]"


def _shade_columns(table: Table, *, first: int, last: int, n_rows: int) -> None:
    """Tint the edit-gated column group so the two families read apart without a second header."""
    for (row, col), cell in table.get_celld().items():
        if first <= col <= last and 0 <= row <= n_rows:
            cell.set_facecolor(_GATED_COL)


def _mark_shipped_row(table: Table, index: int) -> None:
    """Bold the shipped row's text and shade its cells, so the default is spottable."""
    # `get_celld()` keys are (row, col); the header occupies row 0, so data row `index` is cell
    # row `index + 1`. The shipped index is passed in (run_eval owns what "shipped" means).
    for (row, _col), cell in table.get_celld().items():
        if row == index + 1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor(_SHIPPED_ROW)


def _sweep_note(as_shipped: Sequence[PolicyCell], edit_gated: Sequence[PolicyCell]) -> str:
    # A cell that never fired has NO precision (None), not a precision of zero — it is excluded
    # from the ranking rather than given the worst finite value.
    scored = [c for c in (*as_shipped, *edit_gated) if c.precision is not None]
    if not scored:
        return f"{len(as_shipped)} configurations, none of which fired"
    best = max(scored, key=lambda c: c.precision or 0.0)
    family = "edit-gated" if any(best is c for c in edit_gated) else "as-shipped"
    return (
        f"{len(as_shipped)} configurations per family; highest P(fail|fired) is "
        f"{best.precision:.3f} at {family} escalate_after_n={best.escalate_after_n} "
        f"(stale_window={best.stale_window}) against a base rate of {best.base_failure_rate:.3f}"
    )


def _sweep_limits(
    as_shipped: Sequence[PolicyCell], edit_gated: Sequence[PolicyCell]
) -> tuple[str, ...]:
    """Say plainly, with a count, how many cells fail to clear the base rate."""
    cells = [*as_shipped, *edit_gated]
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
    return (
        f"{len(separating)} of {len(cells)} configurations across both families clear the base "
        "failure rate; every other cell in this table does not.",
    )


# ------------------------------------------------------------------- 4. session_value


SESSION_VALUE_SPEC = FigureSpec(
    title="At production cadence, escalating after a cheap failure beats retrying cheap",
    reading=(
        "Read on EVERY trajectory in the corpus, not the per-step-stamped subset the other "
        "escalation figures score: a session outcome comes off the run header, so a run without "
        "per-step stamps still counts here. "
        "Measured on the overlap subset — tasks carrying BOTH a second cheap session and a "
        "frontier session, so the two arms are read on the same tasks. Left: after a cheap "
        "session failed a task, the share of FRONTIER sessions on that task that resolved it "
        "(escalate) against the share of a SECOND cheap session that resolved it (retry). Both "
        "intervals resample whole INSTANCES, because several frontier sessions on one task are "
        "not independent draws. Right: the PAIRED difference, escalate minus retry, on those same "
        "instance resamples, with its 95% interval and zero marked."
    ),
    goal=(
        "The right panel is the answer. Two marginal intervals that fail to overlap is a "
        "conservative test of a difference; the paired distribution IS the difference, and the "
        "claim holds only if its interval excludes zero."
    ),
    definitions=(
        ("frontier", "the two most expensive models present in the corpus"),
        ("cheap", "the cheapest model present — the base pick and the retry counterfactual"),
        ("overlap subset", "tasks with >=2 cheap sessions AND a frontier session"),
    ),
    notes=(
        "At session cadence the detector is trivially satisfied — the failed cheap session carries "
        "the task's target failing-check id — so this measures the LADDER's value, not the "
        "trigger's detection quality.",
        "The dashed line is the cheap model's UNCONDITIONAL base rate. The bars condition on a "
        "cheap failure on the same task, so the line is not a ceiling for them.",
    ),
    limitations=(
        "Observational: the arms ran in parallel and which tasks got frontier coverage was "
        "adaptive. Small n — read the interval, not the point estimate.",
        "Production's ladder steps one price rank at a time; this collapses it to its endpoint.",
    ),
)


def session_value(sc: SessionCadenceReport, axes: Sequence[Axes]) -> Annotations:
    """The escalate-vs-retry contrast and, beside it, the paired difference that decides it."""
    ax_bars, ax_diff = axes
    _draw_session_bars(ax_bars, sc)
    _draw_paired_difference(ax_diff, sc)
    lift = "n/a" if sc.lift is None else f"{sc.lift:.2f}x"
    return Annotations(
        subtitle_facts=(
            f"{sc.n_overlap_instances} overlap tasks · escalate {sc.n_escalated_resolved}/"
            f"{sc.n_escalated} vs retry {sc.n_retried_resolved}/{sc.n_retried}",
            f"lift {lift} · paired difference {sc.diff_estimate:+.3f} "
            f"[{sc.diff_ci[0]:+.3f}, {sc.diff_ci[1]:+.3f}]",
        ),
        # ALWAYS non-None, and that is load-bearing: `_merge` keeps the first caveat, so a
        # figure that leaves it None inherits the run-level one — and the run-level line is about
        # per-step stamping, which this figure does not use (it reads every trajectory, stamped
        # or not). An inherited caveat that is false for the figure carrying it is the exact
        # defect this redesign removes. The same reasoning applies to the subtitle and the
        # limitations, which is why the caller hands this figure `_session_run_annotations`
        # rather than the stamped-corpus `_run_annotations`.
        caveat=(
            "Observational: the arms ran in parallel and frontier coverage was adaptive."
            if sc.diff_excludes_zero
            else "The paired difference's 95% interval spans zero — no measured advantage here."
        ),
        notes=(
            f"instance-level bootstrap over {sc.n_instances_resampled} overlap tasks, not Wilson "
            "over sessions: several frontier sessions on one task are one draw, not several",
        ),
        counts=(
            ("overlap_instances", sc.n_overlap_instances),
            ("escalate_sessions", sc.n_escalated),
            ("retry_sessions", sc.n_retried),
        ),
    )


def _draw_session_bars(ax: Axes, sc: SessionCadenceReport) -> None:
    heights = [sc.escalate_rate, sc.retry_rate]
    errors = [
        [sc.escalate_rate - sc.escalate_ci[0], sc.retry_rate - sc.retry_ci[0]],
        [sc.escalate_ci[1] - sc.escalate_rate, sc.retry_ci[1] - sc.retry_rate],
    ]
    ax.bar(
        ["escalate\nto frontier", "cheap\nretry"],
        heights,
        yerr=np.clip(np.array(errors), 0.0, None),
        capsize=6,
        width=0.55,
        color=[_OBSERVED, _SHIPPED],
    )
    for position, (rate, n, top) in enumerate(
        (
            (sc.escalate_rate, sc.n_escalated, sc.escalate_ci[1]),
            (sc.retry_rate, sc.n_retried, sc.retry_ci[1]),
        )
    ):
        ax.text(position, 0.03, f"n={n}", ha="center", color="white", fontsize=9)
        # Above the interval, not above the bar — see `_draw_arms`.
        ax.text(
            position,
            min(0.95, max(rate, top) + 0.025),
            f"{rate:.3f}",
            ha="center",
            fontsize=9.5,
        )
    ax.axhline(
        sc.cheap_base_rate,
        linestyle="--",
        color=_NULL_CENTRE,
        label=f"cheap unconditional base {sc.cheap_base_rate:.3f}",
    )
    ax.set_ylabel("P(task resolved by the next session)")
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="upper right")
    panel_label(ax, "A · resolution by next-session choice")


def _draw_paired_difference(ax: Axes, sc: SessionCadenceReport) -> None:
    if not sc.diff_draws:
        ax.axis("off")
        ax.text(0.5, 0.5, "no paired draws on this corpus", ha="center", va="center", color=MUTED)
        return
    ax.hist(list(sc.diff_draws), bins=40, color=_NULL_BAND)
    ax.axvline(0.0, color=_NULL_CENTRE, linestyle="--", linewidth=1.2, label="no difference")
    ax.axvline(
        sc.diff_estimate,
        color=_OBSERVED,
        linewidth=2,
        label=f"observed {sc.diff_estimate:+.3f}",
    )
    for bound in sc.diff_ci:
        ax.axvline(bound, color=_OBSERVED, linestyle=":", linewidth=1.2)
    ax.set_xlabel("paired difference: P(resolve | escalate) - P(resolve | retry)", labelpad=30)
    ax.set_ylabel("instance resamples")
    ax.tick_params(labelsize=_TICK_PT)
    # Below the axes: every vertical rule this legend names spans the FULL height, so an
    # in-axes legend necessarily covers the marks it is explaining.
    ax.legend(
        fontsize=_LEGEND_PT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=2,
        frameon=False,
    )
    panel_label(ax, "B · paired difference, 95% interval dotted")


# ------------------------------------------------------------- 5. corpus_and_coverage


@dataclass(frozen=True)
class ModelArm:
    """One model's two outcome arms at the canonical cell — the per-model dumbbell's data."""

    model: str
    n: int
    p_fail_fired: float | None
    p_fail_quiet: float | None


@dataclass(frozen=True)
class Admission:
    """Who the prefix risk model is actually fitted on, and who it silently is not."""

    depth: int
    n_stamped: int
    n_too_short: int
    n_by_margin: int
    n_admitted: int
    admitted_base_rate: float
    corpus_base_rate: float


@dataclass(frozen=True)
class StratifiedAuroc:
    """The recurrence score pooled, then ranked only within a model, then within a challenge."""

    pooled: float
    within_model: float | None
    within_challenge: float | None


CORPUS_COVERAGE_SPEC = FigureSpec(
    title="Who is in the sample, and whether the edge survives the confounds",
    reading=(
        "A: the share of each model's trajectories that carry per-step verified outcomes, with "
        "95% Wilson intervals and the counts printed — a run without them cannot fire the trigger "
        "at all and is excluded from every per-step metric. B: per model, P(run failed | fired) "
        "against P(run failed | quiet) at the canonical cell, drawn as a dumbbell; a model whose "
        "two ends coincide contributes no separation. C: the recurrence score's AUROC pooled, "
        "then computed WITHIN each model and WITHIN each challenge and pooled by comparable pairs "
        "— the drop between them is how much of the pooled number is the confound rather than the "
        "score. D: the prefix risk model's admission waterfall at its reported depth, with the "
        "admitted population's base failure rate against the corpus's."
    ),
    goal=(
        "In C the within-strata bars must stay well above chance: if the pooled edge disappears "
        "once ranking happens inside a model or inside a challenge, the score is reading which "
        "model or which task the run belongs to. In D read the two base rates against each other "
        "— an admitted population failing far more often than the corpus is a different "
        "population, and a null measured on it is a coverage gap, not a falsification."
    ),
    definitions=(
        ("stamped", "the offline container replay wrote per-step verified outcomes for this run"),
        # "canonical cell" appears in this figure's Reading block but was defined only on
        # operating_point.png, so a reader arriving here had no way to learn it is the eval-only
        # counter. A term a canvas uses is defined on that canvas.
        ("canonical cell", "the eval-only edit-gated counter at the shipped knobs (panels B, C)"),
        ("within-model AUROC", "ranked only against runs of the same model, pooled by pair count"),
        ("admission margin", "runs that reached the depth but leave too few steps unread after it"),
    ),
    notes=(
        "Stamping coverage tracks capture DATE, and capture date correlates with model, so model "
        "and coverage are confounded on this corpus and cannot be separated from it.",
        "A single-class stratum contributes no comparable pairs and is DROPPED from the "
        "within-strata AUROCs rather than scored at chance.",
    ),
    limitations=(
        "Panel D's population is length-selected by construction: the anti-leak margin excludes "
        "every short run, and short runs resolve more often, so the admitted base rate is higher "
        "than the corpus's by design rather than by accident.",
    ),
)


def corpus_and_coverage(
    coverages: Sequence[ModelCoverage],
    arms: Sequence[ModelArm],
    strat: StratifiedAuroc,
    admission: Admission | None,
    axes: Sequence[Axes],
) -> Annotations:
    """Sample composition, per-model separation, stratified AUROC, and the prefix scope."""
    ax_cov, ax_arms, ax_strat, ax_admit = axes
    _draw_stamping(ax_cov, coverages)
    _draw_model_arms(ax_arms, arms)
    _draw_stratified(ax_strat, strat)
    _draw_admission(ax_admit, admission)
    total = sum(c.n_trajectories for c in coverages)
    stamped = sum(c.n_stamped for c in coverages)
    facts = [f"{len(coverages)} models · {stamped}/{total} trajectories stamped"]
    if admission is not None:
        facts.append(
            f"prefix depth {admission.depth} admits {admission.n_admitted}/"
            f"{admission.n_stamped} at base rate {admission.admitted_base_rate:.3f} vs corpus "
            f"{admission.corpus_base_rate:.3f}"
        )
    return Annotations(
        subtitle_facts=tuple(facts),
        # Panel D ONLY. The feature-mismatch line used to be appended to every figure in the
        # set, including the recurrence ROC, whose score reads two fields production has.
        # The old wording said "Panel D only: … The recurrence trigger does not", which was false
        # for this canvas: panels B and C are both scored on the EDIT-GATED counter, and that
        # counter reads `StepView.action` — a per-step field production lacks. The claim was
        # narrowed correctly for the as-shipped score and over-broadly for the canonical one.
        caveat=(
            "Panels B/C use the eval-only edit-gated counter; panel D's prefix score "
            "reads per-step fields production lacks."
        ),
        notes=(
            f"AUROC pooled {strat.pooled:.3f} · within-model {_rate(strat.within_model)} · "
            f"within-challenge {_rate(strat.within_challenge)}",
        ),
        limitations=_coverage_limits(coverages),
        counts=(("models", len(coverages)), ("trajectories", total), ("stamped", stamped)),
    )


def _draw_stamping(ax: Axes, coverages: Sequence[ModelCoverage]) -> None:
    names = [c.model for c in coverages]
    shares = [c.n_stamped / c.n_trajectories if c.n_trajectories else 0.0 for c in coverages]
    intervals = [metrics.wilson_interval(c.n_stamped, c.n_trajectories) for c in coverages]
    errors = np.array(
        [
            [max(0.0, s - lo) for s, (lo, _hi) in zip(shares, intervals, strict=True)],
            [max(0.0, hi - s) for s, (_lo, hi) in zip(shares, intervals, strict=True)],
        ]
    )
    ax.barh(names, shares, xerr=errors, color=_SHIPPED, height=0.6, capsize=3)
    for index, cov in enumerate(coverages):
        ax.text(0.02, index, f"{cov.n_stamped}/{cov.n_trajectories}", va="center", fontsize=8)
    ax.set_xlabel("share of runs with per-step verified outcomes")
    ax.set_xlim(0, 1.05)
    ax.tick_params(labelsize=_TICK_PT)
    panel_label(ax, "A · stamping coverage, 95% Wilson")


def _draw_model_arms(ax: Axes, arms: Sequence[ModelArm]) -> None:
    readable = [a for a in arms if a.p_fail_fired is not None and a.p_fail_quiet is not None]
    if not readable:
        ax.axis("off")
        ax.text(0.5, 0.5, "no model has both arms populated", ha="center", va="center", color=MUTED)
        return
    for index, arm in enumerate(readable):
        quiet = arm.p_fail_quiet or 0.0
        fired = arm.p_fail_fired or 0.0
        ax.plot([quiet, fired], [index, index], color=_FAINT, linewidth=2, zorder=1)
        ax.scatter([quiet], [index], color=_SHIPPED, zorder=2, s=34)
        ax.scatter([fired], [index], color=_OBSERVED, zorder=2, s=34)
    ax.set_yticks(range(len(readable)), [f"{a.model} (n={a.n})" for a in readable])
    ax.scatter([], [], color=_SHIPPED, label="quiet", s=34)
    ax.scatter([], [], color=_OBSERVED, label="fired", s=34)
    ax.set_xlabel("P(run ultimately failed)")
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="lower right")
    panel_label(ax, "B · per-model separation at the canonical cell")


def _draw_stratified(ax: Axes, strat: StratifiedAuroc) -> None:
    bars = [
        ("pooled", strat.pooled),
        ("within model", strat.within_model),
        ("within challenge", strat.within_challenge),
    ]
    drawn = [(name, value) for name, value in bars if value is not None]
    ax.bar(
        [name for name, _ in drawn],
        [value for _, value in drawn],
        color=[_OBSERVED, "#EF6C00", _SHIPPED][: len(drawn)],
        width=0.55,
    )
    for index, (_name, value) in enumerate(drawn):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=9.5)
    ax.axhline(0.5, linestyle="--", color=_NULL_CENTRE, label="chance")
    ax.set_ylabel("AUROC of the recurrence score")
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="upper right")
    panel_label(ax, "C · does the edge survive the strata")


# Axes-fraction centre of the waterfall's second column ("too short"), which is the one region
# of that panel no bar reaches: four categories over a [-0.5, 3.5] view puts column 1 at 0.375.
_ADMISSION_NOTE_X = 0.375


def _draw_admission(ax: Axes, admission: Admission | None) -> None:
    if admission is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no prefix depth was estimable", ha="center", va="center", color=MUTED)
        return
    labels = ["stamped", "too short", "anti-leak\nmargin", "admitted"]
    values = [
        admission.n_stamped,
        -admission.n_too_short,
        -admission.n_by_margin,
        admission.n_admitted,
    ]
    bottoms = [0, admission.n_stamped - admission.n_too_short, admission.n_admitted, 0]
    colours = [_SHIPPED, _UNDEFINED, _UNDEFINED, _OBSERVED]
    ax.bar(labels, [abs(v) for v in values], bottom=bottoms, color=colours, width=0.6)
    for index, value in enumerate(values):
        ax.text(
            index,
            bottoms[index] + abs(value) + admission.n_stamped * 0.015,
            f"{abs(value)}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("trajectories")
    ax.set_ylim(0, admission.n_stamped * 1.16)
    ax.tick_params(labelsize=_TICK_PT)
    # In the "too short" column's own empty space. That column's bar is ten trajectories tall, so
    # everything below it is blank by construction — no opaque backing needed, and an opaque
    # backing here would punch a hole through whichever bar it drifted onto.
    ax.text(
        _ADMISSION_NOTE_X,
        0.40,
        f"admitted base rate {admission.admitted_base_rate:.3f}\nvs corpus "
        f"{admission.corpus_base_rate:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=_OBSERVED,
    )
    panel_label(ax, f"D · prefix admission at depth {admission.depth}")


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


# ---------------------------------------------------------------- 6. escalation_budget


ESCALATION_BUDGET_SPEC = FigureSpec(
    # "the trigger", unqualified, read as the SHIPPED trigger — which fires on 727/727 runs and
    # would produce an entirely different ledger. Every number on this canvas comes from the
    # canonical (edit-gated) cell, so the title has to say which trigger it is about.
    title="What firing costs: the eval-only edit-gated trigger pre-empts more than it interrupts",
    reading=(
        "Left: where in a run the trigger fires, as a fraction of the run's total steps, drawn as "
        "an ECDF for the runs that ultimately FAILED and for the runs that were RESOLVED. A curve "
        "that rises early means the trigger fires early in those runs. Right: the steps that sit "
        "AFTER the trigger point, totalled over every fired run and split by how the run ended. "
        "On a failed run that work was spent and lost, so escalating there PRE-EMPTS it; on a "
        "resolved run the agent went on to fix the task, so escalating INTERRUPTS work that was "
        "about to pay off. The ratio between the two bars is the trigger's budget case."
    ),
    goal=(
        "Want the pre-empted bar clearly taller than the interrupted one — that ratio is what the "
        "trigger buys per unit of disruption. In the left panel, want the failed-run curve to the "
        "LEFT of the resolved one: firing earlier on the runs that were going to fail is the "
        "whole point."
    ),
    definitions=(
        ("edit-gated", "failures before the agent's first edit-like action are not counted"),
        ("fire position", "the step index the policy first escalated at, over the run's length"),
        ("pre-empted", "steps after the trigger on runs that ultimately failed"),
        ("interrupted", "steps after the trigger on runs that were ultimately resolved"),
    ),
    notes=(
        "Aggregates only. The per-run timing arrays these summarise are deliberately not kept: "
        "the same reasoning that deleted the lead-time figure — on this corpus a lead time is "
        "largely the run length minus a constant.",
        "Steps are agent decisions, not wall-clock and not dollars. This is a work ledger, not a "
        "cost estimate.",
    ),
    limitations=(
        "COUNTERFACTUAL BY ARITHMETIC, not by measurement: no logged trajectory escalated, so "
        "'pre-empted' is what firing would have cut short assuming the run would otherwise have "
        "continued unchanged. See the scope strip.",
        "The ledger's ratio is driven mostly by ARM SIZE, not by timing: most of it is simply "
        "that more of the fired runs failed. Read it beside the fire-position panel, which is "
        "where a timing claim would have to come from — and where the two curves nearly coincide.",
    ),
)


def escalation_budget(cell: PolicyCell, axes: Sequence[Axes]) -> Annotations:
    """Where the trigger lands and what the steps after it are worth, split by outcome."""
    ax_ecdf, ax_ledger, ax_scope = axes
    budget = cell.budget
    _draw_fire_ecdf(ax_ecdf, budget)
    _draw_step_ledger(ax_ledger, budget)
    scope_strip(ax_scope)
    if not budget.n_fired_positioned:
        return Annotations(
            subtitle_facts=("no fired run carries a trigger position on this report",),
            limitations=("The budget aggregates were not computed for this cell.",),
        )
    ratio = _ratio(budget.steps_after_fire_failed, budget.steps_after_fire_resolved)
    return Annotations(
        subtitle_facts=(
            f"{budget.n_fired_positioned} fired runs · median fire at step "
            f"{_number(budget.fire_step_median)} of {_number(budget.run_length_median)}",
            f"{budget.steps_after_fire_failed} steps pre-empted vs "
            f"{budget.steps_after_fire_resolved} interrupted ({ratio})",
        ),
        notes=(
            f"median fire position {_rate(budget.fire_fraction_median_failed)} of the run on "
            f"failed runs, {_rate(budget.fire_fraction_median_resolved)} on resolved ones",
        ),
        counts=(("fired_positioned", budget.n_fired_positioned),),
    )


def _draw_fire_ecdf(ax: Axes, budget: policy_eval.BudgetAggregates) -> None:
    quantiles = np.linspace(0.0, 1.0, 11)
    drawn = False
    for deciles, name, colour in (
        (budget.fire_fraction_deciles_failed, "ultimately failed", _OBSERVED),
        (budget.fire_fraction_deciles_resolved, "ultimately resolved", _SHIPPED),
    ):
        if not deciles:
            continue
        drawn = True
        ax.step(deciles, quantiles, where="post", color=colour, linewidth=1.8, label=name)
    if not drawn:
        ax.axis("off")
        ax.text(0.5, 0.5, "no positioned fire on this cell", ha="center", va="center", color=MUTED)
        return
    ax.set_xlabel("fire position as a fraction of the run's steps")
    ax.set_ylabel("share of fired runs at or before")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=_TICK_PT)
    ax.legend(fontsize=_LEGEND_PT, loc="lower right")
    panel_label(ax, "A · where the trigger lands")


def _draw_step_ledger(ax: Axes, budget: policy_eval.BudgetAggregates) -> None:
    heights = [budget.steps_after_fire_failed, budget.steps_after_fire_resolved]
    ax.bar(
        ["pre-empted\n(run failed)", "interrupted\n(run resolved)"],
        heights,
        color=[_OBSERVED, _SHIPPED],
        width=0.55,
    )
    top = max(heights) if any(heights) else 1
    for index, value in enumerate(heights):
        ax.text(index, value + top * 0.02, f"{value}", ha="center", fontsize=9.5)
    ax.set_ylabel("agent steps after the trigger point")
    ax.set_ylim(0, top * 1.16)
    ax.tick_params(labelsize=_TICK_PT)
    panel_label(ax, "B · work after the trigger, by outcome")


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f}:1" if denominator else "no interrupted steps"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _rate(value: float | None, digits: int = 3) -> str:
    """Format a proportion for a caption, keeping an undefined one visibly undefined."""
    # A never-firing cell's interval is (nan, nan) — not None — so a bare-value guard is not
    # enough: NaN is the same undefined quantity and must print the same way.
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


__all__ = [
    "CORPUS_COVERAGE_SPEC",
    "ESCALATION_BUDGET_SPEC",
    "ESCALATION_DECISION_SPEC",
    "OPERATING_POINT_SPEC",
    "POLICY_SWEEP_SPEC",
    "SESSION_VALUE_SPEC",
    "Admission",
    "ModelArm",
    "StratifiedAuroc",
    "corpus_and_coverage",
    "escalation_budget",
    "escalation_decision",
    "operating_point",
    "policy_sweep",
    "scope_strip",
    "session_value",
]
