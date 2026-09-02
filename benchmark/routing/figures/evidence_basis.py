"""evidence_basis.png — how much of every number in this set is measured, and how much is filled."""

# NEARLY every imputed cell is pass=True (`impute.py`: the ladder's dominant branch says "a
# pricier model would also have passed"; it also has a fail branch, for a model at or below an
# observed failure, which fires rarely). So imputation almost only ever adds a success, and
# every pass rate in this figure set is biased upward by close to the share drawn here — the
# subtitle carries the measured pass/fail split of the filled cells rather than asserting it.
# Two channels, not one: the DOLLARS a strategy billed on projected cells
# and the PASSES it was credited with on them. A figure that discloses only the dollars
# leaves the quality claim undisclosed, which is the half that matters.
#
# The per-band panel is the one that cannot be replaced by a per-strategy total: imputation
# is not spread evenly. In the band holding half the enabled models the imputed cells
# OUTNUMBER the real ones, and a pooled percentage hides that entirely.

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.patches import Patch

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import usd

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_MEASURED = "#0072B2"
_IMPUTED = "#E69F00"
_UNKNOWN = "#BDBDBD"

SPEC = FigureSpec(
    title="A third of the evidence is filled in, and nearly every filled cell is a pass",
    reading=(
        "Left: per strategy, the share of scored DOLLARS billed to measured cells against "
        "projected ones. Middle: the same split on PASSES — the channel that decides every "
        "quality claim in this set. Right: per capability band, real against imputed against "
        "still-unknown cells; the bands are ordered weakest to strongest."
    ),
    goal=(
        "Look for a strategy whose orange share is large on the PASS panel: its pass rate is "
        "that far from a measurement. Then look at the right panel for a band where orange "
        "exceeds blue — every comparison that crosses that band rests more on the imputer "
        "than on the benchmark."
    ),
    definitions=(
        (
            "imputed cell",
            "filled by the monotone ladder: a model at least as strong as one that passed is "
            "credited with a pass, and a model no stronger than one that failed with a fail. "
            "The pass branch dominates; the subtitle counts how far.",
        ),
        (
            "capability band",
            "models grouped by derived capability rank, weakest band first. A band with more "
            "imputed than real cells is carried by the imputer.",
        ),
        ("unknown", "a cell neither measured nor safely fillable — excluded from scoring."),
    ),
    notes=(
        "The dollar split is PATH-AWARE: a cascade that probed a projected cell on its way to "
        "a measured pick counts as projected, so the measured bar is measured end to end.",
    ),
    limitations=(
        "Imputation is directional. Nothing here corrects the bias; it states its size so a "
        "reader can discount the pass rates by it.",
    ),
)


def strategy_splits(ctx: ctxmod.RoutingContext) -> dict[str, dict[str, float]]:
    """Per strategy: measured/imputed dollars and passes over its scored selections."""
    out: dict[str, dict[str, float]] = {}
    for name, (cells, unscorable) in ctx.by_strategy.items():
        scored = [(p, c, i) for tid, (p, c, i) in cells.items() if tid not in unscorable]
        if not scored:
            continue
        out[name] = {
            "measured_cost": sum(c for _p, c, i in scored if not i),
            "imputed_cost": sum(c for _p, c, i in scored if i),
            "measured_pass": float(sum(1 for p, _c, i in scored if p and not i)),
            "imputed_pass": float(sum(1 for p, _c, i in scored if p and i)),
            "n": float(len(scored)),
        }
    return out


def _draw_split(  # noqa: PLR0913 (one argument per drawn channel plus the shared row order)
    ax: Axes,
    splits: dict[str, dict[str, float]],
    key: str,
    label: str,
    money: bool,
    names: list[str],
) -> None:
    # Both split panels take the SAME row order, so a reader comparing a strategy's
    # dollars against its passes reads across one line rather than hunting for the row.
    # The value string is kept SHORT deliberately. It is data-anchored past the end of the
    # bar, so constrained layout shrinks the panel to make room for whatever it overhangs
    # and `fit_end_labels` then widens the axis against that shrunken geometry — a longer
    # label costs axis span twice over. "(45% projected)" ran the dollars axis to $220 for
    # a $96 maximum; "· 45% proj." lands it at $161 with the panel title already saying
    # "measured vs projected".
    ys = list(range(len(names)))[::-1]
    for y, name in zip(ys, names, strict=True):
        measured = splits[name][f"measured_{key}"]
        imputed = splits[name][f"imputed_{key}"]
        total = measured + imputed
        ax.barh(y, measured, height=0.6, color=_MEASURED, zorder=2)
        ax.barh(y, imputed, left=measured, height=0.6, color=_IMPUTED, zorder=2)
        if total > 0:
            shown = usd(total) if money else f"{total:.0f}"
            ax.text(
                total * 1.03,
                y,
                f"{shown} · {imputed / total:.0%} proj.",
                fontsize=7,
                va="center",
                color="#333333",
            )
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5)
    top = max(
        (splits[n][f"measured_{key}"] + splits[n][f"imputed_{key}"] for n in names), default=1.0
    )
    # A small margin on the bars; `fit_end_labels` then buys exactly the room the value
    # strings need. The old 1.42 was a guess ON TOP of that measurement and left half of
    # panel A empty — nothing is plotted past $96 on an axis that ran to $200.
    ax.set_xlim(0, top * 1.42)
    ax.set_xlabel(label, fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)


def _draw_bands(ax: Axes, rows: list[dict]) -> None:
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        real, imputed, unknown = int(row["real"]), int(row["imputed"]), int(row["unknown"])
        ax.barh(y, real, height=0.6, color=_MEASURED, zorder=2)
        ax.barh(y, imputed, left=real, height=0.6, color=_IMPUTED, zorder=2)
        ax.barh(y, unknown, left=real + imputed, height=0.6, color=_UNKNOWN, zorder=2)
        total = real + imputed + unknown
        # Second line, not a longer first one: the one-line form was wider than the
        # whole panel, which no amount of axis widening can pull back inside.
        flag = "\n← fill exceeds measurement" if imputed > real else ""
        ax.text(
            total * 1.03,
            y,
            f"{real} real / {imputed} imp. / {unknown} unk.{flag}",
            fontsize=7,
            va="center",
            color="#B71C1C" if imputed > real else "#333333",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"band {r['band']}" for r in rows], fontsize=8)
    top = max((int(r["real"]) + int(r["imputed"]) + int(r["unknown"]) for r in rows), default=1)
    # Same rule as the split panels: tight on the data, widened by measurement below.
    ax.set_xlim(0, top * 2.15)
    ax.set_xlabel("cells in the band", fontsize=9)
    ax.legend(
        handles=[
            Patch(color=_MEASURED, label="real"),
            Patch(color=_IMPUTED, label="imputed (nearly always a pass)"),
            Patch(color=_UNKNOWN, label="unknown (unscored)"),
        ],
        fontsize=7,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
    )
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    # Kept SHORT on purpose. A panel label is laid out before constrained layout knows how
    # wide this panel will be, so it cannot be shrunk to fit: when an eighth strategy row
    # widened panel A's category labels, the two narrower panels lost ~20px each and the
    # old 51-character label ran off the canvas, failing the strict layout audit.
    plot_frame.panel_label(ax, "C · per band — where the fill concentrates")


def filled_outcomes(completed: dict) -> tuple[int, int]:
    """(imputed cells filled pass=True, imputed cells in total) over the completed matrix.

    ``completed`` is the FLAT ``{task: {model: cell}}`` form — `ImputedMatrix.matrix`, the
    same population `report.coverage_rows` counts the bands over.
    """
    # DERIVED, never written down. This fact used to be the frozen string "397 of 398 filled
    # cells" sitting on the same subtitle line as a derived imputed-cell count — two numbers
    # about the same cells that no longer agreed (the derived count is 410), and no gate can
    # catch it: SH012's number-provenance scan reads `*.md` lines, never a producer literal.
    # ONE POPULATION, NOT TWO. The wrapped `{"results": ...}` matrix the report also carries is
    # COMPLETE-ONLY (184 of 200 challenges), so counting this over it reintroduced exactly the
    # disagreement above in derived form — 406 filled cells beside 410 imputed ones on one
    # subtitle line. The bands and this count now walk the same matrix.
    filled = [c for cells in completed.values() for c in cells.values() if c.get("imputed")]
    return (sum(1 for c in filled if c.get("pass")), len(filled))


def _annotations(
    splits: dict[str, dict[str, float]], bands: list[dict], completed_matrix: dict
) -> Annotations:
    total_real = sum(int(r["real"]) for r in bands)
    total_imp = sum(int(r["imputed"]) for r in bands)
    total_unk = sum(int(r["unknown"]) for r in bands)
    completed = total_real + total_imp
    worst = (
        max(bands, key=lambda r: int(r["imputed"]) / max(int(r["real"]) + int(r["imputed"]), 1))
        if bands
        else None
    )
    facts = [
        f"{total_imp} of {completed} completed cells ({total_imp / max(completed, 1):.0%}) are "
        f"imputed, {total_unk} unknown",
    ]
    if worst is not None:
        share = int(worst["imputed"]) / max(int(worst["real"]) + int(worst["imputed"]), 1)
        facts.append(f"worst band {worst['band']}: {share:.0%} imputed")
    pass_filled, n_filled = filled_outcomes(completed_matrix)
    if n_filled:
        facts.append(
            f"imputation is overwhelmingly pass-only ({pass_filled} of {n_filled} filled cells)"
        )
    caveat = None
    if worst is not None and int(worst["imputed"]) > int(worst["real"]):
        caveat = (
            f"Band {worst['band']} holds more imputed cells ({worst['imputed']}) than real "
            f"ones ({worst['real']})."
        )
    notes = [
        f"{name}: {usd(v['measured_cost'])} measured + {usd(v['imputed_cost'])} projected; "
        f"{v['measured_pass']:.0f} measured passes + {v['imputed_pass']:.0f} projected"
        for name, v in sorted(splits.items())
    ] + [
        f"band {r['band']}: {r['real']} real / {r['imputed']} imputed / {r['unknown']} unknown"
        for r in bands
    ]
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        counts=(
            ("completed_cells", completed),
            ("imputed_cells", total_imp),
            ("unknown", total_unk),
        ),
    )


def render(ctx: ctxmod.RoutingContext, band_rows: list[dict]) -> Path | None:
    """Draw evidence_basis.png from the per-strategy split and the per-band coverage rows."""
    splits = strategy_splits(ctx)
    if not splits or not band_rows:
        return None
    order = sorted(splits, key=lambda n: -(splits[n]["measured_cost"] + splits[n]["imputed_cost"]))
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 1, 3)
    _draw_split(axes[0], splits, "cost", "scored dollars", True, order)
    plot_frame.panel_label(axes[0], "A · dollars: measured vs projected")
    _draw_split(axes[1], splits, "pass", "scored passes", False, order)
    plot_frame.panel_label(axes[1], "B · passes: nearly all projected are passes")
    _draw_bands(axes[2], band_rows)
    for ax in axes:
        plot_style.fit_end_labels(ax)
    return plot_frame.save(
        fig,
        ctx.out_dir / "evidence_basis.png",
        SPEC,
        extra=_annotations(splits, band_rows, ctx.imputed.matrix if ctx.imputed else {}),
        provenance=ctx.provenance(__name__),
        size=size,
    )
