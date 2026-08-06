"""arm_manipulation.png — the manipulation check first: was the reasoning knob ever turned?"""

# `arm_monotonicity` reported flat pass-rate contrasts across reasoning arms and read as a
# null result: "more reasoning effort does not help". On most of the pairs it is not a null
# result, because the knob NEVER FIRED. Paired on co-measured tasks, the high arm spends
# 0.78x to 1.08x the low arm's output tokens on deepseek, kimi-k2.5, qwen and zai — at or
# below the noise of no change at all. Only gpt-5-mini's minimal/medium -> high steps show a
# real manipulation, at 2.6-2.9x.
#
# A treatment that was never applied cannot have a null effect; it has no effect to measure.
# So the manipulation check is the FIRST panel and it gates the second: a row whose knob did
# not fire is drawn grey and labelled "manipulation never fired — not a null", never as
# evidence about reasoning effort.

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import metrics
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import usd

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_FIRED = "#0072B2"
_NOT_FIRED = "#9E9E9E"
_NO_CHANGE = "#B71C1C"

NOT_A_NULL: str = "never fired — not a null"

SPEC = FigureSpec(
    title="Most reasoning knobs were never turned — the flat contrasts are unmeasured",
    reading=(
        "Left: the manipulation check. For each (model, low arm, high arm) pair, the ratio of "
        "mean output tokens on the tasks where BOTH arms ran. A knob that was turned moves "
        "this well above 1; the red line is where nothing changed. Right: the paired "
        "pass-rate difference for the same pairs, with a 95% paired interval. A row whose "
        "knob did not fire is greyed and carries the label — its flat difference measures "
        "nothing."
    ),
    goal=(
        "Do not read the right panel for a row that is grey on the left. The only row that "
        "supports any claim about reasoning effort is the one whose knob demonstrably moved."
    ),
    definitions=(
        (
            "manipulation check",
            "evidence the treatment was applied at all, measured before any outcome. Output "
            "tokens are what a reasoning-effort setting mechanically controls.",
        ),
        (
            "paired contrast",
            "computed only on tasks where BOTH arms ran and NEITHER was censored, so a "
            "resource-limit stop cannot masquerade as a capability failure.",
        ),
    ),
    notes=(
        "Arm coverage is sparse by design (p(arm|model) sampling), so a pair below the "
        "minimum count is greyed for sample size as well as for manipulation.",
    ),
    limitations=(
        "The ratio is over co-measured tasks only. A knob could fire on tasks neither arm "
        "shares, and this check would not see it.",
    ),
)


def _label(row: dict) -> str:
    return f"{row['model']}\n{row['low_arm']} → {row['high_arm']}"


def _draw_check(ax: Axes, rows: list[dict]) -> None:
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        ratio = row["out_tok_ratio"]
        colour = _FIRED if row["fired"] else _NOT_FIRED
        ax.barh(y, ratio, height=0.55, color=colour, zorder=2)
        ax.text(
            ratio + 0.06,
            y,
            f"{ratio:.2f}×  (n={row['n_pairs']})" + ("" if row["fired"] else "  ← never fired"),
            fontsize=7.5,
            va="center",
            color=colour,
            zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85),
        )
    # Under the labels: the two reference lines cross every row, so drawing them on top
    # struck through the ratio each row exists to report.
    ax.axvline(1.0, color=_NO_CHANGE, lw=1.3, zorder=1)
    ax.axvline(metrics.ARM_FIRED_RATIO, color=_FIRED, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(r) for r in rows], fontsize=7)
    top = max(
        (r["out_tok_ratio"] for r in rows if r["out_tok_ratio"] == r["out_tok_ratio"]), default=1.0
    )
    ax.set_xlim(0, max(top, 1.2) * 1.55)
    ax.set_xlabel("high-arm ÷ low-arm mean output tokens (1.0 = knob did nothing)", fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · manipulation check — was the knob turned?")


def _draw_contrast(ax: Axes, rows: list[dict], pairs: dict[tuple[str, str, str], dict]) -> None:
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0.0, color="#bbbbbb", lw=0.9, zorder=1)
    for y, row in zip(ys, rows, strict=True):
        pair = pairs.get((row["model"], row["low_arm"], row["high_arm"]))
        if pair is None:
            continue
        colour = _FIRED if row["fired"] else _NOT_FIRED
        lo, hi = pair["ci"]
        ax.plot([lo, hi], [y, y], color=colour, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot([pair["delta_pp"]], [y], "o", color=colour, ms=7, zorder=3)
        text = f"{pair['delta_pp']:+.1f}pp (n={pair['n']})"
        if not row["fired"]:
            text = NOT_A_NULL
        ax.text(hi + 1.2, y, text, fontsize=7.5, va="center", color=colour)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    bounds = [
        b
        for r in rows
        if (p := pairs.get((r["model"], r["low_arm"], r["high_arm"])))
        for b in p["ci"]
    ]
    span = max((abs(b) for b in bounds), default=10.0)
    ax.set_xlim(-span * 1.15, span * 2.5)
    ax.set_xlabel("paired pass-rate difference, high − low arm (pp)", fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · effect — only readable where the knob fired")


def _pool(pairs: list[dict]) -> dict[str, float] | None:
    """Net paired effect over a subset of arm pairs (report.arm_pair_totals, inlined)."""
    n = sum(p["n"] for p in pairs)
    if not n:
        return None
    lost = sum(p["violations"] for p in pairs)
    won = sum(p["gains"] for p in pairs)
    return {
        "n": float(n),
        "net_pp": (won - lost) / n * 100,
        "p": metrics.mcnemar_exact_p(lost, won),
    }


def _annotations(
    rows: list[dict], pairs: dict[tuple[str, str, str], dict], totals: dict
) -> Annotations:
    fired = [r for r in rows if r["fired"]]
    # Pool over the FIRED rows only. Pooling all nine put 392 pairs whose knob never moved
    # into the headline, dragging a +8.5pp effect down to +1.7pp — the figure's title, its
    # greying and its caveat all exist to say those rows measure nothing, so the subtitle
    # must not average them back in.
    fired_keys = {(r["model"], r["low_arm"], r["high_arm"]) for r in rows if r["fired"]}
    fired_pairs = [pair for key, pair in pairs.items() if key in fired_keys]
    fired_totals = _pool(fired_pairs)
    facts = [
        f"{len(fired)} of {len(rows)} reasoning knobs demonstrably fired "
        f"(ratio ≥ {metrics.ARM_FIRED_RATIO:g}× output tokens)",
    ]
    if fired_totals is not None:
        facts.append(
            f"over the fired knobs only: {fired_totals['net_pp']:+.2f}pp on "
            f"{int(fired_totals['n'])} co-measured pairs (exact McNemar "
            f"p={fired_totals['p']:.3f})"
        )
    facts.append(
        f"all nine pooled, including the seven that never fired: {totals['net_pp']:+.2f}pp "
        f"on {int(totals['n'])} pairs"
    )
    caveat = (
        f"{len(rows) - len(fired)} of {len(rows)} rows show no manipulation — their flat "
        "contrasts are unmeasured, not null."
        if len(fired) < len(rows)
        else None
    )
    notes = []
    for row in rows:
        pair = pairs.get((row["model"], row["low_arm"], row["high_arm"]))
        state = "FIRED" if row["fired"] else NOT_A_NULL
        line = (
            f"{row['model']} {row['low_arm']}→{row['high_arm']}: output-token ratio "
            f"{row['out_tok_ratio']:.2f}x on n={row['n_pairs']} pairs — {state}"
        )
        if pair is not None:
            line += (
                f"; paired Δ {pair['delta_pp']:+.1f}pp "
                f"[{pair['ci'][0]:+.1f}, {pair['ci'][1]:+.1f}], cost Δ {usd(pair['cost_delta'], 4)}"
            )
        notes.append(line)
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        counts=(("arm_pairs", len(rows)), ("fired", len(fired))),
    )


def render(ctx: ctxmod.RoutingContext, contrasts: list[dict], totals: dict) -> Path | None:
    """Draw arm_manipulation.png; the manipulation check gates the effect panel."""
    if ctx.raw is None or not contrasts:
        return None
    keys = [(c["model"], c["low"], c["high"]) for c in contrasts]
    rows = metrics.arm_manipulation(ctx.raw, keys)
    if not rows:
        return None
    pairs = {(c["model"], c["low"], c["high"]): c for c in contrasts}
    rows.sort(
        key=lambda r: -r["out_tok_ratio"] if r["out_tok_ratio"] == r["out_tok_ratio"] else 0.0
    )
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.0, 1.0))
    _draw_check(axes[0], rows)
    _draw_contrast(axes[1], rows, pairs)
    return plot_frame.save(
        fig,
        ctx.out_dir / "arm_manipulation.png",
        SPEC,
        extra=_annotations(rows, pairs, totals),
        provenance=ctx.provenance(__name__),
        size=size,
    )
