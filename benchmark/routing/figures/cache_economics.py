"""cache_economics.png — the product's spine, and the two numbers the gate invented."""

# Cache-safety is what Shunt sells, and until now no figure drew it. Worse, the gate's
# headline criterion — the cache-aware cost ratio — was computed from a hard-coded
# `cache_hit_rate=0.9, cache_discount=0.5` applied uniformly to six providers. Both halves
# of this figure exist to make that visible: panel A puts the assumption next to what the
# providers actually billed, panel B shows how fast the whole saving evaporates if the
# router switches models inside a cached session.

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import cache_cost, plot_style
from benchmark.routing.figures import context as ctxmod

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_BILLED = "#0072B2"
_REGISTRY = "#009E73"
_UNCERTAINTY = "#666666"
_TAX = "#D55E00"

# The hit rate is the ONE input to the registry prediction that no row in this corpus
# measures, so it is the only thing the green marker can be swept over. The band is drawn
# from a pessimistic half-hit to a perfect-hit ceiling; `cache_cost.ASSUMED_CACHE_HIT_RATE`
# is the point inside it that the gate actually costs with.
_HIT_FLOOR = 0.5
_HIT_CEILING = 1.0
# Share of the panel kept clear on the right for the value column, as a fraction of the
# drawn extent. Derived from the data every run — never a fixed axis limit.
_GUTTER = 0.30
# Vertical offset of the hit-rate band below its row's bar, in y units (rows are 1.0 apart
# and the bar is 0.5 tall, so this clears both the bar and the row below).
_BAND_OFFSET = 0.34

SPEC = FigureSpec(
    title="The modelled cache saving is now the same size as the whole list-to-bill discount",
    reading=(
        "Left: per model, the share of the list-price bill that was actually charged. The "
        "blue bar is MEASURED — every row's real_cost over its estimated_cost in "
        "results.csv, which is the whole gap between list price and the invoice. The green "
        "diamond is the share the registry's cache-read price predicts would survive, priced "
        "at the corpus's measured input/output token mix; the faint grey rule beneath each "
        "bar sweeps the one input nothing here measures, the cache hit rate, from 100% (left "
        "cap) to 50% (right cap). The right-hand column repeats each row as billed → "
        "modelled with its row count. Lower is cheaper. Right: the switch tax — how much of "
        "the MODELLED cache saving survives as the router changes model inside one session."
    ),
    goal=(
        "Read the DISTANCE between the blue bar and the green diamond. It is now small on "
        "every model, so caching alone is a large enough effect to account for essentially "
        "the whole discount. Read that as a SIZE agreement, not as a validation: the bar "
        "mixes every reason the invoice differs from list price, and two quantities landing "
        "on the same number cannot separate them. Then read the right panel at the shipped "
        "operating point — one decision per session."
    ),
    definitions=(
        (
            "billed share",
            "sum(real_cost) / sum(estimated_cost) over every measured row for that model. "
            "1.0 means the invoice matched list price. It mixes EVERY reason the two differ "
            "— caching, negotiated rates, provider-side discounts — not caching alone.",
        ),
        (
            "registry prediction",
            "1 - input_share x hit_rate x (1 - cache_read_price / input_price), the "
            "per-model cache economics benchmark.runner.kill_gate now costs with.",
        ),
        (
            "input share",
            "the share of a model's spend that is INPUT tokens, from the measured in_tok / "
            "out_tok mix in results.csv priced at registry rates. This corpus is "
            "input-dominated, which is why the modelled saving is large.",
        ),
        (
            "switch tax",
            "naive cost minus cache-aware cost. A model switch forfeits the cached prefix, "
            "so the next turn is billed at full input price.",
        ),
    ),
    notes=(
        "The cache-read prices come from the shipped registry (src/shunt/config/models.yaml), "
        "so a provider price change moves this figure rather than silently invalidating it.",
    ),
    limitations=(
        "The hit rate is still assumed: no run in this corpus records a per-turn cache-hit "
        "ratio, so only the DISCOUNT and the INPUT SHARE are measured. The grey whisker is "
        "the whole range that assumption can move the green marker over.",
        "The blue bar and the markers are different quantities drawn on one axis "
        "deliberately. They now land close together, and that is NOT a calibration result: "
        "the bar also contains negotiated rates and provider-side discounts, so agreement in "
        "magnitude cannot attribute the discount to caching.",
        "The right panel is a MODEL of the switch tax, not a measurement — no live session "
        "in this corpus switched model mid-conversation, because the shipped router cannot.",
    ),
)


@dataclass(frozen=True)
class ModelCache:
    """One model's assumed and realised cache economics."""

    model: str
    billed_share: float
    n_rows: int
    registry_share: float
    share_at_hit_floor: float
    share_at_hit_ceiling: float
    input_share: float
    provenance: str
    share_provenance: str

    @property
    def residual(self) -> float:
        """Modelled surviving share minus the billed one: what caching does NOT explain."""
        return self.registry_share - self.billed_share


def billed_shares(results_csv: Path) -> dict[str, tuple[float, int]]:
    """model -> (sum(real_cost)/sum(estimated_cost), n rows) over every priced row."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                real = float(row.get("real_cost") or 0.0)
                estimated = float(row.get("estimated_cost") or 0.0)
            except ValueError:
                continue
            if estimated <= 0:
                continue
            bucket = totals[row["model"]]
            bucket[0] += real
            bucket[1] += estimated
            bucket[2] += 1
    return {m: (r / e, int(n)) for m, (r, e, n) in totals.items() if e > 0}


def model_rows(ctx: ctxmod.RoutingContext) -> list[ModelCache]:
    """Per enabled model, the measured bill next to both cost models."""
    billed = billed_shares(config.results_csv_path())
    prices = cache_cost.cache_prices(ctx.models_by_price)
    rows: list[ModelCache] = []
    for model in ctx.models_by_price:
        share, n = billed.get(model, (float("nan"), 0))
        price = prices[model]
        rows.append(
            ModelCache(
                model=model,
                billed_share=share,
                n_rows=n,
                registry_share=1.0 - price.saving_fraction,
                share_at_hit_floor=1.0 - price.input_share * _HIT_FLOOR * price.discount,
                share_at_hit_ceiling=1.0 - price.input_share * _HIT_CEILING * price.discount,
                input_share=price.input_share,
                provenance=price.provenance,
                share_provenance=price.share_provenance,
            )
        )
    return rows


def _panel_extent(rows: list[ModelCache]) -> float:
    """The rightmost drawn x, so the axis follows the data instead of a fixed 1.12."""
    # Correcting `input_share` from a price-ratio proxy to the measured token mix moved the
    # markers from ~0.85 down to ~0.25, which left a fixed 0..1.12 axis 60% empty. Deriving
    # the limit means the next price change rescales the panel instead of hiding under it.
    drawn = [r.share_at_hit_floor for r in rows]
    drawn += [r.billed_share for r in rows if r.billed_share == r.billed_share]
    return max(drawn) if drawn else 1.0


def _draw_models(ax: Axes, rows: list[ModelCache]) -> None:
    ys = list(range(len(rows)))[::-1]
    extent = _panel_extent(rows)
    xmax = extent * (1.0 + _GUTTER)
    for y, row in zip(ys, rows, strict=True):
        has_bill = row.billed_share == row.billed_share
        # The hit-rate band is UNCERTAINTY, not a peer series, so it is drawn subordinate:
        # thin, translucent, and under everything. Drawn at full weight it is three times
        # longer than the residual the figure is about, and it buries the claim.
        # Offset below the bar rather than through it: at the corrected input share the
        # 100%-hit cap falls INSIDE the bar on five of six models, so an on-axis rule would
        # hide the end the legend names.
        band_y = y - _BAND_OFFSET
        ax.plot(
            [row.share_at_hit_ceiling, row.share_at_hit_floor],
            [band_y, band_y],
            color=_UNCERTAINTY,
            lw=1.0,
            alpha=0.5,
            solid_capstyle="butt",
            zorder=1,
        )
        for cap in (row.share_at_hit_ceiling, row.share_at_hit_floor):
            ax.plot([cap], [band_y], "|", color=_UNCERTAINTY, ms=5, mew=1.2, alpha=0.6, zorder=1)
        if has_bill:
            ax.barh(y, row.billed_share, height=0.5, color=_BILLED, zorder=2)
        ax.plot([row.registry_share], [y], "D", color=_REGISTRY, ms=7, zorder=5)
        if has_bill:
            # Right-aligned value gutter. The old label sat at `billed + 0.015`, which is
            # exactly where the corrected green diamond now lands — a label on top of the
            # mark it describes. A fixed column cannot collide with data by construction.
            ax.text(
                xmax * 0.995,
                y,
                f"{row.billed_share:.2f} → {row.registry_share:.2f}  (n={row.n_rows})",
                fontsize=7.5,
                va="center",
                ha="right",
                color="#333333",
            )
    ax.set_yticks(ys)
    ax.set_yticklabels([r.model for r in rows], fontsize=8)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(
        "share of the list-price bill actually paid (lower = bigger discount)",
        fontsize=9,
        labelpad=18,
    )
    ax.barh([], [], color=_BILLED, label="measured: real_cost / estimated_cost")
    ax.plot([], [], "D", color=_REGISTRY, ms=6, label="registry cache price at the measured mix")
    ax.plot(
        [],
        [],
        color=_UNCERTAINTY,
        lw=1.4,
        marker="|",
        ms=5,
        mew=1.2,
        alpha=0.55,
        label=f"same price at hit rate {_HIT_FLOOR:.0%}–{_HIT_CEILING:.0%} (the one assumed input)",
    )
    # Below the axes, not inside it. At `center right` the frame sat on top of the gpt-5-mini
    # and kimi-k2.5 markers — every point in this panel is data, so there is no free interior.
    ax.legend(
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
    )
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · list price vs the invoice, against the cache model")


def _switch_curve(switches: list[float]) -> list[float]:
    """Share of the achievable cache saving that survives at each switch rate."""
    # One decision per session means a switch happens only BETWEEN sessions, so every
    # turn after the first reuses the prefix: saving_share = 1 - switch_rate.
    return [max(0.0, 1.0 - s) for s in switches]


def _draw_switch_tax(ax: Axes, rows: list[ModelCache]) -> None:
    xs = [i / 40.0 for i in range(41)]
    ys = _switch_curve(xs)
    mean_saving = sum(1.0 - r.registry_share for r in rows) / max(len(rows), 1)
    ax.plot(xs, [y * mean_saving * 100 for y in ys], color=_TAX, lw=2.0, zorder=3)
    ax.fill_between(xs, 0, [y * mean_saving * 100 for y in ys], color=_TAX, alpha=0.10, zorder=1)
    ax.axvline(0.0, color=_REGISTRY, lw=1.6, zorder=4)
    ax.annotate(
        "shipped invariant: one decision per session\n"
        "→ no mid-session switch, so the\n"
        "full modelled saving is retained",
        xy=(0.30, 0.93),
        xycoords="axes fraction",
        fontsize=7.5,
        color=_REGISTRY,
        va="top",
        ha="left",
    )
    ax.set_xlim(-0.02, 1.02)
    # Headroom for the callout, but never above 100: this axis is a percentage OF the bill,
    # and the corrected input share puts the curve's origin near 76% rather than near 17%.
    ax.set_ylim(0.0, min(mean_saving * 100 * 1.35, 100.0))
    ax.set_xlabel("share of turns that switch model inside a session", fontsize=9)
    ax.set_ylabel("cache saving retained (% of the bill)", fontsize=9)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · modelled switch tax (mean over the enabled models)")


def _annotations(rows: list[ModelCache]) -> Annotations:
    measured = [r for r in rows if r.provenance == cache_cost.MEASURED]
    measured_share = [r for r in rows if r.share_provenance == cache_cost.MEASURED]
    billed = [r for r in rows if r.billed_share == r.billed_share]
    mean_billed = sum(r.billed_share for r in billed) / len(billed) if billed else float("nan")
    mean_modelled = sum(r.registry_share for r in rows) / max(len(rows), 1)
    residuals = [r.residual for r in billed]
    facts = [
        f"{len(billed)} models, {sum(r.n_rows for r in billed)} priced rows",
        f"cache-read price measured for {len(measured)}/{len(rows)} models and input share for "
        f"{len(measured_share)}/{len(rows)}; hit rate assumed at "
        f"{cache_cost.ASSUMED_CACHE_HIT_RATE:.0%}",
        f"mean billed share {mean_billed:.2f} of list price vs a modelled {mean_modelled:.2f} "
        f"— residual {min(residuals):+.2f} to {max(residuals):+.2f} per model"
        if residuals
        else f"mean modelled share {mean_modelled:.2f}",
    ]
    notes = [
        f"{r.model}: billed {r.billed_share:.3f} (n={r.n_rows}), registry predicts "
        f"{r.registry_share:.3f} [discount {r.provenance}, input share "
        f"{r.input_share:.3f} {r.share_provenance}], hit-rate band "
        f"{r.share_at_hit_ceiling:.3f}–{r.share_at_hit_floor:.3f}"
        for r in rows
    ]
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=(
            "Agreement in size is not calibration: blue mixes every discount, "
            "green models caching alone."
        ),
        notes=tuple(notes),
        counts=(("models", len(rows)), ("priced_rows", sum(r.n_rows for r in rows))),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw cache_economics.png, or return None when no priced row exists."""
    rows = model_rows(ctx)
    if not any(r.n_rows for r in rows):
        return None
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.25, 1.0))
    _draw_models(axes[0], rows)
    _draw_switch_tax(axes[1], rows)
    plot_style.fit_end_labels(axes[0])
    return plot_frame.save(
        fig,
        ctx.out_dir / "cache_economics.png",
        SPEC,
        extra=_annotations(rows),
        provenance=ctx.provenance(__name__),
        size=size,
    )
