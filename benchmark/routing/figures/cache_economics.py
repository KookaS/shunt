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
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.runner import kill_gate as gate

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_BILLED = "#0072B2"
_REGISTRY = "#009E73"
_LEGACY = "#8a8a8a"
_TAX = "#D55E00"

SPEC = FigureSpec(
    title="The gate's cache model explains a fraction of the list-price-to-bill gap",
    reading=(
        "Left: per model, the share of the list-price bill that was actually charged. The "
        "blue bar is MEASURED — every row's real_cost over its estimated_cost in "
        "results.csv, which is the whole gap between list price and the invoice. The green "
        "marker is the share the registry's cache-read price predicts would survive, and "
        "the grey marker is the flat hit=0.9 x discount=0.5 the gate used to assume for "
        "every provider alike. Lower is cheaper. Right: the switch tax — how much of the "
        "MODELLED cache saving survives as the router changes model inside one session."
    ),
    goal=(
        "Read the DISTANCE between the blue bar and the green marker. It is the part of the "
        "billing gap the gate's cache model does not explain, and it is large: the invoice "
        "is far below list price for reasons caching alone cannot account for. The gate's "
        "cache-aware ratio is therefore a model, not a reconciliation. Then read the right "
        "panel at the shipped operating point — one decision per session."
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
        "ratio, so only the DISCOUNT is measured. Every marker carries that flag.",
        "The blue bar and the markers are different quantities and are drawn on one axis "
        "deliberately, to show how much of the gap the cache model leaves unexplained. It "
        "is not a calibration check and a matching pair would not validate the model.",
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
    legacy_share: float
    provenance: str


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
    prices = gate.cache_prices(ctx.models_by_price)
    rows: list[ModelCache] = []
    for model in ctx.models_by_price:
        share, n = billed.get(model, (float("nan"), 0))
        price = prices[model]
        legacy = 1.0 - price.input_share * gate.ASSUMED_CACHE_HIT_RATE * gate.ASSUMED_CACHE_DISCOUNT
        rows.append(
            ModelCache(
                model=model,
                billed_share=share,
                n_rows=n,
                registry_share=1.0 - price.saving_fraction,
                legacy_share=legacy,
                provenance=price.provenance,
            )
        )
    return rows


def _draw_models(ax: Axes, rows: list[ModelCache]) -> None:
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        if row.billed_share == row.billed_share:
            ax.barh(y, row.billed_share, height=0.5, color=_BILLED, zorder=2)
            ax.text(
                row.billed_share + 0.015,
                y,
                f"{row.billed_share:.2f}  (n={row.n_rows})",
                fontsize=7.5,
                va="center",
                color="#333333",
            )
        ax.plot([row.registry_share], [y], "D", color=_REGISTRY, ms=7, zorder=4)
        ax.plot([row.legacy_share], [y], "x", color=_LEGACY, ms=7, mew=1.6, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.model for r in rows], fontsize=8)
    ax.set_xlim(0.0, 1.12)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(
        "share of the list-price bill actually paid (lower = bigger discount)",
        fontsize=9,
        labelpad=18,
    )
    ax.barh([], [], color=_BILLED, label="measured: real_cost / estimated_cost")
    ax.plot([], [], "D", color=_REGISTRY, ms=6, label="registry cache price (measured discount)")
    ax.plot(
        [], [], "x", color=_LEGACY, ms=6, mew=1.6, label="old flat assumption (hit .9 × disc .5)"
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


def _switch_curve(rows: list[ModelCache], switches: list[float]) -> list[float]:
    """Share of the achievable cache saving that survives at each switch rate."""
    # One decision per session means a switch happens only BETWEEN sessions, so every
    # turn after the first reuses the prefix: saving_share = 1 - switch_rate.
    return [max(0.0, 1.0 - s) for s in switches]


def _draw_switch_tax(ax: Axes, rows: list[ModelCache]) -> None:
    xs = [i / 40.0 for i in range(41)]
    ys = _switch_curve(rows, xs)
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
    ax.set_ylim(0.0, mean_saving * 100 * 1.35)
    ax.set_xlabel("share of turns that switch model inside a session", fontsize=9)
    ax.set_ylabel("cache saving retained (% of the bill)", fontsize=9)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · modelled switch tax (mean over the enabled models)")


def _annotations(rows: list[ModelCache]) -> Annotations:
    measured = [r for r in rows if r.provenance == gate.MEASURED]
    billed = [r for r in rows if r.billed_share == r.billed_share]
    mean_billed = sum(r.billed_share for r in billed) / len(billed) if billed else float("nan")
    facts = [
        f"{len(billed)} models, {sum(r.n_rows for r in billed)} priced rows",
        f"cache-read price measured for {len(measured)}/{len(rows)} models; hit rate assumed "
        f"at {gate.ASSUMED_CACHE_HIT_RATE:.0%}",
        f"mean billed share {mean_billed:.2f} of list price vs a modelled "
        f"{sum(r.registry_share for r in rows) / max(len(rows), 1):.2f}",
    ]
    notes = [
        f"{r.model}: billed {r.billed_share:.3f} (n={r.n_rows}), registry predicts "
        f"{r.registry_share:.3f} [{r.provenance}], old flat assumption {r.legacy_share:.3f}"
        for r in rows
    ]
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=(
            "Blue is the whole list-to-bill gap; the markers model caching only — "
            "different quantities, drawn together on purpose."
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
