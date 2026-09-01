#!/usr/bin/env python3
"""cost_quality_headline.png — the same plane as its parent, at four points and one claim."""

# A COMPANION to cost_quality_frontier.png, not a replacement. That figure is the audit
# instrument: twelve strategies, three stacked magnifications, intervals, a mixture hull, a
# deployability shape vocabulary and a red caveat band. It is the right figure for
# docs/routing.md and the wrong one for a front page — a cold read of it landed on
# Always-Cheap (the cheapest point, and the worst) and concluded the router is cheaper AND
# worse, which is the opposite of what the plane says.
#
# So this one carries ONE claim and pays for it with everything else: four points, direct
# labels, no legend, no intervals, no hull, no mixture region, no context brackets, no shape
# taxonomy beyond the one mark that says the bound is not for sale.
#
# ONE DELIBERATE DEPARTURE FROM THE PARENT: the x axis is LINEAR. The parent spans two
# decades over twelve strategies and needs a log axis to keep five of them out of the left
# margin. Four points across $1.50-$96 separate perfectly well on a linear scale, and a log
# axis is exactly what makes a lay reader misjudge the gap the figure exists to show — on a
# log axis the $67 the router saves looks smaller than the $17 between the cheap point and
# the bound. The y axis still starts above zero, and says so on its own label.
#
# It reads the DERIVED strategy summary rather than re-deriving the strategy rows, so it
# renders in a second from committed data and cannot disagree with the table its parent is
# scored from. Every number on the canvas comes from that CSV; none is written down here.

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from benchmark import plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import plot_style  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.plot_style import usd  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

# `python -m` sets __name__ to "__main__", which would land in the figure manifest
# instead of the module that drew it.
_GENERATOR: Final = "benchmark.routing.scripts.cost_quality_headline"

FIGURE_NAME: Final = "cost_quality_headline.png"
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_OUT: Final = _REPO_ROOT / "docs/assets/figures/routing" / FIGURE_NAME
DEFAULT_SUMMARY: Final = _REPO_ROOT / "benchmark/routing/reports/strategy_summary.csv"

# The four rows, and the ROLE each one plays in the claim. Two names are resolved from the
# figure package rather than spelled here, so moving the shipped default or the baseline is
# one edit in one place and cannot leave this figure quoting the previous one.
HERO: Final = ctxmod.DEFAULT_STRATEGY
BASELINE: Final = ctxmod.BASELINE_STRATEGY
# The cheap floor and the hindsight bound. Neither is resolvable from the figure package:
# `Always-Cheap` is the always-cheapest control point the plane's left edge is anchored on,
# and `Oracle` is the reward-aware bound every other figure in the set names the same way.
CHEAP: Final = "Always-Cheap"
BOUND: Final = "Oracle"

# Okabe-Ito, two hues only: the two points being compared own the colour channel, and the
# two context points are greys. A four-hue palette would have spent colour on the fact that
# there are four points, which the reader can already see.
_HERO_COLOR: Final = plot_style.OKABE_ITO[0]  # blue
_BASELINE_COLOR: Final = plot_style.OKABE_ITO[3]  # vermillion
_CONTEXT_COLOR: Final = "#6b6b6b"
_RULER_COLOR: Final = "#8a8a8a"
_GRID: Final = "#e1e0d9"

_LABEL_PT: Final = 9.5
_RULER_Y: Final = 79.0
# The truncated y-axis is deliberate and is stated on the axis label. It is a floor
# for READABILITY, never a filter: `_y_floor` lowers it whenever a plotted point sits
# beneath, because a point outside the limits is drawn nowhere and says nothing. A
# ladder reaching down to sub-1B models routinely lands well below 72%.
_Y_FLOOR_DEFAULT: Final = 72.0
_Y_FLOOR_PAD: Final = 3.0
_Y_CEIL: Final = 101.5


class SummaryMissingError(FileNotFoundError):
    """The derived strategy summary this figure reads is not on disk."""


def _rows(path: Path) -> dict[str, dict[str, str]]:
    """Every strategy row of the derived summary, keyed by strategy name."""
    if not path.exists():
        raise SummaryMissingError(
            f"{path} is missing — run `make routing-report` to derive it from results.csv"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["strategy"]: row for row in csv.DictReader(handle)}


def _digest(rows: dict[str, dict[str, str]], names: tuple[str, ...]) -> str:
    """A fingerprint of the four rows actually plotted, not of the whole file."""
    # Over the plotted VALUES: a summary re-derived with an extra strategy, or with a column
    # this figure never reads, must not age a figure whose four points did not move.
    blob = "|".join(
        f"{name}:{rows[name]['TotalCost_cacheaware']}:{rows[name]['AvgPerf%']}:"
        f"{rows[name]['n_tasks']}"
        for name in names
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _point(rows: dict[str, dict[str, str]], name: str) -> tuple[float, float]:
    """One strategy's (cache-aware total cost, pass rate) — the plane's two axes."""
    row = rows[name]
    return float(row["TotalCost_cacheaware"]), float(row["AvgPerf%"])


SPEC = FigureSpec(
    title="Shunt's default matches the frontier baseline's quality at a third of the bill",
    subtitle="",  # every fact on the band is derived; see `_annotations`
    caveat=(
        "Oracle is a hindsight bound, not a setting anyone can buy — and the scored tasks are "
        "coverage-selected, not random."
    ),
    reading=(
        "One plane, four points, no legend — each point is named where it sits. Left to right "
        "is the total dollars a strategy spent over the whole scored task set, on a LINEAR "
        "axis; bottom to top is the share of tasks it passed. Cheap is left, good is up, so "
        "the best place to be is the top-left corner. The blue point is what Shunt routes "
        "with by default. The orange point is the baseline it has to beat: send everything to "
        "the strongest enabled frontier model. The grey circle at the far left is the "
        "opposite extreme — send everything to the cheapest model. The grey STAR is a bound, "
        "not a product: it is what a router that already knew each task's outcome would have "
        "paid, and no configuration reproduces it. The measuring bar across the bottom is the "
        "figure's whole point: it spans the horizontal distance between the blue point and "
        "the orange one, which is the money the router does not spend."
    ),
    goal=(
        "Read the measuring bar, then check the two heights it connects. The blue point sits "
        "far to the LEFT of the orange one, at the same height or above it — cheaper for the "
        "same work is the claim, and a router that landed below and to the right of the "
        "baseline would have failed."
    ),
    definitions=(
        ("pass rate", "share of the scored tasks a strategy solved"),
        (
            "cache-aware cost",
            "what a deployment is billed once a repeat of the same model on a consecutive "
            "attempt is charged at the provider's cache-read rate rather than full input price",
        ),
        (
            "hindsight bound",
            "the price of choosing correctly with the answers already known — a floor on what "
            "any router could cost, never a setting an operator can select",
        ),
    ),
    limitations=(
        "This is FOUR of the strategies the benchmark scores. The full plane — every strategy, "
        "its interval, the mixture region a router has to clear, and which rows are not "
        "selectable at all — is cost_quality_frontier.png, and this figure asserts nothing "
        "the parent does not.",
        "The y axis starts above zero and is labelled with the range it shows: the four points "
        "span about twenty points of pass rate, which on a 0-100 axis is a flat line.",
        "The cache-aware x position rests on an ASSUMED cache hit rate; only the per-model "
        "discount and input share are measured — see cache_economics.png for the range that "
        "assumption spans.",
        "Pass rates are scored on the coverage-completed matrix, whose imputed cells are all "
        "pass=True — see evidence_basis.png for how much of each strategy's number that is.",
        "The scored set is chosen by coverage, not at random: the collector runs the expensive "
        "tier only on the discriminating slice, so both axes describe a difficulty-biased "
        "sample.",
    ),
)


def _label(  # noqa: PLR0913 (one argument per placement degree of freedom)
    ax: Axes,
    xy: tuple[float, float],
    text: str,
    offset: tuple[float, float],
    ha: str,
    va: str,
    color: str,
    weight: str = "normal",
) -> None:
    """One direct label beside its point — this figure has no legend to fall back on."""
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        fontweight=weight,
        fontsize=_LABEL_PT,
        zorder=6,
    )


def _draw_ruler(ax: Axes, hero: tuple[float, float], baseline: tuple[float, float]) -> None:
    """The measuring bar: the horizontal distance the figure exists to show."""
    # Drawn in its own empty lane under the data with a dotted extension line from each of the
    # two markers, rather than as an arrow between them. The two points differ by 1.6 points of
    # pass rate, so an arrow drawn marker-to-marker is not horizontal — and "horizontal
    # distance at the same height" is exactly the reading this figure is for.
    for (x, y), colour in ((hero, _HERO_COLOR), (baseline, _BASELINE_COLOR)):
        ax.plot(
            [x, x],
            [_RULER_Y, y],
            color=colour,
            ls=":",
            lw=1.1,
            alpha=0.55,
            zorder=2,
        )
    ax.annotate(
        "",
        xy=(hero[0], _RULER_Y),
        xytext=(baseline[0], _RULER_Y),
        arrowprops={"arrowstyle": "<|-|>", "color": _RULER_COLOR, "lw": 1.7},
        zorder=3,
    )
    saving = baseline[0] - hero[0]
    ax.annotate(
        f"{usd(saving)} not spent\nno measurable loss in pass rate",
        xy=((hero[0] + baseline[0]) / 2.0, _RULER_Y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=plot_frame.INK,
        zorder=6,
    )


def _y_floor(points: Sequence[tuple[float, float]]) -> float:
    """The lowest y the axis must show: the default, or below every plotted point."""
    lowest = min(y for _x, y in points)
    if lowest >= _Y_FLOOR_DEFAULT + _Y_FLOOR_PAD:
        return _Y_FLOOR_DEFAULT
    return max(0.0, lowest - _Y_FLOOR_PAD)


def _draw(ax: Axes, rows: dict[str, dict[str, str]]) -> None:
    """The whole canvas: four marks, four names, one measuring bar."""
    hero, baseline = _point(rows, HERO), _point(rows, BASELINE)
    cheap, bound = _point(rows, CHEAP), _point(rows, BOUND)

    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID, linewidth=0.6)

    _draw_ruler(ax, hero, baseline)

    ax.scatter(*cheap, s=120, color=_CONTEXT_COLOR, zorder=4)
    ax.scatter(
        *bound, s=420, marker="*", facecolor="none", edgecolor=_CONTEXT_COLOR, lw=1.8, zorder=4
    )
    ax.scatter(*baseline, s=200, color=_BASELINE_COLOR, zorder=5)
    ax.scatter(*hero, s=330, color=_HERO_COLOR, zorder=5, edgecolor="white", lw=1.4)

    _label(
        ax,
        bound,
        f"Oracle — perfect hindsight, NOT PURCHASABLE\n{usd(bound[0])} · {bound[1]:.1f}% passed",
        (0, 16),
        "center",
        "bottom",
        _CONTEXT_COLOR,
    )
    _label(
        ax,
        hero,
        f"SHUNT (routed)\n{usd(hero[0])} · {hero[1]:.1f}% passed",
        (15, -6),
        "left",
        "top",
        _HERO_COLOR,
        weight="bold",
    )
    _label(
        ax,
        baseline,
        f"Frontier model on everything\n{usd(baseline[0])} · {baseline[1]:.1f}% passed",
        (0, 14),
        "right",
        "bottom",
        _BASELINE_COLOR,
        weight="bold",
    )
    _label(
        ax,
        cheap,
        f"Cheap model on everything\n{usd(cheap[0])} · {cheap[1]:.1f}% passed",
        (14, 0),
        "left",
        "center",
        _CONTEXT_COLOR,
    )

    floor = _y_floor((hero, baseline, cheap, bound))
    ax.set_xlim(-3.0, baseline[0] * 1.10)
    ax.set_ylim(floor, _Y_CEIL)
    ax.xaxis.set_major_locator(MultipleLocator(20.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:,.0f}"))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.set_xlabel(
        "total spent over the scored task set (USD, cache-aware) — cheaper is left",
        fontsize=10,
        color=plot_frame.INK,
    )
    ax.set_ylabel(
        f"tasks passed (%) — axis starts at {floor:.0f}%, not 0",
        fontsize=10,
        color=plot_frame.INK,
    )


def _annotations(rows: dict[str, dict[str, str]]) -> Annotations:
    """The band's derived facts, plus the per-point record the docs section quotes."""
    hero, baseline = _point(rows, HERO), _point(rows, BASELINE)
    n_tasks = int(rows[HERO]["n_tasks"])
    share = 100.0 * hero[0] / baseline[0]
    return Annotations(
        subtitle_facts=(
            f"4 of {len(rows)} scored strategies, {n_tasks} scored tasks",
            f"{HERO} {usd(hero[0])} at {hero[1]:.1f}% vs {BASELINE} {usd(baseline[0])} "
            f"at {baseline[1]:.1f}% — {share:.0f}% of the bill",
            "cache-aware cost on a LINEAR axis; the full twelve-strategy plane is "
            "cost_quality_frontier.png",
        ),
        notes=tuple(
            f"{name}: {usd(_point(rows, name)[0])} cache-aware, "
            f"{_point(rows, name)[1]:.2f}% passed, n={rows[name]['n_tasks']} ({role})"
            for name, role in (
                (BOUND, "hindsight bound — no router.strategy value reproduces it"),
                (HERO, "the shipped router.strategy default"),
                (BASELINE, "the baseline the kill gate is measured against"),
                (CHEAP, "the cheap floor"),
            )
        )
        + (
            "The four rows are read from the derived strategy summary at render time, so this "
            "figure and cost_quality_frontier.png cannot quote different numbers for one "
            "strategy.",
            "No interval is drawn. This figure states an ordering, not a precision — the "
            "intervals, the mixture region and the eight strategies left out are in "
            "cost_quality_frontier.png.",
        ),
        counts=(("strategies_drawn", 4), ("tasks", n_tasks)),
    )


def plot(rows: dict[str, dict[str, str]], out_path: Path, digest: str) -> Path:
    """Draw cost_quality_headline.png through the shared frame."""

    def draw(ax: Axes) -> Annotations:
        _draw(ax, rows)
        return _annotations(rows)

    return plot_frame.render(
        out_path,
        SPEC,
        draw,
        size=plot_frame.SINGLE,
        provenance=plot_frame.Provenance(_GENERATOR, digest, _REPO_ROOT / ctxmod.MANIFEST),
    )


def main() -> None:
    """Render the headline cost/quality figure from the derived strategy summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = _rows(args.summary)
    names = (BOUND, HERO, BASELINE, CHEAP)
    missing = [name for name in names if name not in rows]
    if missing:
        raise SystemExit(f"{args.summary} has no row for: {', '.join(missing)}")
    plot(rows, args.out, _digest(rows, names))
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
