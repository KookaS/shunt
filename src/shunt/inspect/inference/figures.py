"""Draw functions for the inference figure family — one function per figure, no arithmetic."""

# Everything numeric lives in `data.py`; everything textual in `specs.py`. What is left here is
# geometry and colour, so a figure cannot quietly compute a different number from the one its
# tests assert.
#
# EMPTINESS IS A RESULT. On a seed-only corpus F2 and F6 are entirely empty, F3's live claim and
# F5's live series are empty, and F4's panel C is empty. Every one of those panels draws
# `_empty` with the count that explains it. A blank axes would read as a rendering bug and a
# zero-height bar would read as a measurement; neither is what happened.

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from matplotlib import dates as mdates
from matplotlib.patches import Patch, Rectangle

from shunt.inspect import plot_frame
from shunt.inspect.inference import data as idata
from shunt.inspect.inference import specs
from shunt.inspect.inference.estimators import ESTIMATORS, InstrumentInadmissibleError
from shunt.inspect.plot_frame import Annotations, FigureSize, FigureSpec, Provenance
from shunt.inspect.plot_style import (
    MIN_N_PROVISIONAL,
    OKABE_ITO,
    ci_yerr,
    is_provisional,
    model_color_map,
    usd,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

_SEED_GREY: Final[str] = "#9e9e9e"
_PROVISIONAL_HATCH: Final[str] = "///"
_ALARM: Final[str] = "#B71C1C"
_EMPTY_WRAP: Final[int] = 34
_BAR: Final[float] = 0.38


def _spec(text: specs.FigureText) -> FigureSpec:
    """The frame's spec built from the matplotlib-free record, which is the only source."""
    return FigureSpec(
        title=text.title,
        subtitle=text.subtitle,
        caveat=text.caveat,
        reading=text.reading,
        goal=text.goal,
        definitions=text.definitions,
        notes=text.notes,
        limitations=text.limitations,
    )


def _frame(ax: Axes) -> None:
    """A dashed panel outline with no axes furniture, for a panel carrying prose."""
    # The dashed frame is not decoration. Without it an all-empty figure leaves most of the
    # canvas uninked, which measures as an unfilled canvas and reads as a figure that failed
    # to draw rather than one whose answer is "none". The frame marks the panel's extent, so
    # the reader sees a panel that exists and is empty.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#cccccc",
            linestyle=(0, (6, 6)),
            linewidth=1.0,
        )
    )


def _empty(ax: Axes, message: str) -> None:
    """Say what is absent and how much of it. A blank axes reads as a broken render."""
    _frame(ax)
    ax.text(
        0.5,
        0.5,
        "\n".join(textwrap.wrap(message, _EMPTY_WRAP)),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=plot_frame.MUTED,
        style="italic",
    )


# `matplotlib.dates` ships no annotations, so every call below is untyped in a --strict
# context. The alternative — plotting POSIX seconds against a numeric axis — puts an epoch
# integer under a time series, which is unreadable. Coded ignores on three known-good calls.
def _dates(values: list[datetime]) -> list[float]:
    """Matplotlib's float date form, so a time axis survives a --strict typed call site."""
    return [float(mdates.date2num(value)) for value in values]  # type: ignore[no-untyped-call]


def _date_axis(ax: Axes) -> None:
    """Give an axis carrying `_dates` values real date ticks rather than float ordinals."""
    locator = mdates.AutoDateLocator()  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore[no-untyped-call]
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)


def _headroom(ax: Axes, fraction: float = 0.22) -> None:
    """Room above the tallest bar so an in-axes legend never sits on the data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * fraction)


def _bar_labels(ax: Axes, labels: list[str]) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


# ------------------------------------------------------------------- F1 strata


def draw_strata(out_dir: Path, view: idata.StrataData, provenance: Provenance | None) -> Path:
    """F1 — the two strata sharing one corpus, side by side at every lifecycle stage."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3)
    _strata_stages(axes[0], view)
    _strata_times(axes[1], view)
    _strata_models(axes[2], view)
    # Counted from the adjudicated strata, not from the census: the census partitions on the
    # id prefix alone, so an ambiguous row is inside one of its two `stored` figures AND in
    # `ambiguous`, and summing the three claimed more sessions than the corpus holds.
    extra = Annotations(
        subtitle_facts=(
            f"seeded n={view.n_seeded}",
            f"live n={view.n_live}",
            f"ambiguous n={len(view.ambiguous)}",
        ),
        caveat=_strata_caveat(view),
        counts=(
            ("sessions", view.n_sessions),
            ("seeded", view.n_seeded),
            ("live", view.n_live),
            ("ambiguous", len(view.ambiguous)),
        ),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.STRATA.filename,
        _spec(specs.STRATA),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _strata_caveat(view: idata.StrataData) -> str | None:
    """The red line for F1: rows in neither stratum, rows panel A misfiles, stages that invert."""
    # Kept to one short clause per defect because `plot_frame` hard-caps the caveat and a
    # caveat cut at the limit ships half a sentence. The nesting clause names only the first
    # inverting pair and counts the rest; the full list is the panel-A stage note in the docs.
    parts: list[str] = []
    if view.ambiguous:
        parts.append(f"{len(view.ambiguous)} sessions in neither stratum")
    if view.n_prefix_disagree:
        parts.append(f"panel A is id-prefix derived and misfiles {view.n_prefix_disagree}")
    if view.nesting_breaks:
        first, rest = view.nesting_breaks[0], len(view.nesting_breaks) - 1
        more = f" +{rest} more" if rest else ""
        parts.append(f"stages are not nested — {first}{more}")
    return "; ".join(parts) if parts else None


def _strata_stages(ax: Axes, view: idata.StrataData) -> None:
    stages = idata.CENSUS_STAGES
    for offset, (funnel, colour) in enumerate(
        ((view.census.seeded, _SEED_GREY), (view.census.live, OKABE_ITO[1]))
    ):
        values = [getattr(funnel, stage) for stage in stages]
        ax.bar(
            [i + (offset - 0.5) * _BAR for i in range(len(stages))],
            values,
            width=_BAR,
            color=colour,
            label=funnel.stratum,
            hatch=_PROVISIONAL_HATCH if offset == 0 else None,
            edgecolor="white",
        )
    _bar_labels(ax, list(stages))
    ax.set_ylabel("sessions")
    _headroom(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    # NOT "funnel": `embedded` and `indexed` are the only stages structurally contained in
    # `stored`. `labeled` is counted off `outcome_events` and `tier2` off the `outcomes`
    # view, so the five are five counts over one population, not a chain of survivors.
    plot_frame.panel_label(ax, "A · lifecycle stage counts by stratum")


def _strata_times(ax: Axes, view: idata.StrataData) -> None:
    rows = {idata.SEEDED: 0.0, idata.LIVE: 1.0, idata.AMBIGUOUS: 2.0}
    colours = {idata.SEEDED: _SEED_GREY, idata.LIVE: OKABE_ITO[1], idata.AMBIGUOUS: _ALARM}
    if not view.times:
        _empty(ax, "no timestamped sessions in this corpus")
        plot_frame.panel_label(ax, "B · arrival times")
        return
    for stratum, y in rows.items():
        xs = _dates([when for name, when in view.times if name == stratum])
        ax.scatter(xs, [y] * len(xs), s=180, alpha=0.45, color=colours[stratum], marker="|")
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows), fontsize=8)
    ax.set_ylim(-0.6, 2.6)
    stamps = {when for _s, when in view.times}
    if len(stamps) == 1:
        # A corpus imported at ONE deterministic stamp otherwise gets an auto date axis spanning
        # years around a single invisible tick, which reads as a broken render rather than as
        # the finding: every row arrived at the same instant. Keyed on the timestamps being
        # IDENTICAL, not merely close — a busy day of live traffic is not a seeded burst.
        stamp = next(iter(stamps))
        ax.set_xlim(*_dates([stamp - timedelta(days=1), stamp + timedelta(days=1)]))
        ax.annotate(
            f"every session shares one stamp\n{stamp.date().isoformat()}",
            (0.5, 0.86),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            color=plot_frame.MUTED,
        )
    _date_axis(ax)
    ax.set_xlabel("session timestamp")
    plot_frame.panel_label(ax, "B · arrival times (seeded import is one burst)")


def _strata_models(ax: Axes, view: idata.StrataData) -> None:
    if not view.per_model:
        _empty(ax, "no labeled sessions in this corpus")
    else:
        names = [model for model, _s, _l in view.per_model]
        positions = range(len(names))
        ax.barh(
            [p + _BAR / 2 for p in positions],
            [s for _m, s, _l in view.per_model],
            height=_BAR,
            color=_SEED_GREY,
            hatch=_PROVISIONAL_HATCH,
            edgecolor="white",
            label="seeded",
        )
        ax.barh(
            [p - _BAR / 2 for p in positions],
            [live for _m, _s, live in view.per_model],
            height=_BAR,
            color=OKABE_ITO[1],
            edgecolor="white",
            label="live",
        )
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("labeled sessions")
        ax.legend(fontsize=8, frameon=False)
    plot_frame.panel_label(ax, "C · labeled sessions per model")


# --------------------------------------------------------------------- F2 cost


def draw_cost(out_dir: Path, view: idata.CostData, provenance: Provenance | None) -> Path:
    """F2 — live spend only. On a seed-only corpus this is the flagship honest-empty figure."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3)
    _cost_by_model(axes[0], view)
    _cost_coverage(axes[1], view)
    _cost_cumulative(axes[2], view)
    whole = next((agg for label, agg in view.windows if label == "all"), None)
    # No cost fact when there is no live traffic: printing `live cost $0.0000` directly above a
    # caveat reading "empty, not zero" is the figure contradicting itself on its own canvas.
    cost_fact = (f"live cost {usd(whole.total if whole else 0.0, 4)}",) if view.n_live else ()
    extra = Annotations(
        subtitle_facts=(
            f"seeded rows excluded (n={view.n_seeded_excluded})",
            f"live sessions n={view.n_live}",
            *cost_fact,
        ),
        caveat=(
            "no live sessions in this corpus — every panel is empty, not zero"
            if view.n_live == 0
            else None
        ),
        counts=(
            ("live_sessions", view.n_live),
            ("seeded_excluded", view.n_seeded_excluded),
            ("cost_unknown", whole.n_cost_unknown if whole else 0),
        ),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.COST.filename,
        _spec(specs.COST),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _cost_by_model(ax: Axes, view: idata.CostData) -> None:
    models = sorted({m for _label, agg in view.windows for m, _n, _t in agg.by_model})
    if not models:
        _empty(ax, f"no live sessions in this corpus (n={view.n_live}) — nothing to cost")
    else:
        colours = model_color_map(models)
        width = 0.8 / len(view.windows)
        for offset, (label, agg) in enumerate(view.windows):
            totals = dict((m, t) for m, _n, t in agg.by_model)
            ax.bar(
                [i + offset * width - 0.4 for i in range(len(models))],
                [totals.get(m, 0.0) for m in models],
                width=width,
                label=label,
                color=[colours[m] for m in models],
                alpha=1.0 - 0.25 * offset,
                edgecolor="white",
            )
        _bar_labels(ax, models)
        ax.set_ylabel("live inference cost (USD)")
        ax.legend(fontsize=8, frameon=False, title="window", title_fontsize=8)
    plot_frame.panel_label(ax, "A · live cost per model by window")


def _cost_coverage(ax: Axes, view: idata.CostData) -> None:
    labels = [label for label, _agg in view.windows]
    known = [agg.n_cost_known for _label, agg in view.windows]
    unknown = [agg.n_cost_unknown for _label, agg in view.windows]
    if not any(known) and not any(unknown):
        _empty(ax, "no live sessions, so cost coverage is undefined (0 known, 0 unknown)")
    else:
        ax.bar(labels, known, color=OKABE_ITO[2], label="cost reported", edgecolor="white")
        ax.bar(
            labels,
            unknown,
            bottom=known,
            color=_ALARM,
            label="cost unknown",
            hatch=_PROVISIONAL_HATCH,
            edgecolor="white",
        )
        ax.set_ylabel("live sessions")
        ax.legend(fontsize=8, frameon=False)
    # The windows NEST (7d is inside 30d is inside all), so summing them counts the same session
    # up to three times. The widest window is the store's actual unknown count.
    whole_unknown = unknown[-1] if unknown else 0
    plot_frame.panel_label(ax, f"B · cost coverage (unknown = {whole_unknown})")


def _cost_cumulative(ax: Axes, view: idata.CostData) -> None:
    if not view.cumulative:
        _empty(ax, "no live spend recorded in this corpus")
    else:
        ax.plot(
            _dates([when for when, _total in view.cumulative]),
            [total for _when, total in view.cumulative],
            color=OKABE_ITO[1],
            linewidth=1.6,
        )
        ax.set_ylabel("cumulative live cost (USD)")
        _date_axis(ax)
    plot_frame.panel_label(ax, "C · cumulative live spend")


# ----------------------------------------------------------- F3 unit economics


def draw_unit_economics(
    out_dir: Path, view: idata.UnitEconomicsData, provenance: Provenance | None
) -> Path:
    """F3 — live cost per verified success against the seeded reference band."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2)
    _economics_rate(axes[0], view)
    _economics_cost(axes[1], view)
    live_labeled = sum(row.n_labeled for row in view.live)
    extra = Annotations(
        subtitle_facts=(
            f"seeded models n={len(view.seeded)}",
            f"live labeled sessions n={live_labeled}",
        ),
        counts=(
            ("seeded_labeled", sum(row.n_labeled for row in view.seeded)),
            ("live_labeled", live_labeled),
        ),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.UNIT_ECONOMICS.filename,
        _spec(specs.UNIT_ECONOMICS),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _economics_models(view: idata.UnitEconomicsData) -> list[str]:
    return sorted({row.model for row in (*view.seeded, *view.live)})


def _economics_rate(ax: Axes, view: idata.UnitEconomicsData) -> None:
    models = _economics_models(view)
    if not models:
        _empty(ax, "no labeled sessions in either stratum")
        plot_frame.panel_label(ax, "A · verified-success rate")
        return
    seeded = {row.model: row for row in view.seeded}
    live = {row.model: row for row in view.live}
    handles: list[Patch] = []
    for offset, (source, colour, label) in enumerate(
        ((seeded, _SEED_GREY, "seeded (replayed)"), (live, None, "live"))
    ):
        drawn = _rate_series(ax, models, source, colour, label, offset)
        # An absent stratum is named in the legend rather than dropped from it: a panel whose
        # legend silently loses "live" reads as a figure that only ever had one series.
        handles.append(
            Patch(
                facecolor=colour or OKABE_ITO[0],
                hatch=_PROVISIONAL_HATCH if colour else None,
                edgecolor="white",
                label=label if drawn else f"{label} (none in this corpus)",
                alpha=1.0 if drawn else 0.35,
            )
        )
    # Hatching already means "replayed seeded row" on every figure in this family, so it
    # cannot also mean "provisional" here — one channel, one meaning. Provisionality goes on
    # the tick label instead.
    _bar_labels(ax, [_provisional_mark(model, seeded, live) for model in models])
    ax.set_ylabel("verified-success rate")
    ax.set_ylim(0.0, 1.28)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    plot_frame.panel_label(
        ax, f"A · verified-success rate (hatch = replayed; * = n<{MIN_N_PROVISIONAL})"
    )


def _provisional_mark(
    model: str,
    seeded: dict[str, idata.ModelEconomics],
    live: dict[str, idata.ModelEconomics],
) -> str:
    """Append * where either stratum's cell rests on fewer than the provisional floor."""
    counts = [source[model].n_labeled for source in (seeded, live) if model in source]
    return f"{model} *" if any(is_provisional(n) for n in counts if n) else model


def _rate_series(
    ax: Axes,
    models: list[str],
    source: dict[str, idata.ModelEconomics],
    colour: str | None,
    label: str,
    offset: int,
) -> bool:
    """Draw one stratum's bars. Returns whether it had anything to draw."""
    present = [(i, source[m]) for i, m in enumerate(models) if m in source and source[m].n_labeled]
    if not present:
        return False
    palette = model_color_map(models)
    ax.bar(
        [i + (offset - 0.5) * _BAR for i, _row in present],
        [row.rate for _i, row in present],
        width=_BAR,
        color=colour if colour else [palette[models[i]] for i, _row in present],
        hatch=_PROVISIONAL_HATCH if colour else None,
        edgecolor="white",
        label=label,
        yerr=_yerr([row for _i, row in present]),
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "ecolor": plot_frame.INK},
    )
    return True


def _yerr(rows: list[idata.ModelEconomics]) -> list[list[float]]:
    pairs = [ci_yerr(row.rate, row.lo, row.hi) for row in rows]
    return [[lo for lo, _hi in pairs], [hi for _lo, hi in pairs]]


def _economics_cost(ax: Axes, view: idata.UnitEconomicsData) -> None:
    models = _economics_models(view)
    seeded = {row.model: row.cost_per_success for row in view.seeded}
    live = {row.model: row.cost_per_success for row in view.live}
    drawn = [m for m in models if seeded.get(m) is not None or live.get(m) is not None]
    if not drawn:
        _empty(ax, "no model has a verified success, so cost per success is undefined")
    else:
        palette = model_color_map(drawn)
        ax.barh(
            [i + _BAR / 2 for i in range(len(drawn))],
            [seeded.get(m) or 0.0 for m in drawn],
            height=_BAR,
            color=_SEED_GREY,
            hatch=_PROVISIONAL_HATCH,
            edgecolor="white",
            label="seeded (replayed)",
        )
        ax.barh(
            [i - _BAR / 2 for i in range(len(drawn))],
            [live.get(m) or 0.0 for m in drawn],
            height=_BAR,
            color=[palette[m] for m in drawn],
            edgecolor="white",
            label="live",
        )
        ax.set_yticks(range(len(drawn)))
        ax.set_yticklabels(drawn, fontsize=7)
        ax.set_ylim(-1.0, len(drawn) - 0.2)
        ax.set_xlabel("cost per verified success (USD)")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    live_n = sum(1 for m in models if live.get(m) is not None)
    plot_frame.panel_label(ax, f"B · cost per verified success (live models: {live_n})")


# ------------------------------------------------------------- F4 neighbourhood


def draw_neighbourhood(
    out_dir: Path, view: idata.NeighbourhoodData, provenance: Provenance | None
) -> Path:
    """F4 — does a near neighbour predict an outcome, and whose evidence is it?"""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3)
    _knn_reliability(axes[0], view)
    _knn_distances(axes[1], view)
    _knn_origin(axes[2], view)
    extra = Annotations(
        subtitle_facts=(
            f"k={view.k}",
            f"probed n={view.n_probed}",
            f"live decisions n={view.n_live_decisions}",
        ),
        caveat=(
            "no live decisions in this corpus — panel C is empty, not zero"
            if view.n_live_decisions == 0
            else None
        ),
        counts=(("probed", view.n_probed), ("live_decisions", view.n_live_decisions)),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.NEIGHBOURHOOD.filename,
        _spec(specs.NEIGHBOURHOOD),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _knn_reliability(ax: Axes, view: idata.NeighbourhoodData) -> None:
    if not view.bins:
        _empty(ax, "no indexed session has a labeled neighbour")
    else:
        ax.plot([0, 1], [0, 1], color=plot_frame.MUTED, linewidth=0.9, linestyle="--")
        ax.plot(
            [b.predicted for b in view.bins],
            [b.observed for b in view.bins],
            marker="o",
            color=OKABE_ITO[1],
            linewidth=1.5,
        )
        # Alternating sides: a fixed offset puts every label on the polyline it annotates,
        # and on a monotone curve they all land on the same side of it.
        for index, point in enumerate(view.bins):
            ax.annotate(
                str(point.n),
                (point.predicted, point.observed),
                textcoords="offset points",
                xytext=(0, 9) if index % 2 else (0, -14),
                ha="center",
                fontsize=7,
                color=plot_frame.MUTED,
            )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("mean neighbour success rate")
        ax.set_ylabel("realised success rate")
    plot_frame.panel_label(ax, "A · neighbour reliability (labels are bin n)")


def _knn_distances(ax: Axes, view: idata.NeighbourhoodData) -> None:
    if not view.distances:
        _empty(
            ax,
            f"no neighbour pairs: {view.n_probed} sessions were probed and none had a labeled "
            f"neighbour besides itself",
        )
    else:
        ax.hist(view.distances, bins=30, color=OKABE_ITO[2], edgecolor="white")
        ax.set_xlabel("neighbour distance")
        ax.set_ylabel("neighbour pairs")
    plot_frame.panel_label(ax, "B · neighbour distance distribution")


def _knn_origin(ax: Axes, view: idata.NeighbourhoodData) -> None:
    if not view.origin_mix:
        # Two different absences, and conflating them was the bug: no live rows at all, versus
        # live rows that were never embedded and so have no neighbourhood to attribute.
        message = (
            "no live decisions in this corpus — no neighbourhood to attribute"
            if view.n_live_decisions == 0
            else f"{view.n_live_decisions} live decisions, none of them embedded — "
            "no neighbourhood to attribute"
        )
        _empty(ax, message)
    else:
        ax.hist(view.origin_mix, bins=20, range=(0.0, 1.0), color=_SEED_GREY, edgecolor="white")
        ax.set_xlabel("seeded share of a live decision's top-k")
        ax.set_ylabel("live decisions")
    plot_frame.panel_label(ax, "C · neighbour origin mix (live decisions)")


# ------------------------------------------------------------------- F5 policy


def draw_policy(out_dir: Path, view: idata.PolicyData, provenance: Provenance | None) -> Path:
    """F5 — live model share and the collapse alarms; the seed band is never a decision."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3)
    _policy_share(axes[0], view)
    _policy_alarms(axes[1], view)
    _policy_propensity(axes[2], view)
    extra = Annotations(
        subtitle_facts=(
            f"live sessions n={view.n_live}",
            f"window={view.window}",
            f"seeded models n={len(view.seed_mix)}",
        ),
        caveat=(
            "no live sessions — panel A shows the seeded corpus composition instead"
            if view.n_live == 0
            else None
        ),
        counts=(("live_sessions", view.n_live), ("seeded_models", len(view.seed_mix))),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.POLICY.filename,
        _spec(specs.POLICY),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _policy_share(ax: Axes, view: idata.PolicyData) -> None:
    if view.live_series:
        models = sorted({m for _when, counts in view.live_series for m in counts})
        palette = model_color_map(models)
        times = _dates([when for when, _counts in view.live_series])
        series = [[counts.get(m, 0.0) for _w, counts in view.live_series] for m in models]
        ax.stackplot(times, *series, labels=models, colors=[palette[m] for m in models])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("live model share")
        _date_axis(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        plot_frame.panel_label(ax, "A · live model share over time")
        return
    _policy_seed_band(ax, view)


def _policy_seed_band(ax: Axes, view: idata.PolicyData) -> None:
    if not view.seed_mix:
        _empty(ax, "no sessions in this corpus")
        plot_frame.panel_label(ax, "A · model share over time")
        return
    names = [model for model, _n in view.seed_mix]
    total = sum(n for _m, n in view.seed_mix) or 1
    left = 0.0
    palette = model_color_map(names)
    for model, count in view.seed_mix:
        share = count / total
        ax.barh(
            [0.0],
            [share],
            left=left,
            height=0.34,
            color=palette[model],
            hatch=_PROVISIONAL_HATCH,
            edgecolor="white",
        )
        ax.annotate(
            f"{share:.0%}",
            (left + share / 2, 0.0),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
        left += share
    ax.set_yticks([0.0])
    ax.set_yticklabels(["seeded"], fontsize=8)
    ax.set_xlim(0.0, 1.0)
    # A single-row barh on an auto y-axis fills the panel edge to edge and stops reading as a
    # band at all; fixed limits keep it a band, with the space above it reserved for the legend.
    ax.set_ylim(-0.35, 1.45)
    ax.set_xlabel("share of the seeded corpus")
    ax.legend(
        handles=[
            Patch(facecolor=palette[m], hatch=_PROVISIONAL_HATCH, edgecolor="white") for m in names
        ],
        labels=names,
        fontsize=6.5,
        frameon=False,
        ncol=2,
        loc="upper center",
    )
    plot_frame.panel_label(ax, "A · corpus composition, NOT a routing decision")


def _policy_alarms(ax: Axes, view: idata.PolicyData) -> None:
    if view.candidate_models is None:
        _empty(
            ax,
            "no model registry supplied — normalized entropy needs the number of arms the "
            "router could have picked, and the store does not record it",
        )
    elif not view.entropy:
        _empty(ax, f"no live sessions (n={view.n_live}) — collapse alarms have nothing to read")
    else:
        ax.plot(
            [i for i, _v in view.entropy],
            [v for _i, v in view.entropy],
            color=OKABE_ITO[1],
            label="normalized entropy",
        )
        ax.plot(
            [i for i, _v in view.frontier_share],
            [v for _i, v in view.frontier_share],
            color=OKABE_ITO[3],
            label="frontier share",
        )
        ax.axhline(view.thresholds.entropy_collapse, color=_ALARM, linestyle="--", linewidth=0.9)
        ax.axhline(view.thresholds.frontier_share_alarm, color=_ALARM, linestyle=":", linewidth=0.9)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(f"live session index (rolling window {view.window})")
        ax.legend(fontsize=8, frameon=False)
    arms = view.candidate_models
    plot_frame.panel_label(
        ax,
        "B · rolling entropy and frontier share vs alarms"
        + (f" ({arms} arms)" if arms is not None else ""),
    )


def _policy_propensity(ax: Axes, view: idata.PolicyData) -> None:
    if not view.propensities:
        _empty(ax, "no policy decision carries a selection propensity in this corpus")
    else:
        names = [model for model, _n, _mean, _min in view.propensities]
        palette = model_color_map(names)
        ax.barh(
            range(len(names)),
            [mean for _m, _n, mean, _min in view.propensities],
            color=[palette[m] for m in names],
            edgecolor="white",
            height=0.6,
        )
        ax.axvline(view.thresholds.propensity_epsilon, color=_ALARM, linestyle="--", linewidth=0.9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("mean selection propensity")
    plot_frame.panel_label(ax, "C · propensity support vs the exploration floor")


# --------------------------------------------------------------- F6 escalation


def draw_escalation(
    out_dir: Path, view: idata.EscalationData, provenance: Provenance | None
) -> Path:
    """F6 — escalation rate, rungs, holds (including the untokenised one) and what followed."""
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 2, 2)
    _esc_rate(axes[0][0], view)
    _esc_rungs(axes[0][1], view)
    _esc_holds(axes[1][0], view)
    _esc_outcomes(axes[1][1], view)
    extra = Annotations(
        subtitle_facts=(
            f"live sessions n={view.n_live}",
            f"escalation records n={view.n_records}",
            f"derived undeliverable holds n={view.n_undeliverable}",
        ),
        caveat=(
            "no live escalation records in this corpus — every panel is empty, not zero"
            if view.n_records == 0 and view.n_undeliverable == 0
            else None
        ),
        counts=(
            ("live_sessions", view.n_live),
            ("escalation_records", view.n_records),
            ("undeliverable_holds", view.n_undeliverable),
        ),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.ESCALATION.filename,
        _spec(specs.ESCALATION),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _esc_rate(ax: Axes, view: idata.EscalationData) -> None:
    if not any(total for _label, _n, total in view.rates):
        _empty(ax, f"no live sessions in this corpus (n={view.n_live})")
    else:
        labels = [label for label, _n, _total in view.rates]
        rates = [n / total if total else 0.0 for _label, n, total in view.rates]
        ax.bar(labels, rates, color=OKABE_ITO[1], edgecolor="white", width=0.55)
        for index, (_label, n, total) in enumerate(view.rates):
            ax.annotate(
                f"{n}/{total}",
                (index, rates[index]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color=plot_frame.MUTED,
            )
        ax.set_ylabel("escalation rate")
        ax.set_ylim(0.0, max(0.1, max(rates) * 1.3))
    plot_frame.panel_label(ax, "A · escalation rate by window")


def _esc_rungs(ax: Axes, view: idata.EscalationData) -> None:
    if not any(n for _token, n in view.rungs):
        _empty(ax, "no escalation fired in this corpus — no rung was climbed")
    else:
        ax.bar(
            [token for token, _n in view.rungs],
            [n for _t, n in view.rungs],
            color=OKABE_ITO[2],
            edgecolor="white",
            width=0.55,
        )
        ax.set_ylabel("sessions")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    plot_frame.panel_label(ax, "B · rung climbed when escalation fired")


def _esc_holds(ax: Axes, view: idata.EscalationData) -> None:
    # The derived bar is drawn hatched and last, beside the five tokens rather than among them:
    # it is INFERRED from a voided exploration record, not read from a token the engine wrote.
    entries = [*view.holds, *view.unknown_holds, (specs.UNDELIVERABLE_LABEL, view.n_undeliverable)]
    if not any(n for _label, n in entries):
        _empty(ax, "no live hold recorded — escalation never ran on a flagged boundary here")
    else:
        labels = [label for label, _n in entries]
        colours = [OKABE_ITO[0]] * (len(entries) - 1) + [_SEED_GREY]
        hatches = [None] * (len(entries) - 1) + [_PROVISIONAL_HATCH]
        for index, (label, count) in enumerate(entries):
            ax.barh(
                [index],
                [count],
                color=colours[index],
                hatch=hatches[index],
                edgecolor="white",
                height=0.6,
            )
            del label
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("live sessions (lower bound — see the figure's limitations)")
    plot_frame.panel_label(ax, "C · why escalation held (hatched bar is derived, not a token)")


def _esc_outcomes(ax: Axes, view: idata.EscalationData) -> None:
    labeled = [(name, s, n) for name, s, n in view.outcomes if n]
    if not labeled:
        _empty(ax, "no verified outcome on either cohort in this corpus")
    else:
        names = [name for name, _s, _n in labeled]
        rates = [s / n for _name, s, n in labeled]
        ax.bar(names, rates, color=OKABE_ITO[1], edgecolor="white", width=0.5)
        for index, (_name, s, n) in enumerate(labeled):
            ax.annotate(
                f"{s}/{n}" + ("  provisional" if is_provisional(n) else ""),
                (index, rates[index]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color=plot_frame.MUTED,
            )
        ax.set_ylabel("verified-success rate")
        ax.set_ylim(0.0, 1.15)
    plot_frame.panel_label(ax, "D · verified outcome, escalated vs not")


DRAWERS: Final[tuple[str, ...]] = tuple(text.name for text in specs.FIGURES)


def sizes() -> dict[str, FigureSize]:
    """Which named canvas each figure uses — the manifest and the tests read it from here."""
    return {
        specs.STRATA.name: plot_frame.WIDE,
        specs.COST.name: plot_frame.WIDE,
        specs.UNIT_ECONOMICS.name: plot_frame.WIDE,
        specs.NEIGHBOURHOOD.name: plot_frame.WIDE,
        specs.POLICY.name: plot_frame.WIDE,
        specs.ESCALATION.name: plot_frame.WIDE_TALL,
        specs.OPE.name: plot_frame.WIDE_TALL,
    }


# ---------------------------------------------------------------------- F7 OPE


_EST_LABELS: Final[dict[str, str]] = {"ips": "IPS", "snips": "SNIPS", "dr": "DR"}
_EST_COLOURS: Final[dict[str, str]] = {
    "ips": OKABE_ITO[0],
    "snips": OKABE_ITO[2],
    "dr": OKABE_ITO[4],
}
_CONTRAST_COLOUR: Final[str] = OKABE_ITO[3]
_POLICY_LABELS: Final[dict[str, str]] = {
    idata.ALWAYS_POLICY: "always",
    idata.NEVER_POLICY: "never",
}
_REFUSAL_WRAP: Final[int] = 50
# NOT "verified-success rate". IPS is unnormalised — it divides by n, not by the sum of the
# weights — so its estimate is not bounded by 1, and on a low-propensity log it lands above it
# routinely. A bar at 1.29 under an axis that calls itself a RATE tells the reader the router
# succeeded 129% of the time. The axis therefore names the estimand, and `specs.OPE` carries the
# reason a reader will need when they meet a value above 1.
# Kept SHORT deliberately: the panels are half-canvas, and a y-label carrying the whole
# explanation overruns the title and both neighbouring panels. "estimated" is what licenses
# a value above 1; the reason lives in `specs.OPE`'s definitions, which SH009 publishes.
_VALUE_AXIS: Final[str] = "estimated verified-success rate"
# Panel D divides each measurement by its floor so counts and a propensity share one axis. The
# ratios are then unbounded (a propensity of 0.8 is 80 floors), which would flatten every count
# bar to nothing, so the DRAWN length is capped and the raw pair is printed beside every bar.
_RATIO_CAP: Final[float] = 3.0
_BAR_GAP: Final[float] = 0.7


def draw_ope(out_dir: Path, view: idata.OpeData, provenance: Provenance | None) -> Path:
    """F7 — off-policy value, or the estimator's own refusal drawn where the bars would be."""
    # The gate fires BEFORE a canvas exists. An inadmissible instrument must leave no PNG behind:
    # SH009 then finds a manifest row with no file and the commit stops, which is the point.
    if not view.admissible:
        raise InstrumentInadmissibleError(view.admissibility_reason)
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 2, 2)
    _ope_routing(axes[0][0], view)
    _ope_escalation(axes[0][1], view)
    _ope_weights(axes[1][0], view)
    _ope_floors(axes[1][1], view)
    routing, escalation = view.diagnostics
    refused = [leg for leg in (view.routing, *view.escalation) if not leg.identified]
    extra = Annotations(
        subtitle_facts=(
            f"routing {_status(view.routing)}",
            f"escalation {_status(view.escalation[0])}",
            f"usable rows: routing {routing.n_usable}/{routing.n_logged}, "
            f"escalation {escalation.n_usable}/{escalation.n_logged}",
        ),
        caveat=(
            "off-policy value is NOT IDENTIFIED here — each panel prints the estimator’s own "
            "refusal"
            if refused
            else specs.ROUTING_CONTRAST_NOTE
        ),
        # The instrument verdict travels with the figure, not in a file beside it: this line
        # reaches `figures.json` and, byte-identically, the docs section.
        notes=(view.headline,),
        counts=(
            ("routing_logged", routing.n_logged),
            ("routing_usable", routing.n_usable),
            ("escalation_logged", escalation.n_logged),
            ("escalation_usable", escalation.n_usable),
        ),
    )
    return plot_frame.save(
        fig,
        out_dir / specs.OPE.filename,
        _spec(specs.OPE),
        extra=extra,
        provenance=provenance,
        size=size,
    )


def _status(leg: idata.LegEstimates) -> str:
    return "identified" if leg.identified else "NOT IDENTIFIED"


def _ope_routing(ax: Axes, view: idata.OpeData) -> None:
    leg = view.routing
    if not leg.identified:
        _refusal(ax, [leg], view.diagnostics[0])
    else:
        _value_bars(ax, [leg])
        _on_policy_line(ax, leg)
        ax.set_ylabel(_VALUE_AXIS)
        # Drawn on EVERY identified routing panel, because the number a reader reaches for next
        # is the contrast, and routing's is not the same object as escalation's.
        ax.annotate(
            specs.ROUTING_CONTRAST_NOTE,
            (0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            style="italic",
            color=plot_frame.MUTED,
        )
    plot_frame.panel_label(ax, "A · routing: value of serving the top-scored candidate")


def _ope_escalation(ax: Axes, view: idata.OpeData) -> None:
    always, never = view.escalation
    if not always.identified:
        _refusal(ax, [always, never], view.diagnostics[1])
    else:
        _value_bars(ax, [always, never], contrast=True)
        ax.set_ylabel(_VALUE_AXIS)
    plot_frame.panel_label(ax, "B · escalation: always vs never, and the contrast that decides")


def _refusal(
    ax: Axes, legs: Sequence[idata.LegEstimates], diagnostics: idata.LegDiagnostics
) -> None:
    """Print the estimator's own reason where the bars would be — this IS the result."""
    # Not `_empty`: an empty panel says "nothing happened here", and something did happen — the
    # estimator was asked and refused, in words that name the condition that failed. Quoting the
    # reason verbatim is what stops the next reader re-deriving a bar from the same logs.
    _frame(ax)
    reasons = list(dict.fromkeys(leg.estimate.reason for leg in legs))
    body = "\n\n".join("\n".join(textwrap.wrap(reason, _REFUSAL_WRAP)) for reason in reasons)
    ax.text(
        0.5,
        0.72,
        "NOT IDENTIFIED",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=_ALARM,
    )
    ax.text(
        0.5,
        0.60,
        body,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        style="italic",
        color=plot_frame.MUTED,
    )
    ax.text(
        0.5,
        0.20,
        f"{diagnostics.n_logged} decisions logged · {diagnostics.n_usable} usable · "
        f"{diagnostics.n_escalated} target-arm / {diagnostics.n_held} complement · "
        f"{diagnostics.n_clusters} independent sessions",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=plot_frame.MUTED,
    )


def _bar_entries(
    legs: Sequence[idata.LegEstimates], *, contrast: bool = False
) -> tuple[list[tuple[float, str, float, str]], list[float], list[float], list[str]]:
    """(x, label, value, colour) per drawable estimator, its offsets, and what was left out."""
    entries: list[tuple[float, str, float, str]] = []
    lower: list[float] = []
    upper: list[float] = []
    omitted: list[str] = []
    position = 0.0
    for index, leg in enumerate(legs):
        position += _BAR_GAP if index else 0.0
        for name in ESTIMATORS:
            value = leg.value(name)
            if not leg.identified or name not in leg.quotable or value is None:
                omitted.append(f"{leg.policy}/{_EST_LABELS[name]}")
                continue
            entries.append((position, _bar_label(legs, leg, name), value, _EST_COLOURS[name]))
            interval = _dr_interval(leg) if name == "dr" else (0.0, 0.0)
            lower.append(interval[0])
            upper.append(interval[1])
            position += 1.0
    if contrast:
        _append_contrast(legs[0], position + _BAR_GAP, entries, lower, upper)
    return entries, lower, upper, omitted


def _bar_label(legs: Sequence[idata.LegEstimates], leg: idata.LegEstimates, name: str) -> str:
    """One tick label. The policy is named only where two of them share the panel."""
    if len(legs) == 1:
        return _EST_LABELS[name]
    return f"{_EST_LABELS[name]}\n{_POLICY_LABELS.get(leg.policy, leg.policy)}"


def _append_contrast(
    leg: idata.LegEstimates,
    position: float,
    entries: list[tuple[float, str, float, str]],
    lower: list[float],
    upper: list[float],
) -> None:
    """V(escalate) − V(hold): the decision question, drawn beside the levels it is derived from."""
    # A level cannot answer "does escalating help", and a reader given only levels supplies the
    # missing comparator themselves. It is the same reward-rate unit as the bars beside it, so it
    # shares their axis — and the zero line `_value_bars` draws is what it is read against.
    estimate = leg.estimate
    if estimate.contrast_estimate is None:
        return
    entries.append((position, "DR\nescalate − hold", estimate.contrast_estimate, _CONTRAST_COLOUR))
    if estimate.contrast_ci_low is None or estimate.contrast_ci_high is None:
        lower.append(0.0)
        upper.append(0.0)
        return
    low, high = ci_yerr(
        estimate.contrast_estimate, estimate.contrast_ci_low, estimate.contrast_ci_high
    )
    lower.append(low)
    upper.append(high)


def _dr_interval(leg: idata.LegEstimates) -> tuple[float, float]:
    """The DR estimate's bootstrap offsets — the only estimator here that carries an interval."""
    estimate = leg.estimate
    if estimate.dr_estimate is None or estimate.ci_low is None or estimate.ci_high is None:
        return (0.0, 0.0)
    return ci_yerr(estimate.dr_estimate, estimate.ci_low, estimate.ci_high)


def _value_bars(ax: Axes, legs: Sequence[idata.LegEstimates], *, contrast: bool = False) -> None:
    """One bar per estimator that earned a certificate; everything else is named, not drawn."""
    entries, lower, upper, omitted = _bar_entries(legs, contrast=contrast)
    if entries:
        ax.bar(
            [x for x, _label, _v, _c in entries],
            [value for _x, _label, value, _c in entries],
            width=0.62,
            color=[colour for _x, _label, _v, colour in entries],
            edgecolor="white",
            yerr=[lower, upper],
            capsize=3.0,
            ecolor=plot_frame.MUTED,
        )
        ax.set_xticks([x for x, _label, _v, _c in entries])
        ax.set_xticklabels([label for _x, label, _v, _c in entries], fontsize=8)
        ax.axhline(0.0, color=plot_frame.MUTED, linewidth=0.9)
    if omitted:
        # An estimator whose control failed is NAMED and left undrawn. Averaging it into the
        # others would hide the diagnosis that one estimator disagrees with the other two.
        ax.annotate(
            "not quotable: " + ", ".join(omitted),
            (0.02, 0.98),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7,
            color=_ALARM,
        )
    _headroom(ax)


def _on_policy_line(ax: Axes, leg: idata.LegEstimates) -> None:
    """What the LOGGED policy paid — the baseline a value estimate means nothing without."""
    if leg.on_policy_mean is None:
        return
    ax.axhline(leg.on_policy_mean, color=plot_frame.MUTED, linestyle="--", linewidth=1.1)
    ax.annotate(
        f"logged policy paid {leg.on_policy_mean:.3f} (n={leg.n_rewarded})",
        (0.98, leg.on_policy_mean),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=7,
        color=plot_frame.MUTED,
    )


def _ope_weights(ax: Axes, view: idata.OpeData) -> None:
    drawn = [diagnostics for diagnostics in view.diagnostics if diagnostics.weights]
    if not drawn:
        _empty(ax, "no usable importance weight in this corpus — nothing was randomized to weight")
    else:
        for index, diagnostics in enumerate(drawn):
            total = len(diagnostics.weights)
            ax.step(
                diagnostics.weights,
                [(rank + 1) / total for rank in range(total)],
                where="post",
                color=OKABE_ITO[index],
                linewidth=1.6,
                label=(
                    f"{diagnostics.leg} · n={total} · ESS {diagnostics.ess_fraction:.2f}n · "
                    f"{diagnostics.n_clipped} clipped"
                ),
            )
        ax.set_xlabel(f"importance weight (clipped at {idata.WEIGHT_CLIP:g})")
        ax.set_ylabel("fraction of usable rows at or below w")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    plot_frame.panel_label(ax, "C · importance-weight ECDF, with ESS and clipping")


def _ope_floors(ax: Axes, view: idata.OpeData) -> None:
    entries = [
        (f"{diagnostics.leg} · {label}", measured, floor)
        for diagnostics in view.diagnostics
        for label, measured, floor in diagnostics.floors
    ]
    raw = [measured / floor if floor else 0.0 for _label, measured, floor in entries]
    ratios = [min(value, _RATIO_CAP) for value in raw]
    ax.barh(
        range(len(entries)),
        ratios,
        height=0.62,
        color=[OKABE_ITO[2] if value >= 1.0 else _ALARM for value in raw],
        edgecolor="white",
    )
    ax.axvline(1.0, color=plot_frame.MUTED, linestyle="--", linewidth=1.1)
    for index, ((_label, measured, floor), ratio) in enumerate(zip(entries, ratios, strict=True)):
        ax.annotate(
            f"{_number(measured)} / {_number(floor)}",
            (ratio, index),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=7,
            color=plot_frame.MUTED,
        )
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([label for label, _m, _f in entries], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.0, _RATIO_CAP * 1.25)
    ax.set_xlabel(f"measured ÷ floor (1.0 = the floor, bars capped at {_RATIO_CAP:g})")
    plot_frame.panel_label(ax, "D · identification floors: what the logs measured ÷ the floor")


def _number(value: float) -> str:
    """Counts print as counts and a propensity prints as a propensity, on one shared ledger."""
    return f"{value:.0f}" if value == 0.0 or value >= 1.0 else f"{value:g}"
