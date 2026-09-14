"""The combined model-universe figures: every free and paid model, valid or not.

Three canvases read the ONE table in `benchmark.routing.model_universe`, which itself is a lens
over the ONE inference-valid predicate in `benchmark.routing.model_validity`:

  * `universe_coverage.png`  — coverage matrix (models x verified challenges) plus per-model
    measured-cell counts, for EVERY evidenced model, free and paid.
  * `universe_economics.png` — per-model measured cost and quality, free vs paid, with the
    inference-valid set marked; the same roster on two axes.
  * `invalid_models.png`     — the inference-INVALID subset alone: its coverage and its
    per-model quality/cost, explicitly labelled as outside the inference pool.

The three exist because the main routing figures draw only the inference-valid models; the
universe outside that pool must still be named, measured and explained rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import model_universe, plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.model_universe import Performance, UniverseRow

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_PAID = "#0072B2"
_FREE = "#E69F00"
_EMPTY = "#ECEFF1"
_VALID = "#2E7D32"
_INK = "#1a1a1a"


@dataclass(frozen=True)
class _Data:
    """Everything the three canvases share, read once per render."""

    rows: tuple[UniverseRow, ...]
    challenges: tuple[str, ...]
    coverage: dict[str, frozenset[str]]
    perf: dict[str, Performance]
    corpus: int

    @property
    def evidenced(self) -> tuple[UniverseRow, ...]:
        return tuple(r for r in self.rows if (p := self.perf.get(r.identity)) and p.n > 0)

    @property
    def invalid(self) -> tuple[UniverseRow, ...]:
        return tuple(r for r in self.evidenced if not r.valid)

    @property
    def valid(self) -> tuple[UniverseRow, ...]:
        return tuple(r for r in self.rows if r.valid)


def _load() -> _Data | None:
    rows = tuple(model_universe.universe())
    challenges = model_universe.verified_challenges()
    if not rows or not challenges:
        return None
    return _Data(
        rows=rows,
        challenges=challenges,
        coverage=model_universe.coverage_map(),
        perf=model_universe.performance(),
        corpus=len(challenges),
    )


def _colour(row: UniverseRow) -> str:
    return _FREE if row.channel == model_universe.FREE else _PAID


def _channel_key(*, star: bool = False) -> list[Patch | Line2D]:
    """The paid/free colour key every universe canvas carries on its face.

    ``star`` adds the inference-valid marker, which two canvases carry as a row-left glyph
    but the economics canvas only names in its subtitle — so its key spells the marker out.
    """
    handles: list[Patch | Line2D] = [
        Patch(facecolor=_PAID, edgecolor="white", label="paid channel"),
        Patch(facecolor=_FREE, edgecolor="white", label="free channel"),
    ]
    if star:
        handles.append(
            Line2D([], [], color="none", marker="*", markersize=8, label="★ inference-valid")
        )
    return handles


def _repo_of(challenge: str) -> str:
    return challenge.split("__", 1)[0]


def _repo_spans(challenges: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """(repo, start, end) runs over the sorted challenge tuple — the matrix column groups."""
    spans: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(challenges) + 1):
        if index == len(challenges) or _repo_of(challenges[index]) != _repo_of(challenges[start]):
            spans.append((_repo_of(challenges[start]), start, index))
            start = index
    return spans


# ---------------------------------------------------------------- universe_coverage

COVERAGE_SPEC = FigureSpec(
    title="Every model's evidence on one grid — free and paid, valid and not",
    subtitle="one row per canonical weights identity · verified challenges as columns",
    caveat="Coverage is an evidence status, not a quality score — a covered cell can fail.",
    reading=(
        "One row per canonical weights identity the committed evidence names, one column per "
        "verified challenge, in the corpus's own sorted order. A blue cell is a paid channel's "
        "measured default-arm cell, an orange cell a free one, a pale cell no measured outcome. "
        "Vertical rules mark repository boundaries, and each row label carries its covered/"
        "total count. Rows are ordered inference-valid first (marked ★, above the green rule), "
        "then the paid collection, then the free collection."
    ),
    goal=(
        "Read that coverage is ragged and channel-specific. The four starred rows above the "
        "green rule are the only models the main routing figures draw; every row below it is "
        "still measured and still named, and a row of pale cells says no committed outcome "
        "exists rather than that the model is weak."
    ),
    definitions=(
        (
            "canonical identity",
            "the registry `version` slug, so the same weights served under several channel "
            "listings are ONE row and a provider prefix or `-free` marker is never a name",
        ),
        (
            "measured cell",
            "a committed default-arm outcome for one (challenge, model) pair; imputed cells "
            "are not counted here",
        ),
    ),
    notes=(
        "Cells are measured outcomes only. The completed matrix used by the strategy figures "
        "adds imputed cells, which this canvas deliberately excludes.",
    ),
    limitations=(
        "Paid and free corpora were collected under different campaigns, so a free row's "
        "coverage is not comparable cell-for-cell with a paid row's.",
        "A row with no committed cell is drawn as an empty band and is not evidence that the "
        "model is weak.",
    ),
)


def _draw_coverage_matrix(ax: Axes, data: _Data) -> None:
    rows, challenges = data.rows, data.challenges
    rgba = np.ones((len(rows), len(challenges), 4), dtype=float)
    for i, row in enumerate(rows):
        covered = data.coverage.get(row.identity, frozenset())
        colour = to_rgba(_colour(row))
        rgba[i, :] = to_rgba(_EMPTY)
        for j, challenge in enumerate(challenges):
            if challenge in covered:
                rgba[i, j] = colour
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for i in range(1, len(rows)):
        ax.axhline(i - 0.5, color="#FFFFFF", linewidth=1.1)
    for _repo, start, _end in _repo_spans(challenges):
        ax.axvline(start - 0.5, color="#9E9E9E", linewidth=0.7)
    # Only a repo wide enough to print its name gets a tick: a one-challenge repository's
    # label collides with its neighbour and neither is readable, which is worse than leaving
    # that boundary to the grey rule alone.
    labelled = [(r, s, e) for r, s, e in _repo_spans(challenges) if e - s >= 4]
    ax.set_xticks([(s + e) / 2 for _r, s, e in labelled])
    ax.set_xticklabels([r for r, _s, _e in labelled], fontsize=6.2, rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            f"★ {r.canonical}  {len(data.coverage.get(r.identity, frozenset()))}/{data.corpus}"
            if r.valid
            else f"  {r.canonical}  {len(data.coverage.get(r.identity, frozenset()))}/{data.corpus}"
            for r in rows
        ],
        fontsize=6.6,
        family="monospace",
    )
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # The valid block separator sits BOLD so the reader sees where the inference pool ends.
    valid_n = sum(1 for r in rows if r.valid)
    if 0 < valid_n < len(rows):
        ax.axhline(valid_n - 0.5, color=_VALID, linewidth=1.6)
    ax.set_xlabel("verified challenges, grouped by repository", fontsize=8.5)
    ax.set_ylabel("canonical weights identity", fontsize=8.5)
    # THE COLOUR KEY IS ON THE FACE. The palette is a two-value channel encoding and a reader
    # without the key cannot tell a paid cell from a free one; the invalid family had the same
    # gap. The green separator and the grey vertical rules are named here too, because neither
    # is self-explanatory from a line alone.
    handles: list[Patch | Line2D] = _channel_key()
    if 0 < valid_n < len(rows):
        handles.append(
            Line2D([], [], color=_VALID, linewidth=1.6, label="★ inference-valid above this rule")
        )
    handles.append(Line2D([], [], color="#9E9E9E", linewidth=0.7, label="repository boundary"))
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=6.8,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
        handletextpad=0.8,
        labelspacing=0.35,
    )
    plot_frame.panel_label(ax, "coverage matrix — every canonical weights identity")


def render_coverage(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw universe_coverage.png; None when no model is evidenced."""
    data = _load()
    if data is None:
        return None
    size = plot_frame.FigureSize("universe_coverage", 14.0, 13.5)
    fig = plot_frame.new_figure(size)
    _draw_coverage_matrix(fig.subplots(), data)
    valid_n = len(data.valid)
    free_n = sum(1 for r in data.rows if r.channel == model_universe.FREE)
    return plot_frame.save(
        fig,
        ctx.out_dir / "universe_coverage.png",
        COVERAGE_SPEC,
        extra=Annotations(
            subtitle_facts=(
                f"{len(data.rows)} canonical models: {len(data.rows) - free_n} paid, {free_n} free",
                f"{valid_n} inference-valid, {len(data.rows) - valid_n} outside the pool",
                f"{data.corpus} verified challenges",
            ),
            notes=tuple(
                f"{r.canonical}: {len(data.coverage.get(r.identity, frozenset()))}/"
                f"{data.corpus} challenges, {r.cells} measured cells, {r.channel}, "
                f"{'VALID' if r.valid else r.reason}"
                for r in data.rows
            ),
            counts=(
                ("models", len(data.rows)),
                ("valid", valid_n),
                ("free", free_n),
                ("challenges", data.corpus),
            ),
        ),
        provenance=ctx.provenance(__name__),
        size=size,
    )


# ---------------------------------------------------------------- universe_economics

ECONOMICS_SPEC = FigureSpec(
    title="What every free and paid model costs and delivers",
    subtitle="measured default-arm outcomes only · blue paid, orange free · ★ inference-valid",
    caveat=(
        "A $0 row is a free price or a PAID sub-cent mean; both draw as a hatched floor, not a bar."
    ),
    reading=(
        "Panel A: per canonical identity, the mean measured cost per task on a log axis. A "
        "$0.0000 row — whether a genuinely free channel or a paid model whose mean rounds to "
        "zero — is drawn as a hatched floor stub at the left of the axis, because a log axis "
        "has no width for a zero bar. Panel B: the same identity's measured pass rate with a "
        "95% Wilson interval, same row order, so cost and quality read across one line. Blue is "
        "a paid channel, orange a free one, and a star marks the four inference-valid models."
    ),
    goal=(
        "Find the identities that are both cheap and high-pass: the free channel's rows sit "
        "far left, but their intervals are wide because their coverage is thin. The four "
        "starred rows are the ones the router may actually serve."
    ),
    definitions=(
        (
            "measured cost",
            "the mean `real_cost` over that identity's committed default-arm cells — what was "
            "billed, not list price",
        ),
        ("pass rate", "passes over measured default-arm cells, with a 95% Wilson interval"),
    ),
    notes=(
        "Rows with no committed cell are absent: this canvas is a measurement, and a model "
        "with nothing measured has no point to draw.",
    ),
    limitations=(
        "Free-channel rows are cheap by price and thin by coverage; a wide interval is a "
        "coverage gap, not a quality estimate.",
        "Mean cost is over measured cells only and is not the list price the strategy figures "
        "rank on.",
    ),
)


def _draw_economics_cost(
    ax: Axes, rows: tuple[UniverseRow, ...], perf: dict[str, Performance]
) -> None:
    ys = list(range(len(rows)))[::-1]
    # A log axis cannot draw zero, so a $0 row used to be a float(1e-6) bar the axis swallowed
    # — invisible, which read as missing rather than as zero. Every zero row instead gets a
    # hatched FLOOR STUB in an axes-anchored column at the far left, so the reader sees a mark
    # that says "zero, off this ruler" rather than a gap. The stub is drawn AFTER the log bars
    # so its own fixed width is not stretched by the log transform.
    costs = [max(perf[r.identity].mean_cost, 1e-6) for r in rows]
    zeros: list[tuple[int, UniverseRow]] = []
    for y, row, cost in zip(ys, rows, costs, strict=True):
        if perf[row.identity].mean_cost <= 0.0:
            zeros.append((y, row))
            continue
        ax.barh(y, cost, height=0.62, color=_colour(row), alpha=0.9)
        ax.text(cost * 1.08, y, f"${cost:.4f}", va="center", fontsize=6.0, color=_INK)
    for y, row in zeros:
        ax.barh(
            y,
            0.028,
            left=0.0,
            height=0.62,
            color=_colour(row),
            alpha=0.9,
            hatch="////",
            edgecolor="white",
            linewidth=0.0,
            transform=ax.get_yaxis_transform(),
            zorder=2,
        )
        ax.text(
            0.035,
            y,
            "$0.0000",
            va="center",
            fontsize=6.0,
            color=_INK,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"★ {r.canonical}" if r.valid else r.canonical for r in rows],
        fontsize=6.4,
        family="monospace",
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-6, max(costs) * 6)
    ax.set_xlabel("mean billed cost per measured task (USD, log; hatched = $0)", fontsize=8.5)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4)
    plot_frame.panel_label(ax, "A · cost")
    # The subtitle names the colour encoding in prose; its sibling canvases carry the swatch
    # key on the face, so this panel does the same rather than making the reader decode a
    # sentence. The ★ entry spells out the row-left glyph used in the tick labels.
    ax.legend(
        handles=_channel_key(star=True),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        fontsize=6.8,
        frameon=False,
        handlelength=1.4,
    )


def _draw_economics_quality(
    ax: Axes, rows: tuple[UniverseRow, ...], perf: dict[str, Performance]
) -> None:
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        p = perf[row.identity]
        rate = 100.0 * p.pass_rate
        lo, hi = plot_style.wilson_interval(p.passes, p.n)
        ax.barh(y, rate, height=0.62, color=_colour(row), alpha=0.9)
        ax.plot([lo * 100, hi * 100], [y, y], color=_INK, linewidth=0.8)
        ax.plot([lo * 100, lo * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.plot([hi * 100, hi * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.text(hi * 100 + 1.0, y, f"{rate:.0f}% (n={p.n})", va="center", fontsize=6.0, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlim(0, 108)
    ax.set_xlabel("measured pass rate % (95% Wilson)", fontsize=8.5)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4)
    plot_frame.panel_label(ax, "B · quality")


def render_economics(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw universe_economics.png; None when nothing has a measured outcome."""
    data = _load()
    if data is None:
        return None
    rows = data.evidenced
    if not rows:
        return None
    size = plot_frame.table_size(len(rows) + 2, width_in=13.5)
    fig, axes = plot_frame.subplots(size, 1, 2, gridspec_kw={"width_ratios": [1.15, 1.0]})
    _draw_economics_cost(axes[0], rows, data.perf)
    _draw_economics_quality(axes[1], rows, data.perf)
    free_n = sum(1 for r in rows if r.channel == model_universe.FREE)
    return plot_frame.save(
        fig,
        ctx.out_dir / "universe_economics.png",
        ECONOMICS_SPEC,
        extra=Annotations(
            subtitle_facts=(
                f"{len(rows)} evidenced models ({len(rows) - free_n} paid, {free_n} free)",
                f"{len(data.valid)} inference-valid marked ★",
            ),
            notes=tuple(
                f"{r.canonical}: {data.perf[r.identity].pass_rate:.1%} on "
                f"n={data.perf[r.identity].n}, ${data.perf[r.identity].mean_cost:.4f}/task, "
                f"{r.channel}, {'VALID' if r.valid else r.reason}"
                for r in rows
            ),
            counts=(
                ("evidenced", len(rows)),
                ("free", free_n),
                ("valid", len(data.valid)),
            ),
        ),
        provenance=ctx.provenance(__name__),
        size=size,
    )


# ---------------------------------------------------------------- invalid_models

INVALID_SPEC = FigureSpec(
    title="Outside the inference pool: what the other measured models deliver",
    subtitle="inference-invalid on committed evidence · not selected for routing, not erased",
    caveat="Invalid means not selected on this evidence — not that the model is weak.",
    reading=(
        "Every evidenced model that fails at least one inference criterion, on two axes. "
        "Panel A: verified-challenge coverage, with the measured-cell count printed. Panel B: "
        "measured pass rate with a 95% Wilson interval, the mean billed cost per task printed "
        "beside it. Blue is a paid channel, orange a free one. Each row carries its first "
        "failing reason in the notes below."
    ),
    goal=(
        "Read this page as the boundary of the inference pool: these models have real measured "
        "outcomes, some of them on hundreds of cells, and the reason each is outside is "
        "printed rather than implied. A dominated benchmark model and a collection-only probe "
        "are not the same kind of absence."
    ),
    definitions=(
        (
            "inference-invalid",
            "fails at least one of: live pool, triage KEEP, capability rank measured, "
            "coverage >= K cells, a paid channel. The first failing reason is named per row.",
        ),
    ),
    notes=(
        "The predicate and the first-failing reason are not restated here — they are read "
        "from benchmark.routing.model_validity, the same source model_validity.png draws.",
    ),
    limitations=(
        "A live slot with no committed measurement is flagged invalid here but is a coverage "
        "gap, not evidence of domination; collect it and the verdict can change.",
        "Free-channel rows were collected under a different campaign and their intervals are "
        "wide from thin coverage, not from measured weakness.",
    ),
)


def _draw_invalid_coverage(ax: Axes, rows: tuple[UniverseRow, ...], corpus: int) -> None:
    ys = list(range(len(rows)))[::-1]
    covered = [r.covered for r in rows]
    ax.barh(ys, covered, height=0.62, color=[_colour(r) for r in rows], alpha=0.9)
    top = max(covered) if covered else 1
    for y, row, count in zip(ys, rows, covered, strict=True):
        ax.text(count + top * 0.02, y, f"{count} ({row.cells} cells)", va="center", fontsize=6.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.canonical for r in rows], fontsize=6.4, family="monospace")
    ax.set_xlim(0, top * 1.42)
    ax.set_xlabel(f"verified challenges covered (of {corpus})", fontsize=8.5)
    ax.set_ylabel("canonical weights identity", fontsize=8.5)
    ax.legend(
        handles=_channel_key(),
        loc="lower right",
        fontsize=6.8,
        frameon=False,
        handlelength=1.4,
    )
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4)
    plot_frame.panel_label(ax, "A · coverage")


def _draw_invalid_quality(
    ax: Axes, rows: tuple[UniverseRow, ...], perf: dict[str, Performance]
) -> None:
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        p = perf[row.identity]
        rate = 100.0 * p.pass_rate
        lo, hi = plot_style.wilson_interval(p.passes, p.n)
        ax.barh(y, rate, height=0.62, color=_colour(row), alpha=0.9)
        ax.plot([lo * 100, hi * 100], [y, y], color=_INK, linewidth=0.8)
        ax.plot([lo * 100, lo * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.plot([hi * 100, hi * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.text(
            hi * 100 + 1.2,
            y,
            f"{rate:.0f}% (n={p.n}, ${p.mean_cost:.4f})",
            va="center",
            fontsize=6.0,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlim(0, 118)
    # The pass-rate axis names ONE quantity. The mean cost is already printed beside every bar,
    # so repeating it in the axis title read as though the axis measured both.
    ax.set_xlabel("measured pass rate % (95% Wilson)", fontsize=8.5)
    ax.legend(
        handles=_channel_key(),
        loc="lower right",
        fontsize=6.8,
        frameon=False,
        handlelength=1.4,
    )
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4)
    plot_frame.panel_label(ax, "B · quality and cost")


def render_invalid(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw invalid_models.png; None when no invalid model has a measured outcome."""
    data = _load()
    if data is None:
        return None
    rows = data.invalid
    if not rows:
        return None
    size = plot_frame.table_size(len(rows) + 2, width_in=13.5)
    fig, axes = plot_frame.subplots(size, 1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})
    _draw_invalid_coverage(axes[0], rows, data.corpus)
    _draw_invalid_quality(axes[1], rows, data.perf)
    free_n = sum(1 for r in rows if r.channel == model_universe.FREE)
    return plot_frame.save(
        fig,
        ctx.out_dir / "invalid_models.png",
        INVALID_SPEC,
        extra=Annotations(
            subtitle_facts=(
                f"{len(rows)} inference-invalid evidenced models "
                f"({len(rows) - free_n} paid, {free_n} free)",
                "the four inference-valid models are on model_validity.png",
            ),
            notes=tuple(
                f"{r.canonical}: {r.reason} — {r.covered}/{data.corpus} challenges, "
                f"{r.cells} cells, {r.channel}, capability {r.capability}, triage {r.triage}"
                for r in rows
            ),
            counts=(("invalid_evidenced", len(rows)), ("free", free_n)),
        ),
        provenance=ctx.provenance(__name__),
        size=size,
    )
