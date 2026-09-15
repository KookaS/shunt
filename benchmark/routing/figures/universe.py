"""The combined model-universe figures: every free and paid model, valid or not.

Three canvases read the ONE table in `benchmark.routing.model_universe`, which itself is a lens
over the ONE inference-valid predicate in `benchmark.routing.model_validity`:

  * `universe_coverage.png`  — coverage matrix (models x verified challenges) plus per-model
    measured-cell counts, for every model the committed evidence NAMES, free and paid.
  * `universe_economics.png` — per-model measured cost and quality, free vs paid, with the
    inference-valid set marked; the same roster on two axes.
  * `invalid_models.png`     — the inference-INVALID subset alone: its coverage and its
    per-model quality/cost, explicitly labelled as outside the inference pool.

The three exist because the main routing figures draw only the inference-valid models; the
universe outside that pool must still be named, measured and explained rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

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
# The free-channel row label, darkened from the orange cell hue so the channel cue does not
# cost contrast against the near-white canvas — it reads as dark as the paid rows' ink.
_FREE_INK: Final[str] = "#5C3A00"

# A log axis cannot place a zero, and a paid mean can round below this floor and become one
# visually. Rows at or under it get a floor marker instead of a dot. The column is INSET from
# the y spine so a stack of stubs does not pile onto the axis line.
_COST_FLOOR: Final[float] = 1e-6
_FLOOR_LEFT: Final[float] = 0.014
_FLOOR_WIDTH: Final[float] = 0.012
_FLOOR_TEXT_X: Final[float] = 0.032

# A rate from fewer than two cells is 0% or 100% by construction — it cannot be estimated, so
# it draws a visible minimum stub and an "insufficient n" label, never an invisible bar or a
# full one.
_INSUFFICIENT_N: Final[int] = 2
_INSUFFICIENT_STUB: Final[float] = 2.5
# The quality axis leaves room for the longest printed label — "insufficient n (n=1, $0.0000)"
# is longer than any rate — so a bar label never runs across the right spine.
_QUALITY_XLIM: Final[float] = 150.0

# THE ONE "evidenced" DEFINITION, shared by all three canvases so a count is never quoted
# against a different filter. The committed evidence NAMES more identities than it has
# measured outcomes for; an unmeasured row is an empty band, not a weak model.
_EVIDENCED: Final[tuple[str, str]] = (
    "evidenced",
    "a canonical identity with at least one committed measured default-arm outcome "
    "(Performance.n > 0); a named identity with no measured outcome is drawn as an empty "
    "band and is not a claim that the model is weak",
)


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
        """Rows with at least one committed measured default-arm outcome (Performance.n > 0)."""
        return tuple(r for r in self.rows if (p := self.perf.get(r.identity)) and p.n > 0)

    @property
    def invalid(self) -> tuple[UniverseRow, ...]:
        """Evidenced rows that fail at least one inference criterion."""
        return tuple(r for r in self.evidenced if not r.valid)

    @property
    def valid(self) -> tuple[UniverseRow, ...]:
        """Rows that clear every inference criterion, whether evidenced here or not."""
        return tuple(r for r in self.rows if r.valid)

    @property
    def filter_note(self) -> str:
        """The named / evidenced / invalid-with-outcome counts, so the filter is self-explaining."""
        return (
            f"{len(self.rows)} named · {len(self.evidenced)} evidenced (≥1 measured outcome) · "
            f"{len(self.invalid)} invalid-with-outcome"
        )


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


def _row_label_colour(row: UniverseRow) -> str:
    """The y-tick ink for a row: paid ink, or the darkened free-channel ink."""
    return _FREE_INK if row.channel == model_universe.FREE else _INK


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
        _EVIDENCED,
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
    # The channel cue lives on the row label too, at a contrast that matches the paid rows.
    for label, row in zip(ax.get_yticklabels(), rows, strict=True):
        label.set_color(_row_label_colour(row))
    # A deeper pad under the rotated repository names gives the longest of them bottom margin,
    # so "sphinx-doc" and "sympy" stop reaching the canvas edge.
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=5.0)
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
    size = plot_frame.FigureSize("universe_coverage", 14.0, 14.5)
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
                data.filter_note,
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
        "Panel A draws dots, not bars — a log axis reads position. Free $0 and paid sub-floor "
        "rows are separate hatched markers."
    ),
    reading=(
        "Panel A: per canonical identity, the mean measured cost per task on a log axis. Each "
        "priced row is a dot at its cost — a dot carries position, the one thing a log axis "
        "reads honestly, where a bar's length would encode log(cost) and misstate every ratio. "
        "A `$0.0000 free` row (a genuinely free channel) and a `<$0.000001 paid` row (a paid "
        "mean that rounds below the axis floor) are distinct hatched stubs in a fixed column at "
        "the left, not bars and not each other. Panel B: the same identity's measured pass rate "
        "with a 95% Wilson interval, same row order, so cost and quality read across one line. "
        "Blue is a paid channel, orange a free one, and a star marks the inference-valid models."
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
        _EVIDENCED,
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


def _cost_kind(channel: str, mean_cost: float) -> str:
    """How a mean cost is drawn, from the model's ENTITLEMENT — never from the number alone.

    A free channel is genuinely $0 however its mean rounds; a PAID channel whose mean is at or
    below the log floor has no place on the axis and is labelled sub-floor, never "free". The
    channel is the serving fact; the mean is what happened to be billed.
    """
    if channel == model_universe.FREE:
        return "free"
    return "sub_micro" if mean_cost < _COST_FLOOR else "priced"


def _floor_text(kind: str) -> str:
    """The annotation a floor marker carries — the only thing separating free from sub-floor."""
    return "$0.0000 free" if kind == "free" else "$0.0000 sub-floor (unpriced)"


def _floor_marker(ax: Axes, y: float, row: UniverseRow, kind: str) -> None:
    """A fixed-width hatched stub for a $0 or sub-floor mean: a mark, never a bar.

    Its x is anchored in AXES FRACTION, not data, so the log transform cannot stretch its
    width and the two floor kinds stay distinguishable by hatch and annotation.
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(
        Rectangle(
            (_FLOOR_LEFT, y - 0.31),
            _FLOOR_WIDTH,
            0.62,
            transform=trans,
            facecolor=_colour(row),
            edgecolor="white",
            linewidth=0.0,
            hatch="////" if kind == "free" else "xxxx",
            alpha=0.9,
            zorder=2,
        )
    )
    ax.text(
        _FLOOR_TEXT_X,
        y,
        _floor_text(kind),
        va="center",
        fontsize=6.0,
        color=_INK,
        transform=trans,
    )


def _draw_economics_cost(
    ax: Axes, rows: tuple[UniverseRow, ...], perf: dict[str, Performance]
) -> None:
    ys = list(range(len(rows)))[::-1]
    # A DOT, not a bar. The x axis is logarithmic because these prices span decades, and a bar
    # drawn on a log axis encodes log(cost) − log(floor) as its length: the 64.7x gap between
    # deepseek-v4-flash and kimi-k3 read as 1.46x of length, and moving the axis floor would
    # have rewritten every ratio on the canvas without changing one number. Position is the one
    # channel a log axis carries honestly (the same rationale as figures/live_gap.py:222-228);
    # the grey guide rule behind each dot is deliberately unlabelled so it is not read as a
    # magnitude.
    costs = [perf[r.identity].mean_cost for r in rows]
    kinds = [_cost_kind(r.channel, c) for r, c in zip(rows, costs, strict=True)]
    ax.set_xscale("log")
    top = max(
        (c for c, kind in zip(costs, kinds, strict=True) if kind == "priced"),
        default=_COST_FLOOR,
    )
    ax.set_xlim(_COST_FLOOR, top * 6.0)
    floor_x = ax.get_xlim()[0]
    for y, row, cost, kind in zip(ys, rows, costs, kinds, strict=True):
        if kind != "priced":
            _floor_marker(ax, y, row, kind)
            continue
        ax.plot([floor_x, cost], [y, y], color="#dddddd", lw=0.9, zorder=1, solid_capstyle="butt")
        ax.plot([cost], [y], "o", markersize=6.5, color=_colour(row), zorder=3)
        ax.text(cost * 1.08, y, f"${cost:.4f}", va="center", fontsize=6.0, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"★ {r.canonical}" if r.valid else r.canonical for r in rows],
        fontsize=6.4,
        family="monospace",
    )
    for label, row in zip(ax.get_yticklabels(), rows, strict=True):
        label.set_color(_row_label_colour(row))
    ax.set_xlabel(
        "mean billed cost per measured task (USD, log; dot = cost, hatched = free/sub-floor)",
        fontsize=8.5,
    )
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4)
    plot_frame.panel_label(ax, "A · cost")
    # The subtitle names the colour encoding in prose; its sibling canvases carry the swatch
    # key on the face, so this panel does the same rather than making the reader decode a
    # sentence. The ★ entry spells out the row-left glyph used in the tick labels, and each
    # floor kind present gets its own hatch key so sub-floor is never read as genuinely free.
    handles = _channel_key(star=True)
    for kind in ("free", "sub_micro"):
        if kind in kinds:
            handles.append(
                Patch(
                    # MATCH THE MARKER'S FILL, not a grey stand-in: the plotted floor marker is
                    # `facecolor=_colour(row)`, so a free stub is orange and a sub-micro paid stub
                    # is blue. A grey legend swatch promised a colour no row is drawn with.
                    facecolor=_FREE if kind == "free" else _PAID,
                    edgecolor="white",
                    hatch="////" if kind == "free" else "xxxx",
                    label=_floor_text(kind),
                )
            )
    ax.legend(
        handles=handles,
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
        if p.n < _INSUFFICIENT_N:
            # A one-cell rate is 0% or 100% by construction and would draw as an invisible bar
            # or a full one; a hatched stub plus a label marks "measured but not estimable".
            ax.barh(
                y,
                _INSUFFICIENT_STUB,
                height=0.62,
                color=_colour(row),
                alpha=0.5,
                hatch="////",
                edgecolor="white",
                linewidth=0.0,
            )
            ax.text(
                _INSUFFICIENT_STUB + 1.0,
                y,
                f"insufficient n (n={p.n}, ${p.mean_cost:.4f})",
                va="center",
                fontsize=6.0,
                color=_INK,
            )
            continue
        rate = 100.0 * p.pass_rate
        lo, hi = plot_style.wilson_interval(p.passes, p.n)
        ax.barh(y, rate, height=0.62, color=_colour(row), alpha=0.9)
        ax.plot([lo * 100, hi * 100], [y, y], color=_INK, linewidth=0.8)
        ax.plot([lo * 100, lo * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.plot([hi * 100, hi * 100], [y - 0.16, y + 0.16], color=_INK, linewidth=0.8)
        ax.text(hi * 100 + 1.0, y, f"{rate:.0f}% (n={p.n})", va="center", fontsize=6.0, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlim(0, _QUALITY_XLIM)
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
                data.filter_note,
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
        "Panel A: verified-challenge coverage, with the measured-cell count printed; a row "
        "covering nothing draws a hatched zero stub, not a missing bar. Panel B: measured pass "
        "rate with a 95% Wilson interval, the mean billed cost per task printed beside it. A "
        "row with fewer than the provisional cell floor is drawn hatched and faded in both "
        "panels, so a thin sample is never read as a full measurement. Blue is a paid channel, "
        "orange a free one. Each row carries its first failing reason in the notes below."
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
        _EVIDENCED,
    ),
    notes=(
        "The predicate and the first-failing reason are not restated here — they are read "
        "from benchmark.routing.model_validity, the same source the model-validity figure "
        "draws.",
    ),
    limitations=(
        "A live slot with no committed measurement is flagged invalid here but is a coverage "
        "gap, not evidence of domination; collect it and the verdict can change.",
        "Free-channel rows were collected under a different campaign and their intervals are "
        "wide from thin coverage, not from measured weakness.",
    ),
)


def _coverage_top(covered: list[int]) -> int:
    """The coverage axis top: the maximum, or 1 when every row covers nothing.

    `max(covered) if covered else 1` yielded 0 for a non-empty all-zero roster, collapsing
    the axis and drawing every bar as a bare rule.
    """
    return max(covered) or 1


def _provisional_key(provisional: bool) -> list[Patch | Line2D]:
    """The channel key, plus the provisional-hatch entry when any drawn row is provisional."""
    handles = _channel_key()
    if provisional:
        handles.append(
            Patch(
                facecolor="#BDBDBD",
                edgecolor="white",
                hatch="////",
                label=f"provisional (n < {plot_style.MIN_N_PROVISIONAL} measured cells)",
            )
        )
    return handles


def _draw_invalid_coverage(
    ax: Axes, rows: tuple[UniverseRow, ...], perf: dict[str, Performance], corpus: int
) -> None:
    ys = list(range(len(rows)))[::-1]
    covered = [r.covered for r in rows]
    top = _coverage_top(covered)
    provisional = [plot_style.is_provisional(perf[r.identity].n) for r in rows]
    for y, row, count, prov in zip(ys, rows, covered, provisional, strict=True):
        if count == 0:
            # A zero-coverage row drew a zero-width bar — a bare rule at the axis, which reads
            # as a missing row rather than as a measured zero. A hatched stub marks it as a
            # value, not an absence (the same fix as escalation/plots.py:1451-1465).
            ax.barh(
                y,
                top * 0.01,
                height=0.62,
                color=_colour(row),
                alpha=0.9,
                hatch="////",
                edgecolor="white",
                linewidth=0.0,
            )
        else:
            ax.barh(
                y,
                count,
                height=0.62,
                color=_colour(row),
                alpha=0.5 if prov else 0.9,
                hatch="////" if prov else None,
                edgecolor="white",
                linewidth=0.0,
            )
        ax.text(count + top * 0.02, y, f"{count} ({row.cells} cells)", va="center", fontsize=6.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.canonical for r in rows], fontsize=6.4, family="monospace")
    ax.set_xlim(0, top * 1.42)
    ax.set_xlabel(f"verified challenges covered (of {corpus})", fontsize=8.5)
    ax.set_ylabel("canonical weights identity", fontsize=8.5)
    ax.legend(
        handles=_provisional_key(any(provisional)),
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
    provisional = [plot_style.is_provisional(perf[r.identity].n) for r in rows]
    for y, row, prov in zip(ys, rows, provisional, strict=True):
        p = perf[row.identity]
        if p.n < _INSUFFICIENT_N:
            # A one-cell rate is 0% or 100% by construction: an invisible bar or a full one
            # both read as evidence, so the row draws a hatched stub plus an explicit label.
            ax.barh(
                y,
                _INSUFFICIENT_STUB,
                height=0.62,
                color=_colour(row),
                alpha=0.5,
                hatch="////",
                edgecolor="white",
                linewidth=0.0,
            )
            ax.text(
                _INSUFFICIENT_STUB + 1.2,
                y,
                f"insufficient n (n={p.n}, ${p.mean_cost:.4f})",
                va="center",
                fontsize=6.0,
                color=_INK,
            )
            continue
        rate = 100.0 * p.pass_rate
        lo, hi = plot_style.wilson_interval(p.passes, p.n)
        ax.barh(
            y,
            rate,
            height=0.62,
            color=_colour(row),
            alpha=0.5 if prov else 0.9,
            hatch="////" if prov else None,
            edgecolor="white",
            linewidth=0.0,
        )
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
    # Wide enough for the longest printed label; the key lives on panel A, so this panel draws
    # none (a second copy printed over the step-3.7-flash bar and its label).
    ax.set_xlim(0, _QUALITY_XLIM)
    # The pass-rate axis names ONE quantity. The mean cost is already printed beside every bar,
    # so repeating it in the axis title read as though the axis measured both.
    ax.set_xlabel("measured pass rate % (95% Wilson)", fontsize=8.5)
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
    _draw_invalid_coverage(axes[0], rows, data.perf, data.corpus)
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
                data.filter_note,
                f"the {len(data.valid)} inference-valid models are on the model-validity figure",
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
