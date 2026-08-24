"""Statistical + palette helpers shared by every figure family (see below)."""

# Wilson score CIs on pass-rate estimates, the fixed Okabe-Ito model palette with an
# arm-rank size ramp nested inside each hue, direct-labelling with leader-line fallback,
# the cost-quality Pareto frontier (non-decreasing convex hull) + an AIQ-style area
# scalar, and the tri-state pass/fail/not-sampled encoding.
#
# Ships in the wheel rather than under benchmark/ because the shipped figure code needs
# it and SH006 forbids src/shunt importing benchmark. The corpus-coupled half — anything
# typed on the benchmark's raw challenge x model x arm cache — stays in
# benchmark/routing/plot_style.py, which re-exports this module.
#
# Color note: the model palette is genuine Okabe-Ito (a colorblind-safe standard,
# the categorical convention shared across the benchmark's figures and the task's
# design brief), validated with the dataviz skill's six-check validator — all
# 6 slots PASS in both light (surface #fcfcfb) and dark (#1a1a19) modes on the
# adjacent pairlist; the all-pairs pairlist (scatter/bubble/small-multiples,
# where any two models can be neighbors) lands CVD in the 6-8 floor band, so
# every scatter-form plot using this palette MUST ship secondary encoding
# (direct labels) — never color alone. The same hex values serve both themes:
# Okabe-Ito is an externally fixed standard (not re-stepped per surface);
# contrast, CVD, and the normal-vision floor all clear on both surfaces, only
# the design system's own "lightness band" cosmetic guideline (tuned for its
# own ramps) does not apply.

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.text import Annotation

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Text

Point = tuple[float, float]

# ---------------------------------------------------------------------------
# Wilson score interval — valid near 0/1 and at small n, unlike a normal
# approximation. The #1 fix this design brief calls for: p(arm|model) sampling
# makes per-arm n uneven, so every pass-rate estimate must carry a CI.
# ---------------------------------------------------------------------------

MIN_N_PROVISIONAL: Final[int] = 10

# Shared caveat text for the degrade-gracefully case (only the default arm has
# data yet — the live executor doesn't issue divergent per-arm requests yet).
ARM_SWEEP_PENDING_NOTE: Final[str] = "single-arm data — arm sweep pending"
UNEVEN_COVERAGE_NOTE: Final[str] = "uneven n/N per column is BY DESIGN (p(arm|model) sampling)"


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI (default z=1.96 -> 95%) for a binomial pass rate."""
    if n <= 0:
        return (0.0, 0.0)
    phat = passes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def ci_yerr(rate: float, lo: float, hi: float) -> tuple[float, float]:
    """(lower, upper) error-bar offsets from ``rate`` for matplotlib's ``yerr=``."""
    return (max(0.0, rate - lo), max(0.0, hi - rate))


def is_provisional(n: int, min_n: int = MIN_N_PROVISIONAL) -> bool:
    """True when a sample is too small to trust — render hollow/greyed, not solid."""
    return n < min_n


def ci_footer(method: str = "Wilson", level: float = 0.95) -> str:
    """One-line caption stating the CI method + level (state it, never imply it)."""
    return f"error bars = {level:.0%} {method} CI (binomial)"


# ---------------------------------------------------------------------------
# Okabe-Ito categorical palette — model = hue, fixed order, never cycled.
# Arm-rank is a channel NESTED INSIDE the hue (marker size), never a shared
# axis across models: arms are ordinal WITHIN one model only.
# ---------------------------------------------------------------------------

OKABE_ITO: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#56B4E9",  # sky blue     -- contrast WARN vs white surface: always direct-label
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange       -- contrast WARN vs white surface: always direct-label
    "#F0E442",  # yellow
    "#000000",  # black (reserved; not assigned to a series)
)


def model_color_map(models_in_order: Sequence[str]) -> dict[str, str]:
    """Assign each model its fixed Okabe-Ito hue by POSITION in ``models_in_order``."""
    # Callers must compute this once from the full/stable model list (e.g.
    # tier-then-price order) and pass the same map to every figure THAT USES IT —
    # never re-derive it from a filtered subset, or a model's color would repaint
    # when the subset changes (the recolor-on-filter anti-pattern).
    #
    # NOT every figure uses it, and that is deliberate. Hue is a scarce channel: a
    # panel that already names the model on an axis should spend hue on its OTHER
    # variable, and painting the model twice while the legend explains only the
    # second variable is how a legend ends up contradicting its own bars. Reach for
    # this map where the model has no other channel (a stacked area band); prefer a
    # single hue per series where it does (a grouped or labelled bar).
    return {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(models_in_order)}


# ---------------------------------------------------------------------------
# Arm-rank encoding: marker size grows with within-model reasoning-effort rank.
# ---------------------------------------------------------------------------

_ARM_BASE_SIZE: Final[float] = 45.0
_ARM_SIZE_STEP: Final[float] = 75.0


def arm_marker_size(rank: int, max_rank: int) -> float:
    """Marker area (pts^2) for a scatter point, growing with within-model arm rank."""
    # max_rank is the model's OWN highest rank — never compared across models
    # (a rank-1 arm on a 2-arm model is not the same effort as rank-1 on a
    # 4-arm model; the size only orders arms within one model's facet/hue).
    if max_rank <= 0:
        return _ARM_BASE_SIZE
    frac = max(0.0, min(1.0, rank / max_rank))
    return _ARM_BASE_SIZE + _ARM_SIZE_STEP * frac


def arm_size_legend_values(max_rank: int) -> list[tuple[int, float]]:
    """[(rank, marker_size), ...] spanning 0..max_rank, for a size-legend swatch."""
    return [(r, arm_marker_size(r, max_rank)) for r in range(max_rank + 1)]


# ---------------------------------------------------------------------------
# Tri-state outcome encoding — grey is NEVER a fail, it means "not sampled".
# ---------------------------------------------------------------------------

TRISTATE_PASS: Final[str] = "#2E7D32"
TRISTATE_FAIL: Final[str] = "#C62828"
TRISTATE_UNSAMPLED: Final[str] = "#BDBDBD"


# Candidate label offsets in POINTS, ordered cheapest-first: right, left, above,
# below, then the four diagonals, then the same eight at 2x and 3x radius. A label
# is placed at the first candidate that collides with nothing already on the axes.
_LABEL_DIRECTIONS: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.7, 0.7),
    (-0.7, 0.7),
    (0.0, -1.0),
    (0.7, -0.7),
    (-0.7, -0.7),
)
_LABEL_RADII: Final[tuple[float, ...]] = (9.0, 18.0, 30.0)
# Rough DejaVu Sans advance + line height as a fraction of the point size. Only used
# to reserve a rectangle for a label that has not been drawn yet, so an estimate is
# what is wanted: measuring would need a renderer, and there isn't one until save().
_LABEL_ADVANCE: Final[float] = 0.60
_LABEL_HEIGHT: Final[float] = 1.25
_LABEL_PAD_PT: Final[float] = 2.0


@dataclass(frozen=True)
class LabelPoint:
    """One point to direct-label, with the half-extent of its error bars (0 when none)."""

    x: float
    y: float
    text: str
    yerr_lo: float = 0.0
    yerr_hi: float = 0.0
    # Horizontal extent, for a figure that puts an interval on x as well. The obstacle model
    # knew only vertical bars, so a label was free to land on top of a horizontal one.
    # Defaults keep every caller that has no x interval placing labels exactly as before.
    xerr_lo: float = 0.0
    xerr_hi: float = 0.0
    # Text colour for THIS label, overriding `place_labels`' uniform default. It exists for
    # the crowded-cluster case the offset search cannot solve: when several markers collapse
    # into a few pixels, every label earns a leader, the leaders converge, and which name
    # belongs to which marker stops being readable from geometry. Matching the label to its
    # marker's colour restores the pairing without moving anything. None keeps the caller's
    # uniform colour, which is what every figure with no such cluster wants.
    color: str | None = None


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _inside(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _label_rect(
    cx: float, cy: float, dx: float, dy: float, text: str, fontsize: float
) -> tuple[float, float, float, float]:
    """Pixel rect a label would occupy, anchored so it grows away from its point."""
    w = len(text) * fontsize * _LABEL_ADVANCE + 2 * _LABEL_PAD_PT
    h = fontsize * _LABEL_HEIGHT + 2 * _LABEL_PAD_PT
    x0 = cx + dx - (0.0 if dx > 0 else w if dx < 0 else w / 2.0)
    y0 = cy + dy - (0.0 if dy > 0 else h if dy < 0 else h / 2.0)
    return (x0, y0, x0 + w, y0 + h)


def label_extent(text: str, fontsize: float) -> tuple[float, float]:
    """(width, height) in POINTS that a direct label of this text reserves."""
    # The one place a caller may ask how big a label is. It exists so a panel that has to
    # HOLD a set of labels sizes itself from the same estimate the collision search uses,
    # rather than from a second constant that drifts away from it.
    return (
        len(text) * fontsize * _LABEL_ADVANCE + 2 * _LABEL_PAD_PT,
        fontsize * _LABEL_HEIGHT + 2 * _LABEL_PAD_PT,
    )


def _data_anchored(ax: Axes, text: Text) -> bool:
    """Is this text pinned to a data x, so that widening the axis actually moves it?"""
    # An axes-fraction or figure-anchored label cannot be pulled inside by widening, and
    # chasing one would grow the axis every round without ever converging. An Annotation
    # is data-anchored through its `xy`, not its transform, so it needs its own test.
    if isinstance(text, Annotation):
        return text.xycoords == "data"
    return bool(text.get_transform() is ax.transData)


def fit_end_labels(ax: Axes, *, pad_px: float = 10.0) -> None:
    """Widen the x-axis once so every data-anchored text on it ends inside the axes."""
    # An end-of-bar value label is placed in DATA coordinates but sized in points, so no
    # xlim multiplier picked at authoring time can be right: it depends on the panel's
    # rendered width and the label's character count. `plot_contract` cannot see this
    # class of overflow, so it is measured here instead of guessed.
    #
    # Exactly ONE layout pass, and the widening solved in closed form rather than iterated:
    # a text's pixel WIDTH does not change when the axis widens, only its left edge moves,
    # so the needed span factor follows directly. Iterating meant re-running constrained
    # layout several times per panel, and that is not idempotent — it drifted an axes off
    # the canvas.
    scale = ax.get_xscale()
    fig = ax.get_figure(root=True)
    if fig is None or scale not in ("linear", "log"):
        return
    fig.draw_without_rendering()
    box = ax.get_window_extent()
    width = box.width
    if width <= 1.0:
        return
    factor = 1.0
    for text in ax.texts:
        if not _data_anchored(ax, text):
            continue
        extent = text.get_window_extent()
        room = width - extent.width - pad_px
        if room <= 0.0:
            # Wider than the whole panel: no axis limit can pull it in, and pretending
            # otherwise would squash the bars into the left margin for nothing.
            continue
        factor = max(factor, (extent.x0 - box.x0) / room)
    if factor <= 1.0:
        return
    forward: Callable[[float], float] = math.log10 if scale == "log" else (lambda v: v)
    back: Callable[[float], float] = (lambda v: 10.0**v) if scale == "log" else (lambda v: v)
    lo, hi = ax.get_xlim()
    # Grow the span in the axis's OWN coordinate, so a log panel widens by a factor
    # rather than by an additive amount that a decade would swallow.
    ax.set_xlim(lo, back(forward(lo) + (forward(hi) - forward(lo)) * factor))


def _owns(
    rect: tuple[float, float, float, float],
    own: tuple[float, float],
    markers: Sequence[tuple[float, float]],
) -> bool:
    """Is this label's own marker the nearest one to where the text would sit?"""
    # Not-overlapping is not the same as not-ambiguous. A label placed clear of every
    # obstacle can still land beside a DIFFERENT marker than its own, and a reader
    # assigns a floating name to whatever it is nearest. Rejecting those candidates
    # sends the point to the leader-line fallback, where the arrow says who owns it.
    cx, cy = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
    mine = (cx - own[0]) ** 2 + (cy - own[1]) ** 2
    return all((cx - mx) ** 2 + (cy - my) ** 2 >= mine for mx, my in markers)


def place_labels(  # noqa: C901 (a single offset-search loop; splitting it hides the search)
    ax: Axes,
    points: Sequence[LabelPoint],
    fontsize: float = 8.0,
    color: str = "#333333",
    marker_pad_pt: float = 9.0,
    obstacles: Sequence[tuple[float, float, float, float]] = (),
) -> None:
    """Direct-label points at the nearest free offset; a leader line only as a fallback."""
    # Replaces a margin-column labeller that spread EVERY label down the right edge
    # regardless of where its point sat, so a point on the left of the panel got a
    # leader line crossing the whole figure. Here a label is tried at eight
    # directions x three radii and taken at the first that hits neither another
    # label, nor a marker, nor an error bar; only a point with no free slot at all
    # falls back to a leader line, which is then the exception it was meant to be.
    if not points:
        return
    trans = ax.transData
    # `transform` returns DISPLAY PIXELS while the offsets and font size are in POINTS,
    # and the figure renders at 150 dpi — so every rectangle below was reserved a bit
    # under half its true size and the collision test passed candidates that visibly
    # collide. Everything in this function is done in pixels; only the final `annotate`
    # offset goes back to points.
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    pad_px = marker_pad_pt * scale
    # Anything on the axes this function cannot see for itself — the legend, a magnified
    # inset panel — arrives here as a display-pixel rectangle. Without them a label is free
    # to land on the legend, which is how a key ends up with a strategy name printed across
    # it. Empty by default, so every caller that has no such artist places exactly as before.
    taken: list[tuple[float, float, float, float]] = [tuple(r) for r in obstacles]  # type: ignore[misc]
    # Every marker and every error bar is an obstacle, including those of points
    # that have not been labelled yet — otherwise the first label placed wins a slot
    # the second point's own error bar occupies.
    for p in points:
        px, py = trans.transform((p.x, p.y))
        _, lo_py = trans.transform((p.x, p.y - p.yerr_lo))
        _, hi_py = trans.transform((p.x, p.y + p.yerr_hi))
        left_px, _ = trans.transform((p.x - p.xerr_lo, p.y))
        right_px, _ = trans.transform((p.x + p.xerr_hi, p.y))
        taken.append(
            (
                min(left_px, px) - pad_px,
                min(lo_py, py) - pad_px,
                max(right_px, px) + pad_px,
                max(hi_py, py) + pad_px,
            )
        )

    markers: list[tuple[float, float]] = []
    for q in points:
        qx, qy = trans.transform((q.x, q.y))
        markers.append((float(qx), float(qy)))
    for p in sorted(points, key=lambda q: -q.y):
        px, py = trans.transform((p.x, p.y))
        placed: tuple[float, float] | None = None
        for radius in _LABEL_RADII:
            for dirx, diry in _LABEL_DIRECTIONS:
                # Never annotate below a point that carries a lower error bar: the
                # label would sit on the whisker it is meant to describe.
                if diry < 0 and p.yerr_lo > 0:
                    continue
                dx, dy = dirx * radius, diry * radius
                rect = _label_rect(px, py, dx * scale, dy * scale, p.text, fontsize * scale)
                if any(_rects_overlap(rect, t) for t in taken):
                    continue
                own = (float(px), float(py))
                if not _owns(rect, own, [m for m in markers if m != own]):
                    continue
                placed = (dx, dy)
                taken.append(rect)
                break
            if placed is not None:
                break
        if placed is None:
            _leader_label(ax, p, fontsize, p.color or color, taken, scale)
            continue
        dx, dy = placed
        # A label pushed past the first ring is far enough from its marker that a reader
        # in a crowded region will attach it to a NEIGHBOUR — which is exactly the defect
        # the margin-column labeller produced at scale. Past the first ring the label
        # therefore earns a leader; inside it, proximity is unambiguous on its own.
        far = max(abs(dx), abs(dy)) > _LABEL_RADII[0] + 1e-6
        ax.annotate(
            p.text,
            xy=(p.x, p.y),
            xycoords="data",
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            color=p.color or color,
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            annotation_clip=True,
            arrowprops=(
                dict(arrowstyle="-", color="#aaaaaa", lw=0.6, shrinkA=1, shrinkB=3) if far else None
            ),
        )


def stack_labels(
    ax: Axes,
    points: Sequence[LabelPoint],
    fontsize: float = 8.0,
    color: str = "#333333",
    *,
    obstacles: Sequence[tuple[float, float, float, float]] = (),
    marker_pad_pt: float = 6.0,
) -> list[str]:
    """Label a crowd on stacked levels, each name directly over its own marker on a leader."""
    # WHY THIS EXISTS BESIDE `place_labels`. The offset search is the right answer when a point
    # has room somewhere around it. It is the WRONG answer once markers are closer together than
    # the names are wide: every candidate near the point is nearer some OTHER marker, `_owns`
    # rejects them all, and the leader fallback — which cannot apply an ownership test, because
    # by then no slot passes one — puts the name beside a neighbour with a short line pointing
    # at that neighbour's marker. A confidently mispaired label is worse than a crowded one.
    #
    # A ladder cannot mispair. Each name is CENTRED on its own marker's x and lifted to the
    # lowest level that is free, so the leader is vertical and lands on the marker it names, and
    # reading the levels left to right is reading the markers left to right. It costs vertical
    # room, which is exactly what a magnified panel has and a crowded plane does not.
    #
    # Returns the texts it could NOT place, so a caller is never told a crowd was labelled when
    # part of it was silently dropped.
    if not points:
        return []
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    pad = marker_pad_pt * scale
    frame = (ax.bbox.x0, ax.bbox.y0, ax.bbox.x1, ax.bbox.y1)
    taken: list[tuple[float, float, float, float]] = [tuple(r) for r in obstacles]  # type: ignore[misc]
    marks: list[tuple[float, float]] = []
    for p in points:
        px, py = ax.transData.transform((p.x, p.y))
        marks.append((float(px), float(py)))
        taken.append((float(px) - pad, float(py) - pad, float(px) + pad, float(py) + pad))
    unplaced: list[str] = []
    for (px, py), p in sorted(zip(marks, points, strict=True), key=lambda pair: pair[0][0]):
        width, height = (v * scale for v in label_extent(p.text, fontsize))
        step = height + 0.25 * height
        placed = False
        for level in range(_STACK_LEVELS):
            for direction in (1.0, -1.0):
                dy = direction * (pad + level * step) + (0.0 if direction > 0 else -height)
                dx = _clamp_shift(px, width, frame)
                rect = (px + dx - width / 2, py + dy, px + dx + width / 2, py + dy + height)
                if not _inside(rect, frame) or any(_rects_overlap(rect, t) for t in taken):
                    continue
                taken.append(rect)
                _stacked_annotation(ax, p, fontsize, color, (dx / scale, (dy + height / 2) / scale))
                placed = True
                break
            if placed:
                break
        if not placed:
            unplaced.append(p.text)
    return unplaced


_STACK_LEVELS: Final[int] = 8


def _clamp_shift(px: float, width: float, frame: tuple[float, float, float, float]) -> float:
    """How far a centred label must slide to stay inside the frame (0 when it already is)."""
    left, right = px - width / 2, px + width / 2
    if left < frame[0]:
        return frame[0] - left
    if right > frame[2]:
        return frame[2] - right
    return 0.0


def _stacked_annotation(
    ax: Axes, p: LabelPoint, fontsize: float, color: str, offset: tuple[float, float]
) -> None:
    """One ladder rung: the name, centred, on a leader back to its own marker."""
    ax.annotate(
        p.text,
        xy=(p.x, p.y),
        xycoords="data",
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color=p.color or color,
        ha="center",
        va="center",
        annotation_clip=True,
        arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.6, "shrinkA": 2, "shrinkB": 3},
    )


_LEADER_SLOTS: Final[tuple[tuple[float, float], ...]] = (
    (34.0, 12.0),
    (34.0, -12.0),
    (34.0, 30.0),
    (34.0, -30.0),
    (-34.0, 12.0),
    (-34.0, -12.0),
)


def _leader_label(
    ax: Axes,
    p: LabelPoint,
    fontsize: float,
    color: str,
    taken: list[tuple[float, float, float, float]],
    scale: float,
) -> None:
    """The genuine fallback: a short leader to the first slot the crowd leaves free."""
    # A single fixed slot put every fallback label at the same offset from its own point,
    # so two neighbouring crowded points printed their names on top of each other — the
    # collision the fallback exists to avoid.
    px, py = ax.transData.transform((p.x, p.y))
    dx, dy = _LEADER_SLOTS[0]
    for cand_x, cand_y in _LEADER_SLOTS:
        rect = _label_rect(px, py, cand_x * scale, cand_y * scale, p.text, fontsize * scale)
        if not any(_rects_overlap(rect, t) for t in taken):
            dx, dy = cand_x, cand_y
            taken.append(rect)
            break
    ax.annotate(
        p.text,
        xy=(p.x, p.y),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        ha="left" if dx > 0 else "right",
        va="center",
        annotation_clip=True,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=2, shrinkB=2),
    )


# ---------------------------------------------------------------------------
# Crowd detection and free-space search.
#
# `place_labels` searches 8 directions x 3 radii and then falls back to a leader slot, and
# on a dense enough cluster the FALLBACK SLOTS collide too — which is not crowding, it is
# data loss: two names print on the same pixels and neither is readable. Tuning the offsets
# does not fix that, because the room simply is not there. What fixes it is drawing the
# crowd somewhere else, magnified, and the two functions below are the inputs that decision
# needs: which points are actually crowded, and where the canvas is empty enough to put a
# panel. Both are computed from the rendered geometry, so a figure whose data spreads out
# stops magnifying by itself rather than pointing a hardcoded window at an empty region.
# ---------------------------------------------------------------------------

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class LabelCluster:
    """Points whose preferred labels overlap, plus the DATA box their markers span."""

    members: tuple[int, ...]
    x0: float
    x1: float
    y0: float
    y1: float


def _preferred_rects(
    ax: Axes, points: Sequence[LabelPoint], fontsize: float, scale: float
) -> list[Rect]:
    """Where `place_labels` would put each label on its FIRST try, in display pixels."""
    # The first candidate is _LABEL_DIRECTIONS[0] at _LABEL_RADII[0] — read from the same
    # constants the search uses, so a change there cannot leave this measuring a slot the
    # placer no longer tries.
    dirx, diry = _LABEL_DIRECTIONS[0]
    radius = _LABEL_RADII[0]
    out: list[Rect] = []
    for p in points:
        px, py = ax.transData.transform((p.x, p.y))
        out.append(
            _label_rect(
                float(px),
                float(py),
                dirx * radius * scale,
                diry * radius * scale,
                p.text,
                fontsize * scale,
            )
        )
    return out


def label_clusters(
    ax: Axes,
    points: Sequence[LabelPoint],
    fontsize: float = 8.0,
    *,
    marker_pad_pt: float = 9.0,
) -> list[LabelCluster]:
    """Single-linkage groups of points whose labels cannot all be placed where they sit."""
    # The linkage threshold is the LABEL's own extent — `_label_rect` on the placer's first
    # candidate — never a pixel constant. A longer name, a bigger font or a wider panel all
    # move the threshold on their own, which is what keeps this from rotting when the data,
    # the strategy set or the canvas changes.
    if len(points) < 2:
        return []
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    rects = _preferred_rects(ax, points, fontsize, scale)
    pad = marker_pad_pt * scale
    marks: list[Rect] = []
    for p in points:
        px, py = ax.transData.transform((p.x, p.y))
        marks.append((float(px) - pad, float(py) - pad, float(px) + pad, float(py) + pad))

    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Either the two names would print on each other, or the two markers are so close
            # that no label between them could be attributed to one of them.
            if _rects_overlap(rects[i], rects[j]) or _rects_overlap(marks[i], marks[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)

    clusters = [
        LabelCluster(
            members=tuple(sorted(idx)),
            x0=min(points[k].x for k in idx),
            x1=max(points[k].x for k in idx),
            y0=min(points[k].y for k in idx),
            y1=max(points[k].y for k in idx),
        )
        for idx in groups.values()
        if len(idx) >= 2
    ]
    # Densest first: a caller with room for one panel must magnify the worst crowd, and a
    # caller that runs out of room must be able to say which crowd it could not draw.
    return sorted(clusters, key=lambda c: (-len(c.members), c.members))


def _overlap_area(a: Rect, b: Rect) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _gap(a: Rect, b: Rect) -> float:
    """Euclidean gap between two axis-aligned rects (0 when they touch or overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def free_region(
    width: float,
    height: float,
    obstacles: Sequence[Rect],
    *,
    margin: float = 0.02,
    steps: int = 40,
) -> Rect | None:
    """The emptiest placement in AXES FRACTION for a box of this size, or None if none is."""
    # Emptiest is computed, not chosen: every candidate position on a grid is scored against
    # the obstacle rects the caller declares (markers, intervals, the filled region, the
    # legend, any panel already placed), the ones that touch anything are discarded outright,
    # and the winner is the survivor furthest from its nearest obstacle. Returning None is a
    # real answer — a caller that cannot fit a panel must say so rather than draw it over the
    # legend.
    span_x, span_y = 1.0 - 2.0 * margin - width, 1.0 - 2.0 * margin - height
    if span_x < 0.0 or span_y < 0.0:
        return None
    best: Rect | None = None
    best_gap = -1.0
    for i in range(steps + 1):
        x0 = margin + span_x * i / steps
        for j in range(steps + 1):
            y0 = margin + span_y * j / steps
            cand = (x0, y0, x0 + width, y0 + height)
            if any(_overlap_area(cand, o) > 0.0 for o in obstacles):
                continue
            gap = min((_gap(cand, o) for o in obstacles), default=1.0)
            if gap > best_gap:
                best, best_gap = cand, gap
    return best


# ---------------------------------------------------------------------------
# Cost-quality Pareto frontier: NON-DECREASING CONVEX HULL, not keep-max
# staircase — a mixture router reaches interpolated points on the hull edges.
# ---------------------------------------------------------------------------


def pareto_prune(points: Sequence[Point]) -> list[Point]:
    """Keep only non-dominated (cost, pass_rate) points (lower cost AND higher
    rate beats; ties on both do not dominate)."""
    pts = list(points)
    keep: list[Point] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return keep


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def upper_hull(points: Sequence[Point]) -> list[Point]:
    """Upper convex hull of (cost, pass_rate) points, cost ascending.

    THE CALLER MUST PRE-FILTER to live strategies — see `strategy_class.live_rows`.
    """
    # A mixture router reaches this region by probabilistically interpolating between two
    # strategies' (cost, pass) points — the honest cost-quality frontier, not a keep-max
    # staircase that ignores mixtures. Duplicate costs keep only the highest rate.
    #
    # "Achievable" is a property of the INPUT, not of this function: it takes bare points
    # and cannot tell a shipped router from an oracle that read the answer. Handing it every
    # strategy is how a hull got anchored on Price-Cascade, which the router rejects at boot.
    by_cost: dict[float, float] = {}
    for cost, rate in points:
        if cost not in by_cost or rate > by_cost[cost]:
            by_cost[cost] = rate
    pts = sorted(by_cost.items())
    if len(pts) <= 2:
        return pts
    hull: list[Point] = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def area_under_frontier(hull: Sequence[Point]) -> float:
    """AIQ-style scalar in [0,1]: area under the (cost, pass%) frontier curve,
    normalized by the bounding rectangle (max cost x 100%). Extends flat from
    x=0 at the cheapest hull point's rate when that point's cost is > 0.
    """
    if not hull:
        return 0.0
    pts = list(hull)
    if pts[0][0] > 0:
        pts = [(0.0, pts[0][1]), *pts]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if xs[-1] <= 0:
        return 0.0
    area = float(np.trapezoid(ys, xs))
    return float(area / (xs[-1] * 100.0))


def usd(amount: float, places: int = 2) -> str:
    r"""A dollar amount safe to put in matplotlib text: ``\$12.34``."""
    # Matplotlib treats a PAIR of unescaped ``$`` as mathtext delimiters, so a caption with
    # two amounts silently renders the text between them as italic math (or raises). Every
    # figure footer quoting money must go through this.
    return rf"\${amount:,.{places}f}"
