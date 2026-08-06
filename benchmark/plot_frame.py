"""Figure frame: a claim on the canvas, the full record in the manifest."""

# One contract for every figure under benchmark/ (routing + escalation).
#
# ON THE CANVAS: a claim-style title, a subtitle carrying n / units / operating
# point, and — only where a reader could be actively misled — ONE red caveat line.
# Nothing else. A plot is a display, not a document; the five-section footer this
# module used to render was longer than some of the plots it sat under, and on the
# 30-row sweep table it collided with the data.
#
# OFF THE CANVAS: `reading`, `goal`, `definitions`, `notes` and `limitations` stay
# MANDATORY in code. They are written to benchmark/<half>/figures.json, and SH009
# holds that manifest in bijection with the per-figure sections of docs/routing.md
# and docs/escalation.md. The explanation did not get deleted; it moved to where
# prose belongs, and a gate now proves it is there.
#
# LAYOUT: the title band is drawn in figure coordinates and the content is confined
# to a reserved rect below it, so a single axes, a 2x3 grid, a heatmap with a
# colorbar and a 30-row table all get the same band with no per-figure math and no
# overlap. `bbox_inches="tight"` is deliberately NOT used: it made the size system
# meaningless (the PNG came out however wide the widest artist made it) and it hid
# exactly the clipping that `plot_contract.audit` now asserts against.

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import matplotlib.pyplot as plt
from matplotlib.layout_engine import ConstrainedLayoutEngine

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

INK: Final[str] = "#1a1a1a"
MUTED: Final[str] = "#555555"
CAVEAT_RED: Final[str] = "#B71C1C"

DPI: Final[int] = 150

MAX_TITLE_CHARS: Final[int] = 90
MAX_CAVEAT_CHARS: Final[int] = 120

_TITLE_PT: Final[float] = 13.0
_SUBTITLE_PT: Final[float] = 9.0
_CAVEAT_PT: Final[float] = 9.0
_LINE_SPACING: Final[float] = 1.35
# Glyph advance as a fraction of point size for DejaVu Sans, used to pick a wrap column.
# Deliberately the WORST realistic case, not the mean, and weight-aware. Measured over the
# strings these bands actually carry: regular tops out at 0.599 (a capitalised claim) against
# a 0.524 mean, and BOLD is 12% wider again at 0.670 — which is why one constant silently
# overflowed the bold caveat while the regular subtitle beside it fitted. An under-wrapped
# line runs off the canvas and the layout audit rejects it; an over-wrapped one costs at most
# one extra band line, so erring wide is the cheap direction.
_GLYPH_ADVANCE: Final[float] = 0.62
_GLYPH_ADVANCE_BOLD: Final[float] = 0.70
_BAND_PAD_TOP_IN: Final[float] = 0.10
_BAND_PAD_BOTTOM_IN: Final[float] = 0.14
_BLOCK_GAP_IN: Final[float] = 0.05
_LEFT_X: Final[float] = 0.012
# Constrained layout solves to the rect it is given and then rounds to device pixels,
# so a colorbar or a long tick label lands a few pixels outside a rect flush with the
# canvas. A half-percent inset is cheaper than loosening the clipping assertion.
_RECT_INSET: Final[float] = 0.006

_STRICT_ENV: Final[str] = "SHUNT_PLOT_STRICT"


@dataclass(frozen=True)
class FigureSize:
    """A named canvas. The only accepted figsize, so the set reads as one family."""

    name: str
    width_in: float
    height_in: float

    @property
    def figsize(self) -> tuple[float, float]:
        return (self.width_in, self.height_in)


SINGLE: Final[FigureSize] = FigureSize("single", 10.0, 6.0)
SINGLE_TALL: Final[FigureSize] = FigureSize("single_tall", 10.0, 7.5)
SQUARE: Final[FigureSize] = FigureSize("square", 7.5, 7.0)
WIDE: Final[FigureSize] = FigureSize("wide", 13.0, 6.0)
WIDE_TALL: Final[FigureSize] = FigureSize("wide_tall", 13.0, 8.0)

SIZES: Final[dict[str, FigureSize]] = {
    s.name: s for s in (SINGLE, SINGLE_TALL, SQUARE, WIDE, WIDE_TALL)
}


def table_size(n_rows: int, *, width_in: float = 11.0) -> FigureSize:
    """Rows drive height so a 30-row sweep is legible and a 4-row one is not whitespace."""
    # `width_in` is the one honest escape from the fixed set: a table's width is set by its
    # COLUMN COUNT, which the caller knows and this cannot. Widening is the correct answer to
    # a column that will not fit; dropping the column is not.
    height = min(13.0, max(3.2, 1.15 + 0.30 * max(n_rows, 1)))
    return FigureSize(f"table_{n_rows}", width_in, round(height, 2))


@dataclass(frozen=True)
class Annotations:
    """Runtime-derived content a draw callback computes from the real data in scope."""

    # Rendered:
    subtitle_facts: tuple[str, ...] = ()
    caveat: str | None = None
    # Never rendered — these reach the reader through figures.json and the docs:
    definitions: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Provenance:
    """Which module drew this figure, and over what data."""

    generator: str
    data_digest: str
    manifest: Path


@dataclass(frozen=True)
class FigureSpec:
    """One figure's canvas text plus the full record its docs section is written from."""

    # --- rendered ---
    title: str = ""
    subtitle: str = ""
    caveat: str | None = None
    # --- never rendered; the manifest and the docs carry them ---
    reading: str = ""
    goal: str = ""
    definitions: tuple[tuple[str, str], ...] = field(default=())
    notes: tuple[str, ...] = field(default=())
    limitations: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.reading.strip() or not self.goal.strip():
            raise ValueError("FigureSpec requires both `reading` and `goal`")
        if len(self.title) > MAX_TITLE_CHARS:
            raise ValueError(
                f"title is {len(self.title)} chars, max {MAX_TITLE_CHARS}: {self.title!r}"
            )
        _check_caveat(self.caveat)

    def merged(self, extra: Annotations | None) -> FigureSpec:
        """This spec plus runtime-derived content (duplicates dropped)."""
        if extra is None:
            return self
        subtitle = " · ".join(_dedup((self.subtitle, *extra.subtitle_facts)))
        return FigureSpec(
            title=self.title,
            subtitle=subtitle,
            # First non-None wins, and callers merge most-severe-first, so a corpus
            # that failed its integrity check outranks any per-figure nuance.
            caveat=self.caveat if self.caveat is not None else extra.caveat,
            reading=self.reading,
            goal=self.goal,
            definitions=_dedup_terms(self.definitions + extra.definitions),
            notes=_dedup(self.notes + extra.notes),
            limitations=_dedup(self.limitations + extra.limitations),
        )


def _check_caveat(caveat: str | None) -> None:
    """A caveat cut at the limit ships half a sentence, which is worse than none."""
    if caveat is None:
        return
    if not caveat.strip():
        raise ValueError("caveat is blank — pass None when a figure needs no red line")
    if "\n" in caveat:
        raise ValueError("caveat must be a single line")
    if len(caveat) > MAX_CAVEAT_CHARS:
        raise ValueError(
            f"caveat is {len(caveat)} chars, max {MAX_CAVEAT_CHARS} — shorten it or move "
            f"the detail to the figure's docs section: {caveat!r}"
        )


def _dedup_terms(terms: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Definitions with the first meaning of each term kept (static wins over runtime)."""
    seen: dict[str, str] = {}
    for term, meaning in terms:
        seen.setdefault(term, meaning)
    return tuple(seen.items())


def _dedup(items: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        text = item.strip()
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


# --------------------------------------------------------------------------- band


def _wrap(text: str, width_in: float, pt: float, *, bold: bool = False) -> list[str]:
    # The band starts at _LEFT_X, so the text has that much less room than the canvas — and it
    # needs the same margin on the right, or the last glyph sits flush against the edge. Sizing
    # the wrap to the FULL width (the original bug) guaranteed an overflow on every figure whose
    # subtitle filled its line.
    usable_in = width_in * (1.0 - 2.0 * _LEFT_X)
    advance = _GLYPH_ADVANCE_BOLD if bold else _GLYPH_ADVANCE
    columns = max(30, int(usable_in * 72.0 / (pt * advance)))
    return textwrap.wrap(text, width=columns, break_long_words=True, break_on_hyphens=False) or [
        text
    ]


def _band_blocks(spec: FigureSpec, width_in: float) -> list[tuple[list[str], float, str, str]]:
    """[(lines, point size, colour, weight)] for each rendered block, top to bottom."""
    blocks: list[tuple[list[str], float, str, str]] = []
    if spec.title.strip():
        blocks.append((_wrap(spec.title, width_in, _TITLE_PT, bold=True), _TITLE_PT, INK, "bold"))
    if spec.subtitle.strip():
        blocks.append((_wrap(spec.subtitle, width_in, _SUBTITLE_PT), _SUBTITLE_PT, MUTED, "normal"))
    if spec.caveat:
        blocks.append(
            (_wrap(spec.caveat, width_in, _CAVEAT_PT, bold=True), _CAVEAT_PT, CAVEAT_RED, "bold")
        )
    return blocks


def band_height_inches(spec: FigureSpec, width_in: float) -> float:
    """Height the title band needs. Computed, not measured, so it cannot vary by backend."""
    blocks = _band_blocks(spec, width_in)
    if not blocks:
        return 0.0
    text_in = sum(len(lines) * pt * _LINE_SPACING / 72.0 for lines, pt, _c, _w in blocks)
    gaps_in = _BLOCK_GAP_IN * (len(blocks) - 1)
    return _BAND_PAD_TOP_IN + text_in + gaps_in + _BAND_PAD_BOTTOM_IN


def attach_band(fig: Figure, spec: FigureSpec) -> float:
    """Draw the title band and reserve the rect below it. Returns the band's top in px."""
    width_in, height_in = fig.get_size_inches()
    blocks = _band_blocks(spec, width_in)
    if not blocks:
        return float(fig.bbox.y1)

    band_in = band_height_inches(spec, width_in)
    top = 1.0 - band_in / height_in
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        inset = _RECT_INSET
        engine.set(rect=(inset, inset, 1.0 - 2 * inset, top - 2 * inset))
    else:
        # Legacy path for a figure built outside `new_figure`; constrained layout is
        # the supported one because it also accounts for colorbars.
        fig.subplots_adjust(top=min(top, 0.98))

    y = 1.0 - _BAND_PAD_TOP_IN / height_in
    for lines, pt, colour, weight in blocks:
        fig.text(
            _LEFT_X,
            y,
            "\n".join(lines),
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=pt,
            color=colour,
            fontweight=weight,
            linespacing=_LINE_SPACING,
        )
        y -= (len(lines) * pt * _LINE_SPACING / 72.0 + _BLOCK_GAP_IN) / height_in
    return float(fig.bbox.y1 * top)


# ------------------------------------------------------------------------ figures


def new_figure(size: FigureSize) -> Figure:
    """The one way to make a benchmark figure — named size, constrained layout, fixed dpi."""
    fig = plt.figure(figsize=size.figsize, dpi=DPI)
    fig.set_layout_engine("constrained", w_pad=0.04, h_pad=0.04)
    return fig


def subplots(size: FigureSize, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> tuple[Figure, Any]:
    """`plt.subplots` against a named size, with the layout engine already set."""
    fig = new_figure(size)
    axes = fig.subplots(nrows, ncols, **kwargs)
    return fig, axes


def panel_label(ax: Axes, text: str) -> None:
    """Name one panel of a multi-panel figure. Not a title — the figure has exactly one."""
    ax.set_title(text, fontsize=9.5, color=MUTED, loc="left", pad=4.0)  # noqa: SH007


def save(
    fig: Figure,
    path: Path,
    spec: FigureSpec,
    *,
    extra: Annotations | None = None,
    provenance: Provenance | None = None,
    size: FigureSize | None = None,
) -> Path:
    """Draw the band, audit the layout, write the PNG, record the manifest row."""
    # SH007 denies savefig anywhere else, so every benchmark figure is framed and
    # recorded by construction rather than by whoever remembers to add a caption.
    merged = spec.merged(extra)
    try:
        band_top_px = attach_band(fig, merged)
        if os.environ.get(_STRICT_ENV) == "1":
            from benchmark import plot_contract

            plot_contract.assert_clean(fig, path.name, band_top_px=band_top_px)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI)  # noqa: SH007 (the frame itself)
    finally:
        plt.close(fig)
    if provenance is not None:
        record(path, merged, extra, provenance, size)
    return path


def render(
    path: Path,
    spec: FigureSpec,
    draw: Callable[[Axes], Annotations | None],
    *,
    size: FigureSize = SINGLE,
    provenance: Provenance | None = None,
) -> Path:
    """Single-axes convenience: make the figure, run `draw`, band it, save it."""
    fig = new_figure(size)
    ax = fig.subplots()
    try:
        extra = draw(ax)
    except BaseException:
        plt.close(fig)
        raise
    return save(fig, path, spec, extra=extra, provenance=provenance, size=size)


# ----------------------------------------------------------------------- manifest


def manifest_row(
    spec: FigureSpec,
    extra: Annotations | None,
    provenance: Provenance,
    size: FigureSize | None,
) -> dict[str, Any]:
    """The figure's full record — everything the canvas no longer says."""
    counts = dict(extra.counts) if extra is not None else {}
    return {
        "generator": provenance.generator,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "caveat": spec.caveat,
        "reading": spec.reading,
        "goal": spec.goal,
        "terms": [list(pair) for pair in spec.definitions],
        "notes": list(spec.notes),
        "limitations": list(spec.limitations),
        "n": counts,
        "size": size.name if size is not None else None,
        "figsize": list(size.figsize) if size is not None else None,
        "dpi": DPI,
        "data_digest": provenance.data_digest,
    }


def record(
    path: Path,
    spec: FigureSpec,
    extra: Annotations | None,
    provenance: Provenance,
    size: FigureSize | None,
) -> None:
    """Upsert this figure's row. Eight processes write routing figures, so never truncate."""
    manifest = provenance.manifest
    payload: dict[str, Any] = {"schema": 1, "half": manifest.parent.name, "figures": {}}
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        payload.setdefault("figures", {})
    payload["figures"][path.name] = manifest_row(spec, extra, provenance, size)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic on purpose: no timestamp, no git sha. Either would dirty the file
    # on every regeneration and turn a meaningful diff into noise.
    # Written through a temp file + os.replace: several producers write the routing
    # manifest, and a partial write here loses every row another one just recorded.
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = manifest.with_suffix(manifest.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, manifest)


def prune(manifest: Path, live: Sequence[str]) -> list[str]:
    """Drop rows whose PNG no longer exists. Only a complete run may call this."""
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text())
    figures = payload.get("figures", {})
    orphans = sorted(set(figures) - set(live))
    for name in orphans:
        del figures[name]
    if orphans:
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return orphans
