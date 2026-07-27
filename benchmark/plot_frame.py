"""Self-sufficient figure frame: every benchmark plot carries how to read it."""

# One contract for every figure under benchmark/ (routing + escalation): a
# five-section footer — READ / GOAL / TERMS / NOTE / LIMITS — of which READ and
# GOAL are mandatory. The figures are read far from their docs (README, an issue,
# the reports/ directory), so the canvas must answer "what am I looking at, what
# am I looking for, what does this word mean, what must I not conclude".
#
# Static vs data-derived: reading/goal/definitions describe the figure's
# construction and are fixed per plot; notes/limitations also accept an
# `Annotations` returned by the draw callback, computed from the real data in
# scope (sample counts, coverage gaps, single-arm data, a no-skill detector) so a
# caveat can never go stale when the data grows.
#
# Layout: the footer is drawn BELOW the figure box and picked up by
# bbox_inches="tight". That needs no per-figure layout math, so one mechanism
# serves a single axes, a 2x3 grid, a heatmap with a colorbar, and a suptitle
# alike — none of which survive a fixed subplots_adjust band.

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

INK: Final[str] = "#333333"
MUTED: Final[str] = "#555555"
LIMIT_RED: Final[str] = "#B71C1C"

# Point size of footer body text on a standard 10in-wide figure; narrower figures
# step down so a small escalation plot is not swallowed by its own caption.
_BASE_FONTSIZE: Final[float] = 7.5
_MIN_FONTSIZE: Final[float] = 6.0
_REFERENCE_WIDTH_IN: Final[float] = 10.0
# Mean glyph advance as a fraction of point size for DejaVu Sans at these sizes —
# used only to pick a wrap column, so an approximation is fine.
_GLYPH_ADVANCE: Final[float] = 0.52
_LINE_SPACING: Final[float] = 1.45


@dataclass(frozen=True)
class Annotations:
    """Runtime-derived footer content a draw callback computes from real data."""

    # definitions: a term that only exists on SOME renders (a strategy that was
    # plotted this run, an encoding that degenerates on single-arm data).
    definitions: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureSpec:
    """What a reader needs to read one figure without any surrounding docs."""

    # reading: axes, how to read it, what to expect. goal: what to look FOR,
    # spatially where possible ("aim top-left: high pass rate at low cost").
    reading: str
    goal: str
    definitions: tuple[tuple[str, str], ...] = field(default=())
    notes: tuple[str, ...] = field(default=())
    limitations: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.reading.strip() or not self.goal.strip():
            raise ValueError("FigureSpec requires both `reading` and `goal`")

    def merged(self, extra: Annotations | None) -> FigureSpec:
        """This spec plus runtime-derived sections (duplicates dropped)."""
        if extra is None:
            return self
        return FigureSpec(
            reading=self.reading,
            goal=self.goal,
            definitions=_dedup_terms(self.definitions + extra.definitions),
            notes=_dedup(self.notes + extra.notes),
            limitations=_dedup(self.limitations + extra.limitations),
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


def _fontsize(width_in: float) -> float:
    """Footer point size, scaled down for narrow figures (never below the floor)."""
    scaled = _BASE_FONTSIZE * min(1.0, width_in / _REFERENCE_WIDTH_IN)
    return max(_MIN_FONTSIZE, round(scaled, 1))


def _wrap_columns(width_in: float, fontsize: float) -> int:
    """How many characters fit on one footer line at this width and point size."""
    return max(40, int(width_in * 72.0 / (fontsize * _GLYPH_ADVANCE)))


def _definitions_text(definitions: Sequence[tuple[str, str]]) -> str:
    return " · ".join(f"{term} = {meaning}" for term, meaning in definitions)


def footer_blocks(spec: FigureSpec) -> list[tuple[str, str, str]]:
    """[(label, body, color)] for each non-empty section, in fixed reading order."""
    blocks: list[tuple[str, str, str]] = [
        ("READ", spec.reading, INK),
        ("GOAL", spec.goal, INK),
    ]
    if spec.definitions:
        blocks.append(("TERMS", _definitions_text(spec.definitions), MUTED))
    if spec.notes:
        blocks.append(("NOTE", " · ".join(spec.notes), MUTED))
    if spec.limitations:
        blocks.append(("LIMITS", " · ".join(spec.limitations), LIMIT_RED))
    return blocks


def _wrapped_lines(label: str, body: str, columns: int) -> list[str]:
    """Section text wrapped to `columns`, continuation lines hanging under the label."""
    indent = " " * (len(label) + 2)
    return textwrap.wrap(
        f"{label}: {body}",
        width=columns,
        subsequent_indent=indent,
        # A token longer than the wrap column (a pasted path, a hash) is broken rather
        # than allowed to set the figure width — bbox_inches="tight" sizes the canvas
        # to the widest artist, so one long word would stretch the whole PNG.
        break_long_words=True,
        break_on_hyphens=False,
    ) or [f"{label}: {body}"]


def attach_footer(fig: Figure, spec: FigureSpec) -> None:
    """Draw the footer below the figure box (bbox_inches='tight' picks it up)."""
    width_in, height_in = fig.get_size_inches()
    fontsize = _fontsize(width_in)
    columns = _wrap_columns(width_in, fontsize)
    line_step = (fontsize * _LINE_SPACING) / 72.0 / height_in
    y = -0.045
    for label, body, color in footer_blocks(spec):
        lines = _wrapped_lines(label, body, columns)
        fig.text(
            0.0,
            y,
            "\n".join(lines),
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=color,
            linespacing=_LINE_SPACING,
        )
        y -= line_step * len(lines) + line_step * 0.45


def save(
    fig: Figure,
    path: Path,
    spec: FigureSpec,
    *,
    extra: Annotations | None = None,
    dpi: int = 150,
) -> Path:
    """Attach the footer, write the PNG, close the figure. The ONLY savefig site."""
    # SH007 denies savefig anywhere else, so every benchmark figure is annotated
    # by construction rather than by whoever remembers to add a caption.
    # The figure is closed in `finally`: a report run renders dozens of figures, so a
    # driver that catches one failure and continues must not accumulate open figures.
    try:
        attach_footer(fig, spec.merged(extra))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")  # noqa: SH007 (the frame itself)
    finally:
        plt.close(fig)
    return path


def render(
    path: Path,
    spec: FigureSpec,
    draw: Callable[[Axes], Annotations | None],
    *,
    figsize: tuple[float, float] = (10.0, 6.0),
    dpi: int = 150,
) -> Path:
    """Single-axes convenience: make the figure, run `draw`, footer it, save it."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        extra = draw(ax)
    except BaseException:
        plt.close(fig)
        raise
    return save(fig, path, spec, extra=extra, dpi=dpi)
