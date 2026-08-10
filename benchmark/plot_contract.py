"""Layout audit: prove a rendered figure has no overlap and nothing clipped."""

# Every figure in this repo is read as a PNG, far from the code that drew it. Two
# defects therefore ship silently: an artist drawn on top of another (the sweep
# table rendered its title THROUGH its own rows for months) and an artist drawn off
# the canvas (a heatmap's leftmost cell label came out as ".78"). Both are visible
# to the renderer and invisible to every other gate we have, so they are asserted
# here rather than caught by whoever next opens the image.
#
# This runs INSIDE plot_frame.save() under SHUNT_PLOT_STRICT=1, which the test
# suite and `make benchmark-figures` both set. That is deliberate: it makes every
# existing render test a layout test, so there is no per-figure test list to keep in
# sync with a figure set that is about to change.
#
# WHAT IT CANNOT SEE, stated so nobody reads a pass as more than it is:
#   * text-over-text INSIDE the axes (annotation labels colliding with each other or
#     with data marks). An all-pairs check is O(n^2) on a 500-point scatter and
#     false-positives wherever near-overlap is intentional.
#   * a legend covering data — same reason, and it fights matplotlib's own solver.
#   * an artist with clip_on=True: it is excluded from get_tightbbox, so a label that
#     was drawn but clipped to invisibility passes here.
#   * legibility and truth. It cannot tell you 9pt is too small, that two hues are
#     indistinguishable, or that the title's claim is false.
# It is also font- and backend-dependent: conftest pins Agg and DejaVu Sans, and
# matplotlib's minor version is pinned, because get_tightbbox semantics have moved
# between releases.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from matplotlib.table import Table

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.transforms import BboxBase

# Sub-pixel slack: matplotlib rounds device coordinates, and an artist sitting
# exactly on a boundary is not a defect.
_TOL_PX: Final[float] = 1.0


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


class LayoutError(AssertionError):
    """A figure was about to be written with an overlapping or clipped artist."""


def _contains(outer: BboxBase, inner: BboxBase, tol: float = _TOL_PX) -> bool:
    return (
        inner.x0 >= outer.x0 - tol
        and inner.y0 >= outer.y0 - tol
        and inner.x1 <= outer.x1 + tol
        and inner.y1 <= outer.y1 + tol
    )


def _intersects(a: BboxBase, b: BboxBase, tol: float = _TOL_PX) -> bool:
    return a.x0 < b.x1 - tol and b.x0 < a.x1 - tol and a.y0 < b.y1 - tol and b.y0 < a.y1 - tol


def _label(ax: Axes) -> str:
    for candidate in (ax.get_ylabel(), ax.get_xlabel(), ax.get_label()):
        if candidate:
            return str(candidate)
    return "<unlabelled axes>"


def _audit_canvas(fig: Figure, renderer: RendererBase, canvas: BboxBase) -> list[Violation]:
    """Nothing may leave the canvas — bbox_inches='tight' is gone, so this is real."""
    out: list[Violation] = []
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        if box is not None and not _contains(canvas, box):
            out.append(Violation("outside_canvas", f"axes {_label(ax)!r} extends to {_fmt(box)}"))
    for text in fig.texts:
        if not text.get_visible() or not text.get_text():
            continue
        if not _contains(canvas, text.get_window_extent(renderer)):
            out.append(Violation("outside_canvas", f"figure text {_snip(text.get_text())}"))
    return out


def _audit_band(fig: Figure, renderer: RendererBase, band_top_px: float) -> list[Violation]:
    """No axes may reach into the reserved title band."""
    out: list[Violation] = []
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        if box is not None and box.y1 > band_top_px + _TOL_PX:
            out.append(
                Violation(
                    "band_intruded",
                    f"axes {_label(ax)!r} top {box.y1:.0f}px is above the band floor "
                    f"{band_top_px:.0f}px — the title will be drawn over it",
                )
            )
    return out


def _audit_text_over_axes(fig: Figure, renderer: RendererBase) -> list[Violation]:
    """The band's own text must not land on a plot."""
    out: list[Violation] = []
    boxes = [(ax, ax.get_tightbbox(renderer)) for ax in fig.axes]
    for text in fig.texts:
        if not text.get_visible() or not text.get_text():
            continue
        extent = text.get_window_extent(renderer)
        for ax, box in boxes:
            if box is not None and _intersects(extent, box):
                out.append(
                    Violation(
                        "text_over_axes",
                        f"{_snip(text.get_text())} overlaps axes {_label(ax)!r}",
                    )
                )
    return out


def _audit_tables(fig: Figure, renderer: RendererBase) -> list[Violation]:
    """A table scaled past its axes is not clipped by matplotlib — it just overflows."""
    out: list[Violation] = []
    for ax in fig.axes:
        for child in ax.get_children():
            if not isinstance(child, Table):
                continue
            extent = child.get_window_extent(renderer)
            if not _contains(ax.bbox, extent, tol=2.0):
                out.append(
                    Violation(
                        "table_overflow",
                        f"table in axes {_label(ax)!r} extends to {_fmt(extent)} beyond "
                        f"{_fmt(ax.bbox)} — use bbox=[0,0,1,1] instead of Table.scale()",
                    )
                )
    return out


def _drawn_tick_labels(ax: Axes) -> list[Text]:
    """Tick labels that are actually rendered."""
    # Matplotlib keeps locator ticks beyond the view limits (an axis on [0,1] carries ticks
    # at -0.2 and 1.2); their Text artists report visible but are never drawn, so measuring
    # them reports a clip on every well-formed figure.
    if not ax.axison:  # ax.axis("off") — a table or an annotation-only panel
        return []
    drawn: list[Text] = []
    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(lo, hi), max(lo, hi)
        for tick, label in zip(axis.get_ticklocs(), axis.get_ticklabels(), strict=False):
            if lo <= tick <= hi and label.get_visible() and label.get_text():
                drawn.append(label)
    return drawn


def _audit_tick_labels(fig: Figure, renderer: RendererBase, canvas: BboxBase) -> list[Violation]:
    """A tick label half off the canvas reads as a different number ('.78' for '0.78')."""
    out: list[Violation] = []
    for ax in fig.axes:
        for label in _drawn_tick_labels(ax):
            if not _contains(canvas, label.get_window_extent(renderer)):
                out.append(
                    Violation(
                        "outside_canvas",
                        f"tick label {label.get_text()!r} on axes {_label(ax)!r} is clipped",
                    )
                )
    return out


def audit(fig: Figure, *, band_top_px: float | None = None) -> list[Violation]:
    """Draw once, then check every artist's real device-space extent."""
    fig.canvas.draw()
    renderer: RendererBase = fig.canvas.get_renderer()
    canvas = fig.bbox

    violations = _audit_canvas(fig, renderer, canvas)
    if band_top_px is not None:
        violations += _audit_band(fig, renderer, band_top_px)
        violations += _audit_text_over_axes(fig, renderer)
    violations += _audit_tables(fig, renderer)
    violations += _audit_tick_labels(fig, renderer, canvas)
    return violations


def assert_clean(fig: Figure, name: str, *, band_top_px: float | None = None) -> None:
    """Raise rather than write a figure with an overlapping or clipped artist."""
    violations = audit(fig, band_top_px=band_top_px)
    if violations:
        joined = "\n  ".join(str(v) for v in violations)
        raise LayoutError(f"{name}: {len(violations)} layout violation(s)\n  {joined}")


def _fmt(box: BboxBase) -> str:
    return f"({box.x0:.0f},{box.y0:.0f})-({box.x1:.0f},{box.y1:.0f})"


def _snip(text: str, width: int = 40) -> str:
    flat = " ".join(text.split())
    return repr(flat if len(flat) <= width else flat[: width - 1] + "…")
