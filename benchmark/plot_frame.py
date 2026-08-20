"""Re-export shim: the figure frame now ships in `shunt.inspect.plot_frame`."""

# The frame moved into the wheel so the shipped figure code can draw through the same
# contract without importing `benchmark` (SH006 forbids that direction). It used to be
# mirrored — a trimmed copy under src/shunt/inspect with a "keep them in sync" comment —
# which is a drift bug waiting to be paid for; there is now exactly one implementation.
# Every `benchmark.plot_frame` import in the routing and escalation halves, and in the
# tests, keeps working unchanged.
#
# Importing this module forces the matplotlib Agg backend, because the shipped frame does.

from __future__ import annotations

from shunt.inspect.plot_frame import (
    _GLYPH_ADVANCE,  # noqa: F401 (re-export: tests pin the measured wrap constants)
    _GLYPH_ADVANCE_BOLD,  # noqa: F401 (re-export: tests pin the measured wrap constants)
    CAVEAT_RED,
    DPI,
    INK,
    MAX_CAVEAT_CHARS,
    MAX_TITLE_CHARS,
    MUTED,
    SINGLE,
    SINGLE_TALL,
    SIZES,
    SQUARE,
    WIDE,
    WIDE_TALL,
    Annotations,
    FigureSize,
    FigureSpec,
    Provenance,
    _wrap,  # noqa: F401 (re-export: tests pin the measured wrap constants)
    attach_band,
    band_height_inches,
    manifest_row,
    new_figure,
    panel_label,
    prune,
    record,
    render,
    save,
    subplots,
    table_size,
)

__all__ = [
    "CAVEAT_RED",
    "DPI",
    "INK",
    "MAX_CAVEAT_CHARS",
    "MAX_TITLE_CHARS",
    "MUTED",
    "SINGLE",
    "SINGLE_TALL",
    "SIZES",
    "SQUARE",
    "WIDE",
    "WIDE_TALL",
    "Annotations",
    "FigureSize",
    "FigureSpec",
    "Provenance",
    "attach_band",
    "band_height_inches",
    "manifest_row",
    "new_figure",
    "panel_label",
    "prune",
    "record",
    "render",
    "save",
    "subplots",
    "table_size",
]
