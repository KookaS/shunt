#!/usr/bin/env python3
"""SH007: every figure is created, titled and saved through a plot frame."""

from __future__ import annotations

import ast
import os
import sys
from typing import Final

from _shared import Finding, run

_CODE = "SH007"
# The frame is the only legal site for all three calls below. It reserves the title band (a
# caller-owned title cannot be measured before the axes rect exists), pins the named canvas
# sizes, owns the savefig, and records the figure's full record into the manifest named by its
# `Provenance` — benchmark/<half>/figures.json, which SH009 then holds in bijection with the
# docs. `shunt inspect`'s diagnostics pass no `Provenance`, so they write no row. There is one
# implementation, in src/shunt/inspect/plot_frame.py; benchmark/plot_frame.py re-exports it and
# is listed here because a `# noqa: SH007` opt-out could otherwise be needed on the shim.
_FRAME_MODULES: Final[tuple[str, ...]] = (
    os.path.join("benchmark", "plot_frame.py"),
    os.path.join("src", "shunt", "inspect", "plot_frame.py"),
)

_SAVE_MESSAGE = (
    "save figures through a plot frame (benchmark.plot_frame.save()/render() or "
    "shunt.inspect.plot_frame.save()) so the figure carries its claim title band — a bare "
    "savefig ships a plot no one can read away from the docs, and one no gate can trace "
    "back to its data"
)
_CREATE_MESSAGE = (
    "build figures with a plot frame's new_figure(SIZE)/subplots(SIZE, ...) — a raw "
    "plt.figure/plt.subplots picks an ad-hoc figsize with no layout engine, which is how the "
    "set ended up with 15 different canvas sizes and a title drawn over its own table"
)
_TITLE_MESSAGE = (
    "the frame draws the one title; set it via FigureSpec(title=..., subtitle=...) and use "
    "a plot frame's panel_label(ax, ...) to caption a panel — a caller-owned title cannot be "
    "measured before the axes rect is reserved, so it overlaps the content"
)

_WRITERS = frozenset({"savefig", "print_figure", "print_png"})
# Only the pyplot module-level constructors. `fig.subplots(...)` on a figure the frame already
# made is the supported way to lay panels out, so it must NOT be flagged.
_CREATORS = frozenset({"figure", "subplots", "subplot_mosaic"})
_PYPLOT = frozenset({"plt", "pyplot"})
_TITLERS = frozenset({"set_title", "suptitle"})
# `plt.title(...)` is the pyplot-state spelling of the same defect, but `.title` is also
# `str.title`, so it counts only when the chain roots at pyplot.
_PYPLOT_TITLERS = frozenset({"title"})
# `ax.set(title=...)` sets a title without ever naming `set_title`.
_SETTER = "set"
_SETTER_TITLE_KEYS = frozenset({"title"})


def _root_name(node: ast.expr) -> str | None:
    """Leftmost identifier of an attribute chain (`matplotlib.pyplot.figure` -> `matplotlib`)."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _call_violation(node: ast.Call, *, allow_raw_figures: bool) -> str | None:
    """The rule this CALL breaks, or None. Bare references are handled separately."""
    func = node.func
    if isinstance(func, ast.Attribute):
        # `ax.set(title=...)` — a title that never spells `set_title`.
        if func.attr == _SETTER and any(
            kw.arg in _SETTER_TITLE_KEYS for kw in node.keywords if kw.arg
        ):
            return _TITLE_MESSAGE
        if func.attr in _PYPLOT_TITLERS and _root_name(func.value) in _PYPLOT:
            return _TITLE_MESSAGE
        # `plt.subplots()` is a violation; `fig.subplots()` is the supported path.
        if not allow_raw_figures and func.attr in _CREATORS and _root_name(func.value) in _PYPLOT:
            return _CREATE_MESSAGE
        return None
    if isinstance(func, ast.Name):
        # The bare form a `from matplotlib.pyplot import savefig, subplots` makes available —
        # matching only the attribute form left one import away from bypassing the gate.
        if func.id == "getattr":
            return _getattr_violation(node)
        if not allow_raw_figures and func.id in _CREATORS:
            return _CREATE_MESSAGE
    return None


def _getattr_violation(node: ast.Call) -> str | None:
    """`getattr(fig, "savefig")` reaches the writer without ever naming it as an attribute."""
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return None
    name = node.args[1].value
    if name in _WRITERS:
        return _SAVE_MESSAGE
    if name in _TITLERS:
        return _TITLE_MESSAGE
    return None


def _reference_violation(node: ast.expr) -> str | None:
    """The rule a bare NAME/ATTRIBUTE reference breaks — an alias saves just as well."""
    # Matching only the call form let `writer = fig.savefig; writer(path)`, a bound method
    # handed to a callback, and `functools.partial(fig.savefig, path)` through untouched. The
    # reference itself is the violation, so a call is caught by the same rule.
    if isinstance(node, ast.Attribute):
        if node.attr in _WRITERS:
            return _SAVE_MESSAGE
        if node.attr in _TITLERS:
            return _TITLE_MESSAGE
        return None
    if isinstance(node, ast.Name) and node.id in _WRITERS:
        return _SAVE_MESSAGE
    return None


def _is_test(path: str) -> bool:
    parts = os.path.normpath(path).split(os.sep)
    return "tests" in parts or parts[-1].startswith("test_")


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag figure creation, titling and saving outside a plot frame module."""
    if os.path.normpath(path).endswith(_FRAME_MODULES):
        return []
    # Tests may build a figure by hand: they exercise draw functions in isolation, and the
    # layout-audit tests must be able to construct a DELIBERATELY broken figure to prove the
    # audit catches it. The save and title rules still apply to them — a test that writes a
    # PNG outside the frame is the same escape hatch as production code doing it.
    allow_raw_figures = _is_test(path)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
            continue
        message = None
        if isinstance(node, ast.Call):
            message = _call_violation(node, allow_raw_figures=allow_raw_figures)
        if message is None:
            message = _reference_violation(node)
        if message is not None:
            findings.append(Finding(path, node.lineno, node.col_offset, message))
    return _dedup(findings)


def _dedup(findings: list[Finding]) -> list[Finding]:
    """One finding per source position — a called attribute matches two rules at one spot."""
    seen: set[tuple[int, int]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.line, f.col)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


if __name__ == "__main__":
    # require_subjects: an empty file list is a broken invocation, not a clean tree.
    sys.exit(run("SH007", _CODE, check, sys.argv[1:], require_subjects=True))
