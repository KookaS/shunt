#!/usr/bin/env python3
"""SH007: every benchmark figure is saved through the annotated plot frame."""

from __future__ import annotations

import ast
import os
import sys

from _shared import Finding, run

_CODE = "SH007"
# The frame itself is the one legal savefig site; it is what attaches the
# READ/GOAL/TERMS/NOTE/LIMITS footer that makes a figure readable on its own.
_FRAME_MODULE = os.path.join("benchmark", "plot_frame.py")
_MESSAGE = (
    "save figures through benchmark.plot_frame.save()/render() so the figure "
    "carries its READ/GOAL/TERMS/NOTE/LIMITS footer — a bare savefig ships a "
    "plot no one can read away from the docs"
)


_WRITERS = frozenset({"savefig", "print_figure", "print_png"})


def _is_savefig(node: ast.Call) -> bool:
    """True for any figure-writing call, attribute or bare name."""
    # Attribute form (fig.savefig, plt.savefig, fig.canvas.print_figure) AND the bare
    # form a `from matplotlib.pyplot import savefig` makes available — matching only
    # the attribute form left an import away from bypassing the whole gate.
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in _WRITERS
    return isinstance(node.func, ast.Name) and node.func.id in _WRITERS


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag savefig calls outside the plot frame module."""
    if os.path.normpath(path).endswith(_FRAME_MODULE):
        return []
    return [
        Finding(path, node.lineno, node.col_offset, _MESSAGE)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_savefig(node)
    ]


if __name__ == "__main__":
    sys.exit(run("SH007", _CODE, check, sys.argv[1:]))
