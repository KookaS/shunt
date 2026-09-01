#!/usr/bin/env python3
"""SH002: cap module/class/function docstrings at MAX_LINES non-blank lines."""

from __future__ import annotations

import ast
import sys
from typing import Final

from _shared import Finding, run

_CODE = "SH002"
# The ceiling. It exists to stop a wall of text standing in for code that reads, NOT to push
# design rationale out of the file: at 3 lines every hit was prose moved verbatim into a
# comment three lines below the docstring, which caught no defect and twice nearly cost a
# recorded rationale. 20 non-blank lines still fails a genuine wall of text and never fires on
# an intent line plus a paragraph. Counted on `ast.get_docstring(clean=True)`, so the quote
# lines do not count. Escape one case with a `noqa: SH002` comment after the closing quotes
# (a multi-line docstring's opening line is inside the string, so nothing can be written there).
MAX_LINES: Final[int] = 20
_Documentable = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag any docstring whose non-blank line count exceeds the cap."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, _Documentable):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc is None:
            continue
        n = sum(1 for line in doc.splitlines() if line.strip())
        if n > MAX_LINES:
            line, end = _docstring_span(node)
            findings.append(
                Finding(
                    path,
                    line,
                    0,
                    f"docstring {n} non-blank lines > {MAX_LINES}; lead with one intent line and "
                    "move the rest into prose near the code",
                    noqa_line=end,
                )
            )
    return findings


def _docstring_span(node: ast.AST) -> tuple[int, int]:
    """First and last source line of the docstring expression, or (1, 1) for a module."""
    # The last line matters: a multi-line docstring's FIRST line is inside the string, so
    # a `noqa: SH002` comment can only be written after the closing quotes.
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr):
        return body[0].lineno, body[0].end_lineno or body[0].lineno
    return 1, 1


if __name__ == "__main__":
    sys.exit(run("SH002", _CODE, check, sys.argv[1:]))
