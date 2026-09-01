"""Shared helpers for the SH0xx custom AST lint checks."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """One rule violation at a file location."""

    path: str
    line: int
    col: int
    message: str
    # Where the `noqa` comment for this finding may be written, when that is not `line`. A
    # multi-line docstring's first line is INSIDE the string, so a comment cannot be put
    # there at all — SH002's documented hatch was unreachable for exactly the docstrings
    # that violate it until the check started naming the closing-quote line here.
    noqa_line: int | None = None


def line_has_noqa(source: str, lineno: int, code: str) -> bool:
    """Return True if the 1-based source line carries ``# noqa: <code>``."""
    lines = source.splitlines()
    if not 1 <= lineno <= len(lines):
        return False
    comment = lines[lineno - 1].partition("#")[2]
    return "noqa" in comment and code in comment


CheckFn = Callable[[str, ast.Module], list[Finding]]


def run(
    name: str, code: str, check: CheckFn, argv: list[str], *, require_subjects: bool = False
) -> int:
    """Run ``check`` over the file args; exit non-zero on findings unless advisory."""
    advisory = "--advisory" in argv
    paths = [a for a in argv if not a.startswith("-")]
    # A gate handed nothing reports green, which reads exactly like a gate that ran. A check
    # whose caller is supposed to enumerate its subjects (a file list, a glob) says so with
    # `require_subjects` and fails loudly when that enumeration came back empty.
    if require_subjects and not [p for p in paths if p.endswith(".py")]:
        msg = f"[{name}] no Python files to check — refusing to report green on nothing"
        print(msg, file=sys.stderr)
        return 1
    findings = _collect(paths, code, check)
    _report(name, findings, advisory=advisory)
    if findings and not advisory:
        return 1
    return 0


def _collect(paths: Iterable[str], code: str, check: CheckFn) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.endswith(".py"):
            continue
        try:
            source = open(path, encoding="utf-8").read()  # noqa: SIM115
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        findings.extend(
            f
            for f in check(path, tree)
            if not line_has_noqa(source, f.line, code)
            and not (f.noqa_line is not None and line_has_noqa(source, f.noqa_line, code))
        )
    return findings


def _report(name: str, findings: list[Finding], *, advisory: bool) -> None:
    if not findings:
        return
    tag = "ADVISORY" if advisory else "ERROR"
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{name} {tag}] {f.message}", file=sys.stderr)
