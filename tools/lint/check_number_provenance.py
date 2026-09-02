#!/usr/bin/env python3
"""SH012: every number-bearing site — markdown prose OR producer string literal — is sourced."""

# EXACT SCOPE — read this before quoting a green exit as coverage. The gate has two
# independent dimensions, and it is complete in the first and deliberately partial in
# the second:
#   FILES  — every ``*.md`` in the tree, skip-dirs pruned. Complete: README.md,
#            CHANGELOG.md, docs/**, benchmark/**, examples/**, governance files.
#            PLUS every FIGURE-PRODUCER ``*.py`` (see ``_produces_figure_text``), where
#            the same claims live as string literals rather than prose.
#   LINES  — only lines matching ``_SIGNATURE`` (markdown) or ``_PY_SIGNATURE`` (literals)
#            below — the result-statistic SHAPES. Partial by design: a result number
#            phrased outside those shapes is NOT enforced. A green exit means "no line in
#            a KNOWN shape lacks a marker", never "every number here is sourced".
#
# WHY THE PYTHON DIMENSION EXISTS. The gate scanned ``*.md`` only, and the canvas strings a
# producer freezes into a title, a note or a limitation are published just as widely as the
# prose that quotes them — SH009 byte-locks the docs TO those literals, so a wrong literal
# propagates into the docs as a matched pair and every gate stays green. Two shipped that way:
# a figure title naming a pair of AUROCs the code no longer derives, and a limitation naming an
# imputed-cell count off by a dozen. A number a producer COMPUTES needs no marker (it cannot go
# stale); a number it FREEZES into a literal must say where it came from, or be interpolated.
# The file dimension used to be a hardcoded three-doc list (docs/results.md,
# docs/benchmark.md, docs/benchmark-data.md) whose comment claimed a whole-tree scan.
# CHANGELOG.md and README.md published result statistics under zero enforcement for as
# long as that list stood; widening the walk is what closed it, so do not narrow it back
# to a list — a new doc must be covered by existing, not by remembering to add it.
#
# The story this gate enforces: "Six sites publish numbers that NO committed entrypoint
# reproduces — when the corpus grew they went stale and could only be flagged, not
# corrected." A number-bearing site is a line that publishes a benchmark result
# statistic (a strategy-row variant, a paired comparison, a task-count
# claim, a pp-delta-with-CI claim, a scorable/usable count, or a task-set restatement).
# Each such site must carry EXACTLY ONE provenance marker, placed inline as an HTML
# comment on the same line:
#
#     <!-- generated-by: <entrypoint> -->            a committed entrypoint emits it
#     <!-- frozen-value: n=<N>, date=<YYYY-MM-DD>, run=<RUN> -->   a named snapshot
#
# A frozen marker without a corpus (any of n / date / run missing) is a defect: a frozen
# value is allowed to be old, it is not allowed to be anonymous. A signature match with
# no marker at all is a defect too — that is how a seventh site gets added silently.
#
# The gate is a whole-tree scan (the SH004 shape, not SH007's file filter): the
# failure it catches is an addition or an edit, and only a scan of every committed
# markdown file sees every number-bearing line. ``pass_filenames: false`` +
# ``always_run: true``.
#
# Marker authoring:
#   - generator sites: ``<!-- generated-by: benchmark.routing.report:paired_quality_contrast -->``
#   - frozen sites:    ``<!-- frozen-value: n=180, date=2026-08-10, run=49b8362 -->``

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

from _shared import Finding

_CODE = "SH012"
_ROOT = Path(__file__).resolve().parents[2]

# Directories that are not shipped text: VCS, virtualenvs, caches, build output.
# Mirrors SH004's `_SKIP_DIRS` — the same walk, over `*.md` only.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        "site",
        ".eggs",
    }
)

# The discovery signature: the shapes of the six result-statistic sites. A line matching
# any alternative is a number-bearing site and must carry a marker. Kept deliberately
# narrow — the shapes of the six sites, not every number in the docs — so a NEW site
# added in the same class fails, while incidental prose numbers do not need markers.
# The literal `167 of 175` / `170 of 177` alternatives are a RATCHET, not a guarantee:
# they freeze the exact phrasing of two known sites, so if the corpus grows and prose
# restates those bare numbers differently the gate will miss them — that is the accepted
# gap (shapes only), and a regression in the six known shapes is what it still catches.
_SIGNATURE = re.compile(
    r"^\|.*rank_shortlist=0\b"  # Session-Cascade variant rows
    r"|^\|.*`sl=\d+`\s+vs"  # paired per-task comparison rows
    r"|\b\d+ of \d+ (?:shared |usable )?tasks?"  # task-count claims
    r"|\b\d+ scorable\b"  # set-size claims
    r"|\b\d+ usable tasks?"  # census claims
    r"|[+-]\d+\.\d+ ?pp, CI crosses zero"  # pp-delta-with-CI claims
    r"|\b\d+-task set\b"  # "74-task set" / "175-task set"
    r"|Set [AB] \(\d+ tasks?\)"  # Set A/B comparison rows
    r"|over the 175\b"  # "…over the 175 shared tasks"
    r"|\b167 of 175\b"
    r"|\b170 of 177\b"
)

# The producer-literal signature. Deliberately SHAPES ONLY — no literal from the current
# corpus appears in it. Seeding a threshold from the state it exists to constrain is how the
# markdown half acquired its `\b167 of 175\b` ratchet rows; this half does not repeat that,
# so a number that changes tomorrow is caught by the same pattern that catches it today.
# Percent shapes are excluded on measurement: `95%`/`.0%` in a producer are method vocabulary
# and format specs, ~100 of them, none a frozen result.
_PY_SIGNATURE = re.compile(
    r"\b\d+ of \d+\b"  # count-of-total claims: "397 of 398 filled cells"
    r"|\b\d+/\d+\b"  # ratio claims: "406/1104 scored cells"
    r"|\b\d+\.\d+ vs \d+\.\d+\b"  # two-metric comparisons: "AUROC 0.601 vs 0.781"
)
# The machinery that turns a string into published figure text. A file that names any of it
# draws a canvas, so the scope EXTENDS ITSELF: a new producer is in scope the day it is
# written, with nobody remembering to list it.
_FIGURE_TEXT_MARKERS = ("FigureSpec", "plot_frame", "draw_frame")
_PY_MARKER = re.compile(r"#\s*(?:generated-by:\s*\S+|frozen-value:)")
_PY_FROZEN = re.compile(
    r"#\s*frozen-value:\s*n=(?P<n>\S+),\s*date=(?P<date>\S+),\s*run=(?P<run>\S+)"
)
_PY_FROZEN_ANY = re.compile(r"#\s*frozen-value:")

_GENERATED = re.compile(r"<!--\s*generated-by:\s*(?P<entrypoint>\S+)\s*-->")
# Frozen markers must carry n, date AND run — presence without a corpus is a defect.
_FROZEN = re.compile(
    r"<!--\s*frozen-value:\s*"
    r"n=(?P<n>\S+),\s*date=(?P<date>\S+),\s*run=(?P<run>\S+)\s*-->"
)
_FROZEN_ANY = re.compile(r"<!--\s*frozen-value:")  # presence probe


def _scan_doc(root: Path, rel: str) -> list[Finding]:
    """Return findings for one doc: every signature match must carry a valid marker."""
    path = root / rel
    findings: list[Finding] = []
    if not path.exists():
        return findings
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not _SIGNATURE.search(line):
            continue
        generated = _GENERATED.search(line)
        frozen = _FROZEN.search(line)
        if generated:
            continue  # generator marker present — byte-for-byte regen is a test's job
        if _FROZEN_ANY.search(line):
            # Marker present but the corpus is incomplete — a defect.
            if frozen is None:
                findings.append(
                    Finding(
                        rel,
                        lineno,
                        0,
                        "number-bearing site carries a frozen-value marker WITHOUT a "
                        "complete corpus (n, date and run are all required) — a frozen "
                        "value is allowed to be old, it is not allowed to be anonymous. "
                        f"line: {line.strip()[:80]}",
                    )
                )
            continue
        findings.append(
            Finding(
                rel,
                lineno,
                0,
                "number-bearing site carries NO provenance marker (neither "
                "`<!-- generated-by: … -->` nor `<!-- frozen-value: n=…, date=…, "
                "run=… -->`). Every published result statistic must say whether a "
                "committed entrypoint emits it or which snapshot it is frozen on. "
                f"line: {line.strip()[:80]}",
            )
        )
    return findings


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Every string Constant that is a docstring — prose ABOUT the code, not figure text."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        body = node.body
        first = body[0] if body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.add(id(first.value))
    return out


def _produces_figure_text(text: str) -> bool:
    """Does this module build published figure text? The scope test, content-based."""
    return any(marker in text for marker in _FIGURE_TEXT_MARKERS)


def _scan_producer(root: Path, rel: str) -> list[Finding]:
    """Every frozen number-bearing STRING LITERAL in a producer must carry a marker."""
    # An f-string's interpolated value is not a literal and is never flagged: a number the
    # producer derives cannot go stale. What is flagged is a constant — including the constant
    # PARTS of an f-string, because digits typed between the placeholders are just as frozen.
    path = root / rel
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    docstrings = _docstring_ids(tree)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings or not _PY_SIGNATURE.search(node.value):
            continue
        span = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
        if _PY_FROZEN_ANY.search(span):
            if _PY_FROZEN.search(span) is None:
                findings.append(
                    Finding(
                        rel,
                        node.lineno,
                        0,
                        "frozen figure-text literal carries a frozen-value marker WITHOUT a "
                        "complete corpus (n, date and run are all required). "
                        f"literal: {node.value.strip()[:80]}",
                    )
                )
            continue
        if _PY_MARKER.search(span):
            continue
        findings.append(
            Finding(
                rel,
                node.lineno,
                0,
                "figure-text literal FREEZES a result number and carries no provenance "
                "marker. Derive it from the data instead (an f-string over the computed "
                "value cannot go stale), or add `# generated-by: <entrypoint>` / "
                "`# frozen-value: n=…, date=…, run=…` on the literal's line. "
                f"literal: {node.value.strip()[:80]}",
            )
        )
    return findings


def _iter_producers(root: Path) -> list[str]:
    """Every `*.py` under root that builds figure text, repo-relative, skip-dirs pruned."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            # A test fixture legitimately hardcodes numbers; it publishes nothing.
            if "tests" in rel.parts:
                continue
            try:
                if _produces_figure_text(path.read_text(encoding="utf-8")):
                    found.append(str(rel))
            except (OSError, UnicodeDecodeError):
                continue
    return sorted(found)


def _check_subjects(root: Path, docs: list[str], producers: list[str]) -> list[Finding]:
    """A green exit over an EMPTY subject set proves nothing — refuse to report one."""
    # Both walks are content-discovered, so either can silently collapse to zero (a moved docs
    # tree, a renamed spec class) and the exit code would still be 0. The demand is STRICT on
    # the path that actually executes — the hook runs with no `--root`, so `_ROOT` is the
    # coverage being claimed and BOTH dimensions must have subjects there. Under an explicit
    # `--root` (fixtures) only total emptiness is refused: a fixture legitimately exercises one
    # dimension at a time.
    strict = root == _ROOT
    empty = []
    if not docs and (strict or not producers):
        empty.append("no *.md files")
    if not producers and (strict or not docs):
        empty.append(f"no figure producers (nothing names any of {_FIGURE_TEXT_MARKERS})")
    if not empty:
        return []
    return [
        Finding(
            str(root),
            1,
            0,
            "scanned an EMPTY subject set — " + " and ".join(empty) + ". A green exit here "
            "would report that nothing was checked, not that nothing was wrong.",
        )
    ]


def _iter_docs(root: Path) -> list[str]:
    """Every `*.md` under root, repo-relative, skip-dirs pruned during the walk."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        found.extend(
            str((Path(dirpath) / name).relative_to(root))
            for name in sorted(filenames)
            if name.endswith(".md")
        )
    return sorted(found)


def main(argv: list[str]) -> int:
    """Scan every markdown file; print every finding and exit non-zero on any."""
    root = _ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    findings: list[Finding] = []
    docs, producers = _iter_docs(root), _iter_producers(root)
    findings.extend(_check_subjects(root, docs, producers))
    for rel in docs:
        findings.extend(_scan_doc(root, rel))
    for rel in producers:
        findings.extend(_scan_producer(root, rel))
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{_CODE} ERROR] {f.message}")  # noqa: T201
    if findings:
        print(  # noqa: T201
            f"\n{_CODE}: {len(findings)} problem(s). Every number-bearing site must "
            "carry a provenance marker: `<!-- generated-by: <entrypoint> -->` or "
            "`<!-- frozen-value: n=<N>, date=<YYYY-MM-DD>, run=<RUN> -->`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
