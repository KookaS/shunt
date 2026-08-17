#!/usr/bin/env python3
"""SH012: every number-bearing site in any markdown file carries a provenance marker."""

# EXACT SCOPE — read this before quoting a green exit as coverage. The gate has two
# independent dimensions, and it is complete in the first and deliberately partial in
# the second:
#   FILES  — every ``*.md`` in the tree, skip-dirs pruned. Complete: README.md,
#            CHANGELOG.md, docs/**, benchmark/**, examples/**, governance files.
#   LINES  — only lines matching ``_SIGNATURE`` below (the result-statistic SHAPES).
#            Partial by design: a result number phrased outside those shapes is NOT
#            enforced. A green exit means "no line in a KNOWN shape lacks a marker",
#            never "every number in the docs is sourced".
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
    for rel in _iter_docs(root):
        findings.extend(_scan_doc(root, rel))
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
