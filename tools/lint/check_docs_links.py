#!/usr/bin/env python3
"""SH014: every relative link in the docs tree resolves to a file inside docs_dir."""

# The story this gate enforces: "A link in docs/free-tier-smoke.md pointed at
# examples/integrations/README.md — repo content, not site content. mkdocs resolves a
# relative link against docs_dir, looked for docs/examples/integrations/README.md, found
# nothing, and `mkdocs build --strict` exited 1. It reached `main` because docs.yml was
# push-only and could not run on the PR."
#
# The correct form for a repo file is an absolute URL
# (https://github.com/<org>/<repo>/blob/main/...), which mkdocs does not try to resolve.
# docs/live-smoke-runbook.md carries the same sentence written that way.
#
# SCOPE, deliberately narrower than `mkdocs build --strict`: this checks relative link
# TARGETS only. It buys sub-second offline feedback at commit time with no mkdocs install
# — it does NOT replace the strict build, which also validates nav, anchors and config.
# The `pull_request` trigger on .github/workflows/docs.yml is the complete check; keep
# both.
#
# Whole-tree scan (the SH004/SH011/SH012 shape): a link breaks when its TARGET is deleted
# or moved, which a staged-file list never sees. `pass_filenames: false` + `always_run`.
# File arguments are honoured anyway, so the gate can also be pointed at one document.
#
# TWO LINK GRAMMARS, because two renderers read this repository. A document INSIDE docs_dir is
# published by mkdocs, which resolves a relative target against docs_dir — so escaping the tree
# is an error there however real the file is. A document OUTSIDE docs_dir (CHANGELOG.md, the
# root health files, examples/**/README.md) is read on GitHub, which resolves relative to the
# file — so linking `docs/escalation.md` from CHANGELOG.md is CORRECT, and the only failure
# available is a target that does not exist. Both were previously unchecked: `check()` iterated
# docs_dir alone, so every markdown file outside it was outside the gate by construction.
#
# Escape hatch: `<!-- noqa: SH014 -->` on the same line.

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote

from _shared import Finding

_CODE = "SH014"
_ROOT = Path(__file__).resolve().parents[2]

# Markdown inline links and images: `[text](target)` / `![alt](target)`. The target stops
# at whitespace so a `(path "title")` form yields just the path, and `<...>` is stripped
# below. Reference-style definitions (`[id]: target`) are matched separately.
_INLINE = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)")
_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)", re.MULTILINE)

# Targets mkdocs never resolves as a file on disk.
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "//", "#", "data:")


def _docs_dir(root: Path) -> Path:
    """The site source directory, read from mkdocs.yml rather than hardcoded."""
    config = root / "mkdocs.yml"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.startswith("docs_dir:"):
                return root / line.split(":", 1)[1].strip().strip("\"'")
    return root / "docs"


def _targets(text: str) -> list[tuple[int, str]]:
    """Every (1-based line, raw target) pair in the document, links and images alike."""
    found: list[tuple[int, str]] = []
    for match in _INLINE.finditer(text):
        found.append((text.count("\n", 0, match.start()) + 1, match.group(1)))
    for match in _REFERENCE.finditer(text):
        found.append((text.count("\n", 0, match.start()) + 1, match.group(1)))
    return found


def _resolvable(target: str) -> str | None:
    """The on-disk path a target refers to, or None when it is not a file reference."""
    cleaned = target.strip().strip("<>")
    if not cleaned or cleaned.startswith(_SKIP_PREFIXES):
        return None
    # A fragment or query is addressing within the target, not a different file.
    path = unquote(cleaned.split("#", 1)[0].split("?", 1)[0])
    return path or None


def _check_file(md: Path, docs_dir: Path, root: Path) -> list[Finding]:
    """Every relative link in one document that escapes docs_dir or does not exist."""
    text = md.read_text(encoding="utf-8")
    lines = text.splitlines()
    findings: list[Finding] = []
    # Which renderer publishes THIS file decides which rule applies — see the module header.
    published_by_mkdocs = md.resolve().is_relative_to(docs_dir.resolve())
    for lineno, target in _targets(text):
        if "noqa: SH014" in (lines[lineno - 1] if lineno <= len(lines) else ""):
            continue
        path = _resolvable(target)
        if path is None:
            continue
        resolved = (md.parent / path).resolve()
        rel = md.relative_to(root)
        if published_by_mkdocs and not resolved.is_relative_to(docs_dir.resolve()):
            findings.append(
                Finding(
                    str(rel),
                    lineno,
                    0,
                    f"link {target!r} resolves outside {docs_dir.name}/ — mkdocs cannot "
                    f"publish repo content. Use an absolute "
                    f"https://github.com/KookaS/shunt/blob/main/... URL",
                )
            )
        elif not resolved.exists():
            findings.append(
                Finding(str(rel), lineno, 0, f"link {target!r} points at a missing file")
            )
    return findings


def _tracked_markdown(root: Path) -> list[Path]:
    """Every version-controlled `.md` in the repository, or a glob fallback outside a checkout."""
    # Tracked-only on purpose: an agent wrapper drops untracked CLAUDE.md / scratch notes at the
    # root, and a gate that fails on a file the repository does not ship is noise, not coverage.
    argv = ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"]
    try:
        out = subprocess.run(  # noqa: S603
            argv, capture_output=True, check=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for p in root.rglob("*.md") if ".venv" not in p.parts)
    return sorted((root / name) for name in out.split("\0") if name)


def check(root: Path, files: Sequence[Path] | None = None) -> list[Finding]:
    """Every broken relative link, in `files` when given, else across all tracked markdown."""
    docs_dir = _docs_dir(root)
    # With no file list, scan WIDER than docs_dir: the docs tree is the mkdocs half, and the
    # other half — CHANGELOG.md, the root health files, examples/**/README.md — renders on
    # GitHub and was never scanned at all.
    candidates = _tracked_markdown(root) if files is None else [f.resolve() for f in files]
    findings: list[Finding] = []
    for md in candidates:
        if not md.is_file() or not md.resolve().is_relative_to(root.resolve()):
            continue
        findings.extend(_check_file(md, docs_dir, root))
    return findings


def main(argv: list[str]) -> int:
    """Report every broken link and exit 1 when any exists."""
    root = _ROOT
    rest = list(argv)
    if "--root" in rest:
        i = rest.index("--root")
        root = Path(rest[i + 1]).resolve()
        del rest[i : i + 2]
    # Honour a passed file list (`pass_filenames: true`, or a manual invocation on one doc);
    # with none, scan the whole repository, which is what catches a link broken by a MOVE.
    findings = check(root, [Path(a).resolve() for a in rest] or None)
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{_CODE} ERROR] {f.message}")  # noqa: T201
    if findings:
        print(  # noqa: T201
            f"\n{_CODE}: {len(findings)} broken link(s). A relative link must resolve to "
            "a file inside the docs tree; link repo content by absolute URL."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
