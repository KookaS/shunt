#!/usr/bin/env python3
"""SH008: real embeddings come from the shipped Embedder; no proxy vectorizers."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from _shared import Finding, run

_CODE = "SH008"

# The ONE module allowed to touch fastembed. Everything else reaches the real model
# through `shunt.router.embedder.Embedder`, which is where the model name resolves from
# embedding.yaml (`SHUNT_EMBEDDER_MODEL` wins), the durable cache_dir is set, the
# max_chars clip is applied, `fingerprint()` is derived, and the
# SHUNT_DISALLOW_REAL_EMBEDDER test wall lives. A second raw `TextEmbedding(...)` bypasses
# all five: it pins its own model name, so flipping the active row silently scores two
# strategies in two different embedding spaces with nothing recording the split.
_EMBEDDER_MODULE = "src/shunt/router/embedder.py"

# Proxy featurizers are banned outright outside tests: "real-only in benchmark and live"
# means a benchmark vector is a real embedding, never a bag-of-words stand-in — an
# honestly-captioned fake is still a fake.
_BANNED_MODULES: tuple[str, ...] = ("sklearn.feature_extraction",)
_BANNED_NAMES = frozenset({"TfidfVectorizer", "HashingVectorizer", "CountVectorizer"})

# Tests legitimately patch/fake both: unit-test mocking is the one sanctioned use.
_EXEMPT_ROOTS: tuple[str, ...] = ("tests/",)

_SCAN_ROOTS: tuple[str, ...] = ("src", "benchmark", "tools", "examples", "scripts")


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _is_exempt(path: str) -> bool:
    """True for test code, which may fake or patch the embedder."""
    return any(root in _posix(path) for root in _EXEMPT_ROOTS)


def _is_embedder_module(path: str) -> bool:
    return _posix(path).endswith(_EMBEDDER_MODULE)


def _matches(module: str, prefix: str) -> bool:
    """True iff ``module`` is ``prefix`` or a dotted child of it."""
    return module == prefix or module.startswith(prefix + ".")


def _imported(tree: ast.Module) -> list[tuple[ast.stmt, str, tuple[str, ...]]]:
    """(node, imported module, bound names) for every import statement."""
    out: list[tuple[ast.stmt, str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node, a.name, (a.name,)) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node, node.module, tuple(a.name for a in node.names)))
    return out


def _violation(module: str, names: tuple[str, ...], *, embedder_module: bool) -> str | None:
    """The rule this import breaks, or None."""
    if _matches(module, "fastembed") and not embedder_module:
        return (
            "import fastembed only in src/shunt/router/embedder.py — everything else must "
            "go through shunt.router.embedder.Embedder, which resolves the model from "
            "embedding.yaml and applies the cache dir, char clip and fingerprint"
        )
    if any(_matches(module, banned) for banned in _BANNED_MODULES):
        return (
            f"'{module}' is a proxy featurizer; benchmark and live vectors must be real "
            "embeddings from the shipped Embedder (real-only rule)"
        )
    hit = sorted(_BANNED_NAMES.intersection(names))
    if hit:
        return (
            f"'{hit[0]}' is a proxy featurizer; benchmark and live vectors must be real "
            "embeddings from the shipped Embedder (real-only rule)"
        )
    return None


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag raw fastembed imports outside the Embedder, and any proxy vectorizer import."""
    if _is_exempt(path):
        return []
    embedder_module = _is_embedder_module(path)
    findings: list[Finding] = []
    for node, module, names in _imported(tree):
        message = _violation(module, names, embedder_module=embedder_module)
        if message is not None:
            findings.append(Finding(path, node.lineno, node.col_offset, message))
    return findings


def _default_paths() -> list[str]:
    """Every non-test source file — the scan target when pre-commit passes no filenames."""
    found: set[Path] = set()
    for root in _SCAN_ROOTS:
        found.update(Path(root).rglob("*.py"))
    return sorted(str(p) for p in found if not _is_exempt(str(p)))


if __name__ == "__main__":
    # Mirror SH004/SH005/SH006: scan the whole surface when given no paths, so an
    # unstaged file cannot smuggle a raw embedder construction past the gate.
    _argv = sys.argv[1:]
    if not [a for a in _argv if not a.startswith("-")]:
        _argv = _argv + _default_paths()
    sys.exit(run("SH008", _CODE, check, _argv))
