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
# Exemption is anchored to a real `tests` PATH SEGMENT, not a substring: a file under
# `benchmark/latests/` or `mytests/` is production code and must not ride the exemption.
_EXEMPT_SEGMENTS = ("tests",)

_SCAN_ROOTS: tuple[str, ...] = ("src", "benchmark", "tools", "examples", "scripts")

# --- PROXY-VECTOR patterns (hash / random vectors used as embeddings) ---------
# A benchmark vector must be a real embedding from the shipped Embedder — never a
# hashlib digest coerced into floats, never a numpy random array, never draws from a
# seeded RNG. These detectors are deliberately PRECISE rather than blanket bans,
# because the shipped router legitimately hashes (state keys) and seeds RNGs
# (exploration samplers): a digest only trips the gate when it is CONVERTED into a
# numeric vector, and an RNG only when a draw on it is used as a vector.
_DIGEST_ALGOS = frozenset(
    {
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
    }
)
_DIGEST_METHODS = frozenset({"digest", "hexdigest"})
# Calls that turn raw bytes into a numeric/byte vector — the "hash becomes features" move.
_VECTOR_CONVERTERS = frozenset(
    {
        "numpy.frombuffer",
        "numpy.array",
        "numpy.asarray",
        "numpy.fromiter",
        "numpy.fromstring",
        "list",
        "tuple",
        "array.array",
        "struct.unpack",
    }
)
# numpy.random draws that produce values (excluded: seed/shuffle/permutation/choice,
# which set state or sample existing sequences rather than mint a vector).
_RANDOM_DRAWS = frozenset(
    {
        "rand",
        "randn",
        "random",
        "random_sample",
        "sample",
        "randint",
        "random_integers",
        "uniform",
        "normal",
        "standard_normal",
        "multivariate_normal",
        "poisson",
        "binomial",
        "beta",
        "gamma",
        "exponential",
    }
)
_RNG_CONSTRUCTORS = frozenset(
    {"numpy.random.default_rng", "numpy.random.RandomState", "numpy.random.Generator"}
)


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _is_exempt(path: str) -> bool:
    """True for test code, which may fake or patch the embedder.

    Anchored on a path SEGMENT equal to ``tests`` — ``tests/`` substring matching
    exempted ``benchmark/latests/`` and ``mytests/`` (production code) by accident.
    """
    segments = _posix(path).split("/")
    return any(segment in segments for segment in _EXEMPT_SEGMENTS)


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


# --- name resolution (aliases so the detectors survive `import numpy as np`) ----


def _alias_map(tree: ast.Module) -> dict[str, object]:
    """Local name -> module path or (module, attr) for every import."""
    aliases: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    aliases.setdefault(a.name.split(".")[0], a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                aliases[a.asname or a.name] = (node.module, a.name)
    return aliases


def _resolve_name(name: str, aliases: dict[str, object]) -> str:
    resolved = aliases.get(name)
    if resolved is None:
        return name
    if isinstance(resolved, tuple):
        return f"{resolved[0]}.{resolved[1]}"
    if isinstance(resolved, str):
        return resolved
    return name


def _call_path(func: ast.expr, aliases: dict[str, object]) -> str | None:
    """Dotted path of a call target, resolving import aliases; None if unresolvable."""
    if isinstance(func, ast.Name):
        return _resolve_name(func.id, aliases)
    if isinstance(func, ast.Attribute):
        base = _call_path(func.value, aliases)
        if base is None:
            return None
        return f"{base}.{func.attr}"
    return None


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _is_ancestor(ancestor: ast.AST, node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    cur = node
    while cur in parents:
        if parents[cur] is ancestor:
            return True
        cur = parents[cur]
    return False


def _is_bytes_digest(node: ast.AST) -> bool:
    """True for a ``.digest()`` expression (raw bytes) — a hexdigest is a string.

    ``foo.digest()`` parses as Call(func=Attribute(..., 'digest')), so both the
    attribute and its invocation call qualify; a ``.hexdigest()`` does not.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "digest"
    return isinstance(node, ast.Attribute) and node.attr == "digest"


def _embedding_msg(what: str) -> str:
    """Full finding text for the proxy-vector rules (one shared tail)."""
    return f"{what} — real embeddings come from the shipped Embedder (real-only rule)"


def _digest_vector_msg(
    node: ast.Call, path: str, parents: dict[ast.AST, ast.AST], aliases: dict[str, object]
) -> str | None:
    """Message if the digest at ``node`` is converted or sliced into a feature vector."""
    cur: ast.AST = node
    while True:
        parent = parents.get(cur)
        if parent is None:
            return None
        if isinstance(parent, ast.Call):
            if (
                isinstance(cur, ast.Attribute)
                and cur.attr in _DIGEST_METHODS
                and parent.func is cur
            ):
                cur = parent  # the digest()/hexdigest() method invocation itself
                continue
            parent_path = _call_path(parent.func, aliases)
            # consumed by a numeric-vector conversion with the digest as an argument
            if parent_path in _VECTOR_CONVERTERS and _is_ancestor(parent, cur, parents):
                return _embedding_msg(f"feature vector built from a {path}() digest")
            return None
        if isinstance(parent, ast.Subscript):
            # digest sliced into a fixed-size byte vector (`digest()[:N]`); a
            # HEXDIGEST slice is a string (state keys etc.) and stays unflagged
            if _is_bytes_digest(parent.value):
                return _embedding_msg(f"feature vector built from a {path}() digest")
            return None
        if isinstance(parent, ast.Attribute) and parent.attr in _DIGEST_METHODS:
            cur = parent
            continue
        return None


def _hashlib_vector_nodes(
    tree: ast.Module, parents: dict[ast.AST, ast.AST], aliases: dict[str, object]
) -> list[tuple[ast.Call, str]]:
    """A hashlib digest used AS a feature vector (converted or sliced to bytes)."""
    hits: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node.func, aliases)
        if not path or not path.startswith("hashlib."):
            continue
        if path.rsplit(".", 1)[-1] not in _DIGEST_ALGOS:
            continue
        msg = _digest_vector_msg(node, path, parents, aliases)
        if msg is not None:
            hits.append((node, msg))
    return hits


def _rng_bound_name(node: ast.AST, aliases: dict[str, object]) -> tuple[str, str] | None:
    """``(name, kind)`` if ``node`` assigns a Name to an RNG constructor, else None."""
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        path = _call_path(node.value.func, aliases)
        kind = (
            "generator"
            if path in _RNG_CONSTRUCTORS
            else "random_instance"
            if path == "random.Random"
            else None
        )
        if kind is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    return target.id, kind
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.target, ast.Name)
    ):
        path = _call_path(node.value.func, aliases)
        if path in _RNG_CONSTRUCTORS:
            return node.target.id, "generator"
    return None


def _rng_bindings(tree: ast.Module, aliases: dict[str, object]) -> dict[str, str]:
    """name -> 'generator' | 'random_instance' for names assigned an RNG constructor."""
    # Global (not per-function) on purpose: reusing a name across scopes for a different
    # purpose is rare, and the escape hatch (an SH008 noqa) covers it. A field access
    # like ``stream.rng.random()`` never matches — the receiver is an Attribute, not a bound
    # Name, which is how escalation.py's legit sampler stays clean.
    bound: dict[str, str] = {}

    def scan(body: list[ast.stmt]) -> None:
        for node in body:
            for n in ast.walk(node):
                hit = _rng_bound_name(n, aliases)
                if hit is not None:
                    bound[hit[0]] = hit[1]

    scan(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan(node.body)
    return bound


def _in_vector_context(
    node: ast.AST, parents: dict[ast.AST, ast.AST], aliases: dict[str, object]
) -> bool:
    """True if the node sits inside a comprehension, list/tuple literal or a
    vector-building call — the contexts where a random draw becomes a feature vector."""
    cur: ast.AST = node
    while True:
        parent = parents.get(cur)
        if parent is None:
            return False
        if isinstance(parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return True
        if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
            return True
        if isinstance(parent, ast.Call):
            # a method call ON the tracked expression (`random.Random(0).random()`):
            # keep climbing, the receiver is the thing we are tracing
            if isinstance(parent.func, ast.Attribute) and (
                parent.func.value is node or _is_ancestor(parent.func.value, node, parents)
            ):
                cur = parent
                continue
            parent_path = _call_path(parent.func, aliases)
            # any other call consumes the value without vectorizing it — stop looking
            return parent_path in _VECTOR_CONVERTERS
        if isinstance(parent, ast.stmt):
            return False
        cur = parent


def _bound_draw_msg(node: ast.Call, bound: dict[str, str]) -> str | None:
    """Message if the call draws from a name bound to a generator / random.Random."""
    if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
        return None
    kind = bound.get(node.func.value.id)
    if kind is None or node.func.attr not in _RANDOM_DRAWS:
        return None
    source = "a seeded RNG" if kind == "generator" else "a random.Random"
    return _embedding_msg(f"{node.func.attr}() draw on {source} instance as a feature vector")


def _inline_rng_draw_msg(node: ast.Call, aliases: dict[str, object]) -> str | None:
    """Message for the chained form ``np.random.default_rng(0).random(384)``."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in _RANDOM_DRAWS:
        return None
    if not isinstance(node.func.value, ast.Call):
        return None
    inner = _call_path(node.func.value.func, aliases)
    if inner not in _RNG_CONSTRUCTORS:
        return None
    return _embedding_msg(f"feature vector built from a draw on {inner}()")


def _random_vector_nodes(
    tree: ast.Module, parents: dict[ast.AST, ast.AST], aliases: dict[str, object]
) -> list[tuple[ast.Call, str]]:
    """A numpy.random draw, or a seeded-RNG draw, used as a feature vector."""
    hits: list[tuple[ast.Call, str]] = []
    bound = _rng_bindings(tree, aliases)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node.func, aliases)
        msg = None
        if path is not None and path.startswith("numpy.random."):
            draw = path.rsplit(".", 1)[-1]
            if draw in _RANDOM_DRAWS:
                msg = _embedding_msg(f"feature vector built from {path}()")
        elif path == "random.Random" and _in_vector_context(node, parents, aliases):
            msg = _embedding_msg("random.Random() constructed inside a vector-building expression")
        if msg is None:
            msg = _bound_draw_msg(node, bound)
        if msg is None:
            msg = _inline_rng_draw_msg(node, aliases)
        if msg is not None:
            hits.append((node, msg))
    return hits


def _import_module_nodes(
    tree: ast.Module, aliases: dict[str, object]
) -> list[tuple[ast.Call, str]]:
    """importlib.import_module(<literal>) naming a banned module — invisible to the import scan."""
    hits: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_path(node.func, aliases) != "importlib.import_module":
            continue
        if (
            not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            continue
        target = node.args[0].value
        if not _matches(target, "fastembed") and not any(
            _matches(target, banned) for banned in _BANNED_MODULES
        ):
            continue
        hits.append(
            (
                node,
                _embedding_msg(
                    f"importlib.import_module('{target}') bypasses the import scan — "
                    f"{target} is a proxy featurizer / embedder module"
                ),
            )
        )
    return hits


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag raw fastembed imports, proxy vectorizer imports, and hash/random vectors."""
    if _is_exempt(path):
        return []
    embedder_module = _is_embedder_module(path)
    findings: list[Finding] = []
    for node, module, names in _imported(tree):
        message = _violation(module, names, embedder_module=embedder_module)
        if message is not None:
            findings.append(Finding(path, node.lineno, node.col_offset, message))
    aliases = _alias_map(tree)
    parents = _parents(tree)
    for call, message in _hashlib_vector_nodes(tree, parents, aliases):
        findings.append(Finding(path, call.lineno, call.col_offset, message))
    for call, message in _random_vector_nodes(tree, parents, aliases):
        findings.append(Finding(path, call.lineno, call.col_offset, message))
    for call, message in _import_module_nodes(tree, aliases):
        findings.append(Finding(path, call.lineno, call.col_offset, message))
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
