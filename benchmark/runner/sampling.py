"""Deterministic, diversity-first run ordering for cost-safe partial benchmarks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from benchmark.runner.calibration import holdout_score

# A single canonical order whose every prefix spreads across repos and difficulty strata
# and whose prefixes *nest* — ``order[:10] ⊂ order[:20] ⊂ order[:200]`` for a fixed set —
# so raising ``sample_size`` only ever *adds* challenges and already-computed results.csv
# cells are reused instead of re-spent (the churn-free property the seeded shuffle lacked).
# Distinct from ``calibration`` (kill-gate holdout); this decides *run order*, reusing its
# churn-free hash primitive.

ORDER_SALT = "order-v1"
# Cheapest→hardest so a repo's early picks lead with its easy tasks.
_STRATUM_ORDER = ("easy", "medium", "hard")


def _stratum_rank(stratum: str) -> int:
    """Fixed stratum ordering; unknown strata sort last (stable)."""
    return _STRATUM_ORDER.index(stratum) if stratum in _STRATUM_ORDER else len(_STRATUM_ORDER)


def _round_robin(queues: dict[str, list[str]], keys: list[str]) -> list[str]:
    """Interleave queues in fixed ``keys`` order, one item per key per pass."""
    pointers = {k: 0 for k in keys}
    out: list[str] = []
    remaining = sum(len(queues[k]) for k in keys)
    while remaining:
        for k in keys:
            queue = queues[k]
            pos = pointers[k]
            if pos < len(queue):
                out.append(queue[pos])
                pointers[k] = pos + 1
                remaining -= 1
    return out


def stratified_order(items: Iterable[tuple[str, str, str]], salt: str = ORDER_SALT) -> list[str]:
    """Canonical run order over ``(id, repo, stratum)`` triples (see module docstring)."""
    # Round-robin across repos, then difficulty strata within a repo, then ascending stable
    # hash within each (repo, stratum) cell — diverse prefixes that nest for a fixed set.
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    repos: dict[str, None] = {}
    for iid, repo, stratum in items:
        cells[(repo, stratum)].append(iid)
        repos[repo] = None
    for ids in cells.values():
        ids.sort(key=lambda i: (holdout_score(i, salt), i))

    repo_queues: dict[str, list[str]] = {}
    for repo in repos:
        # Tiebreak on the stratum name so two *unknown* strata (same rank) still
        # order deterministically (never fall to set-iteration order).
        strata = sorted({s for (r, s) in cells if r == repo}, key=lambda s: (_stratum_rank(s), s))
        stratum_lists = {s: cells[(repo, s)] for s in strata}
        repo_queues[repo] = _round_robin(stratum_lists, strata)
    return _round_robin(repo_queues, sorted(repo_queues))


def order_from_manifest(ids: list[str], manifest: dict, salt: str = ORDER_SALT) -> list[str] | None:
    """Canonical order for ``ids`` using repo/stratum from the challenges manifest.

    Returns ``None`` when any id lacks repo metadata, so the caller can fall back.
    """
    tasks = manifest.get("tasks", {})
    triples: list[tuple[str, str, str]] = []
    for iid in ids:
        task = tasks.get(iid)
        if not task or not task.get("repo"):
            return None
        repo = str(task["repo"])
        stratum = str(task.get("difficulty_stratum") or task.get("difficulty") or "medium")
        triples.append((iid, repo, stratum))
    return stratified_order(triples, salt)
