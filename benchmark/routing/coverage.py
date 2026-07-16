"""Cache coverage of a strategy's decisions over the M×P cell matrix.

A decision needing an uncached (challenge, model) cell is flagged, not skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Coverage:
    """Which cells a strategy needed and which were absent from the cache."""

    strategy: str
    needed: list[tuple[str, str]] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when every cell the strategy's decisions required is cached."""
        return not self.missing


def cell_coverage(strategy: object, matrix: dict, tasks: list[str]) -> Coverage:
    """Record the (challenge, model) cells a strategy needs and flag uncached ones."""
    cov = Coverage(strategy=getattr(strategy, "name", str(strategy)))
    results = matrix.get("results", {})
    task_meta_all = matrix.get("tasks", {})
    for tid in tasks:
        model = strategy.select(tid, task_meta_all.get(tid, {}), matrix)  # type: ignore[attr-defined]
        cov.needed.append((tid, model))
        if model not in results.get(tid, {}):
            cov.missing.append((tid, model))
    return cov
