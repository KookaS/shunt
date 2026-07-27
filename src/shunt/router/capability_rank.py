"""Runtime capability-rank resolver — the price-implied prior, session-pinned (cache-safe)."""

# Phase 1 ships the price prior verbatim: the order is derived from list price (process-constant,
# always available at cold-start) and pinned per session. The learned overlay (measured > price)
# is Phase 2 and is deliberately NOT wired here. The load-bearing property is cache-safety: a
# snapshot resolved once at session open is constant for that session's lifetime, so no model
# switch can ever be driven by a mid-cached-turn rank change.
#
# NOT-YET-WIRED (Phase 2 obligation): this resolver is not consumed by the engine yet — the
# router reads the pool's live price order each decision, which is cache-safe ONLY because that
# order is process-constant in Phase 1. When Phase 2 makes rank move within a process (the learned
# overlay), the engine's escalation MUST switch to reading this pinned per-session snapshot instead
# of the live order, or the cache-safety wall is bypassed. Until then the snapshot's stability test
# is necessarily trivial (price can't drift mid-process) and does not by itself prove the wall.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shunt.models.config import ModelConfig, RankedModel, _total_price


class RankedPool(Protocol):
    """The pool capability the resolver reads: the price-ordered live models."""

    def ranked_models(self) -> list[ModelConfig]: ...


@dataclass(frozen=True)
class CapabilityRank:
    """A pinned per-session capability order (weakest -> strongest) + each slot's source."""

    ordered: list[RankedModel]  # weakest -> strongest — the pinned snapshot
    source: dict[str, str]  # model -> "price" (Phase 1) | "learned" (Phase 2)

    def names(self) -> list[str]:
        """Model names weakest -> strongest."""
        return [r.model for r in self.ordered]

    def rank_of(self, model: str) -> int | None:
        """The model's slot in this snapshot (0 = weakest), or None if absent."""
        for r in self.ordered:
            if r.model == model:
                return r.rank
        return None

    def strongest(self) -> str | None:
        """The strongest model in the snapshot, or None when the pool is empty."""
        return self.ordered[-1].model if self.ordered else None


class CapabilityRankResolver:
    """Resolve a session's capability order. Phase 1: price prior only, pinned per session.

    ``snapshot_for_session`` is called ONCE at session open and is constant for its lifetime;
    the Phase-2 learned overlay that could reorder it lands behind this same pinned-snapshot wall.
    """

    def __init__(self, pool: RankedPool) -> None:
        self._pool = pool

    def snapshot_for_session(self) -> CapabilityRank:
        """The pinned capability order for a new session (price-implied, weakest -> strongest)."""
        ordered = [
            RankedModel(model=m.name, rank=i, price=_total_price(m))
            for i, m in enumerate(self._pool.ranked_models())
        ]
        source = {r.model: "price" for r in ordered}
        return CapabilityRank(ordered=ordered, source=source)


__all__ = ["CapabilityRank", "CapabilityRankResolver", "RankedPool"]
