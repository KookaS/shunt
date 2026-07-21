"""Pending-outcome queue for delayed / censored verification feedback.

The Beta update is applied only when a *verified* label later arrives; an outcome that
never arrives is forgotten on eviction, never counted as a failure (Vernade et al. 2017).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingDecision:
    """A routed decision awaiting its verified outcome."""

    key: str
    model: str
    exploratory: bool
    propensity: float


@dataclass(frozen=True)
class ResolvedOutcome:
    """A pending decision paired with its arrived verified label."""

    decision: PendingDecision
    success: bool


class PendingOutcomes:
    """Bounded FIFO of decisions awaiting verified labels."""

    # resolve() applies only to a real label; an unknown key is a no-op, not an error.
    # Eviction drops the oldest unlabeled decision to bound memory — forgetting, never a
    # synthesized failure.

    def __init__(self, max_pending: int = 10_000) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be > 0")
        self._max = max_pending
        self._pending: OrderedDict[str, PendingDecision] = OrderedDict()

    def record(self, decision: PendingDecision) -> None:
        """Register a routed decision as awaiting its verified outcome."""
        self._pending[decision.key] = decision
        self._pending.move_to_end(decision.key)
        while len(self._pending) > self._max:
            self._pending.popitem(last=False)  # evict oldest unlabeled — forget, don't fail

    def resolve(self, key: str, *, success: bool) -> ResolvedOutcome | None:
        """Apply a verified label to a pending decision, returning it (or None if unknown)."""
        decision = self._pending.pop(key, None)
        if decision is None:
            return None
        return ResolvedOutcome(decision=decision, success=success)

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, key: object) -> bool:
        return key in self._pending
