"""Session-cadence escalation policies: what the NEXT session should be, per instance.

Eval-side only: a name -> builder map mirroring the router's strategy registry, so the
benchmark can backtest alternative escalation decisions without touching product code.
"""

# Each policy's `decide` returns the sessions it runs next for one overlap instance, so the
# session-cadence eval (`session_eval.session_cadence`) can headline any registered policy —
# including the never-escalate hold policy the always-cheap arm is — without editing its arm
# construction. The `random_escalate` arm is deliberately NOT registered: it fires on a seeded
# subset of the whole instance set sized to the headline arm's fire rate, so its decision is not
# a per-instance one and no user would backtest it. It is the eval's own null construction.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from benchmark.escalation.schema import Trajectory

ARM_ESCALATE: Final[str] = "escalate"
ARM_RETRY: Final[str] = "cheap_retry"
ARM_ALWAYS_FRONTIER: Final[str] = "always_frontier"
ARM_ALWAYS_CHEAP: Final[str] = "always_cheap"
ARM_RANDOM: Final[str] = "random_escalate"
# The per-instance policies a user could backtest; `random_escalate` is excluded (see the
# module docstring — it is a cross-instance null, not a per-instance decision).
REGISTERED_POLICIES: Final[tuple[str, ...]] = (
    ARM_ESCALATE,
    ARM_RETRY,
    ARM_ALWAYS_FRONTIER,
    ARM_ALWAYS_CHEAP,
)


@dataclass(frozen=True)
class ArmSession:
    """One arm's session at one instance: its outcome and the attempts that arm pays for."""

    resolved: bool
    attempts: tuple[Trajectory, ...]


class EscalationPolicy(Protocol):
    """One session-cadence escalation decision: which sessions run next on an instance."""

    @property
    def name(self) -> str:
        """The policy's registry name — what a caller selects it by."""
        ...

    def decide(
        self,
        cheap: Sequence[Trajectory],
        frontier: Sequence[Trajectory],
    ) -> Sequence[ArmSession]:
        """The sessions this policy runs for one overlap instance; empty means it stays put."""
        ...


def _by_length(cheap: Sequence[Trajectory]) -> tuple[Trajectory, ...]:
    """Cheap sessions ordered by length — the within-price ordering the retry arm always used."""
    return tuple(sorted(cheap, key=lambda s: s.header.n_steps))


def _escalate_decide(
    cheap: Sequence[Trajectory], frontier: Sequence[Trajectory]
) -> Sequence[ArmSession]:
    """Escalate the next session to a frontier model once a cheap session failed."""
    first = _by_length(cheap)[0]
    if all(s.header.terminal_resolved for s in cheap):
        return ()
    return tuple(ArmSession(f.header.terminal_resolved, (first, f)) for f in frontier)


def _always_cheap_decide(
    cheap: Sequence[Trajectory], frontier: Sequence[Trajectory]
) -> Sequence[ArmSession]:
    """Never escalate: hold the first cheap session's outcome, whatever the frontier did."""
    del frontier
    first = _by_length(cheap)[0]
    return (ArmSession(first.header.terminal_resolved, (first,)),)


def _always_frontier_decide(
    cheap: Sequence[Trajectory], frontier: Sequence[Trajectory]
) -> Sequence[ArmSession]:
    """Never be cheap: the frontier sessions' outcomes, whatever the cheap arm did."""
    del cheap
    return tuple(ArmSession(f.header.terminal_resolved, (f,)) for f in frontier)


def _retry_decide(
    cheap: Sequence[Trajectory], frontier: Sequence[Trajectory]
) -> Sequence[ArmSession]:
    """Retry cheap after a cheap failure — the same-cost incumbent, escalated to nothing."""
    del frontier
    ordered = _by_length(cheap)
    return tuple(
        ArmSession(ordered[i].header.terminal_resolved, (ordered[i - 1], ordered[i]))
        for i in range(1, len(ordered))
        if not ordered[i - 1].header.terminal_resolved
    )


class _NamedPolicy:
    """One registered policy: its name plus the pure per-instance decide function."""

    def __init__(
        self,
        name: str,
        decide: Callable[[Sequence[Trajectory], Sequence[Trajectory]], Sequence[ArmSession]],
    ) -> None:
        self._name = name
        self._decide = decide

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self,
        cheap: Sequence[Trajectory],
        frontier: Sequence[Trajectory],
    ) -> Sequence[ArmSession]:
        return self._decide(cheap, frontier)


_PolicyBuilder = Callable[[], EscalationPolicy]


def _build_escalate() -> EscalationPolicy:
    return _NamedPolicy(ARM_ESCALATE, _escalate_decide)


def _build_retry() -> EscalationPolicy:
    return _NamedPolicy(ARM_RETRY, _retry_decide)


def _build_always_frontier() -> EscalationPolicy:
    return _NamedPolicy(ARM_ALWAYS_FRONTIER, _always_frontier_decide)


def _build_always_cheap() -> EscalationPolicy:
    return _NamedPolicy(ARM_ALWAYS_CHEAP, _always_cheap_decide)


_BUILDERS: Final[dict[str, _PolicyBuilder]] = {
    ARM_ESCALATE: _build_escalate,
    ARM_RETRY: _build_retry,
    ARM_ALWAYS_FRONTIER: _build_always_frontier,
    ARM_ALWAYS_CHEAP: _build_always_cheap,
}


def build_policy(name: str) -> EscalationPolicy:
    """Construct the session-cadence policy for *name*; unknown names fail listing the allowed."""
    builder = _BUILDERS.get(name)
    if builder is None:
        allowed = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"unknown session escalation policy {name!r}; allowed: {allowed}")
    return builder()


__all__ = [
    "ARM_ALWAYS_CHEAP",
    "ARM_ALWAYS_FRONTIER",
    "ARM_ESCALATE",
    "ARM_RANDOM",
    "ARM_RETRY",
    "REGISTERED_POLICIES",
    "ArmSession",
    "EscalationPolicy",
    "build_policy",
]
