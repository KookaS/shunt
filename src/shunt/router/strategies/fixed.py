"""Fixed live strategies — one model class per session, chosen by capability rank
(``always_cheap`` = lowest rank / cheapest price, ``always_frontier`` = highest rank).
"""

from __future__ import annotations

from shunt.router.selection import ModelPoolProtocol, NeighborResult


def _first_healthy(model_pool: ModelPoolProtocol, *, strongest_first: bool) -> str | None:
    """First healthy model scanning the rank order; else the endpoint (health ignored)."""
    ranked = list(model_pool.ranked_models())
    if not ranked:
        return None
    scan = list(reversed(ranked)) if strongest_first else ranked
    for model in scan:
        if model_pool.is_healthy(model.name):
            return model.name
    return scan[0].name  # none healthy → the intended endpoint anyway


class AlwaysCheapStrategy:
    """Route every session to the cheapest (lowest-rank) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    @property
    def participates_in_escalation(self) -> bool:
        """A PINNED CONTROL by contract: no verified-failure signal may move this pick."""
        # This is the baseline every routing comparison is read against, so it has to be the
        # same policy on turn 1 and turn 600. A control that climbs a ladder is not a control.
        # `session_cascade` is the version of this that DOES escalate, and it is a separate
        # strategy precisely so that choosing one never silently changes the other.
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the lowest-rank healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_healthy(model_pool, strongest_first=False)
        if chosen is None:
            raise ValueError("no models configured for always_cheap routing")
        return (chosen, "always_cheap")


class AlwaysFrontierStrategy:
    """Route every session to the strongest (highest-rank) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    @property
    def participates_in_escalation(self) -> bool:
        """A PINNED CONTROL by contract — see ``AlwaysCheapStrategy``. Also already at the top."""
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the highest-rank healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_healthy(model_pool, strongest_first=True)
        if chosen is None:
            raise ValueError("no models configured for always_frontier routing")
        return (chosen, "always_frontier")


class SessionCascadeStrategy(AlwaysCheapStrategy):
    """The cheap-first cascade at session cadence: ``always_cheap`` that escalates."""

    # Start on the cheapest model and let a repeated verified failure walk a rung at the next
    # session boundary. The base pick is the parent's, unchanged; the whole difference is the
    # escalation layer, which is why this is a subclass and not a copy.
    # Kept apart from `always_cheap` rather than folded in behind a flag: the parent is a
    # PINNED CONTROL and this is the opposite of one, so a single class with a knob would
    # make "is the baseline still the baseline" a runtime question. It also earns its own
    # reason token, so an analysis that filters on `always_cheap` to find control sessions
    # cannot silently scoop up cascade sessions.

    @property
    def participates_in_escalation(self) -> bool:
        """The ladder IS this strategy — without it there is nothing here but always_cheap."""
        return True

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """The cheapest healthy model, reported under this strategy's own reason token."""
        chosen, _reason = super().select(neighbors, model_pool)
        return (chosen, "session_cascade")
