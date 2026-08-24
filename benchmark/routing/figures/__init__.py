"""The routing half's committed figure set — one module per figure, twelve in all.

Act I asks whether the gate is met, Act II how the mechanism works, Act III whether the
embedding carries signal at all, Act IV whether the instruments are valid.
"""

from __future__ import annotations

from benchmark.routing.figures.context import (
    BASELINE_STRATEGY,
    DEFAULT_STRATEGY,
    ROUTER_STRATEGY,
    RoutingContext,
    build_context,
)

__all__ = [
    "BASELINE_STRATEGY",
    "DEFAULT_STRATEGY",
    "ROUTER_STRATEGY",
    "RoutingContext",
    "build_context",
]
