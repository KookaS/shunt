from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class SessionState(enum.StrEnum):
    """Session lifecycle states.

    Transitions: open → closing → closed → verifying
    """

    open = "open"
    closing = "closing"
    closed = "closed"
    verifying = "verifying"


@dataclass
class Session:
    """Represents a single tool session tracked by the router."""

    session_id: str
    tool_identity: str
    start_time: datetime
    # Bumped on every turn. `start_time` alone made an actively-used session expire a
    # fixed span after it BEGAN, which re-routed mid-work and broke cache safety.
    last_activity: datetime | None = None
    end_time: datetime | None = None
    model_chosen: str | None = None
    total_cost: float = 0.0
    cache_tax: float = 0.0
    prompt_length_tokens: int = 0
    # Cumulative, so the hit ratio divides like-for-like. Dividing the running
    # `cache_tax` by a single turn's `prompt_length_tokens` clamped to a fake 1.0.
    prompt_tokens_total: int = 0
    state: SessionState = SessionState.open
    metadata: dict[str, Any] = field(default_factory=dict)
    decision_provenance: dict[str, Any] | None = None
