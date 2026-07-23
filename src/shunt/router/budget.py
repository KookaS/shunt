"""Cost-aware guardrails around exploration: a cumulative exploration-spend cap and a
conservative baseline-slack gate. Both bound the real-money / real-quality risk of
exploring on live user tasks. No module globals; state lives on the instances.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TypedDict

logger = logging.getLogger(__name__)


class BudgetState(TypedDict):
    """Serialized mutable budget counters — config (``_frac``) is deliberately excluded."""

    explore_cost: float
    exploit_cost: float
    decisions: int
    explorations: int


class GateState(TypedDict):
    """Serialized mutable gate state — config (``_alpha``) is deliberately excluded."""

    slack: float


def _as_number(value: object) -> float | None:
    """Coerce a JSON scalar to a finite float, rejecting bool/non-numbers/Infinity/NaN."""
    # json round-trips Infinity/NaN by default, so a tampered snapshot could carry them.
    # Rejecting non-finite routes them into the fresh-zero fallback — closing both the money-cap
    # disable (exploit_cost=inf → always explorable) and the int(inf/nan) startup crash.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


class ExplorationBudget:
    """Cumulative cap keeping exploratory spend within a fraction of exploit spend."""

    # NOT thread-safe on its own: can_explore() then record() is a check-then-act pair.
    # RouterEngine serializes both under its lock; any other caller must synchronize too.

    # Enforces total spend within ~(1 + explore_budget_frac) of pure-exploit spend over the
    # deployment: every decision contributes its *baseline* (greedy) cost, and an exploratory
    # decision additionally contributes only the EXTRA it cost over that baseline. Refuse
    # exploration once cumulative extra exceeds the fraction of cumulative baseline.
    #
    # Recording the baseline on every decision — not only on exploit decisions — is what
    # makes the cap enforceable. Crediting exploit spend solely from the exploit branch left
    # the denominator at 0 in an all-exploratory run, so the bootstrap allowance never closed
    # and the fraction never bound (measured: 2000 exploratory decisions, cap never applied).
    # Cumulative (not windowed) so the baseline can never age out and re-open exploration.

    def __init__(self, explore_budget_frac: float = 0.4) -> None:
        if explore_budget_frac < 0:
            raise ValueError("explore_budget_frac must be >= 0")
        self._frac = explore_budget_frac
        self._explore_cost = 0.0
        self._exploit_cost = 0.0
        self._decisions = 0
        self._explorations = 0

    def can_explore(self) -> bool:
        """True if an exploratory route is within the cumulative budget fraction."""
        if self._frac <= 0.0:
            # frac=0 means NEVER explore. This must precede the bootstrap branch: otherwise
            # exploratory spend (which never raises exploit_cost) keeps the bootstrap open
            # forever and frac=0 silently permits unbounded exploration.
            return False
        if self._exploit_cost <= 0.0:
            if self._decisions == 0:
                return True  # genuine bootstrap: nothing recorded yet
            # Cost-blind regime, and it is the DEFAULT for a provider that omits
            # `usage.cost` (DeepSeek direct) or a model with no known pricing: every
            # baseline lands at 0.0, so the spend ratio has no denominator and can never
            # bind. Bound the exploration RATE instead — the alternative is the spend cap
            # silently permitting unbounded exploration, which is how it behaved before.
            return self._explorations <= self._frac * self._decisions
        return self._explore_cost <= self._frac * self._exploit_cost

    def record(self, *, baseline_cost: float, actual_cost: float, is_exploratory: bool) -> None:
        """Log one decision: its greedy baseline cost, and any extra an exploration paid."""
        # Called for EVERY decision, exploratory or not. A downshift that costs less than the
        # baseline consumes no budget (the extra is clamped at 0) — exploring cheap is free.
        self._exploit_cost += max(0.0, baseline_cost)
        self._decisions += 1
        if is_exploratory:
            self._explore_cost += max(0.0, actual_cost - baseline_cost)
            self._explorations += 1

    @property
    def explore_ratio(self) -> float:
        """Cumulative extra-exploration/baseline spend ratio (0 if no baseline yet)."""
        return self._explore_cost / self._exploit_cost if self._exploit_cost > 0 else 0.0

    def snapshot(self) -> BudgetState:
        """Serialize the mutable counters so the cap survives a process restart."""
        return {
            "explore_cost": self._explore_cost,
            "exploit_cost": self._exploit_cost,
            "decisions": self._decisions,
            "explorations": self._explorations,
        }

    def restore(self, state: Mapping[str, object] | None) -> None:
        """Load a prior snapshot; a missing/partial/corrupt one leaves a fresh zero state."""
        # Robust by contract: never crash startup on a bad snapshot — fall back to zero and
        # log once. All-or-nothing: a partial snapshot must not half-apply (a stale exploit
        # denominator with a zeroed explore numerator would silently re-open the cap).
        if state is None:
            return
        explore = _as_number(state.get("explore_cost"))
        exploit = _as_number(state.get("exploit_cost"))
        decisions = _as_number(state.get("decisions"))
        explorations = _as_number(state.get("explorations"))
        if None in (explore, exploit, decisions, explorations):
            logger.warning("exploration budget snapshot unreadable; starting from zero")
            return
        assert explore is not None and exploit is not None  # narrowed by the guard above
        assert decisions is not None and explorations is not None
        if min(explore, exploit, decisions, explorations) < 0:
            logger.warning("exploration budget snapshot has negative counters; starting from zero")
            return
        self._explore_cost = explore
        self._exploit_cost = exploit
        self._decisions = int(decisions)
        self._explorations = int(explorations)


class ConservativeGate:
    """Pragmatic v1 Conservative-Bandit gate (Wu et al. 2016) for exploring *down*."""

    # Guards exploring to a cheaper/weaker model than the greedy pick (risks failing a
    # real task). Permits a downshift only while banked "slack" (verified successes minus
    # failures on prior *downshift* exploratory routes — evidence the weaker model works)
    # covers the risk; alpha scales how much slack unlocks. A bounded approximation, not
    # the full regret-bounded algorithm.

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._alpha = alpha
        self._slack = 0.0

    def allows_downshift(self) -> bool:
        """True if there is enough banked slack to risk exploring a weaker model."""
        # alpha=0 → never risk a downshift; alpha=1 → the "no conservatism" extreme
        # (slack >= 0 always true). Between, larger alpha unlocks a downshift with less slack.
        return self._alpha > 0.0 and self._slack >= (1.0 - self._alpha)

    def record_outcome(self, *, downshift: bool, success: bool) -> None:
        """Bank slack only from verified *downshift*-exploration outcomes (+1 / -1)."""
        # Slack from up-exploration (escalation) is NOT evidence the weaker model is safe,
        # so only downshift outcomes move the gate — the property the design requires.
        if not downshift:
            return
        self._slack += 1.0 if success else -1.0

    @property
    def slack(self) -> float:
        return self._slack

    def snapshot(self) -> GateState:
        """Serialize the banked slack so downshift evidence survives a process restart."""
        return {"slack": self._slack}

    def restore(self, state: Mapping[str, object] | None) -> None:
        """Load a prior snapshot; a missing/corrupt one leaves slack at zero (log once)."""
        if state is None:
            return
        slack = _as_number(state.get("slack"))  # slack may be negative (banked failures)
        if slack is None:
            logger.warning("conservative gate snapshot unreadable; starting from zero")
            return
        self._slack = slack
