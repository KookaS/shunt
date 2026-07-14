"""Oracle: for each task, pick the cheapest model that passes it.

Upper bound — no real router can beat this. Provides the reference for
cumulative regret calculations.
"""

from __future__ import annotations

from . import Strategy


class Oracle(Strategy):
    @property
    def name(self) -> str:
        return "Oracle"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        results = matrix["results"].get(task_id, {})

        candidates: list[tuple[str, float]] = []
        for model, outcome in results.items():
            if outcome.get("pass"):
                cost = outcome.get("cost", 0.0)
                candidates.append((model, cost))

        if not candidates:
            # All models fail — return the cheapest (minimum cost)
            if results:
                return min(results, key=lambda m: results[m].get("cost", 0.0))
            return ""

        return min(candidates, key=lambda x: x[1])[0]


class OracleRewardAware(Strategy):
    """Picks the model maximizing reward = pass - gamma * cost,
    rather than cheapest-passing. This can differ when a slightly
    more expensive model passes and the cheapest fails."""

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "Oracle-reward"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        results = matrix["results"].get(task_id, {})

        best_reward = -float("inf")
        best_model = list(results.keys())[0] if results else ""

        for model, outcome in results.items():
            passed = 1.0 if outcome.get("pass") else 0.0
            cost = outcome.get("cost", 0.0)
            reward = passed - self.gamma * cost
            if reward > best_reward:
                best_reward = reward
                best_model = model

        return best_model
