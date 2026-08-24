"""Price-ascending cascade: try the cheapest measured models first, escalate on failure."""

from __future__ import annotations

from . import BilledAttempt, Strategy
from ._cascade_common import (
    billed_attempt,
    cheapest_priced_model,
    frontier_model,
    measured_models_by_price,
)


class PriceCascade(Strategy):
    """Try the ``max_tries`` cheapest measured models in ascending price order, stop on
    the first verified pass, and fall back to the frontier model when none pass.
    """

    def __init__(self, max_tries: int = 3):
        self._max_tries = max_tries

        # Cascade metadata (reset per :meth:`select` call), read by summary.evaluate
        # so the reported cost is the whole cascade, not just the returned model's cell.
        self.cascade_total_cost: float = 0.0
        self.cascade_tried_models: list[str] = []
        # Per-attempt (model, cost), in billing order — what the cache-aware cost model reads.
        # A cascade re-serving one model on consecutive attempts is exactly where the discount
        # lands, and the collapsed `cascade_total_cost` cannot express that adjacency.
        self.cascade_attempts: list[BilledAttempt] = []
        # False when any tried cell is unmeasured — the true cost/outcome is unknown, so
        # this decision is a coverage gap, not a real fail@$0.
        self.cascade_scorable: bool = True

    @property
    def name(self) -> str:
        return "Price-Cascade"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        self.cascade_tried_models = []
        self.cascade_attempts = []
        self.cascade_total_cost = 0.0
        self.cascade_scorable = True

        by_price = measured_models_by_price(matrix)
        frontier = frontier_model(matrix)
        if not by_price or frontier is None:
            self.cascade_scorable = False
            return cheapest_priced_model(matrix)

        # Frontier last: the shortlist is the cheap end of the ladder, and the priciest
        # measured model is the escalation target of last resort.
        order = by_price[: self._max_tries]
        if frontier not in order:
            order = [*order, frontier]

        task_results = matrix.get("results", {}).get(task_id, {})
        for model in order:
            self.cascade_tried_models.append(model)
            outcome = task_results.get(model, {})
            if not outcome:
                self.cascade_scorable = False
            attempt_cost = float(outcome.get("cost", 0.0))
            self.cascade_attempts.append(billed_attempt(model, outcome))
            self.cascade_total_cost += attempt_cost
            if outcome.get("pass", False):
                return model
        return order[-1]
