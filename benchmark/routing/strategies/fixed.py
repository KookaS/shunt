"""Fixed-model strategies: always use the cheapest or frontier model.

Provides the simple baselines that any adaptive router must beat.
The cheapest and frontier models are derived from the pricing matrix,
not hardcoded — making the baselines self-consistent with any matrix.
"""

from __future__ import annotations

import random
import zlib

from . import Strategy


def _model_pricing(matrix: dict) -> dict[str, float]:
    """Return {model: total_cost_per_1M} from whichever matrix format.

    Only includes models that have been evaluated (appear in at least one
    task's results), so fixed baselines never route to unevaluated models.
    """
    models = matrix.get("models")
    if models:
        evaluated: set[str] = set()
        for task_results in matrix.get("results", {}).values():
            evaluated.update(task_results.keys())
        return {
            m: models[m].get("input_price", 0) + models[m].get("output_price", 0)
            for m in models
            if m in evaluated
        }
    # Flat pricing dict (e.g. pricing.json loaded directly)
    candidates = {
        k: v for k, v in matrix.items() if isinstance(v, dict) and "input_cost_per_1m" in v
    }
    if candidates:
        return {
            m: candidates[m].get("input_cost_per_1m", 0)
            + candidates[m].get("output_cost_per_1m", 0)
            for m in candidates
        }
    return {}


class AlwaysCheap(Strategy):
    """Always route to the cheapest model (derived from the pricing matrix)."""

    def __init__(self, model: str | None = None):
        self._model = model

    @property
    def name(self) -> str:
        return "Always-Cheap"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if self._model is not None:
            return self._model
        pricing = _model_pricing(matrix)
        if pricing:
            return min(pricing, key=lambda m: pricing[m])
        return "deepseek-v4-flash"


class AlwaysFrontier(Strategy):
    """Always route to the most expensive model (derived from the pricing matrix)."""

    def __init__(self, model: str | None = None):
        self._model = model

    @property
    def name(self) -> str:
        return "Always-Frontier"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if self._model is not None:
            return self._model
        pricing = _model_pricing(matrix)
        if pricing:
            return max(pricing, key=lambda m: pricing[m])
        return "claude-opus-4-6"


class Random(Strategy):
    """Uniform random model selection (deterministic per seed)."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "Random"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        models = list(matrix.get("results", {}).get(task_id, {}).keys())
        if not models:
            return ""
        rng = random.Random(self.seed ^ zlib.adler32(task_id.encode()))
        return rng.choice(models)
