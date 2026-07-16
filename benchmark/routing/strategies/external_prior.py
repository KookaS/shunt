"""External-prior routing: use the SWE-bench leaderboard as a per-task difficulty prior."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark import config

from . import Strategy
from .fixed import _model_pricing

_PRIOR_PATH = Path(__file__).resolve().parents[1] / "data" / "external_swebench.csv"


def _load_prior(path: Path) -> dict[str, float]:
    """{instance_id: p_solve} from the external prior CSV (field-wide difficulty)."""
    # We key on p_solve, NOT p_cheap: the cheap cohort reports only resolved instances so
    # p_cheap is a degenerate 1.0. p_solve (median 0.86, real hard tail) ranks difficulty.
    prior: dict[str, float] = {}
    if not path.exists():
        return prior
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            val = row.get("p_solve", "")
            if val not in ("", None):
                prior[row["instance_id"]] = float(val)
    return prior


class ExternalPriorCascade(Strategy):
    """Cheap-by-default; escalate one step when the external cheap cohort fails."""

    def __init__(self, threshold: float = 0.5, prior_path: Path | None = None):
        self._threshold = threshold
        self._prior = _load_prior(prior_path or _PRIOR_PATH)

    @property
    def name(self) -> str:
        return "External-Prior"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        pricing = _model_pricing(matrix)
        if not pricing:
            return "deepseek-v4-flash"
        by_cost = sorted(pricing, key=lambda m: pricing[m])  # cheapest first
        cheapest = by_cost[0]

        difficulty = self._prior.get(task_id)  # field-wide p_solve (see _load_prior)
        # No external signal, or the field mostly solves it → stay cheap.
        if difficulty is None or difficulty >= self._threshold:
            return cheapest

        # The field struggles on this instance → escalate to the cheapest model
        # whose tier is NOT "cheap" (mid before frontier, by cost).
        tiers = {m: _tier_of(m) for m in by_cost}
        for model in by_cost:
            if tiers.get(model) != "cheap":
                return model
        return cheapest


def _tier_of(model: str) -> str:
    """Tier for *model* from the pricing registry (models.json), 'cheap' if unknown."""
    info = config.load_pricing().get(model)
    return info.get("tier", "cheap") if isinstance(info, dict) else "cheap"
