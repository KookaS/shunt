"""Golden-path scaffold for a new routing strategy (see AGENTS.md)."""

from __future__ import annotations

from . import Strategy


class TemplateStrategy(Strategy):
    """One-line intent: what routing rule this strategy encodes."""

    def __init__(self, fallback_model: str = "deepseek-v4-flash") -> None:
        # Config in the constructor; no module-level globals (SH001).
        self._fallback_model = fallback_model

    @property
    def name(self) -> str:
        """Display name shown in eval output and plots."""
        return "Template"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """Return the model to route this task to (a key of matrix['results'][task_id])."""
        candidates = self._passing_models(task_id, matrix)
        return candidates[0] if candidates else self._fallback_model

    def _passing_models(self, task_id: str, matrix: dict) -> list[str]:
        # Small, single-purpose helper — keep methods short (≤40 stmts, ≤10 complexity).
        outcomes = matrix.get("results", {}).get(task_id, {})
        return [model for model, o in outcomes.items() if o.get("pass", False)]
