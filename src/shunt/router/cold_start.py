from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Final

from shunt.models import TIER_ORDER

if TYPE_CHECKING:
    from shunt.router.selection import ModelPoolProtocol

logger = logging.getLogger(__name__)

_COLD_START_MODEL = "qwen3.7-plus"
_DEFAULT_FALLBACK_MODELS: Final = ["deepseek-v4-flash", "zai-glm-5.2"]


class ColdStartStrategy:
    """Cold-start routing policy: while active, route to cheap qwen3.7-plus
    (falling back through the chain if unhealthy); kNN takes over once inactive.
    Ends when count_tier2 >= threshold_tier2 OR count_labeled >= threshold_tier1.
    """

    def __init__(
        self,
        threshold_tier2: int | None = None,
        threshold_tier1: int | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self._threshold_tier2 = (
            threshold_tier2
            if threshold_tier2 is not None
            else int(os.environ.get("SHUNT_COLD_START_THRESHOLD_TIER2", "20"))
        )
        self._threshold_tier1 = (
            threshold_tier1
            if threshold_tier1 is not None
            else int(os.environ.get("SHUNT_COLD_START_THRESHOLD_TIER1", "50"))
        )
        self._fallback_models = (
            fallback_models if fallback_models is not None else list(_DEFAULT_FALLBACK_MODELS)
        )

    @property
    def threshold_tier2(self) -> int:
        return self._threshold_tier2

    @property
    def threshold_tier1(self) -> int:
        return self._threshold_tier1

    @property
    def fallback_models(self) -> list[str]:
        return list(self._fallback_models)

    def is_active(self, count_labeled: int, count_tier2: int) -> bool:
        """Return True if cold-start routing is still active. ``count_labeled``:
        sessions with any labeled outcome; ``count_tier2``: sessions with Tier-2
        (verified) outcomes.
        """
        if count_tier2 >= self._threshold_tier2:
            return False
        return count_labeled < self._threshold_tier1

    def select(self, model_pool: ModelPoolProtocol) -> str:
        """Return the cold-start model — prefers qwen3.7-plus, falling back
        through the configured chain then escalating through the pool if
        unhealthy.
        """
        if model_pool.is_healthy(_COLD_START_MODEL):
            return _COLD_START_MODEL

        for fallback in self._fallback_models:
            if model_pool.is_healthy(fallback):
                logger.info(
                    "Cold-start primary %s unhealthy, falling back to %s",
                    _COLD_START_MODEL,
                    fallback,
                )
                return fallback

        for tier in TIER_ORDER:
            for model in model_pool.get_tier_models(tier):
                if model_pool.is_healthy(model.name):
                    logger.warning(
                        "Cold-start fallback chain exhausted, escalating to %s",
                        model.name,
                    )
                    return model.name

        logger.warning("No healthy models found, returning cold-start default")
        return _COLD_START_MODEL
