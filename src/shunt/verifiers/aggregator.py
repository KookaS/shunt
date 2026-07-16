from __future__ import annotations

import math
from typing import Final

from .base import VerifierResult

_OUTCOME_ORDER: Final = ["success", "weak_success", "unknown", "failure", "infra_failure"]


def _outcome_rank(outcome: str) -> int:
    try:
        return _OUTCOME_ORDER.index(outcome)
    except ValueError:
        return len(_OUTCOME_ORDER)


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


class VerifierAggregator:
    def __init__(self, k: int = 3, index_threshold: float = 0.3) -> None:
        self._k = k
        self._index_threshold = index_threshold

    def aggregate(self, results: list[VerifierResult]) -> VerifierResult:
        if not results:
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail="no verifier results to aggregate",
            )

        successes = sum(1 for r in results if r.outcome in ("success", "weak_success"))
        total = len(results)
        failures = total - successes

        alpha0, beta0 = 1.0, 1.0
        alpha_n = alpha0 + successes
        beta_n = beta0 + failures

        aggregated_confidence = alpha_n / (alpha_n + beta_n)

        best = max(results, key=lambda r: (r.confidence, -_outcome_rank(r.outcome)))

        p_empirical = successes / total
        entropy = _bernoulli_entropy(p_empirical)

        detail = (
            f"aggregated {total} results ({successes} success, {failures} failure); "
            f"Beta({int(alpha_n)},{int(beta_n)}) posterior; "
            f"entropy={entropy:.3f}"
        )

        return VerifierResult(
            outcome=best.outcome,
            confidence=aggregated_confidence,
            detail=detail,
            is_infra_failure=best.is_infra_failure,
            matched_pattern=best.matched_pattern,
        )

    def should_index(self, result: VerifierResult) -> bool:
        return result.confidence > self._index_threshold
