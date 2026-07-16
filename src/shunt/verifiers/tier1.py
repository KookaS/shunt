from __future__ import annotations

import re
from typing import Final

from .base import Verifier, VerifierResult

_REGEX_PATTERNS: Final[list[tuple[str, str, float, bool]]] = [
    ("success", r"tests passed", 0.4, False),
    ("success", r"All tests succeed", 0.4, False),
    ("success", r"\u2713", 0.4, False),
    ("failure", r"Error:", 0.3, False),
    ("failure", r"Traceback", 0.3, False),
    ("failure", r"failed", 0.3, False),
    ("infra_failure", r"ModuleNotFound", 0.2, True),
    ("infra_failure", r"ImportError", 0.2, True),
    ("infra_failure", r"No module", 0.2, True),
    ("weak_success", r"implementation is complete", 0.2, False),
    ("weak_success", r"works correctly", 0.2, False),
]


class RegexVerifier(Verifier):
    def __init__(self) -> None:
        self._patterns = [
            (outcome, re.compile(pattern, re.IGNORECASE), confidence, infra)
            for outcome, pattern, confidence, infra in _REGEX_PATTERNS
        ]

    def verify(self, text: str, work_dir: str | None = None) -> VerifierResult:
        best_outcome = "unknown"
        best_confidence = 0.0
        best_detail = ""
        best_infra = False
        best_pattern: str | None = None

        for outcome, regex, confidence, is_infra in self._patterns:
            match = regex.search(text)
            if match is not None and confidence > best_confidence:
                best_outcome = outcome
                best_confidence = confidence
                best_detail = f"matched pattern: {regex.pattern!r}"
                best_infra = is_infra
                best_pattern = regex.pattern

        return VerifierResult(
            outcome=best_outcome,
            confidence=best_confidence,
            detail=best_detail,
            is_infra_failure=best_infra,
            matched_pattern=best_pattern,
        )
