from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class VerifierResult:
    outcome: str
    confidence: float
    detail: str = ""
    is_infra_failure: bool = False
    matched_pattern: str | None = None


class Verifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, text: str, work_dir: str | None = None) -> VerifierResult: ...
