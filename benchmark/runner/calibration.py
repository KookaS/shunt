"""Deterministic calibration-holdout sampling for the tiered benchmark.

Uses hash-thresholding on each challenge's immutable id for stable, churn-free membership.

"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

DEFAULT_SALT = "calib-v1"
DEFAULT_FRACTION = 0.18


def holdout_score(instance_id: str, salt: str = DEFAULT_SALT) -> float:
    """Stable uniform score in [0, 1) for one challenge id.

    Uses a NUL separator (absent from salts and SWE-bench ids) so distinct
    (salt, id) pairs can't alias — ``("a", "b:c")`` must not hash like ``("a:b", "c")``.
    """
    digest = hashlib.sha256(f"{salt}\x00{instance_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def in_calibration_holdout(
    instance_id: str, fraction: float = DEFAULT_FRACTION, salt: str = DEFAULT_SALT
) -> bool:
    """True iff this challenge is in the full-matrix calibration holdout.

    Threshold test ``score < fraction`` — monotone in ``fraction`` (raising it only
    adds members) and independent per id (adding challenges never moves others).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return holdout_score(instance_id, salt) < fraction


def calibration_holdout(
    instance_ids: Iterable[str], fraction: float = DEFAULT_FRACTION, salt: str = DEFAULT_SALT
) -> list[str]:
    """Sorted, de-duplicated subset of ids selected for the calibration holdout."""
    return sorted({i for i in instance_ids if in_calibration_holdout(i, fraction, salt)})
