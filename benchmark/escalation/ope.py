"""Re-export shim: the estimator now ships in `shunt.analysis.ope`.

It moved because SH006 forbids `src/shunt/` importing `benchmark/`, and a shipped figure
must quote this estimator. Mirroring it would let a drifted copy silently change a number.
"""

from shunt.analysis.ope import (
    IDENTIFIED,
    NOT_IDENTIFIED,
    ExplorationLogRow,
    PolicyValueEstimate,
    _crossfit_qhat,
    _direct_method,
    always_escalate,
    estimate_policy_value,
    never_escalate,
    rows_from_records,
)

__all__ = [
    "IDENTIFIED",
    "NOT_IDENTIFIED",
    "ExplorationLogRow",
    "PolicyValueEstimate",
    "_crossfit_qhat",
    "_direct_method",
    "always_escalate",
    "estimate_policy_value",
    "never_escalate",
    "rows_from_records",
]
