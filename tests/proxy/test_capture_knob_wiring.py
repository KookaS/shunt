"""The capture loop's knobs must reach the objects that use them.

`verify_timeout_seconds` and `rerun_confirmations` were constructor defaults with no
config path, so a repo whose suite outrun 120s captured nothing and no knob could fix it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shunt.proxy.server import _build_verifier
from shunt.router.policy import CapturePolicy, EscalationPolicy, RouterPolicy
from shunt.verifiers.rerun import RerunConfirmingVerifier


def _policy(**capture: object) -> RouterPolicy:
    return RouterPolicy(capture=CapturePolicy(**capture))  # type: ignore[arg-type]


def test_the_configured_timeout_reaches_the_subprocess_verifier() -> None:
    verifier = _build_verifier(_policy(verify_timeout_seconds=900.0))
    assert isinstance(verifier, RerunConfirmingVerifier)
    assert verifier._inner._timeout == 900.0  # type: ignore[attr-defined]


def test_an_unset_timeout_keeps_the_shipped_default() -> None:
    verifier = _build_verifier(_policy())
    assert verifier._inner._timeout == 120  # type: ignore[attr-defined,union-attr]


def test_the_configured_rerun_count_reaches_the_flake_guard() -> None:
    verifier = _build_verifier(_policy(rerun_confirmations=5))
    assert isinstance(verifier, RerunConfirmingVerifier)
    assert verifier._reruns == 5


def test_zero_reruns_is_rejected_rather_than_silently_disabling_escalation() -> None:
    # An unconfirmed failure is discarded by the escalation gate, so 0 would leave
    # escalation enabled and permanently unable to fire, with nothing in the logs.
    with pytest.raises(ValidationError):
        CapturePolicy(rerun_confirmations=0)


def test_escalation_off_skips_the_rerun_wrapper_but_keeps_the_timeout() -> None:
    policy = RouterPolicy(
        escalation=EscalationPolicy(enabled=False),
        capture=CapturePolicy(verify_timeout_seconds=42.0),
    )
    verifier = _build_verifier(policy)
    assert not isinstance(verifier, RerunConfirmingVerifier)
    assert verifier._timeout == 42.0  # type: ignore[attr-defined]
