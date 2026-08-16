"""The live sink's committable whitelist (src/shunt) must equal the offline evaluator's schema
whitelist, or a field could be committable in one plane but not the other — re-opening the
secret-leak boundary the two are meant to close together.
"""

from __future__ import annotations

from benchmark.escalation.schema import COMMITTABLE_FIELDS as OFFLINE_FIELDS
from shunt.capture.trajectory import COMMITTABLE_FIELDS as LIVE_FIELDS


def test_live_and_offline_committable_whitelists_match() -> None:
    assert LIVE_FIELDS == OFFLINE_FIELDS
