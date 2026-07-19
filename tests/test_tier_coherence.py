"""Tier vocabulary has ONE source of truth (shunt.models.TIER_ORDER) — no drift.

Guards against a new/renamed tier being registered in the ``Tier`` literal but not
in a consumer's hardcoded copy (or vice versa), which would silently mis-sort.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark import config
from shunt.models import TIER_ORDER
from shunt.models.config import ModelEntry
from shunt.router import selection


class TestTierSingleSourceOfTruth:
    def test_selection_derives_from_canonical(self):
        # Product routing iterates the canonical constant, not a private copy —
        # a new tier propagates automatically instead of being silently ignored.
        assert selection.TIER_ORDER is TIER_ORDER

    def test_benchmark_tier_order_matches_canonical(self):
        for i, tier in enumerate(TIER_ORDER):
            assert config._tier_order(tier) == i

    def test_benchmark_tier_order_rejects_unregistered(self):
        # The drift guard: a tier absent from TIER_ORDER fails loud, never silently
        # collapses to a sort-last sentinel.
        with pytest.raises(ValueError, match="unknown tier"):
            config._tier_order("ultra")

    def test_registry_schema_rejects_unregistered_tier(self):
        # The other half: the registry's `Tier` literal makes an invalid tier
        # impossible to load, so the two files can't disagree on what a tier is.
        with pytest.raises(ValidationError):
            ModelEntry(model_id="m", tier="ultra", provider="p")  # type: ignore[arg-type]
