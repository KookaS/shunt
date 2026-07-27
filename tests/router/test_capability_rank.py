"""CapabilityRankResolver — Phase 1 price prior + the session-pinned cache-safety spine."""

from __future__ import annotations

import dataclasses

import pytest

from shunt.models import ModelPool
from shunt.router.capability_rank import CapabilityRank, CapabilityRankResolver


def _live_pool() -> ModelPool:
    pool = ModelPool()
    pool.restrict_to_live(["kimi-k3", "deepseek-v4-flash", "gpt-5-mini", "qwen3.7-plus"])
    return pool


class TestPricePrior:
    def test_snapshot_is_price_order(self) -> None:
        pool = _live_pool()
        snap = CapabilityRankResolver(pool).snapshot_for_session()
        assert snap.names() == ["deepseek-v4-flash", "qwen3.7-plus", "gpt-5-mini", "kimi-k3"]

    def test_snapshot_matches_pool_ranked_models(self) -> None:
        pool = _live_pool()
        snap = CapabilityRankResolver(pool).snapshot_for_session()
        assert snap.names() == [m.name for m in pool.ranked_models()]

    def test_every_slot_sourced_from_price_in_phase1(self) -> None:
        snap = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        assert set(snap.source.values()) == {"price"}

    def test_rank_of_and_strongest(self) -> None:
        snap = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        assert snap.rank_of("deepseek-v4-flash") == 0
        assert snap.rank_of("kimi-k3") == 3
        assert snap.rank_of("absent") is None
        assert snap.strongest() == "kimi-k3"

    def test_prices_ascend_with_rank(self) -> None:
        snap = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        prices = [r.price for r in snap.ordered]
        assert prices == sorted(prices)


class TestCacheSafetySessionPinned:
    """A snapshot resolved at session open is constant for the session (cache-safety)."""

    def test_snapshot_is_a_frozen_value(self) -> None:
        snap = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        # A pinned snapshot must be immutable so nothing can retarget it mid-session.
        assert dataclasses.is_dataclass(snap)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.ordered = []  # type: ignore[misc]

    def test_held_snapshot_unchanged_after_a_later_resolve(self) -> None:
        # Phase-1 STRUCTURAL check only: a held snapshot is a frozen value. It does NOT prove the
        # cache-safety wall against a *changing* signal, because in Phase 1 the source (list price)
        # cannot drift within a process — so a re-resolve is trivially identical. The real
        # anti-drift proof belongs to the Phase-2 gate that introduces a mutable learned rank and
        # wires this snapshot into the engine (see capability_rank.py NOT-YET-WIRED).
        resolver = CapabilityRankResolver(_live_pool())
        pinned = resolver.snapshot_for_session()
        before = pinned.names()
        later = resolver.snapshot_for_session()
        assert pinned.names() == before  # the held snapshot did not change
        assert later.names() == before  # and the new one agrees (price order is process-constant)

    def test_escalation_climbs_a_constant_rank_within_a_session(self) -> None:
        # The escalation ladder reads pool.rank_of / models_from_rank (the price order the
        # snapshot pins). It is process-constant, so a within-session escalation can only
        # ever step to a strictly-higher, deterministic rank — never a mid-turn retarget.
        pool = _live_pool()
        assert pool.rank_of("deepseek-v4-flash") == 0
        higher = [m.name for m in pool.models_from_rank(1)]
        assert higher == ["qwen3.7-plus", "gpt-5-mini", "kimi-k3"]

    def test_snapshot_is_deterministic(self) -> None:
        a = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        b = CapabilityRankResolver(_live_pool()).snapshot_for_session()
        assert a == b
        assert isinstance(a, CapabilityRank)
