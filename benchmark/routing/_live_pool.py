"""Shared read of the packaged live pool: registry restricted to router.yaml's models."""

# Used by ladder_rungs.png (panel B) and the model-triage rule (SH015) so both draw the SAME
# pool the product builds at boot. Reads the PACKAGED registry and router.yaml, never the host
# overrides under SHUNT_CONFIG_DIR / ~/.config: a host override would make either a judgement
# about a pool the shipped router does not build.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shunt.router.policy import RouterPolicy


def _policy() -> RouterPolicy:
    """The packaged router.yaml policy (the shipped defaults, not host overrides)."""
    from shunt.router.policy import load_router_policy, packaged_policy_path  # noqa: PLC0415

    return load_router_policy(packaged_policy_path())


def packaged_live_pool() -> list[str]:
    """Price-ordered names of the shipped router's live pool (router.yaml models: list)."""
    from shunt.models.config import ModelPool, default_registry_path  # noqa: PLC0415

    pool = ModelPool(str(default_registry_path()))
    pool.restrict_to_live(_policy().models)
    return [m.name for m in pool.ranked_models()]


def packaged_rank_shortlist() -> int:
    """The packaged router.yaml escalation rank_shortlist (the ladder's walk shape)."""
    return _policy().escalation.rank_shortlist
