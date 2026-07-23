"""Production router policy — the schema for ``router.yaml`` (active strategy +
knobs + exploration), shared with the benchmark so both configure one algorithm.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shunt.models.config import strict_yaml_load
from shunt.router.escalation import EscalationConfig

logger = logging.getLogger(__name__)

# Live-eligible routing strategies — the set ``router.strategy`` may name. Benchmark-only
# strategies (oracle, external_prior, random) are deliberately absent: they need ground
# truth / external priors and cannot run on live traffic. ``knn_cascade`` is also absent:
# a true quality-cascade needs mid-session verify-then-escalate, which is not one cache-safe
# per-session decision (routing once per session is the product's spine), and the upstream
# fallback chain is availability-only, not quality-based — so it stays benchmark-only.
LIVE_STRATEGIES: Final[tuple[str, ...]] = (
    "knn",
    "always_cheap",
    "always_frontier",
)

_CONFIG_DIR_ENV: Final[str] = "SHUNT_CONFIG_DIR"
_CONFIG_FILENAME: Final[str] = "router.yaml"


class KnnPolicy(BaseModel):
    """kNN selection knobs — shared schema, distinct value-sets per environment."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=20, gt=0)
    success_rate_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_samples: int = Field(default=3, ge=0)


class ExplorationPolicy(BaseModel):
    """Cost-aware Thompson-sampling exploration knobs (see the research doc)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # prior_alpha/beta must be > 0 — Beta(0, .) crashes np.random.beta on the live path,
    # so the wall is here (reject a bad router.yaml at load), not at first request.
    prior_alpha: float = Field(default=1.0, gt=0.0)
    prior_beta: float = Field(default=1.0, gt=0.0)
    explore_budget_frac: float = Field(default=0.4, ge=0.0)
    conservative_alpha: float = Field(default=0.1, ge=0.0, le=1.0)
    propensity_mc_samples: int = Field(default=100, ge=0)
    # Cap on the offline-seeded prior's pseudo-count strength (empirical-Bayes shrinkage):
    # even a large model history contributes at most this many prior observations, so it
    # regularizes the sparse local neighborhood rather than swamping it.
    prior_strength_cap: float = Field(default=20.0, ge=0.0)


class EscalationPolicy(BaseModel):
    """Auto-escalation knobs. Shipped OFF — enabling is a config change."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    escalate_after_n: int = Field(default=2, gt=0)
    stale_window: int = Field(default=10, gt=0)
    blocking_exit_code: int = Field(default=2, ge=0)
    ladder: str = Field(default="effort_then_tier")

    @model_validator(mode="after")
    def _check_ladder(self) -> EscalationPolicy:
        allowed = ("effort_then_tier", "tier_only")
        if self.ladder not in allowed:
            joined = ", ".join(allowed)
            raise ValueError(f"unknown escalation.ladder {self.ladder!r}; allowed: {joined}")
        return self

    def to_config(self) -> EscalationConfig:
        """Bridge the config-file schema to the pure-logic ``EscalationConfig`` the engine reads."""
        return EscalationConfig(
            enabled=self.enabled,
            escalate_after_n=self.escalate_after_n,
            stale_window=self.stale_window,
            blocking_exit_code=self.blocking_exit_code,
            ladder=self.ladder,
        )


class CapturePolicy(BaseModel):
    """Off-wire capture config: where the router re-runs the repo's tests.

    ``work_dir`` is a single repo root; ``work_dirs`` maps ``tool_identity`` → repo.
    Both are operator config only — never a wire path (RCE via subprocess cwd).
    """

    model_config = ConfigDict(extra="forbid")

    work_dir: str | None = None
    work_dirs: dict[str, str] = Field(default_factory=dict)


class RefitPolicy(BaseModel):
    """Batch offline re-fit cadence: rebuild the kNN index from the append-only log."""

    # Learning is batch-first (research pattern #4) — the index is a rebuildable projection,
    # re-fit every every_n_outcomes captured outcomes. 0 disables the trigger (the boot-time
    # rebuild still runs); mid-decision safety is the store lock.
    model_config = ConfigDict(extra="forbid")

    every_n_outcomes: int = Field(default=50, ge=0)


class RouterPolicy(BaseModel):
    """Top-level ``router.yaml`` schema: one active strategy + its knobs + exploration."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "knn"
    policy: KnnPolicy = Field(default_factory=KnnPolicy)
    exploration: ExplorationPolicy = Field(default_factory=ExplorationPolicy)
    escalation: EscalationPolicy = Field(default_factory=EscalationPolicy)
    capture: CapturePolicy = Field(default_factory=CapturePolicy)
    refit: RefitPolicy = Field(default_factory=RefitPolicy)
    # Which registry models are live-routable. Empty = every model in models.yaml
    # (backward compatible). Each name must exist in the registry; that cross-check
    # happens at ModelPool wiring (this schema has no registry access). Benchmark
    # model selection is separate (benchmark/benchmark.yaml).
    models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_strategy(self) -> RouterPolicy:
        if self.strategy not in LIVE_STRATEGIES:
            allowed = ", ".join(LIVE_STRATEGIES)
            raise ValueError(f"unknown router.strategy {self.strategy!r}; live-eligible: {allowed}")
        return self


def parse_router_policy(data: dict[str, object] | None) -> RouterPolicy:
    """Validate a ``router:`` config mapping into a RouterPolicy (defaults if empty)."""
    if not data:
        return RouterPolicy()
    router_section = data.get("router", data)
    if router_section is None:  # a present-but-null `router:` key → defaults, not a crash
        return RouterPolicy()
    return RouterPolicy.model_validate(router_section)


def _env_bool(raw: str) -> bool:
    """Parse a truthy env string (``1/true/yes/on``, case-insensitive)."""
    return raw.strip().lower() in ("1", "true", "yes", "on")


def apply_env_overrides(policy: RouterPolicy) -> RouterPolicy:
    """Overlay ``SHUNT_ROUTER_*`` env vars on *policy* (env > file > packaged default).

    Re-validates through the schema so a bad override (unknown strategy, negative
    budget) fails loudly at boot, mirroring ``router.yaml`` parsing.
    """
    strategy = os.environ.get("SHUNT_ROUTER_STRATEGY")
    enabled = os.environ.get("SHUNT_EXPLORATION_ENABLED")
    budget = os.environ.get("SHUNT_EXPLORE_BUDGET_FRAC")
    if strategy is None and enabled is None and budget is None:
        return policy

    data = policy.model_dump()
    if strategy is not None:
        data["strategy"] = strategy
    if enabled is not None:
        data["exploration"]["enabled"] = _env_bool(enabled)
    if budget is not None:
        data["exploration"]["explore_budget_frac"] = float(budget)
    return RouterPolicy.model_validate(data)


def _user_config_path() -> Path:
    config_dir = os.environ.get(_CONFIG_DIR_ENV)
    base = Path(config_dir) if config_dir else Path.home() / ".config" / "shunt"
    return base / _CONFIG_FILENAME


def packaged_policy_path() -> Path:
    """Path to the router policy shipped inside the package."""
    import importlib.resources

    ref = importlib.resources.files("shunt.config") / _CONFIG_FILENAME
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def load_router_policy(path: str | Path | None = None) -> RouterPolicy:
    """Explicit path → $SHUNT_CONFIG_DIR/router.yaml → packaged router.yaml → defaults."""
    # Env-var / CLI-flag overlays are applied by the server layer, not here.
    resolved = Path(path) if path is not None else _user_config_path()
    if not resolved.exists():
        logger.debug("router policy: %s absent, falling back to packaged", resolved)
        resolved = packaged_policy_path()
    if resolved.exists():
        # Which FILE won matters: a rig can serve a config that differs from the one
        # you last edited, and nothing else in the logs distinguishes them.
        logger.debug("router policy: loaded from %s", resolved)
        return parse_router_policy(strict_yaml_load(resolved.read_text()))
    logger.debug("router policy: no file found, using built-in defaults")
    return RouterPolicy()
