"""Production router policy — the schema for ``router.yaml`` (active strategy +
knobs + exploration), shared with the benchmark so both configure one algorithm.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from shunt.models.config import strict_yaml_load
from shunt.proxy.context_transfer import (
    CONTEXT_TRANSFER_FULL,
    CONTEXT_TRANSFER_MODES,
)
from shunt.router.escalation import ESCALATION_LADDERS, EscalationConfig

logger = logging.getLogger(__name__)

# Live-eligible routing strategies — the set ``router.strategy`` may name. Benchmark-only
# strategies (oracle, random) are deliberately absent: they need ground truth or are
# not routers at all, so they cannot run on live traffic. The WITHIN-TASK quality cascades
# are absent and stay absent: they take a verified outcome per attempt mid-session, which is
# not one cache-safe per-session decision (routing once per session is the product's spine),
# and the upstream fallback chain is availability-only, not quality-based.
#
# The two SESSION-cadence cascades below are the cache-safe operating points those approximate,
# and they ARE nameable, because they pace the same ladder one attempt per SESSION: a base pick
# first, a verified failure walks a rung at the next boundary. They are PRESETS rather than new
# selection rules — a base selection rule plus ``escalation.enabled`` — and ``_check_strategy``
# below refuses either with escalation off so the name cannot mean a bare non-escalating router.
#
# ``session_cascade`` is the SHIPPED DEFAULT (see ``router.yaml``): the ladder opens at the
# cheapest model and consults no neighbourhood, so a default install never embeds.
# ``knn_cascade`` is the OPT-IN routing strategy, and its name is the honest one for what a kNN
# install has always run: the kNN pick has carried ``participates_in_escalation = True`` since
# escalation shipped ON, so a config reading ``strategy: knn`` was already a kNN pick plus the
# ladder. Plain kNN WITHOUT the ladder is no longer a selectable deployable option; ``knn`` is
# accepted only as a migration alias (see ``parse_router_policy``).
LIVE_STRATEGIES: Final[tuple[str, ...]] = (
    "knn_cascade",
    "always_cheap",
    "always_frontier",
    "session_cascade",
)

# The presets above are only themselves with the escalation layer on: without it each config
# resolves to its plain base selection rule under a name promising a cascade.
SESSION_CASCADE_STRATEGY: Final[str] = "session_cascade"
KNN_CASCADE_STRATEGY: Final[str] = "knn_cascade"
_CASCADE_IDS: Final[frozenset[str]] = frozenset({SESSION_CASCADE_STRATEGY, KNN_CASCADE_STRATEGY})

# The pre-rename spelling of ``knn_cascade``. Accepted for at least one minor release so an
# existing router.yaml keeps booting; see ``parse_router_policy`` for the resolution rules.
LEGACY_STRATEGY_ALIASES: Final[dict[str, str]] = {"knn": KNN_CASCADE_STRATEGY}

# Per-cascade error vocabulary: the base selection rule the preset wraps, and the id an operator
# who genuinely wants that rule WITHOUT the ladder should name instead. `knn_cascade` has no such
# alternative — plain kNN stopped being selectable with this rename — so it points at the nearest
# non-escalating fixed router rather than at a name that no longer exists.
_CASCADE_BASES: Final[dict[str, tuple[str, str]]] = {
    SESSION_CASCADE_STRATEGY: ("always_cheap", "always_cheap"),
    KNN_CASCADE_STRATEGY: ("the kNN pick", "always_cheap"),
}

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
    """Auto-escalation knobs. Ships ON (owner choice, 2026-08-08) but only fires once a
    `capture.work_dir` gives it a repo to verify (a boot warning covers enabled-without-one);
    the trigger is a null detector at the live cadence — see EscalationConfig.enabled."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    escalate_after_n: int = Field(default=2, gt=0)
    stale_window: int = Field(default=10, gt=0)
    ladder: str = Field(default="effort_then_rank")
    # The cheapest N ranks are walked one rung at a time; the rung that leaves them jumps to the
    # top rank instead of paying every model in between. 0 restores the every-rank walk.
    rank_shortlist: int = Field(default=3, ge=0)
    # Additionally opt-in on top of `enabled` — enabling escalation must never silently
    # randomize. 0.0 = deterministic; above 0, flagged checkpoints are randomized so the
    # policy's value becomes identifiable. Bounded below 1.0: at 1.0 the policy arm is never
    # taken, which loses overlap on the other side.
    exploration_epsilon: float = Field(default=0.0, ge=0.0, lt=1.0)
    exploration_seed: int | None = Field(default=None)
    # What the ESCALATED model receives of the conversation that ran on the cheaper one.
    # `full` (default) is pure pass-through: Shunt forwards the client's messages untouched,
    # which is what has always shipped. `summary` makes Shunt AUTHOR content — it replaces the
    # prior conversation with a compaction on the first turn after an escalation — so it is
    # opt-in, disclosed at boot, on `shunt doctor`, on the wire (`X-Shunt-Context`) and in
    # `shunt explain`. There is deliberately no `none`: Shunt can drop context from the request
    # it forwards but cannot make the CLI forget, so `none` would have to strip on EVERY turn
    # and the strong model could never accumulate state — a broken router, not a variant.
    context_transfer: str = Field(default=CONTEXT_TRANSFER_FULL)
    # Ceiling on the authored summary, in tokens. The summariser is asked for this budget and
    # an over-budget answer degrades to `full` rather than being truncated mid-sentence.
    context_transfer_max_tokens: int | None = Field(default=2000, gt=0)
    # Which model writes the summary. null ⇒ the OUTGOING, pre-escalation model: it is the
    # cheaper one and its prefix is already warm, so it re-serves a cached prefix. Naming the
    # INCOMING model would pay the full uncached prefill this feature exists to avoid.
    context_transfer_model: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_ladder(self) -> EscalationPolicy:
        if self.ladder not in ESCALATION_LADDERS:
            joined = ", ".join(ESCALATION_LADDERS)
            raise ValueError(f"unknown escalation.ladder {self.ladder!r}; allowed: {joined}")
        if self.context_transfer not in CONTEXT_TRANSFER_MODES:
            joined = ", ".join(CONTEXT_TRANSFER_MODES)
            extra = (
                ""
                if self.context_transfer != "none"
                else (
                    " — `none` is deliberately not offered: Shunt cannot make the client forget, "
                    "so it would strip context on every turn"
                )
            )
            raise ValueError(
                f"unknown escalation.context_transfer {self.context_transfer!r}; "
                f"allowed: {joined}{extra}"
            )
        return self

    def to_config(self) -> EscalationConfig:
        """Bridge the config-file schema to the pure-logic ``EscalationConfig`` the engine reads."""
        return EscalationConfig(
            enabled=self.enabled,
            escalate_after_n=self.escalate_after_n,
            stale_window=self.stale_window,
            ladder=self.ladder,
            rank_shortlist=self.rank_shortlist,
            exploration_epsilon=self.exploration_epsilon,
            exploration_seed=self.exploration_seed,
        )


class CapturePolicy(BaseModel):
    """Off-wire capture config: where the router re-runs the repo's tests.

    ``work_dir`` is a single repo root; ``work_dirs`` maps ``tool_identity`` → repo.
    Both are operator config only — never a wire path (RCE via subprocess cwd).
    """

    model_config = ConfigDict(extra="forbid")

    work_dir: str | None = None
    work_dirs: dict[str, str] = Field(default_factory=dict)
    # Opt-in full-content per-step trajectory capture. Shipped OFF (kill-gate posture): unset
    # ⇒ only the behaviour-only verified-outcome fields are recorded. When true, the recorder
    # redacts every free-text field then encrypts it to a LOCAL, gitignored dir at rest —
    # never on the wire, never mid-cached-turn, and it never alters a routing decision.
    full_content: bool = False
    # Where the encrypted local plane is written. Unset ⇒ the recorder resolves a default
    # OUTSIDE the repo ($SHUNT_HOME/trajectories); never inside the tree, never committed.
    trajectory_dir: str | None = None
    # Wall-clock budget for ONE off-wire verification run. null ⇒ the verifier's built-in
    # 1800s (30 min — the floor for a real repo suite). A suite slower than the budget times
    # out, returns `unknown` + infra-failure and writes nothing — so on a big repo the whole
    # loop is a silent no-op until this is raised.
    verify_timeout_seconds: float | None = Field(default=None, gt=0.0)
    # How many times a FAILING suite is re-run before the red is trusted (flake guard).
    # Total worst-case runs = 1 + rerun_confirmations, each up to verify_timeout_seconds.
    # Floored at 1, not 0: an unconfirmed failure is dropped by the escalation gate, so 0
    # ("stop re-running my slow suite") would silently turn auto-escalation into a null
    # detector with nothing in the logs. Turn escalation off explicitly instead.
    rerun_confirmations: int = Field(default=2, ge=1)
    # Trust Shunt's OWN launch directory (promoted to its git root) as the last work_dir
    # layer, so `cd myrepo && shunt start` captures out of the box. It is checked for
    # capability (git root · test framework present), NOT for trust — the directory is one
    # the operator chose by starting the binary in it. Arming ANY work_dir arms arbitrary
    # code execution: the verifier runs the tree's own test command, which executes its
    # `conftest.py` / `build.rs` / npm scripts. This switch is how you refuse that on a
    # shared or multi-tenant host, and it is the only gate on the layer.
    trust_launch_dir: bool = True


class RefitPolicy(BaseModel):
    """Batch offline re-fit cadence: rebuild the kNN index from the append-only log."""

    # Learning is batch-first (research pattern #4) — the index is a rebuildable projection,
    # re-fit every every_n_outcomes captured outcomes. 0 disables the trigger (the boot-time
    # rebuild still runs); mid-decision safety is the store lock.
    model_config = ConfigDict(extra="forbid")

    every_n_outcomes: int = Field(default=50, ge=0)


class BudgetPolicy(BaseModel):
    """Refuse routing once a session's cumulative reported spend reaches the cap."""

    # A SOFT ceiling enforced at the NEXT request boundary: the check runs before the
    # upstream call, so the request that crosses ``max_spend_usd`` completes and the
    # following one is refused. The bound is one session's cumulative PROVIDER-REPORTED
    # charge (``usage.cost`` from the upstream, never a locally derived price); at the cap
    # the router refuses further routing for that session with a clean error instead of a
    # fabricated success. null = unlimited (the default). This is the knob the live tier's
    # spend cap (compose.live.yaml) documents; the exploration budget is a separate, softer knob.

    model_config = ConfigDict(extra="forbid")

    # null = unlimited. A number >= 0 is a ceiling on one session's cumulative spend,
    # enforced at the next request boundary; 0.0 refuses every request for a session
    # (explicit "no spend allowed").
    max_spend_usd: float | None = Field(default=None, ge=0.0)


class RouterPolicy(BaseModel):
    """Top-level ``router.yaml`` schema: one active strategy + its knobs + exploration."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = SESSION_CASCADE_STRATEGY
    policy: KnnPolicy = Field(default_factory=KnnPolicy)
    exploration: ExplorationPolicy = Field(default_factory=ExplorationPolicy)
    escalation: EscalationPolicy = Field(default_factory=EscalationPolicy)
    capture: CapturePolicy = Field(default_factory=CapturePolicy)
    refit: RefitPolicy = Field(default_factory=RefitPolicy)
    # Per-session spend cap; absent ⇒ no cap (unlimited). The live tier's
    # documented cap key (compose.live.yaml `router.budget.max_spend_usd`) is this.
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    # Which registry models are live-routable. Empty = every model in models.yaml
    # (backward compatible). Each name must exist in the registry; that cross-check
    # happens at ModelPool wiring (this schema has no registry access). Benchmark
    # model selection is separate (benchmark/benchmark.yaml).
    models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_strategy(self, info: ValidationInfo) -> RouterPolicy:
        if self.strategy not in LIVE_STRATEGIES:
            allowed = ", ".join(LIVE_STRATEGIES)
            raise ValueError(f"unknown router.strategy {self.strategy!r}; live-eligible: {allowed}")
        # A COHERENCE rule, not a precondition (contrast the work_dir NOTE below). The whole
        # content of the `session_cascade` name is `always_cheap` + the ladder; with the ladder
        # off the config silently resolves to a fixed cheap router under a name that promises a
        # cascade, which is the deployability-honesty defect this id exists to remove. It is safe
        # as a load ERROR — unlike the work_dir case it can never be the default state, because
        # nothing selects this id by accident.
        if self.strategy in _CASCADE_IDS and not self.escalation.enabled:
            if not _strategy_was_written(info):
                # NOT an error, on purpose: the operator turned the ladder off and never named a
                # cascade — `router.strategy` merely DEFAULTS to one. Refusing here would brick
                # the commonest way to disable escalation (`escalation: {enabled: false}` and
                # nothing else), which is a supported, documented config. The name is still
                # misleading in that state, so it is a loud warning rather than silence.
                logger.warning(
                    "router.escalation.enabled is false while router.strategy defaults to %r, "
                    "so the ladder that name promises does not run — this install is a bare "
                    "%s. That is a supported configuration; name the strategy explicitly "
                    "to make the intent visible, or set escalation.enabled: true.",
                    self.strategy,
                    _CASCADE_BASES[self.strategy][0],
                )
                return self
            # Name the likeliest CAUSE, not just the state. The commonest way to reach here is
            # not "the operator wrote enabled: false" — it is a hand-written router.yaml
            # carrying only `strategy: <a cascade id>`, because a user file replaces the
            # packaged one wholesale and `parse_router_policy` reads an ABSENT escalation block
            # as OFF. Without that sentence the error reads as a contradiction of the docs.
            base, alternative = _CASCADE_BASES[self.strategy]
            raise ValueError(
                f"router.strategy {self.strategy!r} requires "
                f"router.escalation.enabled: true — the preset IS {base} plus the "
                f"escalation ladder, and with the ladder off it is nothing but {base}, "
                f"wearing a cascade's name. If your router.yaml has no `escalation:` block "
                f"at all, that is why: a user file replaces the packaged one wholesale, and "
                f"an absent block means OFF. Add `escalation: {{enabled: true}}`, or use "
                f"{alternative!r}."
            )
        return self

    # NOTE — the escalation/work_dir precondition is deliberately NOT a load error. Escalation's
    # only signal is a verified failure keyed by the resolved work_dir (the repo): without one the
    # engine's `_task_key` resolves to None and no FailureEvent is ever logged, so escalation is
    # INERT. While escalation shipped OFF, an enabled-without-work_dir config was an operator
    # footgun worth rejecting at load. Escalation now ships ON by default, which makes
    # "enabled without a work_dir" the COMMON default state (a plain install has no repo to test)
    # — a load error there would brick every install until a work_dir is set. The never-silently-
    # inert guarantee now lives as a prominent boot WARNING in `server.py`'s disclosure
    # (`_log_capture_disclosure` warns when escalation is enabled but no work_dir is resolvable)
    # and in the docs. Revisit as a hard error only if escalation gains a signal needing no repo.


# The validation-context key recording whether `router.strategy` was WRITTEN by the operator
# (as opposed to defaulted, or migrated from the retired `knn` spelling). Only an explicitly
# written cascade id turns the escalation-coherence rule into a load error; see `_check_strategy`.
_STRATEGY_WRITTEN: Final[str] = "strategy_was_written"


def _strategy_was_written(info: ValidationInfo) -> bool:
    """True unless the caller declared the strategy defaulted/migrated. Absent context ⇒ True."""
    # Fail CLOSED: a direct `RouterPolicy(...)` construction carries no context, and there the
    # caller did name the strategy, so the strict reading is the correct one.
    context = info.context
    if not isinstance(context, dict):
        return True
    return bool(context.get(_STRATEGY_WRITTEN, True))


def parse_router_policy(data: dict[str, object] | None) -> RouterPolicy:
    """Validate a ``router:`` config mapping into a RouterPolicy (defaults if empty)."""
    if not data:
        return RouterPolicy()
    router_section = data.get("router", data)
    if router_section is None:  # a present-but-null `router:` key → defaults, not a crash
        return RouterPolicy()
    if not isinstance(router_section, dict):
        return RouterPolicy.model_validate(router_section)
    written = isinstance(router_section.get("strategy"), str) and (
        router_section["strategy"] not in LEGACY_STRATEGY_ALIASES
    )
    return RouterPolicy.model_validate(
        _migrate_router_section(router_section), context={_STRATEGY_WRITTEN: written}
    )


def _migrate_router_section(raw: dict[str, object]) -> dict[str, object]:
    """Rewrite a pre-rename ``router:`` mapping so it still boots, with a warning per rewrite."""
    written = raw.get("strategy")
    migrated = LEGACY_STRATEGY_ALIASES.get(written) if isinstance(written, str) else None
    aliased = migrated is not None
    section = dict(raw)
    if migrated is not None:
        section["strategy"] = migrated
        logger.warning(
            "router.strategy %r is the pre-rename spelling of %r and has been migrated "
            "automatically — behaviour unchanged, because a kNN install has always run the "
            "escalation ladder on top of the kNN pick. Note this is NOT the shipped default, "
            "which is %r. Update router.yaml; the alias is kept for at least one minor "
            "release, then removed.",
            written,
            migrated,
            SESSION_CASCADE_STRATEGY,
        )
    if "escalation" in section:
        return section
    # ESCAPE HATCH — an `escalation:` block that is ABSENT is an OLD config, not an opt-in.
    # A user policy file replaces the packaged one wholesale (no key-by-key merge), so a
    # config that predates the escalation block never had a chance to write
    # `escalation.enabled` — yet the schema's default_factory would silently flip it ON.
    # An absent key therefore means OFF (the operator never opted in); only an explicit
    # `enabled: true`, the packaged file, or the bare code default ships escalation ON.
    #
    # THE ONE EXCEPTION, and why it is not a hole in the hatch: a strategy that was ALIASED or
    # DEFAULTED never named a cascade, so the operator cannot have meant "the cascade without
    # the ladder". Resolving those to OFF would hand every pre-rename config to the coherence
    # rule below and brick it at boot — a rename that stops existing installs is a worse defect
    # than the one it fixes. An EXPLICIT cascade id still falls through to OFF and still raises,
    # because there the error is the point: the operator typed a cascade name and the file says
    # nothing about the ladder, which is exactly what that message explains.
    if written is None or aliased:
        section["escalation"] = {"enabled": True}
        logger.warning(
            "router.yaml names no `escalation:` block, and its strategy was defaulted or "
            "migrated, so the escalation ladder resolves to ENABLED — that is what %r means "
            "and what this install has always run. Write `escalation: {enabled: false}` to "
            "turn it off.",
            section.get("strategy", SESSION_CASCADE_STRATEGY),
        )
        return section
    section["escalation"] = {"enabled": False}
    return section


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
    # The overlay re-validates, so it has to carry the same explicitness the file did — otherwise
    # a policy that loaded fine (defaulted strategy, ladder off) would be refused on re-entry.
    context = {_STRATEGY_WRITTEN: strategy is not None}
    if strategy is not None:
        # The same alias the file path takes: an operator exporting the pre-rename
        # SHUNT_ROUTER_STRATEGY=knn must not be refused at boot by a rename.
        data["strategy"] = LEGACY_STRATEGY_ALIASES.get(strategy, strategy)
    if enabled is not None:
        data["exploration"]["enabled"] = _env_bool(enabled)
    if budget is not None:
        data["exploration"]["explore_budget_frac"] = float(budget)
    return RouterPolicy.model_validate(data, context=context)


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


def resolved_policy_path(path: str | Path | None = None) -> Path | None:
    """The router.yaml `load_router_policy` reads, or None when it falls back to code defaults."""
    # Public so an inspection surface can name the file a value came from. Same resolution as the
    # loader below, once — a second copy would let the CLI cite a file the server never read.
    resolved = Path(path) if path is not None else _user_config_path()
    if not resolved.exists():
        logger.debug("router policy: %s absent, falling back to packaged", resolved)
        resolved = packaged_policy_path()
    return resolved if resolved.exists() else None


def load_router_policy(path: str | Path | None = None) -> RouterPolicy:
    """Explicit path → $SHUNT_CONFIG_DIR/router.yaml → packaged router.yaml → defaults."""
    # Env-var / CLI-flag overlays are applied by the server layer, not here.
    resolved = resolved_policy_path(path)
    if resolved is not None:
        # Which FILE won matters: a rig can serve a config that differs from the one
        # you last edited, and nothing else in the logs distinguishes them.
        logger.debug("router policy: loaded from %s", resolved)
        return parse_router_policy(strict_yaml_load(resolved.read_text()))
    logger.debug("router policy: no file found, using built-in defaults")
    return RouterPolicy()
