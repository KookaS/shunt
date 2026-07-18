from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _NoDuplicateKeyLoader(yaml.SafeLoader):  # type: ignore[misc] # SafeLoader is untyped
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last."""


def _construct_mapping_no_dups(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    """Reject a duplicate key, then build the mapping normally (nesting handled recursively)."""
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            line = key_node.start_mark.line + 1
            raise ValueError(f"duplicate key {key!r} in YAML mapping at line {line}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=True)  # type: ignore[no-any-return]


_NoDuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_dups
)


def strict_yaml_load(text: str) -> dict[str, Any]:
    """Like ``yaml.safe_load``, but a duplicate mapping key is an error, not silent last-wins.

    Config files with a copy-pasted duplicate provider/model row would otherwise
    silently shadow the earlier one — the exact class of silent config bug to reject.
    """
    data: dict[str, Any] = yaml.load(text, Loader=_NoDuplicateKeyLoader)  # noqa: S506 - SafeLoader subclass
    return data


Tier = Literal["cheap", "mid", "frontier"]

TIER_ORDER: Final[list[Tier]] = ["cheap", "mid", "frontier"]

DEFAULT_PROBE_ENDPOINT: Final[str] = "/v1/chat/completions"
# The authenticated (200) check GETs this — a model listing, billed to no one.
# Overridden per-provider where the default is public (OpenRouter) or set to None
# where no free authenticated endpoint exists (Requesty).
DEFAULT_POSITIVE_ENDPOINT: Final[str] = "/v1/models"


class AuthProbe(BaseModel):
    """Measured auth behaviour (rejection + acceptance) — validation-only, never runtime."""

    # Not a Provider field: the signature lives in tools/provider_auth_signatures.yaml
    # and is read only by the probe and its tests, not the router. Kept here as an
    # importable schema so both the probe's loader and the test mock validate against
    # one definition.
    model_config = ConfigDict(extra="forbid")

    # Per-provider, because neither status nor endpoint is universal: Requesty
    # answers 403, xAI 400, and Fireworks only signals auth on /v1/models.
    endpoint: str = DEFAULT_PROBE_ENDPOINT
    expect_status: list[int] = [401]  # noqa: RUF012, SH001 (pydantic field default, copied per-instance)
    expect_body_pattern: str | None = None
    measured_as_of: str | None = None
    # Positive (authenticated) check: with a REAL key, prove the provider ACCEPTS
    # it (200) — the complement of the keyless rejection above. This endpoint is
    # GET-only and must never bill: `/v1/models` for most, `/v1/auth/key` for
    # OpenRouter (whose /v1/models is a public catalog). `None` means the provider
    # exposes no free authenticated endpoint (Requesty), so its positive check is
    # skipped rather than run a billable completion. Used only in --authenticated.
    positive_endpoint: str | None = DEFAULT_POSITIVE_ENDPOINT


class Provider(BaseModel):
    """One access channel: where to send a request and how to authenticate it."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key_env_var: str
    litellm_prefix: str


class Pricing(BaseModel):
    """Benchmark-only list prices + provenance. Absent ⇒ routable but unscored."""

    model_config = ConfigDict(extra="forbid")

    input_cost_per_1m: float
    output_cost_per_1m: float
    cache_read_cost_per_1m: float | None = None
    cache_write_cost_per_1m: float | None = None
    version: str
    price_provider: str
    price_source: str
    price_as_of: str
    price_note: str | None = None


class ReasoningArm(BaseModel):
    """One native reasoning setting for a model: id, within-model rank, raw API params."""

    # `api` stays a free dict[str, Any] blob deliberately un-typed — the four
    # provider forms (boolean flag, effort label, thinking object, hybrid) are
    # genuinely heterogeneous.
    model_config = ConfigDict(extra="forbid")

    id: str
    rank: int
    api: dict[str, Any]  # noqa: ANN401 (heterogeneous provider params)


class ReasoningConfig(BaseModel):
    """A model's reasoning bracket: its declared arms + which one is the default."""

    model_config = ConfigDict(extra="forbid")

    default_arm: str
    arms: list[ReasoningArm]

    @model_validator(mode="after")
    def _check_arms(self) -> ReasoningConfig:
        ids = [arm.id for arm in self.arms]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate reasoning arm id(s): {dupes}")
        if self.default_arm not in ids:
            raise ValueError(
                f"default_arm {self.default_arm!r} does not match any arm id in {sorted(ids)}"
            )
        return self


class ModelEntry(BaseModel):
    """A model row as written in the registry file (`provider` is an FK)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    tier: Tier
    provider: str
    supports_streaming: bool = True
    supports_cache_control: bool = False
    pricing: Pricing | None = None
    reasoning: ReasoningConfig | None = None


class Registry(BaseModel):
    """The parsed registry file: a provider table and a model table."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, Provider]
    models: dict[str, ModelEntry]


class ModelConfig(BaseModel):
    """A model with its provider row resolved onto it — the runtime view."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model_id: str | None = None
    tier: Tier
    provider: str
    base_url: str
    api_key_env_var: str
    litellm_prefix: str = "openai"
    supports_streaming: bool = True
    supports_cache_control: bool = False
    pricing: Pricing | None = None
    reasoning: ReasoningConfig | None = None

    @property
    def route(self) -> str:
        """The litellm target string: `<litellm_prefix>/<model_id>`."""
        return f"{self.litellm_prefix}/{self.model_id or self.name}"


def parse_registry(data: dict[str, Any]) -> Registry:
    """Parse + validate registry data, resolving every model's provider FK."""
    registry = Registry.model_validate(data)
    for name, entry in registry.models.items():
        if entry.provider not in registry.providers:
            known = ", ".join(sorted(registry.providers))
            raise ValueError(
                f"model {name!r} names unknown provider {entry.provider!r} (known: {known})"
            )
    return registry


def resolve_models(registry: Registry) -> dict[str, ModelConfig]:
    """Flatten the registry into ModelConfigs, preserving registry file order."""
    resolved: dict[str, ModelConfig] = {}
    for name, entry in registry.models.items():
        provider = registry.providers[entry.provider]
        resolved[name] = ModelConfig(
            name=name,
            model_id=entry.model_id,
            tier=entry.tier,
            provider=entry.provider,
            base_url=provider.base_url,
            api_key_env_var=provider.api_key_env_var,
            litellm_prefix=provider.litellm_prefix,
            supports_streaming=entry.supports_streaming,
            supports_cache_control=entry.supports_cache_control,
            pricing=entry.pricing,
            reasoning=entry.reasoning,
        )
    return resolved


def arm_api_params(model: ModelConfig, arm_id: str) -> dict[str, Any]:
    """Resolve one model's reasoning arm to its verbatim request params."""
    # {} for the implicit "default" arm of a model with no declared reasoning
    # block (back-compat); raises for any other unknown arm id — this is the
    # EXTRACT seam shared by the benchmark runner and (later) the production router.
    if model.reasoning is None:
        if arm_id == "default":
            return {}
        raise ValueError(f"unknown reasoning arm {arm_id!r} for model {model.name!r}")
    for arm in model.reasoning.arms:
        if arm.id == arm_id:
            return dict(arm.api)
    raise ValueError(f"unknown reasoning arm {arm_id!r} for model {model.name!r}")


def default_registry_path() -> Path:
    """Path to the registry file shipped inside the package."""
    import importlib.resources

    ref = importlib.resources.files("shunt.models") / "default_config.yaml"
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def load_registry(path: str | Path | None = None) -> Registry:
    """Load + validate the registry file (defaults to the packaged one)."""
    path_obj = Path(path) if path is not None else default_registry_path()
    return parse_registry(strict_yaml_load(path_obj.read_text()))


class ModelPool:
    """Thread-safe pool of model configurations with health tracking."""

    DEFAULT_CONFIG_DIR_ENV: ClassVar[str] = "SHUNT_CONFIG_DIR"
    DEFAULT_CONFIG_FILENAME: ClassVar[str] = "models.yaml"
    DEFAULT_HEALTH_CHECK_INTERVAL: ClassVar[int] = 60

    def __init__(self, config_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._health: dict[str, dict[str, bool | float | None]] = {}
        self._health_check_interval: int = self.DEFAULT_HEALTH_CHECK_INTERVAL
        self._config: dict[str, ModelConfig] = {}
        self._load_config(config_path)

    @classmethod
    def load(cls, path: str | None = None) -> ModelPool:
        """Create a ModelPool from an optional config path."""
        return cls(path)

    def _load_config(self, path: str | None) -> None:
        if path is None:
            config_dir = os.environ.get(self.DEFAULT_CONFIG_DIR_ENV)
            if config_dir:
                path = str(Path(config_dir) / self.DEFAULT_CONFIG_FILENAME)
            else:
                path = str(Path.home() / ".config" / "shunt" / self.DEFAULT_CONFIG_FILENAME)

        path_obj = Path(path)
        source = path_obj if path_obj.exists() else default_registry_path()

        self._parse_config(strict_yaml_load(source.read_text()))

    def _parse_config(self, data: dict[str, Any]) -> None:
        # extra="forbid" makes a stale-schema config fail loudly here, at boot,
        # naming the offending key — never silently at request time.
        self._config = resolve_models(parse_registry(data))

        with self._lock:
            for name in self._config:
                self._health[name] = {"healthy": True, "last_failure": None}

    def get_model(self, name: str) -> ModelConfig | None:
        """Look up a model by name, returning None if unknown."""
        return self._config.get(name)

    def get_tier(self, name: str) -> str | None:
        """Return the capability tier for a model, or None if unknown."""
        model = self.get_model(name)
        if model is None:
            return None
        return model.tier

    def get_tier_models(self, tier: str) -> list[ModelConfig]:
        """Return all models in a given tier, in config order."""
        return [m for m in self._config.values() if m.tier == tier]

    def is_healthy(self, name: str) -> bool:
        """Check if a model is healthy (auto-recovers after cooldown)."""
        with self._lock:
            health = self._health.get(name)
            if health is None:
                return False
            if health["healthy"]:
                return True
            last = health.get("last_failure")
            if last is not None and (time.monotonic() - last) >= self._health_check_interval:
                health["healthy"] = True
                health["last_failure"] = None
                return True
            return False

    def mark_unhealthy(self, name: str) -> None:
        """Mark a model as unhealthy (starts cooldown timer)."""
        with self._lock:
            if name in self._health:
                self._health[name]["healthy"] = False
                self._health[name]["last_failure"] = time.monotonic()

    @property
    def health_check_interval(self) -> int:
        """Return the health cooldown interval in seconds."""
        return self._health_check_interval

    def fallback_chain(self, name: str) -> list[str]:
        """Return ordered fallback chain: same-tier first, then cross-tier."""
        model = self.get_model(name)
        if model is None:
            return []

        tier: Tier = model.tier
        tier_idx = TIER_ORDER.index(tier)

        chain: list[str] = []
        seen: set[str] = set()

        chain.append(name)
        seen.add(name)

        same_tier_models = self.get_tier_models(tier)
        for m in same_tier_models:
            if m.name not in seen:
                chain.append(m.name)
                seen.add(m.name)

        higher_tiers = TIER_ORDER[tier_idx + 1 :]
        lower_tiers = TIER_ORDER[:tier_idx]
        cross_tier_order = higher_tiers + lower_tiers

        for other_tier in cross_tier_order:
            for m in self.get_tier_models(other_tier):
                if m.name not in seen:
                    chain.append(m.name)
                    seen.add(m.name)

        return chain
