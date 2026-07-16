from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import yaml
from pydantic import BaseModel

Tier = Literal["cheap", "mid", "frontier"]

TIER_ORDER: Final[list[Tier]] = ["cheap", "mid", "frontier"]


class ModelConfig(BaseModel):
    """Configuration for a single model in the pool."""

    name: str
    model_id: str | None = None
    tier: Tier
    provider: str
    base_url: str
    api_key_env_var: str
    supports_streaming: bool = True
    supports_cache_control: bool = False


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

        if path_obj.exists():
            with open(path_obj) as f:
                data = yaml.safe_load(f)
        else:
            import importlib.resources

            ref = importlib.resources.files("shunt.models") / "default_config.yaml"
            with importlib.resources.as_file(ref) as default_path, open(default_path) as f:
                data = yaml.safe_load(f)

        self._parse_config(data)

    def _parse_config(self, data: dict[str, Any]) -> None:
        models_data = data.get("models", {})
        for name, cfg in models_data.items():
            self._config[name] = ModelConfig(name=name, **cfg)

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
