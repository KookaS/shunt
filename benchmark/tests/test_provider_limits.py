"""The per-lane limits registry: resolution, provenance, and cross-file consistency."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmark import config
from benchmark.runner import lane_scheduler as ls

_DATA = Path(__file__).resolve().parents[1] / "routing" / "data"
FREE_CATALOGS = _DATA / "free_catalogs.yaml"
PROVIDER_LIMITS = _DATA / "provider_limits.yaml"


def test_registry_path_resolves_to_the_declared_file() -> None:
    assert config.lane_registry_path() == PROVIDER_LIMITS


def test_groq_lane_resolves_to_its_measured_limits() -> None:
    resolved = config.lane_limits_from_registry("groq-gpt-oss-120b", "groq")
    assert resolved["rpm"] == 30
    assert resolved["rpd"] == 1000
    assert resolved["tpm"] == 8000
    assert resolved["max_request_tokens"] == 8026
    assert resolved["daily_token_budget"] == 200000


def test_an_unregistered_lane_with_no_provider_fails_closed() -> None:
    # A providerless lane has no catalogue that can vouch for a free tier, so the declared-free
    # gate must refuse it rather than fall through to the scheduler's free_access=True default.
    resolved = config.lane_limits_from_registry("lane-a", None)
    assert resolved.get("free_access") is False
    assert "providerless" in str(resolved.get("access_note"))
    reason = ls.structural_refusal(ls.LaneLimits.from_mapping(resolved))
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason


def test_an_unregistered_provider_is_structurally_refused() -> None:
    # A provider absent from free_catalogs.yaml has no free lane; the scheduler refuses it
    # through the SAME LANE_NO_FREE_ACCESS limitation, never a separate branch.
    resolved = config.lane_limits_from_registry("lane-a", "not-a-provider")
    assert resolved.get("free_access") is False
    reason = ls.structural_refusal(ls.LaneLimits.from_mapping(resolved))
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason


def test_lane_limits_merge_precedence(monkeypatch) -> None:
    registry = {
        "defaults": {"rpm": 1, "rpd": 1},
        "providers": {"p": {"rpm": 2, "tpm": 100, "lanes": {"lane": {"tpm": 50}}}},
        "lanes": {"lane": {"rpd": 9}},
    }
    monkeypatch.setattr(config, "load_lane_registry", lambda: registry)
    monkeypatch.setattr(config, "free_provider_access", lambda: {"p": None})
    assert config.lane_limits_from_registry("lane", "p") == {"rpm": 2, "rpd": 9, "tpm": 50}


def test_groq_lane_is_structurally_refused_by_the_scheduler() -> None:
    resolved = config.lane_limits_from_registry("groq-gpt-oss-120b", "groq")
    reason = ls.structural_refusal(ls.LaneLimits.from_mapping(resolved))
    assert reason is not None and ls.LANE_TPM_TOO_SMALL in reason


def test_vercel_free_lanes_resolve_limits_and_are_admitted() -> None:
    # The two confirmed free chat lanes: their provider block publishes no rpm/rpd, so the
    # lane falls back to the campaign's unknown defaults and is admitted (no structural block).
    for lane in ("vercel-ling-3.0-flash-fin-free", "vercel-ling-3.0-flash-sante-free"):
        resolved = config.lane_limits_from_registry(lane, "vercel_ai_gateway")
        assert resolved.get("rpm") is None and resolved.get("rpd") is None
        assert resolved.get("free_access") is not False
        limits = ls.LaneLimits.from_mapping(resolved)
        assert limits.free_access is True
        sched = ls.LaneScheduler(limits={lane: limits})
        assert sched.can_admit(lane, now=0.0)


def test_opencode_zen_lane_is_refused_as_no_free_access() -> None:
    resolved = config.lane_limits_from_registry(
        "opencode-nemotron-3.5-lightning-free", "opencode_zen"
    )
    assert resolved.get("free_access") is False
    assert "MissingSessionID" in str(resolved.get("access_note"))
    reason = ls.structural_refusal(ls.LaneLimits.from_mapping(resolved))
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason
    assert "MissingSessionID" in reason


def test_a_lane_on_a_provider_with_no_free_lane_is_refused_at_the_scheduler() -> None:
    # Together has no free subset; the overlay row stays for provenance, but a lane reaching
    # the scheduler from it must be refused through the existing LANE_NO_FREE_ACCESS rule.
    lane = "together-muse-glimmer-30b"
    resolved = config.lane_limits_from_registry(lane, "together")
    assert resolved.get("free_access") is False
    reason = ls.structural_refusal(ls.LaneLimits.from_mapping(resolved))
    assert reason is not None and ls.LANE_NO_FREE_ACCESS in reason
    limits = ls.LaneLimits.from_mapping(resolved)
    sched = ls.LaneScheduler(limits={lane: limits})
    assert not sched.can_admit(lane, now=0.0)
    assert not sched.is_available(lane, now=0.0)


def test_expires_at_is_an_enforced_limit_field(monkeypatch) -> None:
    """`expires_at` rides the registry/overlay mapping and reaches LaneLimits.

    Regression: the `_EXPIRED` limitation could never fire because `expires_at` was absent from
    `_LIMIT_FIELDS`, so a registry (or `lanes.limits`) expiry was dropped during resolution.
    """
    registry = {"providers": {"p": {"expires_at": "2026-10-01T00:00:00+00:00"}}}
    monkeypatch.setattr(config, "load_lane_registry", lambda: registry)
    monkeypatch.setattr(config, "free_provider_access", lambda: {"p": None})
    resolved = config.lane_limits_from_registry("lane", "p")
    assert resolved["expires_at"] == "2026-10-01T00:00:00+00:00"
    assert ls.LaneLimits.from_mapping(resolved).expires_at == "2026-10-01T00:00:00+00:00"


def test_a_declared_free_provider_lane_is_not_refused() -> None:
    # The negative control: a genuinely free provider stays admitted through the same path.
    lane = "kilo-laguna-xs-2.1-free"
    resolved = config.lane_limits_from_registry(lane, "kilo_gateway")
    assert resolved.get("free_access") is not False
    limits = ls.LaneLimits.from_mapping(resolved)
    assert ls.structural_refusal(limits) is None
    sched = ls.LaneScheduler(limits={lane: limits})
    assert sched.can_admit(lane, now=0.0)


def test_provider_limits_covers_every_scanned_provider() -> None:
    scanned = set((yaml.safe_load(FREE_CATALOGS.read_text()) or {}).get("providers", {}))
    limits = set((yaml.safe_load(PROVIDER_LIMITS.read_text()) or {}).get("providers", {}))
    assert scanned <= limits


def test_shared_rpm_rpd_scope_agree_between_registries() -> None:
    # `free_catalogs.yaml` is the scanner's copy of the same published fact; where both name a
    # value they must agree, so scheduler enforcement cannot silently drift from the scan.
    catalogs = (yaml.safe_load(FREE_CATALOGS.read_text()) or {}).get("providers", {})
    limits = (yaml.safe_load(PROVIDER_LIMITS.read_text()) or {}).get("providers", {})
    for provider, block in catalogs.items():
        free_tier = block.get("free_tier") or {}
        registry = limits.get(provider) or {}
        for field in ("rpm", "rpd", "scope", "free_access"):
            left = free_tier.get(field)
            right = registry.get(field)
            if left is not None and right is not None:
                assert left == right, f"{provider}.{field}: {left!r} != {right!r}"
