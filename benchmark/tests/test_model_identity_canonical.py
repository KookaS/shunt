"""Canonical model identity across providers, and the (model, provider) archive.

The owner rule: the same weights served by different providers share ONE canonical identity,
and identity NEVER encodes the provider or a `-free`/`:free` marker. A provider variant is
merged only when the provider specs prove the same weights; an ambiguous pair is left distinct
(the concordance subset exists to settle those). A retired `(model, provider)` row is archived
— retained for provenance, withheld from scheduling — without touching the same weights on a
provider that still serves them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.routing.scripts import refresh_free_campaign as refresh
from benchmark.routing.scripts import scan_free_models as scan

OVERLAY: Path = Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "overlay.yaml"


def _row(provider: str, listing_id: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider": provider,
        "listing_id": listing_id,
        "hugging_face_id": "",
        "canonical_slug": "",
        "prices": None,
        "context_length": None,
        "supports_tools": True,
        "expiration_date": None,
        "last_seen": "2026-09-12",
        "withdrawn_at": None,
    }
    row.update(over)
    return row


def _overlay_versions() -> dict[str, str]:
    overlay = yaml.safe_load(OVERLAY.read_text())
    return {name: str(row["version"]) for name, row in overlay["models"].items()}


# ── the owner's known merges ──────────────────────────────────────────────────────────


def test_nvidia_kimi_resolves_to_the_benchmark_kimi_identity() -> None:
    # The same model already in the benchmark: `nvidia-kimi-k3` (NIM's `moonshotai/kimi-k3`)
    # is `kimi-k3`, never a provider-namespaced id.
    assert _overlay_versions()["nvidia-kimi-k3"] == "kimi-k3"


def test_nvidia_deepseek_pro_0813_resolves_to_deepseek_v4_pro() -> None:
    # The 0813 build is deepseek-v4-pro's newest release, not a distinct model. Both free
    # mirrors (NIM and Cloudflare) canonicalise onto the shipped registry identity.
    assert _overlay_versions()["nvidia-deepseek-v4-pro-0813"] == "deepseek-v4-pro"
    assert _overlay_versions()["cloudflare_workers_ai-deepseek-v4-pro-0813"] == "deepseek-v4-pro"


def test_kilo_step_free_resolves_to_the_publisher_slug_without_free_marker() -> None:
    assert _overlay_versions()["kilo-step-3.7-flash-free"] == "step-3.7-flash"


def test_cloudflare_gpt_oss_resolves_to_the_publisher_identity() -> None:
    assert _overlay_versions()["cf-gpt-oss-120b"] == "gpt-oss-120b"
    assert _overlay_versions()["groq-gpt-oss-120b"] == "gpt-oss-120b"
    assert _overlay_versions()["sambanova-gpt-oss-120b"] == "gpt-oss-120b"


def test_propose_confirms_the_known_merges_as_one_identity() -> None:
    identity = scan.load_identity()
    snapshot = {
        "scan_as_of": "2026-09-12",
        "listings": [
            _row("nvidia_nim", "moonshotai/kimi-k3"),
            _row("nvidia_nim", "deepseek-ai/deepseek-v4-pro-0813"),
            _row("cloudflare_workers_ai", "@cf/deepseek-ai/deepseek-v4-pro-0813"),
            _row("groq", "openai/gpt-oss-120b"),
            _row("sambanova", "gpt-oss-120b"),
            _row("cloudflare_workers_ai", "@cf/openai/gpt-oss-120b"),
            _row("kilo_gateway", "stepfun/step-3.7-flash:free"),
        ],
    }
    proposal = scan.propose(snapshot, identity, "2026-09-12")
    confirmed = proposal["confirmed"]
    assert "kimi-k3" in confirmed
    assert "deepseek-v4-pro" in confirmed
    assert set(confirmed["deepseek-v4-pro"]["listings"]) == {
        "nvidia_nim:deepseek-ai/deepseek-v4-pro-0813",
        "cloudflare_workers_ai:@cf/deepseek-ai/deepseek-v4-pro-0813",
    }
    assert set(confirmed["gpt-oss-120b"]["listings"]) == {
        "groq:openai/gpt-oss-120b",
        "sambanova:gpt-oss-120b",
        "cloudflare_workers_ai:@cf/openai/gpt-oss-120b",
    }
    assert set(confirmed["step-3.7-flash"]["listings"]) == {
        "kilo_gateway:stepfun/step-3.7-flash:free"
    }
    # The dated Tier-1 group must not survive beside the curated merge.
    assert "deepseek-v4-pro-0813" not in proposal["proposed"]


def test_admitted_identities_use_the_canonical_slug_for_every_channel() -> None:
    identity = scan.load_identity()
    snapshot = {
        "scan_as_of": "2026-09-12",
        "listings": [
            _row("nvidia_nim", "deepseek-ai/deepseek-v4-pro-0813"),
            _row("cloudflare_workers_ai", "@cf/deepseek-ai/deepseek-v4-pro-0813"),
        ],
    }
    proposal = scan.propose(snapshot, identity, "2026-09-12")
    admitted = scan.admitted_identities(snapshot, proposal, identity)
    assert set(admitted) == {"deepseek-v4-pro"}
    assert (("nvidia_nim", "deepseek-ai/deepseek-v4-pro-0813")) in admitted["deepseek-v4-pro"]


def test_a_rescan_and_apply_preserves_the_canonicalization() -> None:
    # The durable path: derive the identity from the curated map, stamp it via merge_overlay,
    # and confirm a second apply leaves it canonical (idempotent) rather than reverting to the
    # channel id or the dated Tier-1 slug.
    identity = scan.load_identity()
    snapshot = {
        "scan_as_of": "2026-09-12",
        "listings": [_row("nvidia_nim", "deepseek-ai/deepseek-v4-pro-0813")],
    }
    proposal = scan.propose(snapshot, identity, "2026-09-12")
    admitted = scan.admitted_identities(snapshot, proposal, identity)
    overlay: dict[str, Any] = {
        "providers": {"nvidia_nim": {}},
        "models": {
            "nvidia-deepseek-v4-pro-0813": {
                "provider": "nvidia_nim",
                "model_id": "deepseek-ai/deepseek-v4-pro-0813",
                "version": "deepseek-v4-pro-0813",
            }
        },
    }
    scan.merge_overlay(overlay, admitted, scan_as_of="2026-09-12")
    assert overlay["models"]["nvidia-deepseek-v4-pro-0813"]["version"] == "deepseek-v4-pro"
    first = yaml.safe_dump(overlay, sort_keys=False)
    scan.merge_overlay(overlay, admitted, scan_as_of="2026-09-12")
    assert yaml.safe_dump(overlay, sort_keys=False) == first


# ── a distinct pair is NOT merged ──────────────────────────────────────────────────────


def test_a_dated_flash_snapshot_is_not_merged_into_the_unpinned_flash() -> None:
    # `deepseek-v4-flash-0731` is a DIFFERENT product from `deepseek-v4-flash` (the price
    # channel map says so). The curated map must leave them distinct, never fuzzy-join.
    identity = scan.load_identity()
    snapshot = {
        "scan_as_of": "2026-09-12",
        "listings": [
            _row("explabs", "deepseek-v4-flash"),
            _row("nvidia_nim", "deepseek-ai/deepseek-v4-flash-0731"),
        ],
    }
    proposal = scan.propose(snapshot, identity, "2026-09-12")
    admitted = scan.admitted_identities(snapshot, proposal, identity)
    assert "deepseek-v4-flash" in admitted
    assert "deepseek-v4-flash-0731" in admitted
    assert set(admitted) == {"deepseek-v4-flash", "deepseek-v4-flash-0731"}


def test_the_two_gpt_oss_sizes_are_not_cross_joined() -> None:
    identity = scan.load_identity()
    snapshot = {
        "scan_as_of": "2026-09-12",
        "listings": [
            _row("groq", "openai/gpt-oss-120b"),
            _row("groq", "openai/gpt-oss-20b"),
        ],
    }
    proposal = scan.propose(snapshot, identity, "2026-09-12")
    assert set(proposal["confirmed"]) == {"gpt-oss-120b", "gpt-oss-20b"}


# ── the archive mechanism ──────────────────────────────────────────────────────────────


def _engine() -> Any:
    from benchmark.routing import collection_priority as cp

    return cp.CollectionPriority(
        channels={"p": cp.Channel("p", "p")},
        covered={},
        verified_total=10,
        base_importance={"p": 1.0},
        multimodal_eligible=lambda _model: False,
    )


def _archived_overlay() -> dict[str, Any]:
    return {
        "providers": {"p": {}},
        "models": {
            "p-live": {"provider": "p", "model_id": "live", "version": "live"},
            "p-dead": {
                "provider": "p",
                "model_id": "dead",
                "version": "dead",
                "archived": True,
                "archived_at": "2026-09-12",
                "archive_reason": "provider left the model",
            },
        },
    }


def test_archived_row_is_excluded_from_the_runnable_set_but_retained() -> None:
    overlay = _archived_overlay()
    snapshot = {
        "listings": [_row("p", "live"), _row("p", "dead")],
        "channels": {"p": {"status": "ok (2 free)"}},
    }
    lanes = refresh.runnable_lanes(overlay, snapshot, engine=_engine(), declared_free={"p": None})
    assert [lane.name for lane in lanes] == ["p-live"]
    assert refresh.lane_exclusions(overlay, snapshot, declared_free={"p": None}) == {
        "p-dead": refresh.ARCHIVED
    }
    # Retained for provenance and the corpus: the exclusion is scheduling-only.
    assert "p-dead" in overlay["models"]


def test_merge_overlay_never_clears_an_archive() -> None:
    overlay = _archived_overlay()
    scan.merge_overlay(overlay, {"canonical": [("p", "dead")]})
    row = overlay["models"]["p-dead"]
    assert row["version"] == "canonical"
    assert row["archived"] is True
    assert row["archived_at"] == "2026-09-12"
    assert row["archive_reason"] == "provider left the model"


def test_archive_entry_stamps_one_provider_scoped_row() -> None:
    overlay: dict[str, Any] = {
        "models": {
            "a": {"provider": "vendor", "model_id": "m", "version": "m"},
            "b": {"provider": "reseller", "model_id": "m", "version": "m"},
        }
    }
    row = scan.archive_entry(
        overlay, "vendor", "m", reason="vendor retired it", archived_at="2026-09-12"
    )
    assert row is overlay["models"]["a"]
    assert row["archived"] is True
    # The same weights on a provider that still serves them are untouched.
    assert "archived" not in overlay["models"]["b"]
    assert scan.is_archived(overlay["models"]["a"]) is True


def test_archive_entry_requires_provenance_and_a_real_row() -> None:
    overlay: dict[str, Any] = {"models": {"a": {"provider": "vendor", "model_id": "m"}}}
    with pytest.raises(ValueError, match="archive_reason"):
        scan.archive_entry(overlay, "vendor", "m", reason="", archived_at="2026-09-12")
    with pytest.raises(ValueError, match="archived_at"):
        scan.archive_entry(overlay, "vendor", "m", reason="nope", archived_at="")
    with pytest.raises(KeyError):
        scan.archive_entry(overlay, "vendor", "missing", reason="nope", archived_at="2026-09-12")


def test_committed_overlay_row_still_loads_through_the_registry_parser() -> None:
    # The archive fields are additive on the shared schema; an unarchived overlay must parse
    # exactly as before (the rescan/apply validation path calls this).
    from shunt.models.config import parse_registry

    overlay = yaml.safe_load(OVERLAY.read_text())
    registry = parse_registry(overlay)
    assert registry.models["nvidia-deepseek-v4-pro-0813"].version == "deepseek-v4-pro"
    assert registry.models["nvidia-deepseek-v4-pro-0813"].archived is False


# ── committed-overlay canonical invariants ─────────────────────────────────────────────


def test_no_overlay_version_encodes_a_free_marker() -> None:
    # Regression (2026-09-12): the Vercel ling rows carried `-free` in their
    # `version` because Tier-1 only strips `:free`; both were committed for weeks. The
    # canonical rule bans the promo marker in the identity.
    for name, version in _overlay_versions().items():
        assert "-free" not in version, (name, version)
        assert ":free" not in version, (name, version)


# The residual raw listing ids the free-overlay audit found used verbatim as `version`
# (2026-09-12): a SambaNova/UPPER id, a Cloudflare serving-prefixed `@cf/...` id, or a
# Tier-1 slug whose publisher-issued form differs. Each now canonicalises through the
# curated `model_identity.yaml`.
_RAW_RESIDUALS: tuple[tuple[str, str], ...] = (
    ("Meta-Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct"),
    ("MiniMax-M2.7", "minimax-m2.7"),
    ("DeepSeek-V3.1", "deepseek-v3.1"),
    ("DeepSeek-V3.2", "deepseek-v3.2"),
    ("MiniMax-M3", "minimax-m3"),
    ("nemotron-3-super-120b-a12b", "nemotron-3-super-120b-a12b"),
    ("qwen3.8-27b", "qwen3.8-27b"),
    ("qwen3-8-27b", "qwen3.8-27b"),
    ("@cf/meta/llama-3.2-1b-instruct", "llama-3.2-1b-instruct"),
    ("@cf/zai-org/glm-5.2", "glm-5.2"),
    ("@cf/qwen/qwen3-30b-a3b-fp8", "qwen3-30b-a3b-fp8"),
    ("@cf/ibm-granite/granite-4.0-h-micro", "granite-4.0-h-micro"),
)


def test_no_overlay_version_is_a_raw_provider_listing_id() -> None:
    # A `version` is a weights identity: it never carries the Cloudflare serving prefix or a
    # provider's shouting-case listing id.
    for name, version in _overlay_versions().items():
        assert not version.startswith("@cf/"), (name, version)
        assert version == version.lower(), (name, version)


def test_version_aliases_carry_every_raw_residual() -> None:
    identity = scan.load_identity()
    for raw, canonical in _RAW_RESIDUALS:
        assert identity.version_aliases.get(raw) == canonical, (raw, canonical)


def test_raw_residual_listing_ids_resolve_to_their_canonical_identity() -> None:
    from benchmark.routing import model_universe

    for raw, canonical in _RAW_RESIDUALS:
        if raw.startswith("@cf/") or raw[:1].isupper():
            assert model_universe.resolve_identity(raw) == canonical, (raw, canonical)


def test_retired_zai_serving_slot_resolves_to_the_bare_glm_identity() -> None:
    # `zai-` is a retired serving slot, never part of the weights identity. `canonical_label`
    # already stripped it; `resolve_identity` returned the raw label, so a caller that
    # membership-tests the resolver's output against the bare universe forked from the display
    # path. Both must reduce to the bare `glm-*` slug.
    from benchmark.routing import model_universe

    for raw, canonical in (("zai-glm-5.2", "glm-5.2"), ("zai-glm-5.3-flash", "glm-5.3-flash")):
        assert model_universe.resolve_identity(raw) == canonical, raw
        assert model_universe.canonical_label(raw) == canonical, raw


def test_committed_model_priority_keys_all_name_a_live_identity() -> None:
    # Every allowlist key must name an identity the collection engine can see — an overlay
    # `version`, a shipped model, or a committed corpus `model_version` — or the value is
    # dead and silently falls to `default_importance`.
    import csv

    priority = yaml.safe_load(
        (OVERLAY.parents[2] / "benchmark" / "routing" / "data" / "model_priority.yaml").read_text()
    )
    overlay = yaml.safe_load(OVERLAY.read_text())["models"]
    known = {str(row.get("version")) for row in overlay.values()}
    shipped = yaml.safe_load(
        (OVERLAY.parents[2] / "src" / "shunt" / "config" / "models.yaml").read_text()
    )["models"]
    known |= set(shipped)
    for name in ("results.csv", "results_free.csv"):
        path = OVERLAY.parents[2] / "benchmark" / "routing" / name
        if path.exists():
            with path.open(newline="") as handle:
                known |= {row["model_version"] for row in csv.DictReader(handle)}
    identity = scan.load_identity()
    known |= set(identity.version_aliases) | set(identity.version_aliases.values())
    keys = set(priority.get("importance") or {}) | set(priority.get("frontier") or [])
    assert keys <= known, sorted(keys - known)


def test_concordance_subset_channels_carry_the_declared_identity() -> None:
    # The named measurement pairs rows on `model_version`; the subset's declared `identity`
    # must therefore equal each channel's committed overlay `version`. gemma-4-31b-it fell
    # to the Tier-1 slug while the subset declared `google/gemma-4-31b-it`, so the declared
    # identity never matched the registry until the curated entry was added.
    cfg_path = OVERLAY.parents[2] / "configs" / "free-tier" / "benchmark.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    versions = _overlay_versions()
    for entry in cfg["concordance"]["subset"]:
        for channel in entry["channels"]:
            assert versions[channel] == entry["identity"], (
                channel,
                versions[channel],
                entry["identity"],
            )


def test_collected_lane_versions_agree_with_the_committed_overlay() -> None:
    # The resume/dedupe key is `(challenge_id, lane, ...)`, where `lane` is the overlay row's
    # KEY (the channel); identity grouping keys on `model_version`. If the overlay `version`
    # for a lane diverges from the already-collected corpus value, the next run re-splits the
    # identity and orphans the collected cells. The channel column is `lane`; `model` is the
    # bare identity and does not name an overlay row.
    import csv

    versions = _overlay_versions()
    results = OVERLAY.parents[2] / "benchmark" / "routing" / "results_free.csv"
    asserted = 0
    with results.open(newline="") as handle:
        for row in csv.DictReader(handle):
            lane = row["lane"]
            if lane in versions:
                asserted += 1
                assert versions[lane] == row["model_version"], (
                    lane,
                    versions[lane],
                    row["model_version"],
                )
    assert asserted > 0, "no collected lane matched an overlay row"
