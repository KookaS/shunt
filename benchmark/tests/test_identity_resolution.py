"""Identity resolution: Tier 2 curated, Tier 0 publisher-issued, Tier 1 proposed-only, deny."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.routing.scripts import scan_free_models as scan


def _row(provider: str, listing_id: str, hugging_face_id: str = "", **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider": provider,
        "listing_id": listing_id,
        "hugging_face_id": hugging_face_id,
        "prices": None,
        "canonical_slug": "",
        "context_length": None,
        "supports_tools": False,
        "expiration_date": None,
        "last_seen": "2026-09-10",
        "withdrawn_at": None,
    }
    row.update(over)
    return row


def _glm() -> scan.IdentityMap:
    return scan.IdentityMap(
        entries={
            "glm-5.3": scan.IdentityEntry(
                slug="glm-5.3",
                hugging_face_id="zai-org/GLM-5.3",
                aliases={"groq": ("glm-5.3",)},
                deny=frozenset({"z-ai/glm-5.3-0731"}),
            )
        },
        deny=frozenset({"z-ai/glm-5.3-0731"}),
    )


# ── Tier 0 ────────────────────────────────────────────────────────────────────────────


def test_tier0_joins_only_on_non_empty_shared_hugging_face_id() -> None:
    rows = [
        _row("openrouter", "a", "org/model"),
        _row("requesty", "b", "org/model"),
        _row("x", "c", ""),
        _row("y", "d", ""),
        _row("z", "solo", "solo/only"),
    ]
    joins = scan.tier0_joins(rows)
    assert len(joins) == 1
    assert joins[0].identity == "model"
    assert joins[0].members == (("openrouter", "a"), ("requesty", "b"))
    assert "org/model" in joins[0].evidence


def test_tier0_proposes_with_evidence_and_a_content_hash() -> None:
    snapshot = {
        "listings": [
            _row("openrouter", "nex:free", "nex-agi/Nex-N2.5-mini"),
            _row("requesty", "nvidia/nex", "nex-agi/Nex-N2.5-mini"),
        ]
    }
    proposal = scan.propose(snapshot, scan.IdentityMap(), "2026-09-10")
    entry = proposal["proposed"]["nex-n2.5-mini"]
    assert entry["tier"] == 0
    assert "hugging_face_id=nex-agi/Nex-N2.5-mini" in entry["evidence"]
    assert entry["content_hash"]


# ── Tier 1 ────────────────────────────────────────────────────────────────────────────


def test_tier1_normalisation_drops_prefix_and_declared_suffixes() -> None:
    assert scan.tier1_normalise("z-ai/glm-5.3:free") == "glm-5.3"
    assert scan.tier1_normalise("poolside/laguna-xs-2.1:free") == "laguna-xs-2.1"
    assert scan.tier1_normalise("google/gemma-4-31b-it") == "gemma-4-31b-it"
    assert scan.tier1_normalise("org/Model_Chat") == "model-chat"


def test_tier1_candidates_are_proposed_but_never_confirmed() -> None:
    snapshot = {
        "listings": [
            _row("openrouter", "z-ai/glm-5.3:free"),
            _row("groq", "glm-5.3"),
        ]
    }
    proposal = scan.propose(snapshot, scan.IdentityMap(), "2026-09-10")
    assert proposal["confirmed"] == {}
    assert proposal["proposed"]["glm-5.3"]["tier"] == 1
    assert all(entry["tier"] != 1 for entry in proposal["confirmed"].values())


def test_a_lone_listing_is_never_proposed() -> None:
    proposal = scan.propose(
        {"listings": [_row("groq", "one-off")]}, scan.IdentityMap(), "2026-09-10"
    )
    assert proposal["proposed"] == {}


# ── Tier 2 + deny ─────────────────────────────────────────────────────────────────────


def test_tier2_alias_confirms_a_listing() -> None:
    proposal = scan.propose({"listings": [_row("groq", "glm-5.3")]}, _glm(), "2026-09-10")
    assert proposal["confirmed"]["glm-5.3"]["tier"] == 2
    assert proposal["confirmed"]["glm-5.3"]["listings"] == ["groq:glm-5.3"]


def test_a_deny_entry_is_never_reproposed() -> None:
    snapshot = {
        "listings": [
            _row("openrouter", "z-ai/glm-5.3-vision-exp"),
            _row("openrouter", "z-ai/glm-5.3:free"),
            _row("groq", "glm-5.3"),
        ]
    }
    identity = scan.IdentityMap(deny=frozenset({"z-ai/glm-5.3-vision-exp"}))
    proposal = scan.propose(snapshot, identity, "2026-09-10")
    assert "z-ai/glm-5.3-vision-exp" not in json.dumps(proposal["proposed"])
    assert any(
        item["listing_id"] == "z-ai/glm-5.3-vision-exp" and item["reason"] == "deny"
        for item in proposal["dropped"]
    )


def test_committed_identity_file_loads_and_denies_near_namesakes() -> None:
    identity = scan.load_identity()
    assert "deepseek-v4-flash" in identity.entries
    assert identity.is_denied("z-ai/glm-5.3-0731")


# ── apply guard ───────────────────────────────────────────────────────────────────────


def test_apply_refuses_an_entry_absent_from_the_proposal() -> None:
    with pytest.raises(scan.ApplyRefusedError, match="absent"):
        scan.apply_proposal({"confirmed": {}, "proposed": {}}, {"new": "abc"})


def test_apply_refuses_a_hash_moved_entry() -> None:
    proposal = {"confirmed": {"m": {"content_hash": "old", "listings": ["p:m"]}}, "proposed": {}}
    with pytest.raises(scan.ApplyRefusedError, match="hash moved"):
        scan.apply_proposal(proposal, {"m": "new"})


def test_apply_returns_confirmed_only_when_hashes_match() -> None:
    proposal = {
        "confirmed": {"m": {"content_hash": "h", "listings": ["p:m"]}},
        "proposed": {"n": {"content_hash": "h2", "listings": ["q:n"]}},
    }
    assert scan.apply_proposal(proposal, {"m": "h", "n": "h2"}) == {"m": [("p", "m")]}


def test_main_apply_refusal_returns_the_documented_code_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CONFIRMED-BUG 2: an ApplyRefusedError must be a clean rc-2 refusal, not a traceback.

    Drives `main(["--apply"])` through the real guard: an entry absent from the reviewed
    proposal is refused, the documented `refusing --apply:` message lands on stderr, the
    process would exit 2, and the overlay is never written.
    """
    listing = {
        "prices": None,
        "canonical_slug": "",
        "context_length": None,
        "supports_tools": False,
        "expiration_date": None,
        "last_seen": "2026-09-10",
        "withdrawn_at": None,
    }
    snapshot = {
        "scan_as_of": "2026-09-10",
        "shape_regressions": [],
        "channels": {},
        "listings": [
            dict(listing, provider="openrouter", listing_id="a", hugging_face_id="org/m"),
            dict(listing, provider="groq", listing_id="b", hugging_face_id="org/m"),
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    proposal_path = tmp_path / "proposal.yaml"
    proposal_path.write_text(yaml.safe_dump({"confirmed": {}, "proposed": {}}))
    overlay_path = tmp_path / "overlay.yaml"
    original = "models: {}\n"
    overlay_path.write_text(original)
    monkeypatch.setattr(scan, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(scan, "PROPOSAL_PATH", proposal_path)
    monkeypatch.setattr(scan, "OVERLAY_PATH", overlay_path)

    code = scan.main(["--apply"])

    captured = capsys.readouterr()
    assert code == 2
    assert "refusing --apply" in captured.err
    assert "Traceback" not in captured.err
    assert overlay_path.read_text() == original


def test_content_hash_moves_when_a_fact_changes() -> None:
    before = scan.propose({"listings": [_row("groq", "glm-5.3")]}, _glm(), "2026-09-10")
    changed = _row("groq", "glm-5.3", context_length=999)
    after = scan.propose({"listings": [changed]}, _glm(), "2026-09-10")
    assert (
        before["confirmed"]["glm-5.3"]["content_hash"]
        != after["confirmed"]["glm-5.3"]["content_hash"]
    )


def test_merge_overlay_sets_version_only() -> None:
    overlay = {
        "providers": {"groq": {"base_url": "https://api.groq.com/openai/v1"}},
        "models": {"groq-glm": {"provider": "groq", "model_id": "glm-5.3", "version": "old"}},
    }
    scan.merge_overlay(overlay, {"glm-5.3": [("groq", "glm-5.3")]})
    assert overlay["models"]["groq-glm"]["version"] == "glm-5.3"
    assert overlay["providers"]["groq"]["base_url"] == "https://api.groq.com/openai/v1"


def test_merge_overlay_adds_a_confirmed_row_with_a_real_price_and_provenance() -> None:
    overlay: dict[str, Any] = {"providers": {"openrouter": {}}, "models": {}}
    pair = ("openrouter", "google/gemma-4-31b-it:free")
    rows = {pair: {"cost": {"input": 0.09, "output": 0.34}}}
    scan.merge_overlay(overlay, {"gemma-4-31b-it": [pair]}, rows=rows, scan_as_of="2026-09-10")
    row = overlay["models"]["openrouter-gemma-4-31b-it"]
    assert row["version"] == "gemma-4-31b-it"
    assert row["model_id"] == "gemma-4-31b-it"
    assert row["lane"] == pair[1]
    pricing = row["pricing"]
    assert pricing["input_cost_per_1m"] == 0.09
    assert pricing["output_cost_per_1m"] == 0.34
    assert "cache_read_cost_per_1m" not in pricing  # HARD RULE 3
    assert pricing["price_source"] == "https://models.dev/api.json"
    assert pricing["price_as_of"] == "2026-09-10"


def test_merge_overlay_refuses_to_add_a_zero_cost_row() -> None:
    overlay: dict[str, Any] = {"providers": {"openrouter": {}}, "models": {}}
    pair = ("openrouter", "free-but-unpriced")
    rows = {pair: {"cost": {"input": 0.0, "output": 0.0}}}
    scan.merge_overlay(overlay, {"m": [pair]}, rows=rows, scan_as_of="2026-09-10")
    assert overlay["models"] == {}


def test_merge_overlay_never_writes_a_zero_price_from_the_prices_argument() -> None:
    """CONFIRMED-BUG 1: a `$0` `prices` argument cannot bypass HARD RULE 2.

    The helper's whole job is to enforce the invariant, so a caller price map is validated
    exactly like a snapshot `cost`: the zero entry is refused and the real paid-twin list
    price is used instead, so no row can carry `input_cost_per_1m`/`output_cost_per_1m` of 0.
    """
    overlay: dict[str, Any] = {"providers": {"openrouter": {}}, "models": {}}
    pair = ("openrouter", "nex-agi/nex-n2.5-mini:free")
    rows = {pair: {"cost": {"input": 0.09, "output": 0.34}}}
    scan.merge_overlay(
        overlay,
        {"nex-n2.5-mini": [pair]},
        rows=rows,
        prices={pair: {"input": 0.0, "output": 0.0}},
        scan_as_of="2026-09-10",
    )
    assert overlay["models"]
    for row in overlay["models"].values():
        assert row["pricing"]["input_cost_per_1m"] > 0
        assert row["pricing"]["output_cost_per_1m"] > 0


def test_merge_overlay_refuses_a_priced_row_with_blank_provenance() -> None:
    """A direct call omitting `scan_as_of` must fail, not write `price_as_of: ''`."""
    overlay: dict[str, Any] = {"providers": {"openrouter": {}}, "models": {}}
    pair = ("openrouter", "google/gemma-4-31b-it:free")
    rows = {pair: {"cost": {"input": 0.09, "output": 0.34}}}
    with pytest.raises(ValueError, match="price_as_of"):
        scan.merge_overlay(overlay, {"gemma-4-31b-it": [pair]}, rows=rows)
    assert overlay["models"] == {}


def test_merge_overlay_never_adds_a_row_without_an_active_snapshot_row() -> None:
    overlay: dict[str, Any] = {"providers": {"openrouter": {}}, "models": {}}
    pair = ("openrouter", "withdrawn")
    scan.merge_overlay(
        overlay,
        {"m": [pair]},
        rows={},
        prices={pair: {"input": 1.0, "output": 2.0}},
        scan_as_of="2026-09-10",
    )
    assert overlay["models"] == {}


def test_withdrawn_rows_reports_only_successfully_scanned_channels_and_keeps_the_row() -> None:
    """A vanished listing is MARKED (snapshot `withdrawn_at`), never deleted from the overlay.

    The overlay keeps the row's price/identity provenance; the campaign's runnable set is
    built from the active snapshot rows, so the lane quiesces without losing history. A channel
    that was UNREACHABLE proves nothing, so its row is neither reported nor removed.
    """
    overlay: dict[str, Any] = {
        "providers": {"openrouter": {}, "groq": {}},
        "models": {
            "openrouter-gone": {"provider": "openrouter", "model_id": "gone"},
            "groq-gone": {"provider": "groq", "model_id": "gone"},
        },
    }
    snapshot = {
        "channels": {
            "openrouter": {"status": "ok (1 free)"},
            "groq": {"status": "UNREACHABLE: HTTPError: HTTP Error 401"},
        }
    }
    withdrawn = scan.withdrawn_rows(overlay, snapshot, {})
    assert withdrawn == ["openrouter-gone"]
    # Retained, not deleted: a re-listing resumes with its provenance intact.
    assert set(overlay["models"]) == {"openrouter-gone", "groq-gone"}


def test_laguna_channels_resolve_to_one_identity_through_apply() -> None:
    """All three laguna-s channels canonicalise to ONE version through propose+apply.

    Regression: Vercel's channel id carried a `-free` suffix and was stamped
    `poolside/laguna-s-2.1-free` while OpenRouter/Kilo used `laguna-s-2-1`, so the promised
    3-provider laguna comparison yielded 7 provider pairs instead of 9. A Tier-2 alias in
    `model_identity.yaml` makes the apply path re-derive the canonical slug, so a later
    `refresh_free_campaign --write` cannot reintroduce the split.
    """
    identity = scan.load_identity()
    pairs = [
        ("openrouter", "poolside/laguna-s-2.1:free"),
        ("kilo_gateway", "poolside/laguna-s-2.1:free"),
        ("vercel_ai_gateway", "poolside/laguna-s-2.1-free"),
    ]
    rows = [_row(provider, listing_id, supports_tools=True) for provider, listing_id in pairs]
    snapshot = {"scan_as_of": "2026-09-11", "listings": rows}
    proposal = scan.propose(snapshot, identity, "2026-09-11")
    entry = proposal["confirmed"]["laguna-s-2.1"]
    assert set(entry["listings"]) == {f"{provider}:{listing_id}" for provider, listing_id in pairs}
    admitted = scan.admitted_identities(snapshot, proposal, identity)
    assert set(admitted) == {"laguna-s-2.1"}

    overlay = yaml.safe_load(scan.OVERLAY_PATH.read_text())
    row_map = {
        (provider, listing_id): row for (provider, listing_id), row in zip(pairs, rows, strict=True)
    }
    scan.merge_overlay(overlay, admitted, rows=row_map, scan_as_of="2026-09-11")
    for name in (
        "openrouter-laguna-s-2.1-free",
        "kilo-laguna-s-2.1-free",
        "vercel-laguna-s-2.1-free",
    ):
        assert overlay["models"][name]["version"] == "laguna-s-2.1"


def test_committed_curated_aliases_carry_the_canonical_version() -> None:
    """Every curated alias present in the committed overlay carries its entry's slug.

    The same identity-split check as laguna, applied to every Tier-2 entry: a future alias
    added to `model_identity.yaml` whose committed overlay row carries a different `version`
    fails here before it can silently split a provider group.
    """
    identity = scan.load_identity()
    overlay = yaml.safe_load(scan.OVERLAY_PATH.read_text())
    versions = {
        (str(row["provider"]), str(row["model_id"])): str(row["version"])
        for row in overlay["models"].values()
    }
    for slug, entry in identity.entries.items():
        for provider, ids in entry.aliases.items():
            for listing_id in ids:
                resolved = versions.get((provider, listing_id))
                if resolved is not None:
                    assert resolved == slug, (slug, provider, listing_id, resolved)


def test_withdrawn_rows_excludes_a_row_still_present() -> None:
    overlay: dict[str, Any] = {
        "models": {"openrouter-live": {"provider": "openrouter", "model_id": "live"}}
    }
    snapshot = {"channels": {"openrouter": {"status": "ok (1 free)"}}}
    withdrawn = scan.withdrawn_rows(overlay, snapshot, {("openrouter", "live"): {}})
    assert withdrawn == []
    assert "openrouter-live" in overlay["models"]
