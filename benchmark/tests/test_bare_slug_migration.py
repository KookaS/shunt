"""The bare-slug canonical id convention: overlay/identity/CSV migration, idempotent.

The convention is defined once in `scan_free_models.canonical_slug`; this module pins that the
committed surfaces (overlay, identity map, priority, results CSVs) actually carry its output —
and that re-running the migration is a byte-stable fixpoint, so a drift re-introduced by a
writer is caught here rather than shipping.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml

from benchmark.routing.scripts import migrate_bare_slugs as migrate
from benchmark.routing.scripts.scan_free_models import canonical_slug

OVERLAY: Path = migrate.OVERLAY_PATH
IDENTITY: Path = migrate.IDENTITY_PATH
PRIORITY: Path = migrate.PRIORITY_PATH
CSVS: tuple[Path, ...] = migrate.CSV_PATHS

# "requests" the convention bans from an identity field: a provider prefix, a promo marker, a
# serving prefix, or any uppercase. `lane` is allowed to carry them (it is the wire id).
_BANNED = ("@cf/", "-free", ":free", "requesty-", "kilo-", "openrouter-", "nvidia-", ":")


def test_committed_surfaces_report_no_pending_migration() -> None:
    assert migrate.report() == {}


def test_every_overlay_id_field_is_a_bare_slug_while_lane_keeps_the_wire_id() -> None:
    overlay = yaml.safe_load(OVERLAY.read_text())
    for name, row in overlay["models"].items():
        for field in ("model", "model_id", "version"):
            value = str(row[field])
            assert value == canonical_slug(value), (name, field, value)
            assert value == value.lower(), (name, field, value)
            assert not any(token in value for token in _BANNED), (name, field, value)
            assert "/" not in value and "@" not in value, (name, field, value)
        # `lane` is the raw provider listing: the wire call still needs it.
        assert str(row.get("lane") or ""), name


def test_every_identity_key_is_a_bare_slug() -> None:
    payload = yaml.safe_load(IDENTITY.read_text())
    for key in payload["models"]:
        assert key == canonical_slug(key), key
        assert key == str(key).lower(), key
        assert not re.search(r"[/:@]", key), key


def test_priority_allowlist_keys_are_bare() -> None:
    payload = yaml.safe_load(PRIORITY.read_text())
    for key in payload.get("importance") or {}:
        assert key == canonical_slug(key), key
    for item in payload.get("frontier") or []:
        assert item == canonical_slug(item), item


def test_csvs_carry_lane_and_one_bare_identity_per_row() -> None:
    for path in CSVS:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert "lane" in (reader.fieldnames or []), path
            for row in reader:
                lane, model, version = row["lane"], row["model"], row["model_version"]
                assert lane, (path, row["challenge_id"])
                assert model == canonical_slug(model), (path, model)
                assert version == canonical_slug(version), (path, version)
                assert model == version, (path, model, version)


def test_the_four_audited_duplicate_groups_now_share_one_identity() -> None:
    overlay = yaml.safe_load(OVERLAY.read_text())["models"]
    assert (
        overlay["nvidia-nemotron-3-ultra-550b-a55b"]["version"]
        == (overlay["requesty-nemotron-3-ultra-550b-a55b"]["version"])
    )
    assert (
        overlay["requesty-muse-glimmer-30b"]["version"]
        == (overlay["together-muse-glimmer-30b"]["version"])
    )
    assert overlay["requesty-laguna-xs.2"]["version"] == "laguna-xs-2.1"
    assert (
        overlay["requesty-nemotron-3.5-lightning-30b-a3b"]["version"]
        == (overlay["kilo-nemotron-3.5-lightning-free"]["version"])
    )


def test_overlay_migration_is_a_fixpoint() -> None:
    overlay = yaml.safe_load(OVERLAY.read_text())
    once = migrate.migrate_overlay(overlay)
    assert migrate.migrate_overlay(once) == once


def test_identity_migration_is_a_fixpoint() -> None:
    payload = yaml.safe_load(IDENTITY.read_text())
    once = migrate.migrate_identity(payload)
    assert migrate.migrate_identity(once) == once


def test_csv_migration_is_a_fixpoint() -> None:
    fields, rows = migrate._migrated_csv_rows(migrate.read_csv(CSVS[0]))
    fields2, rows2 = migrate._migrated_csv_rows(rows)
    assert fields2 == fields
    assert rows2 == rows


def test_bare_slot_drops_the_serving_prefix_but_never_a_wire_id() -> None:
    assert migrate.bare_slot("zai-glm-5.2") == "glm-5.2"
    assert migrate.bare_slot("zai-glm-5.3-flash") == "glm-5.3-flash"
    assert migrate.bare_slot("glm-5.2") == "glm-5.2"  # fixpoint on its own output
    # Real provider namespaces/wire ids and unqualified publisher prefixes are untouched.
    for wire in ("@cf/zai-org/glm-5.2", "zai/glm-5.2", "z-ai/glm-5.2", "zai-org/GLM-5.2"):
        assert migrate.bare_slot(wire) == wire, wire


def test_csv_lane_migration_rewrites_the_serving_slot_and_is_idempotent() -> None:
    rows = [
        {"challenge_id": "t", "model": "glm-5.2", "lane": "zai-glm-5.2", "model_version": "glm-5.2"}
    ]
    fields, once = migrate._migrated_csv_rows(rows)
    assert once[0]["lane"] == "glm-5.2"
    assert once[0]["model"] == "glm-5.2"
    fields2, twice = migrate._migrated_csv_rows(once)
    assert fields2 == fields
    assert twice == once


def test_bare_slot_text_rewrites_a_trajectory_id_without_touching_wire_ids() -> None:
    line = '{"trajectory_id": "t__zai-glm-5.3-flash__default", "id": "z-ai/glm-5.3-flash"}'
    out = migrate.bare_slot_text(line)
    assert "zai-glm" not in out
    assert '"t__glm-5.3-flash__default"' in out
    assert '"z-ai/glm-5.3-flash"' in out  # the wire id is untouched
    assert migrate.bare_slot_text(out) == out  # idempotent


def test_committed_escalation_corpus_and_price_sheet_carry_no_legacy_slot() -> None:
    assert migrate.escalation_pending() == []
