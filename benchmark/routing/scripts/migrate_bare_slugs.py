"""One-time, idempotent migration to the bare-slug canonical model-id convention.

    uv run python -m benchmark.routing.scripts.migrate_bare_slugs --check   # report, write nothing
    uv run python -m benchmark.routing.scripts.migrate_bare_slugs --write    # apply in place

WHAT CHANGES. `scan_free_models.canonical_slug` collapses every provider/channel/promo spelling
onto ONE bare slug, and the retired `zai-` serving-slot prefix is dropped from model NAMES
(never from a provider wire id). This moves the committed surfaces that predate it:

* `configs/free-tier/overlay.yaml` — adds `lane` (the raw channel/wire id); `model`/`model_id`/
  `version` become the bare slug.
* `model_identity.yaml` / `model_priority.yaml` — bare keys.
* `results.csv` / `results_free.csv` — `model`/`model_version` bare, and a retired serving-slot
  spelling in `lane` becomes the bare `glm-*`.
* `artifacts/results_history.csv` (gitignored), the tracked escalation corpus
  (`data/{live,probe}/*.jsonl`, whose `trajectory_id` is the cost-join key into results.csv),
  its manifests and the routing price sheet.
* on-disk lane-state artifacts (`benchmark/runner/artifacts/**`).

IDEMPOTENT. Re-running writes byte-identical files; the test drives `--check` after `--write`
and asserts no pending change.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from benchmark.routing.scripts import scan_free_models as scan
from benchmark.routing.scripts.scan_free_models import canonical_slug

_REPO: Final[Path] = Path(__file__).resolve().parents[3]
OVERLAY_PATH: Final[Path] = _REPO / "configs" / "free-tier" / "overlay.yaml"
IDENTITY_PATH: Final[Path] = scan.IDENTITY_PATH
PRIORITY_PATH: Final[Path] = scan._DATA_DIR / "model_priority.yaml"
CSV_PATHS: Final[tuple[Path, ...]] = (
    scan._DATA_DIR.parent / "results.csv",
    scan._DATA_DIR.parent / "results_free.csv",
)
HISTORY_PATH: Final[Path] = scan._DATA_DIR.parent / "artifacts" / "results_history.csv"
ARTIFACTS_DIR: Final[Path] = _REPO / "benchmark" / "runner" / "artifacts"
_ESCALATION_ROOT: Final[Path] = _REPO / "benchmark" / "escalation"
ESCALATION_TRAJECTORY_DIRS: Final[tuple[Path, ...]] = (
    _ESCALATION_ROOT / "data" / "live",
    _ESCALATION_ROOT / "data" / "probe",
)
ESCALATION_META_PATHS: Final[tuple[Path, ...]] = (
    _ESCALATION_ROOT / "data" / "live" / "manifest.json",
    _ESCALATION_ROOT / "data" / "live" / "stamp_ledger.json",
    _ESCALATION_ROOT / "data" / "live" / "state_capture.json",
    _ESCALATION_ROOT / "data" / "probe" / "manifest.json",
    _ESCALATION_ROOT / "reports" / "metrics.json",
)
# The generated routing price sheet is keyed by the registry slot names (its VALUES keep the
# provider wire ids), so it carries the same legacy slot text as the escalation metadata.
PRICE_SHEET_PATH: Final[Path] = scan._DATA_DIR / "price_sheet.json"
# The tracked JSON surfaces whose model keys/ids carry the retired serving-slot text.
SLOT_JSON_PATHS: Final[tuple[Path, ...]] = (*ESCALATION_META_PATHS, PRICE_SHEET_PATH)

# The retired serving-slot prefix. It is dropped only when it precedes a `glm-` model name, so
# a provider wire id/namespace (`@cf/zai-org/glm-5.2`, `zai/glm-5.2`) is never touched. The
# regex holds a PREFIX, never a concrete slot name.
_LEGACY_SLOT_RE: Final[re.Pattern[str]] = re.compile(r"zai-(?=glm-)")


def bare_slot_text(text: str) -> str:
    """Drop the retired ``zai-`` serving-slot prefix before a ``glm-`` name (idempotent)."""
    return _LEGACY_SLOT_RE.sub("", text)


def bare_slot(name: str) -> str:
    """Bare registry/serving name for a model name (see :func:`bare_slot_text`)."""
    return bare_slot_text(str(name))


_IDENTITY_HEADER: Final[str] = """\
# CANONICAL MODEL IDENTITY — bare-slug keys, raw provider listing ids as aliases.
#
# THE CONVENTION (one true id per model). The identity is the model's bare OFFICIAL slug:
# lowercase, tokens joined by `-`, NO `/`, `:`, `@`, no provider prefix and no
# `-free`/`-explabs`/`:free`/`:nitro`/`:floor`/`:extended` promo suffix. The normalisation
# grammar is `scan_free_models.canonical_slug`:
#
#   <creator>[/-:]*<model>…  ->  strip `@cf/` + leading `publisher/` ->
#   strip promo suffix -> lowercase -> `_`/`:` -> `-` (dots PRESERVED) ->
#   CURATED override (the audited ambiguous spellings only).
#
# WHY. The same weights are served under near-name-sake ids on every channel
# (`@cf/zai-org/glm-5.3` / `glm-5.3` / `z-ai/glm-5.3:free` can be one model or three). This file is
# the curated answer: one bare key per model, every raw channel id that serves it listed as an
# alias. The resolved identity is the `version` slug the overlay/corpus rows carry.
#
# TIERS (see scan_free_models.py):
#   Tier 0  auto-proposed ONLY on a publisher-issued identifier (OpenRouter `hugging_face_id`
#           / `canonical_slug`); the proposed key is the bare slug of that identifier.
#   Tier 1  the mechanical `canonical_slug` candidate. PROPOSED only — it is the fuzzy match
#           this repo bans, so it has no write path.
#   Tier 2  the curated aliases below. `--propose` confirms them; `--apply` is the only path
#           that writes them into the overlay's `version` field.
#
# `version_aliases` maps a raw legacy version slug (a provider-qualified or promo-marked
# identity left in a committed corpus) onto its bare canonical. `deny` matches a raw listing id
# and is never proposed, at any tier. models.dev is NEVER a join source (list prices only).
"""


def _canon_unique(values: list[str]) -> list[str]:
    """Canonicalise a list of raw ids, dropping blanks and preserving first-seen order."""
    out: list[str] = []
    for value in values:
        slug = canonical_slug(str(value))
        if slug and slug not in out:
            out.append(slug)
    return out


# ── overlay ─────────────────────────────────────────────────────────────────────────


def migrate_overlay(overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return a migrated copy of the overlay: add `lane`, set the three id fields bare."""
    out: dict[str, Any] = {k: v for k, v in overlay.items() if k != "models"}
    models: dict[str, Any] = dict()
    for name, row in (overlay.get("models") or {}).items():
        if not isinstance(row, Mapping):
            models[str(name)] = row
            continue
        migrated = dict(row)
        raw = str(migrated.get("lane") or migrated.get("model_id") or name)
        bare = canonical_slug(str(migrated.get("version") or migrated.get("model_id") or name))
        migrated["lane"] = raw
        migrated["model"] = bare
        migrated["model_id"] = bare
        migrated["version"] = bare
        models[str(name)] = migrated
    out["models"] = models
    return out


def overlay_pending(overlay: Mapping[str, Any]) -> list[str]:
    """Names of overlay rows that still carry a pre-convention id field."""
    pending: list[str] = []
    for name, row in (overlay.get("models") or {}).items():
        if not isinstance(row, Mapping):
            continue
        version = str(row.get("version") or "")
        model_id = str(row.get("model_id") or "")
        if (
            "lane" not in row
            or version != canonical_slug(version)
            or model_id != canonical_slug(model_id)
        ):
            pending.append(str(name))
    return pending


# ── identity map ─────────────────────────────────────────────────────────────────────


def _merge_identity_entry(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    """Union one curated entry into another (aliases per provider, deny, hf id, modelsdev)."""
    if not dst.get("hugging_face_id") and src.get("hugging_face_id"):
        dst["hugging_face_id"] = src["hugging_face_id"]
    aliases: dict[str, list[str]] = {str(k): list(v) for k, v in (dst.get("aliases") or {}).items()}
    for provider, ids in (src.get("aliases") or {}).items():
        bucket = aliases.setdefault(str(provider), [])
        for item in ids:
            if item not in bucket:
                bucket.append(item)
    dst["aliases"] = aliases
    deny = list(dict.fromkeys([*(dst.get("deny") or []), *(src.get("deny") or [])]))
    dst["deny"] = deny
    if not dst.get("modelsdev") and src.get("modelsdev"):
        dst["modelsdev"] = src["modelsdev"]
    return dst


def migrate_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a migrated identity map: bare-slug keys, raw aliases, bare version aliases."""
    entries: dict[str, Any] = dict()
    for slug, body in (payload.get("models") or {}).items():
        body = body or {}
        canonical = canonical_slug(str(slug))
        aliases = {
            str(provider): [str(item) for item in (ids or [])]
            for provider, ids in (body.get("aliases") or {}).items()
        }
        entry = {
            "hugging_face_id": str(body.get("hugging_face_id") or ""),
            "aliases": aliases,
            "deny": [str(x) for x in (body.get("deny") or [])],
        }
        modelsdev = body.get("modelsdev")
        if isinstance(modelsdev, Mapping) and modelsdev.get("provider") and modelsdev.get("id"):
            entry["modelsdev"] = {
                "provider": str(modelsdev["provider"]),
                "id": str(modelsdev["id"]),
            }
        if canonical in entries:
            _merge_identity_entry(entries[canonical], entry)
        else:
            entries[canonical] = entry
    version_aliases: dict[str, str] = dict()
    for raw, value in (payload.get("version_aliases") or {}).items():
        version_aliases[str(raw)] = canonical_slug(str(value))
    for old_slug in payload.get("models") or {}:
        slug_str = str(old_slug)
        if slug_str != canonical_slug(slug_str):
            version_aliases.setdefault(slug_str, canonical_slug(slug_str))
    out: dict[str, Any] = {"schema": 1}
    out["version_aliases"] = version_aliases
    out["models"] = entries
    return out


# ── priority ─────────────────────────────────────────────────────────────────────────


def migrate_priority(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Dedupe `importance`/`frontier`/`vision` keys onto the bare canonical slug."""
    out: dict[str, Any] = dict(payload)
    if isinstance(payload.get("importance"), Mapping):
        importance: dict[str, Any] = dict()
        for key, value in payload["importance"].items():
            importance.setdefault(canonical_slug(str(key)), value)
        out["importance"] = importance
    for field in ("frontier", "vision"):
        if isinstance(payload.get(field), list):
            out[field] = _canon_unique([str(x) for x in payload[field]])
    return out


# ── CSVs ─────────────────────────────────────────────────────────────────────────────


def _migrated_csv_rows(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    """The new fieldnames and rows: insert `lane` after `model`; ids become bare slugs."""
    if not rows:
        return [], []
    fields = list(rows[0].keys())
    lane_present = "lane" in fields
    if not lane_present:
        fields = (
            fields[: fields.index("model") + 1] + ["lane"] + fields[fields.index("model") + 1 :]
        )
    migrated: list[dict[str, str]] = []
    for row in rows:
        lane = bare_slot(str(row.get("lane") or row.get("model") or ""))
        version = canonical_slug(
            bare_slot(str(row.get("model_version") or row.get("model") or lane))
        )
        new_row = dict(row)
        new_row["lane"] = lane
        new_row["model_version"] = version
        new_row["model"] = version
        migrated.append(new_row)
    return fields, migrated


def csv_pending(path: Path) -> bool:
    """True when *path* still lacks a `lane` column or a non-bare `model`/`model_version`."""
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "lane" not in (reader.fieldnames or []):
            return True
        for row in reader:
            if str(row.get("lane") or "") != bare_slot(str(row.get("lane") or "")):
                return True
            if str(row.get("model") or "") != canonical_slug(
                bare_slot(str(row.get("model") or ""))
            ):
                return True
            if str(row.get("model_version") or "") != canonical_slug(
                bare_slot(str(row.get("model_version") or ""))
            ):
                return True
    return False


def migrate_history() -> None:
    """Rewrite the gitignored superseded-row log's id columns to the bare slot."""
    if not HISTORY_PATH.exists():
        return
    rows = read_csv(HISTORY_PATH)
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        for field in ("model", "model_version"):
            if field in row:
                row[field] = canonical_slug(bare_slot(str(row[field])))
    write_csv(HISTORY_PATH, fields, rows)


def history_pending() -> bool:
    if not HISTORY_PATH.exists():
        return False
    for row in read_csv(HISTORY_PATH):
        for field in ("model", "model_version"):
            if field in row and str(row[field]) != canonical_slug(bare_slot(str(row[field]))):
                return True
    return False


# ── escalation corpus (trajectory_id is the cost-join key into results.csv) ──────────


def escalation_pending() -> list[str]:
    """Files whose names or contents still carry the retired serving-slot prefix."""
    pending: list[str] = []
    for directory in ESCALATION_TRAJECTORY_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            content = path.read_text(encoding="utf-8")
            if path.name != bare_slot_text(path.name) or bare_slot_text(content) != content:
                pending.append(str(path.relative_to(_REPO)))
    for path in SLOT_JSON_PATHS:
        if path.exists() and bare_slot_text(path.read_text(encoding="utf-8")) != path.read_text(
            encoding="utf-8"
        ):
            pending.append(str(path.relative_to(_REPO)))
    return pending


def apply_escalation() -> list[Path]:
    """Rename legacy-slot trajectory files and rewrite their ids + metadata in place."""
    touched: list[Path] = []
    for directory in ESCALATION_TRAJECTORY_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.name != bare_slot_text(path.name):
                path = path.rename(path.with_name(bare_slot_text(path.name)))
                touched.append(path)
            text = path.read_text(encoding="utf-8")
            new = bare_slot_text(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
    for path in SLOT_JSON_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new = bare_slot_text(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            touched.append(path)
    return touched


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".migrate.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


# ── report / apply ───────────────────────────────────────────────────────────────────


def report() -> dict[str, list[str]]:
    """The pending-change census: which surfaces still carry pre-convention ids."""
    pending: dict[str, list[str]] = {}
    overlay = yaml.safe_load(OVERLAY_PATH.read_text())
    pending["overlay"] = overlay_pending(overlay)
    pending["csv"] = [str(path.name) for path in CSV_PATHS if csv_pending(path)]
    pending["identity"] = [str(IDENTITY_PATH.name)] if identity_pending() else []
    pending["priority"] = [str(PRIORITY_PATH.name)] if priority_pending() else []
    pending["history"] = [str(HISTORY_PATH.name)] if history_pending() else []
    pending["escalation"] = escalation_pending()
    return {key: value for key, value in pending.items() if value}


def identity_pending() -> bool:
    payload = yaml.safe_load(IDENTITY_PATH.read_text())
    for key in payload.get("models") or {}:
        if str(key) != canonical_slug(str(key)):
            return True
    for value in (payload.get("version_aliases") or {}).values():
        if str(value) != canonical_slug(str(value)):
            return True
    return False


def priority_pending() -> bool:
    payload = yaml.safe_load(PRIORITY_PATH.read_text())
    for field in ("importance",):
        for key in payload.get(field) or {}:
            if str(key) != canonical_slug(str(key)):
                return True
    for field in ("frontier", "vision"):
        for item in payload.get(field) or []:
            if str(item) != canonical_slug(str(item)):
                return True
    return False


def apply_overlay() -> None:
    if not overlay_pending(yaml.safe_load(OVERLAY_PATH.read_text())):
        return
    header = scan._overlay_header(OVERLAY_PATH.read_text())
    overlay = yaml.safe_load(OVERLAY_PATH.read_text())
    OVERLAY_PATH.write_text(header + scan._write_yaml_body(migrate_overlay(overlay)))


def apply_identity() -> None:
    if not identity_pending():
        return
    payload = yaml.safe_load(IDENTITY_PATH.read_text())
    body = yaml.safe_dump(
        migrate_identity(payload), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    IDENTITY_PATH.write_text(_IDENTITY_HEADER + body)


def apply_priority() -> None:
    if not priority_pending():
        return
    payload = yaml.safe_load(PRIORITY_PATH.read_text())
    PRIORITY_PATH.write_text(yaml.safe_dump(migrate_priority(payload), sort_keys=False))


def apply_csvs() -> None:
    for path in CSV_PATHS:
        if not path.exists():
            continue
        fields, rows = _migrated_csv_rows(read_csv(path))
        write_csv(path, fields, rows)


def apply_lane_state() -> list[Path]:
    """Rewrite any `model`-keyed lane-state artifact onto its `lane` name (usually a no-op).

    The lane names in the committed artifacts already ARE the channel/lane ids, so the
    migration is a fixpoint; the function exists so a future rename of the naming scheme has a
    single place to land.
    """
    touched: list[Path] = []
    if not ARTIFACTS_DIR.exists():
        return touched
    for path in ARTIFACTS_DIR.rglob("lane_state.json"):
        payload = json.loads(path.read_text())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        touched.append(path)
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report pending changes, write nothing")
    mode.add_argument("--write", action="store_true", help="apply the migration in place")
    args = parser.parse_args(argv)

    if args.check:
        pending = report()
        if not pending:
            print("bare-slug convention: no pending changes")
            return 0
        print(json.dumps(pending, indent=2, sort_keys=True))
        return 1

    apply_overlay()
    apply_identity()
    apply_priority()
    apply_csvs()
    migrate_history()
    apply_escalation()
    apply_lane_state()
    print(
        "bare-slug convention: migrated overlay, identity, priority, CSVs, history, "
        "escalation corpus, lane-state"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run(argv: list[str] | None = None) -> int:
    """Alias kept for symmetry with the other script mains."""
    return main(argv)


def _write(*, write: bool) -> None:  # pragma: no cover - convenience for callers
    if write:
        main(["--write"])
    else:
        main(["--check"])
    if not sys.stdout:
        raise RuntimeError
