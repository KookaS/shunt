"""Refresh the free-campaign overlay and report the runnable lane set, ordered by value.

The discovery->campaign loop in one entrypoint:

    python -m benchmark.routing.scripts.refresh_free_campaign --dry-run   # scan, print the set
    python -m benchmark.routing.scripts.refresh_free_campaign --write     # scan + propose + apply

`--dry-run` is the safe scheduled mode: it GETs the public provider catalogues, builds the
facts-only snapshot, derives the identity proposal, prints the ordered runnable set, and writes
nothing tracked. `--write` additionally rewrites `latest_free_models.json`,
`free_models_proposal.yaml` and `configs/free-tier/overlay.yaml`; the workflow runs it on
`workflow_dispatch` only and uploads the three files as artifacts, because a `contents: read`
token cannot push and the owner commits the diff deliberately.

NETWORK IS CATALOGUE GETS ONLY. No benchmark prompt is sent and no model completion runs, so
the refresh is $0. The runnable set is filtered by `collection_priority`, so a lane that is
dominated, denylisted, or has no benchmark left is reported out and never scheduled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from benchmark import config
from benchmark.routing import collection_priority
from benchmark.routing.scripts import scan_free_models as scan

# The scheduling-only refusal tokens for an overlay row whose provider is not a declared free
# lane. The row STAYS in the overlay for provenance; it is never placed in the runnable set.
NO_FREE_LANE: Final[str] = "no-free-lane"
FREE_ACCESS_FALSE: Final[str] = "free_access:false"
NON_CHAT: Final[str] = "non-chat"
ARCHIVED: Final[str] = "archived"


@dataclass(frozen=True)
class RunnableLane:
    """One schedulable overlay row and its value rank, ready for the campaign scheduler."""

    name: str
    version: str
    provider: str
    priority: float
    reason: str
    # The listing's own `expiration_date`, carried so the lane scheduler can refuse a lapsed
    # time-boxed promo (LANE_EXPIRED). None for a listing that declares no expiry.
    expires_at: str | None = None

    def to_row(self) -> dict[str, Any]:
        """A JSON-serialisable view for the workflow artifact and the campaign log."""
        return {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
            "priority": self.priority,
            "reason": self.reason,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class RefreshResult:
    """The refresh's outcome: where the scan landed, the set to schedule, and its exclusions."""

    snapshot: dict[str, Any]
    runnable: tuple[RunnableLane, ...]
    excluded: dict[str, str]
    applied: bool
    overlay_path: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "scan_as_of": self.snapshot.get("scan_as_of"),
            "applied": self.applied,
            "overlay": self.overlay_path,
            "runnable": [lane.to_row() for lane in self.runnable],
            "excluded": dict(sorted(self.excluded.items())),
        }


def _snapshot_listing_index(
    snapshot: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Every snapshot listing keyed by ``(provider, listing_id)``, withdrawn rows included."""
    return {
        (str(row.get("provider", "")), str(row.get("listing_id", ""))): row
        for row in snapshot.get("listings", [])
    }


def lane_refusal(
    provider: str,
    model_id: str,
    declared: Mapping[str, str | None],
    listings: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str | None:
    """The named reason this overlay lane is withheld from scheduling, or ``None``.

    A provider absent from ``free_catalogs.yaml`` has no free lane at all
    (``no-free-lane``); one the catalogues mark ``free_access: false`` is refused as
    ``free_access:false`` even though its overlay row stays for provenance. A non-text-chat
    model (tts / image / computer-use / deep-research / aqa) is refused as ``non-chat`` so a
    stale overlay row can never be scheduled. Where the snapshot carries a matching listing
    that is not schedulable, its own named reason is used.
    """
    if provider not in declared:
        return NO_FREE_LANE
    if declared[provider] is not None:
        return FREE_ACCESS_FALSE
    if scan.is_non_chat_listing(model_id):
        return NON_CHAT
    row = listings.get((provider, model_id))
    if row is not None and row.get("schedulable") is False:
        return str(row.get("schedulable_reason") or f"{provider}: listing is not schedulable")
    return None


def lane_exclusions(
    overlay: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    declared_free: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Overlay lane name -> named reason it is withheld from the runnable set (scheduling-only).

    The provenance-preserving companion of :func:`runnable_lanes`: a Together or OpenCode Zen
    row is still in the overlay, but never scheduled, and this is where its reason is named.
    """
    declared = config.free_provider_access() if declared_free is None else declared_free
    listings = _snapshot_listing_index(snapshot)
    exclusions: dict[str, str] = {}
    for name, row in (overlay.get("models") or {}).items():
        if not isinstance(row, Mapping):
            continue
        if scan.is_archived(row):
            exclusions[str(name)] = ARCHIVED
            continue
        refusal = lane_refusal(
            str(row.get("provider", "")),
            str(row.get("lane") or row.get("model_id", "")),
            declared,
            listings,
        )
        if refusal is not None:
            exclusions[str(name)] = refusal
    return exclusions


def runnable_lanes(
    overlay: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    identity: scan.IdentityMap | None = None,
    *,
    engine: collection_priority.CollectionPriority | None = None,
    declared_free: Mapping[str, str | None] | None = None,
) -> list[RunnableLane]:
    """The schedulable lane set for one overlay + snapshot, ordered by descending priority.

    A row is withdrawn out when its SCANNED channel no longer carries it (the snapshot's
    `withdrawn_at` mark), denied by `model_identity.yaml`, or stopped by the value model
    (`worth_collecting`). A lane on a provider that is not a declared free lane (absent from
    `free_catalogs.yaml`, or marked `free_access: false`) is refused up front with a named
    reason, never scheduled. One identity served by several channel rows is credited to its
    single highest-priority row, so the campaign never schedules a duplicate. `engine` is
    injectable so a caller can rank against an in-memory overlay before it is written.
    """
    active = {
        (str(row["provider"]), str(row["listing_id"]))
        for row in snapshot.get("listings", [])
        if not row.get("withdrawn_at")
    }
    scanned_ok = {
        str(provider)
        for provider, info in (snapshot.get("channels") or {}).items()
        if str((info or {}).get("status", "")).startswith("ok")
    }
    declared = config.free_provider_access() if declared_free is None else declared_free
    listings = _snapshot_listing_index(snapshot)
    rank = engine or collection_priority.default_engine()
    best: dict[str, RunnableLane] = {}
    for name, row in (overlay.get("models") or {}).items():
        if not isinstance(row, Mapping):
            continue
        if scan.is_archived(row):
            continue
        provider = str(row.get("provider", ""))
        channel = str(row.get("lane") or row.get("model_id", ""))
        if lane_refusal(provider, channel, declared, listings) is not None:
            continue
        if provider in scanned_ok and (provider, channel) not in active:
            continue
        if identity is not None and (identity.is_denied(channel) or identity.is_denied(str(name))):
            continue
        version = str(row.get("version") or name)
        keep, reason = rank.worth_collecting(version)
        if not keep:
            continue
        listing = listings.get((provider, channel))
        expires_at = str(listing.get("expiration_date") or "") if listing else ""
        candidate = RunnableLane(
            name=str(name),
            version=version,
            provider=provider,
            priority=rank.priority(version),
            reason=reason,
            expires_at=expires_at or None,
        )
        current = best.get(version)
        if current is None or candidate.priority > current.priority:
            best[version] = candidate
    return sorted(best.values(), key=lambda lane: (-lane.priority, lane.name))


def _load_overlay() -> dict[str, Any]:
    return yaml.safe_load(scan.OVERLAY_PATH.read_text()) or {}


def _rank_engine() -> collection_priority.CollectionPriority:
    """A priority engine rebuilt against the on-disk overlay after an apply."""
    config.set_free_registry(scan.OVERLAY_PATH)
    collection_priority.clear_cache()
    return collection_priority.default_engine()


def refresh(*, write: bool) -> RefreshResult:
    """Scan, (optionally) propose + apply, then return the ordered runnable set.

    `write=False` writes nothing tracked and is the cron mode. `write=True` rewrites the
    snapshot, the proposal and the overlay, then rebuilds the ranking against the new overlay.
    """
    identity = scan.load_identity()
    snapshot = scan.scan_snapshot(scan.SNAPSHOT_PATH)
    if write:
        scan.write_snapshot(snapshot, scan.SNAPSHOT_PATH)
        scan.write_proposal(snapshot, identity)
        code = scan.apply_snapshot(identity, str(snapshot.get("scan_as_of") or ""))
        if code != 0:
            raise SystemExit(code)
    overlay = _load_overlay()
    declared = config.free_provider_access()
    lanes = runnable_lanes(
        overlay, snapshot, identity, engine=_rank_engine(), declared_free=declared
    )
    excluded = lane_exclusions(overlay, snapshot, declared_free=declared)
    return RefreshResult(
        snapshot=snapshot,
        runnable=tuple(lanes),
        excluded=excluded,
        applied=write,
        overlay_path=str(scan.OVERLAY_PATH),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and print the runnable set; write nothing tracked",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="scan, write the snapshot + proposal, apply the overlay, and print the set",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="print the raw facts-only snapshot JSON instead of the runnable set",
    )
    parser.add_argument(
        "--emit-snapshot",
        type=Path,
        default=None,
        help="also write the raw snapshot JSON to this (untracked) path",
    )
    args = parser.parse_args(argv)

    result = refresh(write=args.write)
    if args.emit_snapshot is not None:
        args.emit_snapshot.write_text(json.dumps(result.snapshot, indent=2, sort_keys=True) + "\n")
    if args.snapshot:
        print(json.dumps(result.snapshot, indent=2, sort_keys=True))
        return 0
    print(json.dumps(result.to_payload(), indent=2, sort_keys=True))
    print(
        f"{len(result.runnable)} runnable lane(s) as of "
        f"{result.snapshot.get('scan_as_of') or datetime.now(UTC).date().isoformat()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
