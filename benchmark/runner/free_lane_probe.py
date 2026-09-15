#!/usr/bin/env python3
"""Free-lane admission probe: static admission gates plus a live tool-call probe.

One question — may this free listing be admitted as a collection lane? — answered by static
gates that need no network (declares tool-calling; model age from models.dev; seen in two
consecutive scans; no expiration inside the run window; a resolved identity; a daily budget of
at least one cell reservation) combined with live gates (the 5-case tool-call probe, measured
availability, and app-gating).

The 5-case probe reuses the scaffold's own ``BASH_TOOL`` and ``parse_toolcall_actions``, so the
instrument under test is the one production uses. The live transport is INJECTED, which makes
the whole pipeline hermetic in tests and keeps the real run behind provider keys: with no key
the entrypoint refuses rather than fabricating a verdict.

Deliberately NOT in ``tools/provider_probe.py``: that module's contract is a keyless wiring
check that never bills and runs on fork PRs. This one issues a real (free-lane) completion, so
it stays off PR CI, behind the $0 interlocks.

Instrument validity: this gate emits a verdict, so it carries a positive control (a known
tool-calling model clears the probe) and a destroyed-signal null (a model without tool support
does not), adjudicated by the shared ``admissibility`` adjudicator, never a local re-derivation.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

from benchmark.runner.scaffold_model import ANONYMOUS_API_KEY
from shunt.analysis.admissibility import (
    AdmissibilityResult,
    admissibility_verdict,
)
from shunt.models.config import ModelConfig, load_registry, resolve_models

# The committed scanner outputs the live entrypoint reads facts and identities from.
_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "routing" / "data"
SNAPSHOT_PATH: Final[Path] = _DATA_DIR / "latest_free_models.json"
PROPOSAL_PATH: Final[Path] = _DATA_DIR / "free_models_proposal.yaml"

# The scaffold's format-error template, required by parse_toolcall_actions.
_FORMAT_ERROR_TEMPLATE: Final[str] = "Format error: {{ error }}"

# A provider that publishes no limits is admitted with these conservative defaults. Declaring
# "unknown" beats inventing a number; the values are the `lanes.unknown_limits` reservation.
UNKNOWN_LIMITS_RPM: Final[int] = 10
UNKNOWN_LIMITS_RPD: Final[int] = 100
DEFAULT_CELL_CALL_RESERVATION: Final[int] = 85
DEFAULT_AVAILABILITY_THRESHOLD: Final[float] = 0.8
RELEASE_AGE_LIMIT_MONTHS: Final[int] = 18
DEFAULT_CHANCE_LEVEL: Final[float] = 0.0
DEFAULT_CHANCE_BAND: Final[float] = 0.15
# `ANONYMOUS_API_KEY` (imported above from `benchmark.runner.scaffold_model`) is the one
# canonical harmless placeholder for an anonymous free lane whose key variable is unset (Kilo
# Gateway serves `:free` ids with no `Authorization`). It lives next to the scaffold's injection
# seam so the live path and this probe cannot drift; it is never a real credential and never
# opens a billed path. The refusal for a provider that genuinely requires a key is unaffected —
# only `Provider.key_optional` lanes reach it.

# Wall-clock arithmetic (the pause document's headline). The unit is requests per completed
# cell: measured ~85 at p90. The campaign target is 20 cells per model. The host bound is a
# single fleet number: 1 host-hour per cell at 8-way parallelism.
TARGET_CELLS_PER_MODEL: Final[int] = 20
REQUESTS_PER_CELL_P90: Final[int] = 85
HOST_HOURS_PER_CELL: Final[float] = 1.0
HOST_PARALLELISM: Final[int] = 8

# The artifact directory the pause document is written to (never committed into `shunt/`).
# A neutral, cwd-relative default keeps the probe free of any host-specific absolute path;
# `$SHUNT_FREE_SCAN_DIR` overrides it so a CI or sandbox run can redirect it.
FREE_SCAN_DIR_ENV: Final[str] = "SHUNT_FREE_SCAN_DIR"
DEFAULT_SCAN_DIR: Final[Path] = Path("artifacts") / "free-tier-scan"
LIVE_HALF_STATUS: Final[str] = "pending owner keys — static half only"

# A free listing that only speaks to agentic harnesses, not the API surface this lane calls.
_APP_GATE_PATTERNS: Final[tuple[str, ...]] = (
    "only available on agentic",
    "only available through agentic",
    "only available in agentic",
    "agentic harness",
    "not available via the api",
)

# The live probe's closed transport: (messages, tools) -> an OpenAI-shaped chat response.
ToolcallTransport = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]


class GateHalf(StrEnum):
    """Whether a gate is decided from the snapshot (static) or from a live call (live)."""

    STATIC = "static"
    LIVE = "live"


@dataclass(frozen=True)
class ToolCallCase:
    """One of the probe's five prompts: the instruction a tool-calling model must act on."""

    name: str
    instruction: str


TOOLCALL_CASES: Final[tuple[ToolCallCase, ...]] = (
    ToolCallCase("pwd", "Use the bash tool to run exactly: pwd"),
    ToolCallCase("list", "Use the bash tool to run exactly: ls -la"),
    ToolCallCase("echo", "Use the bash tool to run exactly: echo ready"),
    ToolCallCase("git-status", "Use the bash tool to run exactly: git status --short"),
    ToolCallCase("dispatch", 'Use the bash tool to run exactly: python -c "print(1)"'),
)


@dataclass(frozen=True)
class LiveProbeResult:
    """The outcome of the 5-case tool-call probe for one lane."""

    attempts: int
    passes: int
    app_gated: bool
    cases: tuple[tuple[str, bool], ...]
    errors: tuple[str, ...]

    @property
    def availability(self) -> float:
        """Fraction of probe cases that produced a parseable bash call."""
        return self.passes / self.attempts if self.attempts else 0.0


@dataclass(frozen=True)
class ListingFacts:
    """The per-listing snapshot facts the static admission gates read."""

    listing_id: str
    provider: str
    declares_tools: bool | None
    release_date: str | None
    first_seen: str
    expiration_date: str | None
    identity: str | None
    published_limits: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneLimits:
    """A lane's resolved request limits; ``known`` is False when either side was published null."""

    rpm: int
    rpd: int
    known: bool


@dataclass(frozen=True)
class AdmissionContext:
    """The clock and budget a single admission decision is made against."""

    scan_as_of: date
    run_window_end: date
    cell_reservation_calls: int = DEFAULT_CELL_CALL_RESERVATION
    availability_threshold: float = DEFAULT_AVAILABILITY_THRESHOLD


@dataclass(frozen=True)
class GateCheck:
    """One admission gate's verdict, named and with the reason it produced it."""

    name: str
    half: GateHalf
    passed: bool
    reason: str


@dataclass(frozen=True)
class AdmissionVerdict:
    """The assembled static + live verdict for one listing."""

    facts: ListingFacts
    checks: tuple[GateCheck, ...]
    identity: str | None
    limits: LaneLimits
    live: LiveProbeResult

    @property
    def admitted(self) -> bool:
        """True only when EVERY gate passes."""
        return all(check.passed for check in self.checks)

    @property
    def refusals(self) -> tuple[GateCheck, ...]:
        """The gates that refused this listing, each carrying its own reason."""
        return tuple(check for check in self.checks if not check.passed)


# ── limits ──────────────────────────────────────────────────────────────────────────


def resolve_limits(published: Mapping[str, Any]) -> LaneLimits:
    """Resolve published limits, substituting the ``unknown`` defaults for any null side."""
    rpm = published.get("rpm")
    rpd = published.get("rpd")
    known = isinstance(rpm, int) and isinstance(rpd, int)
    return LaneLimits(
        rpm=rpm if isinstance(rpm, int) else UNKNOWN_LIMITS_RPM,
        rpd=rpd if isinstance(rpd, int) else UNKNOWN_LIMITS_RPD,
        known=known,
    )


# ── static gates ────────────────────────────────────────────────────────────────────


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _minus_months(anchor: date, months: int) -> date:
    """*anchor* moved back by *months* calendar months, clamped to a valid day of month."""
    index = anchor.month - 1 - months
    year = anchor.year + index // 12
    month = index % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _gate_declares_tools(facts: ListingFacts) -> GateCheck:
    """The static tool gate: an explicit no refuses, an unknown DEFERS to the live probe.

    ``None`` is not ``False``: a catalogue that simply does not declare a tool surface is not
    evidence that the model cannot call tools, so the static half passes and the live
    ``tool_call_probe`` gate decides. Only an explicit ``supports_tools is False`` refuses here.
    """
    if facts.declares_tools is False:
        return GateCheck(
            "declares_tool_calling",
            GateHalf.STATIC,
            False,
            "listing explicitly declares no tool-calling support",
        )
    if facts.declares_tools is True:
        return GateCheck(
            "declares_tool_calling", GateHalf.STATIC, True, "listing declares tool-calling"
        )
    return GateCheck(
        "declares_tool_calling",
        GateHalf.STATIC,
        True,
        "tool support UNDECLARED — the live tool-call probe gates admission",
    )


def _gate_release_age(facts: ListingFacts, context: AdmissionContext) -> GateCheck:
    released = _parse_date(facts.release_date)
    if released is None:
        return GateCheck(
            "release_date_within_18_months",
            GateHalf.STATIC,
            False,
            "release_date unknown (models.dev has no entry) — cannot verify model age",
        )
    cutoff = _minus_months(context.scan_as_of, RELEASE_AGE_LIMIT_MONTHS)
    passed = released >= cutoff
    reason = (
        f"release_date {released.isoformat()} is on/after {cutoff.isoformat()}"
        if passed
        else f"release_date {released.isoformat()} is before {cutoff.isoformat()}"
    )
    return GateCheck("release_date_within_18_months", GateHalf.STATIC, passed, reason)


def _gate_consecutive_scans(facts: ListingFacts, context: AdmissionContext) -> GateCheck:
    # The snapshot carries first_seen, so a listing seen at an EARLIER scan has crossed at least
    # one scan boundary — the two-consecutive-scans evidence available without a scan calendar.
    first_seen = _parse_date(facts.first_seen)
    if first_seen is None:
        return GateCheck(
            "seen_in_2_consecutive_scans",
            GateHalf.STATIC,
            False,
            "no first_seen on record — a new listing is not admitted on its first sighting",
        )
    passed = first_seen < context.scan_as_of
    reason = (
        f"present since {first_seen.isoformat()}, before the current scan "
        f"{context.scan_as_of.isoformat()}"
        if passed
        else f"first seen {first_seen.isoformat()} at the current scan — needs a second scan"
    )
    return GateCheck("seen_in_2_consecutive_scans", GateHalf.STATIC, passed, reason)


def _gate_expiration(facts: ListingFacts, context: AdmissionContext) -> GateCheck:
    expires = _parse_date(facts.expiration_date)
    passed = expires is None or expires > context.run_window_end
    expires_text = expires.isoformat() if expires is not None else "none"
    reason = (
        "no expiration inside the run window"
        if passed
        else f"expires {expires_text} inside the run window ending {context.run_window_end}"
    )
    return GateCheck("no_expiration_in_window", GateHalf.STATIC, passed, reason)


def _gate_identity(facts: ListingFacts) -> GateCheck:
    passed = bool(facts.identity)
    reason = (
        f"identity resolved to {facts.identity!r}"
        if passed
        else "identity UNRESOLVED — the listing is not joined to a canonical model slug"
    )
    return GateCheck("identity_resolved", GateHalf.STATIC, passed, reason)


def _gate_budget(limits: LaneLimits, reservation: int) -> GateCheck:
    passed = limits.rpd >= reservation
    side = (
        "published"
        if limits.known
        else f"UNKNOWN (default {UNKNOWN_LIMITS_RPM} RPM / {UNKNOWN_LIMITS_RPD} RPD)"
    )
    reason = (
        f"daily budget {limits.rpd} ({side}) covers one cell reservation of {reservation}"
        if passed
        else f"daily budget {limits.rpd} ({side}) is below one cell reservation of {reservation}"
    )
    return GateCheck("daily_budget_ge_one_cell", GateHalf.STATIC, passed, reason)


def static_gate_checks(facts: ListingFacts, context: AdmissionContext) -> tuple[GateCheck, ...]:
    """The six static admission gates, in the order the admission table lists them."""
    limits = resolve_limits(facts.published_limits)
    return (
        _gate_declares_tools(facts),
        _gate_release_age(facts, context),
        _gate_consecutive_scans(facts, context),
        _gate_expiration(facts, context),
        _gate_identity(facts),
        _gate_budget(limits, context.cell_reservation_calls),
    )


# ── live probe ──────────────────────────────────────────────────────────────────────


def _response_text(response: object) -> str:
    """The assistant text of an OpenAI-shaped response, for app-gate detection."""
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, Mapping) else None
        return str(content or "")
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    return str(getattr(message, "content", None) or "")


def _extract_tool_calls(response: object) -> list[Any]:
    """The tool calls from an OpenAI-shaped chat response, object or mapping form."""
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else None
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        return list(calls or [])
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    return list(getattr(message, "tool_calls", None) or [])


def run_toolcall_probe(
    transport: ToolcallTransport,
    *,
    cases: Sequence[ToolCallCase] = TOOLCALL_CASES,
) -> LiveProbeResult:
    """Run the 5-case tool-call probe, parsing each reply with the scaffold's own parser."""
    outcomes: list[tuple[str, bool]] = []
    errors: list[str] = []
    app_gated = False
    for case in cases:
        messages = [{"role": "user", "content": case.instruction}]
        try:
            response = transport(messages, [BASH_TOOL])
            if any(p in _response_text(response).lower() for p in _APP_GATE_PATTERNS):
                app_gated = True
            actions = parse_toolcall_actions(
                _extract_tool_calls(response), format_error_template=_FORMAT_ERROR_TEMPLATE
            )
            ok = bool(actions and actions[0].get("command"))
        except Exception as exc:  # noqa: BLE001 - one case must not abort the probe
            outcomes.append((case.name, False))
            errors.append(f"{case.name}: {type(exc).__name__}: {exc}")
            continue
        outcomes.append((case.name, ok))
    passes = sum(1 for _, ok in outcomes if ok)
    return LiveProbeResult(len(cases), passes, app_gated, tuple(outcomes), tuple(errors))


def live_gate_checks(live: LiveProbeResult, availability_threshold: float) -> tuple[GateCheck, ...]:
    """The three live gates, from one probe result."""
    return (
        GateCheck(
            "tool_call_probe",
            GateHalf.LIVE,
            live.passes >= 1,
            f"{live.passes}/{live.attempts} probe cases produced a bash tool call",
        ),
        GateCheck(
            "availability_at_threshold",
            GateHalf.LIVE,
            live.availability >= availability_threshold,
            f"availability {live.availability:.3f} vs threshold {availability_threshold:.3f}",
        ),
        GateCheck(
            "not_app_gated",
            GateHalf.LIVE,
            not live.app_gated,
            "reply is app-gated (agentic-harness-only)"
            if live.app_gated
            else "no app-gating language in any probe reply",
        ),
    )


def verdict_from_live(
    facts: ListingFacts, context: AdmissionContext, live: LiveProbeResult
) -> AdmissionVerdict:
    """Assemble the static + live admission verdict from an already-run probe result."""
    checks = static_gate_checks(facts, context) + live_gate_checks(
        live, context.availability_threshold
    )
    return AdmissionVerdict(
        facts=facts,
        checks=checks,
        identity=facts.identity,
        limits=resolve_limits(facts.published_limits),
        live=live,
    )


def evaluate_admission(
    transport: ToolcallTransport, facts: ListingFacts, context: AdmissionContext
) -> AdmissionVerdict:
    """Run the live probe and assemble the static + live admission verdict for one listing."""
    return verdict_from_live(facts, context, run_toolcall_probe(transport))


# ── instrument-validity controls ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdmissionControls:
    """The two transports a plant-signal / destroy-signal control runs through the pipeline."""

    positive_transport: ToolcallTransport
    null_transport: ToolcallTransport


@dataclass(frozen=True)
class ControlOutcome:
    """Both assembled-pipeline verdicts plus the numeric adjudication of the instrument."""

    positive: AdmissionVerdict
    null: AdmissionVerdict
    adjudication: AdmissibilityResult


def evaluate_controls(
    controls: AdmissionControls,
    *,
    positive_facts: ListingFacts,
    null_facts: ListingFacts,
    context: AdmissionContext,
    chance_level: float = DEFAULT_CHANCE_LEVEL,
    chance_band: float = DEFAULT_CHANCE_BAND,
) -> ControlOutcome:
    """Run both controls through the ASSEMBLED pipeline, then adjudicate the live instrument.

    The instrument under test is the live tool-call probe, whose verdict must be trusted before
    any admission is quoted. Its score is availability; the planted-signal fixture must recover
    it and the destroyed-signal fixture (a model without tool support) must collapse to chance.
    """
    # Each control's transport is run ONCE: the score and the assembled verdict share the
    # same probe result, so a live run costs no more than the controls themselves.
    positive_live = run_toolcall_probe(controls.positive_transport)
    null_live = run_toolcall_probe(controls.null_transport)
    positive = verdict_from_live(positive_facts, context, positive_live)
    null = verdict_from_live(null_facts, context, null_live)
    adjudication = admissibility_verdict(
        positive_live.availability,
        null_live.availability,
        chance_level=chance_level,
        chance_band=chance_band,
    )
    return ControlOutcome(positive=positive, null=null, adjudication=adjudication)


# ── wiring: snapshot/proposal loaders and the live entrypoint ────────────────────────


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Load the committed free-listing snapshot produced by the scanner."""
    return dict(json.loads(path.read_text()))


def load_proposal(path: Path = PROPOSAL_PATH) -> dict[str, Any]:
    """Load the reviewed identity proposal (confirmed / proposed / dropped)."""
    return dict(yaml.safe_load(path.read_text()) or {})


def _snapshot_row(
    snapshot: Mapping[str, Any], provider: str, listing_id: str
) -> Mapping[str, Any] | None:
    """The active (non-withdrawn) snapshot row for a provider/listing pair, if present."""
    for row in snapshot.get("listings") or []:
        if (
            str(row.get("provider")) == provider
            and str(row.get("listing_id")) == listing_id
            and not row.get("withdrawn_at")
        ):
            return row
    return None


def identity_lookup(proposal: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Map ``(provider, listing_id)`` to its resolved identity from the reviewed proposal."""
    lookup: dict[tuple[str, str], str] = {}
    for section in ("confirmed", "proposed"):
        for identity, entry in (proposal.get(section) or {}).items():
            for item in entry.get("listings") or []:
                provider, _, listing_id = str(item).partition(":")
                lookup[(provider, listing_id)] = str(identity)
    return lookup


def facts_from_snapshot_row(row: Mapping[str, Any], *, identity: str | None) -> ListingFacts:
    """Adapt one ``latest_free_models.json`` listing row into :class:`ListingFacts`.

    ``supports_tools`` is preserved as a tri-state: ``True``/``False`` when the snapshot
    declares it, ``None`` when the field is absent or not a bool (unknown — the live probe is
    the gate), never coerced to ``False`` by truthiness.
    """
    declares = row.get("supports_tools")
    return ListingFacts(
        listing_id=str(row.get("listing_id") or ""),
        provider=str(row.get("provider") or ""),
        declares_tools=declares if isinstance(declares, bool) else None,
        release_date=row.get("release_date"),
        first_seen=str(row.get("first_seen") or ""),
        expiration_date=row.get("expiration_date"),
        identity=identity,
        published_limits=dict(row.get("published_limits") or {}),
    )


def litellm_transport(
    *, route: str, api_base: str, api_key: str, max_tokens: int = 512
) -> ToolcallTransport:
    """A live transport over litellm; issued only by the key-gated ``--live`` entrypoint."""

    def transport(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        import litellm  # noqa: PLC0415

        return litellm.completion(
            model=route,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            # One attempt per probe case: litellm defaults num_retries to 3, which would make a
            # throttled lane cost three requests (and three backoffs) per case before the probe
            # records the failure, defeating the single-shot admission contract.
            num_retries=0,
        )

    return transport


def _verdict_payload(verdict: AdmissionVerdict) -> dict[str, Any]:
    """A JSON-serialisable view of an admission verdict, with every gate's reason."""
    return {
        "listing_id": verdict.facts.listing_id,
        "provider": verdict.facts.provider,
        "admitted": verdict.admitted,
        "identity": verdict.identity,
        "limits": {
            "rpm": verdict.limits.rpm,
            "rpd": verdict.limits.rpd,
            "known": verdict.limits.known,
        },
        "availability": verdict.live.availability,
        "gates": [
            {"name": c.name, "half": c.half.value, "passed": c.passed, "reason": c.reason}
            for c in verdict.checks
        ],
    }


def _context(scan_as_of: date, run_window_days: int) -> AdmissionContext:
    return AdmissionContext(
        scan_as_of=scan_as_of, run_window_end=scan_as_of + timedelta(days=run_window_days)
    )


# ── the pause document: every listing, its static verdict, and the wall-clock estimate ─


@dataclass(frozen=True)
class ListingReport:
    """One discovered listing's static admission half, ready for the pause document."""

    facts: ListingFacts
    limits: LaneLimits
    checks: tuple[GateCheck, ...]
    live_status: str = LIVE_HALF_STATUS

    @property
    def statically_admitted(self) -> bool:
        """True only when EVERY static gate passes; the live half is never assumed."""
        return all(check.passed for check in self.checks)

    @property
    def refusals(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True)
class ModelWallClock:
    """One identity's API-bound wall clock: its own lanes' aggregate daily capacity."""

    identity: str
    providers: tuple[str, ...]
    capacity_rpd: int
    limits_known: bool
    api_days: float


@dataclass(frozen=True)
class WallClockEstimate:
    """The pause document's headline: `max(host_bound, slowest model's lane bound)`."""

    models: int
    host_bound_days: float
    slowest_model_days: float
    estimate_days: float
    per_model: tuple[ModelWallClock, ...]


def build_listing_reports(
    snapshot: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: AdmissionContext,
) -> list[ListingReport]:
    """Adapt every active snapshot listing into its static admission report, no keys needed."""
    identities = identity_lookup(proposal)
    reports: list[ListingReport] = []
    for row in snapshot.get("listings") or []:
        if row.get("withdrawn_at"):
            continue
        listing_id = str(row.get("listing_id") or "")
        provider = str(row.get("provider") or "")
        facts = facts_from_snapshot_row(row, identity=identities.get((provider, listing_id)))
        reports.append(
            ListingReport(
                facts=facts,
                limits=resolve_limits(facts.published_limits),
                checks=static_gate_checks(facts, context),
            )
        )
    return sorted(reports, key=lambda r: (r.facts.provider, r.facts.listing_id))


def estimate_wall_clock(
    reports: Sequence[ListingReport],
    *,
    cells_per_model: int = TARGET_CELLS_PER_MODEL,
    requests_per_cell: int = REQUESTS_PER_CELL_P90,
    host_hours_per_cell: float = HOST_HOURS_PER_CELL,
    host_parallelism: int = HOST_PARALLELISM,
) -> WallClockEstimate:
    """The wall-clock bound: the larger of the host floor and the slowest model's own-lane bound.

    A model can only draw on lanes that serve it, so the API bound is per identity, not a
    fleet total. Capacity is summed over DISTINCT providers: lanes on different providers
    are independent quotas (the multi-provider thesis), while two listings of one identity on
    one provider share that provider's quota and count once. A provider that publishes no
    limit is reserved at the declared ``unknown`` default, never invented.

    A lane is a CAPACITY candidate when it is otherwise admissible and merely awaits a second
    scan: the two-consecutive-scans gate is a scan-calendar wait, not a capacity fact, so it
    does not remove a lane from the bound. Every hard static refusal (no tools, no identity,
    stale/unknown release, expiring, budget below one cell) excludes the lane.
    """
    admitted = [
        r
        for r in reports
        if r.facts.identity
        and not [c for c in r.refusals if c.name != "seen_in_2_consecutive_scans"]
    ]
    capacity_by_identity: dict[str, dict[str, int]] = {}
    known_by_identity: dict[str, bool] = {}
    for report in admitted:
        assert report.facts.identity is not None  # narrowed by the filter above
        identity = report.facts.identity
        per_provider = capacity_by_identity.setdefault(identity, {})
        prior = per_provider.get(report.facts.provider, 0)
        per_provider[report.facts.provider] = max(prior, report.limits.rpd)
        known_by_identity[identity] = known_by_identity.get(identity, True) and report.limits.known

    per_model: list[ModelWallClock] = []
    slowest = 0.0
    for identity, per_provider in capacity_by_identity.items():
        capacity = sum(per_provider.values())
        api_days = cells_per_model * requests_per_cell / capacity if capacity else float("inf")
        slowest = max(slowest, api_days)
        per_model.append(
            ModelWallClock(
                identity=identity,
                providers=tuple(sorted(per_provider)),
                capacity_rpd=capacity,
                limits_known=known_by_identity.get(identity, False),
                api_days=api_days,
            )
        )
    per_model.sort(key=lambda m: m.api_days, reverse=True)
    models = len(capacity_by_identity)
    host_bound = models * cells_per_model * host_hours_per_cell / host_parallelism / 24.0
    estimate = max(host_bound, slowest) if models else 0.0
    return WallClockEstimate(
        models=models,
        host_bound_days=host_bound,
        slowest_model_days=slowest,
        estimate_days=estimate,
        per_model=tuple(per_model),
    )


def _report_payload(
    reports: Sequence[ListingReport],
    wall_clock: WallClockEstimate,
    *,
    scan_as_of: str,
    run_window_end: str,
) -> dict[str, Any]:
    """The JSON pause document: a verdict + reason per listing and the wall-clock block."""
    return {
        "schema": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "scan_as_of": scan_as_of,
        "run_window_end": run_window_end,
        "live_half": LIVE_HALF_STATUS,
        "summary": {
            "listings": len(reports),
            "statically_admitted": sum(1 for r in reports if r.statically_admitted),
            "refused": sum(1 for r in reports if not r.statically_admitted),
        },
        "wall_clock": {
            "cells_per_model": TARGET_CELLS_PER_MODEL,
            "requests_per_cell_p90": REQUESTS_PER_CELL_P90,
            "host_hours_per_cell": HOST_HOURS_PER_CELL,
            "host_parallelism": HOST_PARALLELISM,
            "candidate_rule": (
                "resolved identity and every static gate except seen_in_2_consecutive_scans "
                "(a scan-calendar wait, not a capacity fact)"
            ),
            "models": wall_clock.models,
            "host_bound_days": round(wall_clock.host_bound_days, 3),
            "slowest_model_days": round(wall_clock.slowest_model_days, 3),
            "estimate_days": round(wall_clock.estimate_days, 3),
            "per_model": [
                {
                    "identity": m.identity,
                    "providers": list(m.providers),
                    "capacity_rpd": m.capacity_rpd,
                    "limits_known": m.limits_known,
                    "api_days": round(m.api_days, 3),
                }
                for m in wall_clock.per_model
            ],
        },
        "listings": [
            {
                "listing_id": r.facts.listing_id,
                "provider": r.facts.provider,
                "identity": r.facts.identity,
                "admitted_static": r.statically_admitted,
                "live": r.live_status,
                "limits": {"rpm": r.limits.rpm, "rpd": r.limits.rpd, "known": r.limits.known},
                "gates": [
                    {
                        "name": c.name,
                        "half": c.half.value,
                        "passed": c.passed,
                        "reason": c.reason,
                    }
                    for c in r.checks
                ],
                "refusal_reasons": [c.reason for c in r.refusals],
            }
            for r in reports
        ],
    }


def _report_markdown(payload: Mapping[str, Any]) -> str:
    """A human-readable rendering of the pause document, one row per listing."""
    wall = payload["wall_clock"]
    summary = payload["summary"]
    lines = [
        "# Free-lane admission scan — pause document",
        "",
        f"- scan_as_of: `{payload['scan_as_of']}`  (run window ends `{payload['run_window_end']}`)",
        f"- listings: **{summary['listings']}**, statically admitted:"
        f" **{summary['statically_admitted']}**, refused: **{summary['refused']}**",
        f"- live half: **{payload['live_half']}**",
        "",
        "## Wall-clock estimate (host bound vs slowest model's lane bound)",
        "",
        f"`max(host_bound, slowest_model)` = **{wall['estimate_days']} days**"
        f" (host bound {wall['host_bound_days']} d over {wall['models']} models;"
        f" slowest model {wall['slowest_model_days']} d).",
        "",
        "| model | providers | capacity req/day | limits | api days |",
        "|---|---|---:|---|---:|",
    ]
    for model in wall["per_model"]:
        known = "published" if model["limits_known"] else "unknown→default"
        lines.append(
            f"| {model['identity']} | {', '.join(model['providers']) or '—'} |"
            f" {model['capacity_rpd']} | {known} | {model['api_days']} |"
        )
    lines += [
        "",
        "## Listings",
        "",
        "| provider | listing | identity | static | refusal |",
        "|---|---|---|---|---|",
    ]
    for item in payload["listings"]:
        verdict = "ADMIT" if item["admitted_static"] else "refuse"
        reason = "; ".join(item["refusal_reasons"]) or "—"
        lines.append(
            f"| {item['provider']} | `{item['listing_id']}` | {item['identity'] or '—'} |"
            f" {verdict} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    snapshot: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    run_window_days: int = 30,
    out_dir: Path | None = None,
) -> Path:
    """Build the pause document from the snapshot alone (no keys) and write it to *out_dir*."""
    scan_as_of = _snapshot_scan_date(snapshot)
    context = _context(scan_as_of, run_window_days)
    reports = build_listing_reports(snapshot, proposal, context)
    wall_clock = estimate_wall_clock(reports)
    payload = _report_payload(
        reports,
        wall_clock,
        scan_as_of=scan_as_of.isoformat(),
        run_window_end=context.run_window_end.isoformat(),
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = (out_dir or _default_scan_dir()) / stamp
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (target / "report.md").write_text(_report_markdown(payload))
    return target


def _snapshot_scan_date(snapshot: Mapping[str, Any]) -> date:
    raw = str(snapshot.get("scan_as_of") or "")
    parsed = _parse_date(raw)
    return parsed or date.today()


def _default_scan_dir() -> Path:
    raw = os.environ.get(FREE_SCAN_DIR_ENV)
    return Path(raw) if raw else DEFAULT_SCAN_DIR


def _transport_for(model: ModelConfig) -> ToolcallTransport:
    """Build a live transport for a resolved overlay model.

    A missing key refuses — UNLESS the provider declares anonymous free access, in which case a
    harmless placeholder is sent. Kilo Gateway answers `:free` requests with no credentials, so
    an unset ``KILO_API_KEY`` is an anonymous free lane, not a reason to abort; a provider that
    genuinely requires a key still refuses.
    """
    key = os.environ.get(model.api_key_env_var)
    if not key and not model.key_optional:
        raise SystemExit(
            f"aborted: ${model.api_key_env_var} is not set — a live admission probe needs a "
            "real key and is never run without one"
        )
    return litellm_transport(
        route=model.route,
        api_base=model.base_url,
        api_key=key or ANONYMOUS_API_KEY,
    )


def _facts_for(model: ModelConfig, name: str) -> ListingFacts:
    """Adapt the model's snapshot row + resolved identity into :class:`ListingFacts`."""
    listing_id = model.model_id or name
    row = _snapshot_row(load_snapshot(), model.provider, listing_id)
    if row is None:
        raise SystemExit(f"aborted: no snapshot row for {model.provider}/{listing_id}")
    identity = identity_lookup(load_proposal()).get((model.provider, listing_id))
    return facts_from_snapshot_row(row, identity=identity)


def _named(models: Mapping[str, ModelConfig], name: str, registry: Path) -> ModelConfig:
    if name not in models:
        raise SystemExit(f"aborted: {name!r} is not in {registry}")
    return models[name]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Live probing needs real keys; absent keys refuse, never fabricate."""
    parser = argparse.ArgumentParser(
        prog="free_lane_probe",
        description=(
            "Admission probe for a free lane. --all builds the pause document from the "
            "snapshot with NO keys (static half only). A live probe needs --live plus real "
            "provider keys: --model admits one overlay listing, --controls validates the "
            "probe instrument (positive + destroyed-signal controls)."
        ),
    )
    parser.add_argument("--live", action="store_true", help="opt in to a live probe")
    parser.add_argument("--registry", type=Path, default=None, help="overlay registry path")
    parser.add_argument("--model", help="admit this registry model name")
    parser.add_argument("--controls", action="store_true", help="validate the probe instrument")
    parser.add_argument("--positive-model", default=None, help="known tool-calling control")
    parser.add_argument("--null-model", default=None, help="known no-tool control")
    parser.add_argument(
        "--all",
        "--report",
        dest="report",
        action="store_true",
        help="build the pause document for every discovered listing (no keys, static half only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"report directory (default ${FREE_SCAN_DIR_ENV} or {DEFAULT_SCAN_DIR})",
    )
    parser.add_argument("--run-window-days", type=int, default=30)
    args = parser.parse_args(argv)

    if args.report:
        snapshot = load_snapshot()
        target = write_report(
            snapshot, load_proposal(), run_window_days=args.run_window_days, out_dir=args.out_dir
        )
        print(f"wrote {target}", file=sys.stderr)
        return 0

    if not args.live:
        print(
            "aborted: pass --live to opt in; the probe issues a real completion and is never "
            "run implicitly",
            file=sys.stderr,
        )
        return 2

    if args.registry is None:
        print("aborted: --registry is required for a live probe", file=sys.stderr)
        return 2

    models = resolve_models(load_registry(args.registry))
    context = _context(date.today(), args.run_window_days)

    if args.model:
        model = _named(models, args.model, args.registry)
        verdict = evaluate_admission(_transport_for(model), _facts_for(model, args.model), context)
        print(json.dumps(_verdict_payload(verdict), indent=2))
        return 0 if verdict.admitted else 1

    if args.controls:
        if not (args.positive_model and args.null_model):
            print("aborted: --controls needs --positive-model and --null-model", file=sys.stderr)
            return 2
        positive = _named(models, args.positive_model, args.registry)
        null = _named(models, args.null_model, args.registry)
        outcome = evaluate_controls(
            AdmissionControls(_transport_for(positive), _transport_for(null)),
            positive_facts=_facts_for(positive, args.positive_model),
            null_facts=_facts_for(null, args.null_model),
            context=context,
        )
        print(
            json.dumps(
                {
                    "admissible": outcome.adjudication.admissible,
                    "reason": outcome.adjudication.reason,
                    "positive_admitted": outcome.positive.admitted,
                    "null_admitted": outcome.null.admitted,
                },
                indent=2,
            )
        )
        return 0 if outcome.adjudication.admissible else 1

    print("nothing to do: pass --model or --controls", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
