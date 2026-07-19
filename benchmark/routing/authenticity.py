"""Layer-1 authenticity for results.csv: recompute every derivable field and
cross-check each raw row, so corruption and naive fabrication fail CI. Cannot catch
a forger reproducing every invariant (needs Layer 2 signing / Layer 3 re-exec).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from benchmark import config
from benchmark.routing import integrity

# Exact tolerance for the pricing-INDEPENDENT `cost` derivation check (cost == its rule).
_COST_REL_TOL: Final[float] = 1e-6
_COST_ABS_TOL: Final[float] = 1e-9

# real_cost plausibility vs the token-based estimate. real_cost is cache-aware so it runs
# BELOW the list-price estimate; observed ratios across all models AND reasoning arms are
# 0.042–0.685 (recalibrated on the first arm-sweep live run — high-reasoning arms `think`/
# `high`/`max` do NOT breach the band; max 0.685 sits well under the 1.5 WARN-high). Only
# the DOWNWARD egregious case is a hard ERROR — a real cost far below the estimate means an
# expensive run billed as ~free (the kill-gate attack). Observed ratios bottom at 0.042; the
# theoretical floor for a 100%-cache-read run is ~0.02 (cache_read/input), so the 0.005 ERR
# floor keeps ~4x margin below even that extreme. The floor is robust to price edits — the
# band anchors on the row's stored estimated_cost, not a live recompute. The UPWARD
# direction stays WARN not ERROR: a provider that bills reasoning tokens outside
# completion_tokens could still push a LEGIT ratio high — not worth a false CI failure.
_REAL_COST_ERR_LOW: Final[float] = 0.005  # < this ⇒ ERROR: "expensive run billed as ~free"
_REAL_COST_WARN_LOW: Final[float] = 0.02
_REAL_COST_WARN_HIGH: Final[float] = 1.5

# Advisory upper bounds — plausible-but-suspicious magnitudes (WARN, never gate-failing).
_MAX_TOKENS: Final[int] = 100_000_000
_MAX_COST: Final[float] = 10_000.0

# A future timestamp beyond this clock-skew allowance is fabricated, not just early.
_CLOCK_SKEW_S: Final[float] = 300.0

# Severity: ERROR fails the gate; WARN is reported but advisory (plausible-but-odd).
ERROR: Final[str] = "ERROR"
WARN: Final[str] = "WARN"

# Outcome + cost columns every real row carries (the writer always emits them, and the
# committed legacy file has them). Requiring them means DROPPING a cost column can't
# silently disable the cost checks (check_cost_columns skips a None field). Only the
# per-arm anchor (arm_hash) and image_digest were added later and may be absent — those
# are validated when present / when required, not here.
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "challenge_id",
    "model",
    "pass",
    "cost",
    "in_tok",
    "out_tok",
    "calls",
    "real_cost",
    "estimated_cost",
)

# Strict numeric grammar — reject underscores ("1_000"), unicode digits, signs, and
# surrounding whitespace that Python int()/float() would silently accept, so the
# validator can never "see" a different value than a downstream CSV/pandas/Go reader.
_INT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]+$")
_FLOAT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
_IMAGE_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_BOOL_LITERALS: Final[frozenset[str]] = frozenset({"True", "False"})


@dataclass(frozen=True)
class Finding:
    """One authenticity violation on a results row (or the file as a whole)."""

    severity: str
    rule: str
    key: str
    detail: str


def _key(row: dict[str, str], default_arms: dict[str, str] | None = None) -> str:
    """Canonical cache key (challenge:model:arm), resolving the ``default`` alias to the
    model's real default_arm id so the two spellings name the same cell (dup-evasion).
    """
    cid = row.get("challenge_id", "?")
    model = row.get("model", "?")
    reasoning = row.get("reasoning") or integrity.DEFAULT_REASONING
    if reasoning == integrity.DEFAULT_REASONING and default_arms:
        reasoning = default_arms.get(model, integrity.DEFAULT_REASONING)
    return f"{cid}:{model}:{reasoning}"


def _as_int(value: str) -> int | None:
    """Parse a non-negative int under a strict grammar, or None."""
    if not _INT_RE.match(value or ""):
        return None
    return int(value)


def _as_float(value: str) -> float | None:
    """Parse a finite non-negative float under a strict grammar, or None."""
    if not _FLOAT_RE.match(value or ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def check_schema(row: dict[str, str]) -> list[Finding]:
    """Required columns present; pass/timeout_flag are bool literals; counts/costs parse."""
    out: list[Finding] = []
    missing = [c for c in _REQUIRED_COLUMNS if c not in row]
    if missing:
        return [Finding(ERROR, "schema.missing_columns", _key(row), f"missing {missing}")]
    if row["pass"] not in _BOOL_LITERALS:
        out.append(Finding(ERROR, "schema.pass_not_bool", _key(row), f"pass={row['pass']!r}"))
    if row.get("timeout_flag", "") not in _BOOL_LITERALS | {""}:
        out.append(Finding(ERROR, "schema.timeout_not_bool", _key(row), f"{row['timeout_flag']!r}"))
    for col in ("in_tok", "out_tok", "calls"):
        if _as_int(row[col]) is None:
            out.append(Finding(ERROR, "schema.bad_int", _key(row), f"{col}={row[col]!r}"))
    for col in ("cost", "real_cost", "estimated_cost"):
        if col in row and _as_float(row[col]) is None:
            out.append(Finding(ERROR, "schema.bad_float", _key(row), f"{col}={row[col]!r}"))
    return out


def check_registered(row: dict[str, str], specs: dict[str, str], resolved: dict) -> list[Finding]:
    """Challenge is a materialised spec; model is registered; reasoning is a known arm."""
    out: list[Finding] = []
    cid, model = row.get("challenge_id", ""), row.get("model", "")
    if cid not in specs:
        out.append(Finding(ERROR, "registered.unknown_challenge", _key(row), f"{cid!r}"))
    mc = resolved.get(model)
    if mc is None:
        out.append(Finding(ERROR, "registered.unknown_model", _key(row), f"{model!r}"))
        return out
    reasoning = row.get("reasoning") or integrity.DEFAULT_REASONING
    if reasoning not in _valid_arms(mc):
        out.append(Finding(ERROR, "registered.unknown_arm", _key(row), f"{model}:{reasoning}"))
    return out


def _valid_arms(mc: object) -> set[str]:
    """The arm ids acceptable for a model: its declared arms plus the ``default`` alias."""
    arms = {integrity.DEFAULT_REASONING}
    reasoning = getattr(mc, "reasoning", None)
    if reasoning is not None:
        arms |= {a.id for a in reasoning.arms}
    return arms


def check_anchors(
    row: dict[str, str], specs: dict[str, str], resolved: dict, versions: dict[str, str]
) -> list[Finding]:
    """Identity anchors: version_hash + model_version present and matching; arm_hash
    present for explicit arms and matching; image_digest well-formed. Blanking an
    anchor no longer skips it.
    """
    out: list[Finding] = []
    stored_vh = row.get("version_hash", "")
    if not stored_vh:
        out.append(Finding(ERROR, "anchor.version_missing", _key(row), "empty version_hash"))
    else:
        current = specs.get(row.get("challenge_id", ""))
        if current and stored_vh != current:
            out.append(
                Finding(
                    ERROR, "anchor.version_mismatch", _key(row), f"{stored_vh[:12]}≠{current[:12]}"
                )
            )
    out.extend(_check_model_version_anchor(row, versions))
    out.extend(_check_arm_anchor(row, resolved))
    digest = row.get("image_digest", "")
    if digest and not _IMAGE_DIGEST_RE.match(digest):
        out.append(Finding(ERROR, "anchor.bad_image_digest", _key(row), f"{digest!r}"))
    return out


def _check_model_version_anchor(row: dict[str, str], versions: dict[str, str]) -> list[Finding]:
    """A row's model_version must equal the registry's CURRENT version for its model."""
    # New-id-per-version convention: each model id carries one current version, so a
    # stored version that differs is a stale (re-run-signal) or fabricated row. Mismatch
    # is checked only when a current version is known — unregistered models are already
    # flagged by check_registered.
    model = row.get("model", "")
    if model not in versions:
        return []  # unregistered model — check_registered owns it; don't double-flag
    stored = row.get("model_version", "")
    if not stored:
        return [Finding(ERROR, "anchor.model_version_missing", _key(row), "empty model_version")]
    current = versions[model]
    if current and stored != current:
        return [Finding(ERROR, "anchor.model_version_mismatch", _key(row), f"{stored}≠{current}")]
    return []


def _check_arm_anchor(row: dict[str, str], resolved: dict) -> list[Finding]:
    """An EXPLICIT (non-default) reasoning arm must carry a matching arm_hash."""
    model = row.get("model", "")
    mc = resolved.get(model)
    reasoning = row.get("reasoning") or integrity.DEFAULT_REASONING
    if mc is None or mc.reasoning is None or reasoning == integrity.DEFAULT_REASONING:
        return []  # legacy/default rows predate arm_hash; tolerated
    if reasoning not in {a.id for a in mc.reasoning.arms}:
        return []  # unknown-arm already flagged by check_registered
    stored = row.get("arm_hash", "")
    if not stored:
        return [
            Finding(ERROR, "anchor.arm_missing", _key(row), f"{model}:{reasoning} empty arm_hash")
        ]
    expected = integrity.arm_hash_value(mc, reasoning)
    if stored != expected:
        return [Finding(ERROR, "anchor.arm_mismatch", _key(row), f"{stored[:12]}≠{expected[:12]}")]
    return []


def check_cost_columns(row: dict[str, str]) -> list[Finding]:
    """`cost` matches its derivation rule and real_cost sits in a plausible band around
    the row's FROZEN estimated_cost — both checks independent of live pricing.
    """
    real_cost, est, cost = (
        _as_float(row.get("real_cost", "")),
        _as_float(row.get("estimated_cost", "")),
        _as_float(row.get("cost", "")),
    )
    if real_cost is None or est is None or cost is None:
        return []  # schema check already flagged the malformed field
    out: list[Finding] = []
    expected_cost = real_cost if real_cost > 0 else est
    if not math.isclose(cost, expected_cost, rel_tol=_COST_REL_TOL, abs_tol=_COST_ABS_TOL):
        out.append(Finding(ERROR, "cost.derivation_mismatch", _key(row), f"{cost}≠{expected_cost}"))
    if real_cost > 0:
        out.extend(_check_real_cost_band(row, real_cost, est))
    return out


def _check_real_cost_band(row: dict[str, str], real_cost: float, est: float) -> list[Finding]:
    """real_cost / estimated_cost must sit in a plausible band (cache-discount aware).

    Anchored to the row's FROZEN estimated_cost, not a live-pricing recompute, so a later
    price correction can't retroactively fail CI on rows authentic when written.
    """
    if est <= 0:
        return []  # unpriced/zero estimate — no band to check
    ratio = real_cost / est
    if ratio < _REAL_COST_ERR_LOW:
        return [Finding(ERROR, "cost.real_cost_implausible", _key(row), f"real/est={ratio:.4g}")]
    if ratio < _REAL_COST_WARN_LOW or ratio > _REAL_COST_WARN_HIGH:
        return [Finding(WARN, "cost.real_cost_unusual", _key(row), f"real/est={ratio:.4g}")]
    return []


def check_pass_plausibility(row: dict[str, str]) -> list[Finding]:
    """A resolved cell must have emitted output and must not also be a timeout."""
    if row.get("pass") != "True":
        return []
    out: list[Finding] = []
    calls, out_tok = _as_int(row.get("calls", "")), _as_int(row.get("out_tok", ""))
    if calls is not None and out_tok is not None and (calls == 0 or out_tok == 0):
        out.append(
            Finding(
                ERROR, "pass.no_output", _key(row), f"pass with calls={calls} out_tok={out_tok}"
            )
        )
    if row.get("timeout_flag") == "True":
        out.append(Finding(ERROR, "pass.timeout_contradiction", _key(row), "pass=True + timeout"))
    return out


def check_bounds(row: dict[str, str]) -> list[Finding]:
    """Advisory (WARN) upper bounds on token counts and cost — implausible but not proof."""
    out: list[Finding] = []
    for col in ("in_tok", "out_tok"):
        n = _as_int(row.get(col, ""))
        if n is not None and n > _MAX_TOKENS:
            out.append(Finding(WARN, "bounds.tokens_huge", _key(row), f"{col}={n}"))
    for col in ("cost", "real_cost"):
        c = _as_float(row.get(col, ""))
        if c is not None and c > _MAX_COST:
            out.append(Finding(WARN, "bounds.cost_huge", _key(row), f"{col}={c}"))
    return out


def check_timestamp(row: dict[str, str], now: datetime) -> list[Finding]:
    """computed_at parses as ISO-8601 and is not in the future (beyond clock skew)."""
    raw = row.get("computed_at", "")
    if not raw:
        return []
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return [Finding(ERROR, "time.unparseable", _key(row), f"computed_at={raw!r}")]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if ts.timestamp() > now.timestamp() + _CLOCK_SKEW_S:
        return [Finding(ERROR, "time.future", _key(row), f"computed_at={raw} > now")]
    return []


def check_duplicate_keys(rows: list[dict[str, str]], default_arms: dict[str, str]) -> list[Finding]:
    """No two rows resolve to the same (challenge, model, arm) cache key."""
    counts = Counter(_key(r, default_arms) for r in rows)
    return [
        Finding(ERROR, "file.duplicate_key", key, f"{n} rows share this key")
        for key, n in sorted(counts.items())
        if n > 1
    ]


def verify_rows(rows: list[dict[str, str]], now: datetime | None = None) -> list[Finding]:
    """Run every Layer-1 authenticity check over the raw results rows."""
    reference = now if now is not None else datetime.now(UTC)
    specs = integrity.swebench_spec_hashes()
    resolved = config.resolved_models()
    versions = integrity.model_versions()
    default_arms = config.default_arm_ids()
    findings: list[Finding] = check_duplicate_keys(rows, default_arms)
    for row in rows:
        schema = check_schema(row)
        findings.extend(schema)
        if any(f.severity == ERROR for f in schema):
            continue  # a malformed row can't be meaningfully cross-checked
        findings.extend(check_registered(row, specs, resolved))
        findings.extend(check_anchors(row, specs, resolved, versions))
        findings.extend(check_cost_columns(row))
        findings.extend(check_pass_plausibility(row))
        findings.extend(check_bounds(row))
        findings.extend(check_timestamp(row, reference))
    return findings


def errors(findings: list[Finding]) -> list[Finding]:
    """Only the gate-failing (ERROR) findings."""
    return [f for f in findings if f.severity == ERROR]


def warnings(findings: list[Finding]) -> list[Finding]:
    """Only the advisory (WARN) findings."""
    return [f for f in findings if f.severity == WARN]
