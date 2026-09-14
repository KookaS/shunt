#!/usr/bin/env python3
"""SH018: a non-shipped free-overlay row must record a REAL list price, never $0."""

# HARD RULE 2 of configs/free-tier/overlay.yaml: every row carries the paid twin's real
# list price. A $0 list price would make the row the pareto global minimum and rank 0 in
# any cost comparison. HARD RULE 3: no cache_read_cost_per_1m, because a shared gateway
# cache namespace makes a harvested cache rate an artefact of other tenants' traffic.
#
# SCOPE IS DELIBERATELY overlay.yaml ONLY. configs/free-tier/models.yaml is the separate,
# deliberate $0 smoke registry (OpenRouter :free) and must keep working — scanning it here
# would fail its first commit.

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from _shared import Finding

_CODE = "SH018"

# The overlay and only the overlay: pre-commit runs this with pass_filenames:false +
# always_run so an unstaged edit cannot smuggle a $0 price past the gate.
_OVERLAY_PATH = Path("configs/free-tier/overlay.yaml")
_PRICE_KEYS = ("input_cost_per_1m", "output_cost_per_1m")
_REQUIRED_PROVENANCE = ("price_source", "price_as_of")


def _is_zero(value: object) -> bool:
    """True iff *value* parses as exactly 0 — a missing/None value is not a $0 claim."""
    try:
        return float(value) == 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _model_line(source: str, model: str) -> int:
    """The 1-based line the model is declared on, for a clickable finding."""
    needle = f"  {model}:"
    for number, line in enumerate(source.splitlines(), start=1):
        if line.rstrip() == needle:
            return number
    return 1


def _check_pricing(path: str, source: str, model: str, pricing: object) -> list[Finding]:
    """Every HARD-RULE violation on one overlay row's pricing block."""
    line = _model_line(source, model)
    if not isinstance(pricing, dict):
        return [Finding(path, line, 2, f"model '{model}' has no pricing block")]
    findings: list[Finding] = []
    for key in _PRICE_KEYS:
        if _is_zero(pricing.get(key)):
            findings.append(
                Finding(
                    path,
                    line,
                    4,
                    f"model '{model}' declares {key} == 0; an overlay row must record the "
                    "paid twin's REAL list price (HARD RULE 2) — a $0 row becomes the pareto "
                    "global minimum",
                )
            )
    if "cache_read_cost_per_1m" in pricing:
        findings.append(
            Finding(
                path,
                line,
                4,
                f"model '{model}' declares cache_read_cost_per_1m; gateway cache namespaces "
                "can be shared across tenants, so a harvested cache rate would look measured "
                "without being so (HARD RULE 3)",
            )
        )
    for key in _REQUIRED_PROVENANCE:
        if not str(pricing.get(key) or "").strip():
            findings.append(
                Finding(
                    path,
                    line,
                    4,
                    f"model '{model}' omits {key}; every list price needs its provenance",
                )
            )
    return findings


def check(path: str) -> list[Finding]:
    """Every overlay row under *path* that violates a HARD RULE."""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        return [Finding(path, 1, 0, f"overlay does not parse as YAML: {exc}")]
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, dict):
        return []
    findings: list[Finding] = []
    for model, body in models.items():
        pricing = body.get("pricing") if isinstance(body, dict) else None
        findings.extend(_check_pricing(path, source, str(model), pricing))
    return findings


def _default_paths() -> list[str]:
    """The single scan target: the non-shipped overlay, never the $0 smoke registry."""
    return [str(_OVERLAY_PATH)]


def main(argv: list[str]) -> int:
    """Report every finding and exit 1 when an overlay row misstates its price."""
    paths = [a for a in argv if not a.startswith("-")] or _default_paths()
    findings = [f for path in paths for f in check(path)]
    for finding in findings:
        print(  # noqa: T201 - a lint gate reports on stderr
            f"{finding.path}:{finding.line}:{finding.col}: [{_CODE} ERROR] {finding.message}",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
