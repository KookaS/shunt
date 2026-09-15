#!/usr/bin/env python3
"""SH019: the listing `billing` entitlement and the observed results `channel` are coherent.

Four invariants, all pass/fail:

1. EVERY model row in the shipped registry, the non-shipped free overlay, and the `$0` smoke
   registry declares `billing` in {free, paid}. The census and the accounting walls read this
   field, so a row that omits it is a silent hole. The smoke registry is $0 by design, so
   invariant 2 does not apply to it — only its billing declaration is required.
2. An overlay row that declares `billing: free` still records the paid twin's REAL list price
   (never $0) with its `price_source` / `price_as_of` provenance — HARD RULE 2 (this gate does
   not replace SH018; SH018 scans the whole overlay).
3. On every committed results CSV: `channel == free` implies `real_cost == 0`, and
   `channel == paid` implies `real_cost > 0`. A blank/malformed `real_cost` is read as 0.0, the
   same write-time default `backfill_channel` uses.
4. Every observed `channel` is in {free, paid, blank} and every `channel_source` is in the
   declared `benchmark.routing.channel.SOURCES` vocabulary. A results CSV missing either
   column is malformed and FAILS rather than reading green on nothing.

The registry/overlay/smoke MODEL rows are the entitlement; the results `channel` column is the
observed per-row view. The two are allowed to disagree in the promo direction; the gate checks
each against its own rule. A run over no subjects at all refuses to report green, and the gate
is run from the repo root with no arguments.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml
from _shared import Finding

_CODE = "SH019"
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

REGISTRY: Path = Path("src/shunt/config/models.yaml")
OVERLAY: Path = Path("configs/free-tier/overlay.yaml")
SMOKE: Path = Path("configs/free-tier/models.yaml")
RESULTS: tuple[Path, ...] = (
    Path("benchmark/routing/results.csv"),
    Path("benchmark/routing/results_free.csv"),
)
_BILLING = frozenset({"free", "paid"})
_CHANNEL = frozenset({"free", "paid", ""})
_PRICE_KEYS = ("input_cost_per_1m", "output_cost_per_1m")
_REQUIRED_PROVENANCE = ("price_source", "price_as_of")


def _bootstrap() -> None:
    """Prepend the repo root and `src/` so `benchmark` resolves as a script."""
    # `benchmark` is deliberately NOT installed, and the editable install that provides `shunt`
    # lives in whichever interpreter pre-commit uses. Prepend both so the gate is hermetic over
    # the checkout, install-independent. This is the sanctioned mutation (SH003 is advisory).
    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
        if path not in sys.path:  # noqa: TID251 (banned-api read; tooling bootstrap)
            sys.path.insert(0, path)  # noqa: TID251, SH003 (tooling bootstrap; packages not installed)


def _source_vocab() -> frozenset[str]:
    """The declared `channel_source` vocabulary, read from its one definition."""
    _bootstrap()
    from benchmark.routing.channel import SOURCES  # noqa: PLC0415 (deferred until _bootstrap ran)

    return frozenset(SOURCES)


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


def _check_registry(path: Path, *, require_nonzero_free: bool) -> tuple[list[Finding], int]:
    """Invariant 1 for every model row, plus invariant 2 when `require_nonzero_free`.

    Returns the findings and the number of model rows actually read, so a missing file cannot
    masquerade as a clean one.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return [], 0
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        return [Finding(str(path), 1, 0, f"does not parse as YAML: {exc}")], 0
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, dict):
        return [], 0
    findings: list[Finding] = []
    for model, body in models.items():
        line = _model_line(source, str(model))
        row = body if isinstance(body, dict) else {}
        billing = row.get("billing")
        if billing not in _BILLING:
            findings.append(
                Finding(
                    str(path),
                    line,
                    4,
                    f"model '{model}' declares billing={billing!r}; every row must declare "
                    "billing in {free, paid}",
                )
            )
            continue
        if require_nonzero_free and billing == "free":
            findings.extend(_check_free_price(str(path), line, str(model), row))
    return findings, len(models)


def _check_free_price(path: str, line: int, model: str, row: dict[str, object]) -> list[Finding]:
    """Invariant 2: an overlay `billing: free` row keeps the paid twin's REAL list price."""
    pricing = row.get("pricing")
    if not isinstance(pricing, dict):
        return [Finding(path, line, 4, f"model '{model}' is billing:free with no pricing block")]
    findings: list[Finding] = []
    for key in _PRICE_KEYS:
        if _is_zero(pricing.get(key)):
            findings.append(
                Finding(
                    path,
                    line,
                    4,
                    f"model '{model}' is billing:free but declares {key} == 0; HARD RULE 2 "
                    "requires the paid twin's REAL list price",
                )
            )
    for key in _REQUIRED_PROVENANCE:
        if not str(pricing.get(key) or "").strip():
            findings.append(
                Finding(
                    path,
                    line,
                    4,
                    f"model '{model}' is billing:free but omits {key}",
                )
            )
    return findings


def _num(value: object) -> float:
    """Parse a CSV cell as a float; blank/non-numeric is 0.0 (the column's write-time default).

    This mirrors `backfill_channel._to_float` exactly: the observed channel was derived from a
    missing cost read as 0.0, so the gate must read the same cell the same way or it flags its
    own writer's output.
    """
    try:
        return float(str(value).strip()) if str(value).strip() else 0.0
    except ValueError:
        return 0.0


def _check_results(path: Path) -> tuple[list[Finding], int]:
    """Invariants 3-4: observed `channel` agrees with `real_cost` and is in vocabulary.

    A CSV missing `channel` or `channel_source` cannot be checked at all, so it is a hard
    failure: the guard `if "channel" in row` used to let a legacy/malformed file read green.
    """
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except OSError:
        return [], 0
    missing = [name for name in ("channel", "channel_source") if name not in columns]
    if missing:
        return (
            [
                Finding(
                    str(path),
                    1,
                    0,
                    f"results CSV omits required column(s) {missing}; a legacy/malformed file "
                    "cannot be checked and must not read green",
                )
            ],
            len(rows),
        )
    sources = _source_vocab()
    findings: list[Finding] = []
    for index, row in enumerate(rows, start=2):  # 1-based + header line
        label = f"row {row.get('challenge_id')}/{row.get('lane')}"
        observed = str(row.get("channel") or "").strip()
        if observed not in _CHANNEL:
            findings.append(
                Finding(
                    str(path),
                    index,
                    0,
                    f"{label} has channel={observed!r}; must be one of {sorted(_CHANNEL)}",
                )
            )
        raw_source = row.get("channel_source")
        if raw_source is not None and str(raw_source).strip() not in sources:
            findings.append(
                Finding(
                    str(path),
                    index,
                    0,
                    f"{label} has channel_source={str(raw_source).strip()!r}; must be one of "
                    f"{sorted(sources)}",
                )
            )
        real_cost = _num(row.get("real_cost"))
        if observed == "free" and real_cost != 0:
            findings.append(
                Finding(
                    str(path),
                    index,
                    0,
                    f"{label} has channel=free but real_cost={real_cost}",
                )
            )
        if observed == "paid" and real_cost <= 0:
            findings.append(
                Finding(
                    str(path),
                    index,
                    0,
                    f"{label} has channel=paid but real_cost={real_cost}",
                )
            )
    return findings, len(rows)


def check_all(
    *,
    registry: Path = REGISTRY,
    overlay: Path = OVERLAY,
    smoke: Path = SMOKE,
    results: tuple[Path, ...] = RESULTS,
) -> tuple[list[Finding], int]:
    """Every finding across the registries, the `$0` smoke registry and the results CSVs, plus
    a subject count so an empty run can refuse to report green."""
    findings: list[Finding] = []
    subjects = 0
    for path, require_nonzero_free in ((registry, False), (overlay, True), (smoke, False)):
        path_findings, count = _check_registry(path, require_nonzero_free=require_nonzero_free)
        findings.extend(path_findings)
        subjects += count
    for path in results:
        path_findings, count = _check_results(path)
        findings.extend(path_findings)
        subjects += count
    return findings, subjects


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--overlay", type=Path, default=OVERLAY)
    # The `$0` smoke registry is checked for its `billing` declaration only (invariant 1):
    # invariant 2 must not apply, since its real list price is genuinely $0 by design.
    parser.add_argument("--smoke", type=Path, default=SMOKE)
    parser.add_argument("--results", type=Path, action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report every finding and exit 1 when a billing/channel invariant is broken."""
    args = _arg_parser().parse_args(argv)
    results = tuple(args.results) if args.results else RESULTS
    findings, subjects = check_all(
        registry=args.registry, overlay=args.overlay, smoke=args.smoke, results=results
    )
    if subjects == 0:
        print(  # noqa: T201 - a lint gate reports on stderr
            f"[{_CODE}] no registry/overlay models or results rows to check — refusing to "
            "report green on nothing",
            file=sys.stderr,
        )
        return 1
    for finding in findings:
        print(  # noqa: T201 - a lint gate reports on stderr
            f"{finding.path}:{finding.line}:{finding.col}: [{_CODE} ERROR] {finding.message}",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
