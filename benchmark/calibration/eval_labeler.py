"""CLI: score the automatic labeler against the owner-labeled calibration set.

Joins owner labels (session_id -> good|bad) with the store's verified verdicts and prints
the metrics; runs with a clear message even before the owner has labeled anything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import yaml

from benchmark.calibration.labeler_metrics import compute_metrics, format_metrics

_VALID = frozenset({"good", "bad"})
_DEFAULT_LABELS = Path(__file__).parent / "labels.yaml"
_TEMPLATE = Path(__file__).parent / "labels.template.yaml"


class _StoreLike(Protocol):
    def get_outcome(self, session_id: str) -> dict[str, Any] | None: ...


def load_owner_labels(path: Path) -> dict[str, str]:
    """Parse the owner label YAML; keep only entries with a valid good/bad value."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    labels = raw.get("labels") if isinstance(raw, dict) else None
    if not isinstance(labels, dict):
        return {}
    return {str(k): v for k, v in labels.items() if isinstance(v, str) and v in _VALID}


def auto_labels_from_store(store: _StoreLike, session_ids: list[str]) -> dict[str, str]:
    """The automatic labeler's verdict per session: verified 'success' -> good, else bad."""
    out: dict[str, str] = {}
    for session_id in session_ids:
        outcome = store.get_outcome(session_id)
        if outcome is None or outcome.get("tier2_outcome") is None:
            continue  # no verified label yet — nothing to compare
        out[session_id] = "good" if outcome["tier2_outcome"] == "success" else "bad"
    return out


def _empty_template_message(path: Path) -> str:
    return (
        f"No owner labels found in {path}.\n"
        f"Copy the template and fill it in:\n"
        f"  cp {_TEMPLATE} {_DEFAULT_LABELS}\n"
        f"then map at least 30 session_ids to 'good' or 'bad' (session_id only — no "
        f"prompts). Re-run this script to score the automatic labeler."
    )


def run(store: _StoreLike | None, labels_path: Path) -> tuple[int, str]:
    """Compute and format the labeler report. Returns (exit_code, text)."""
    owner = load_owner_labels(labels_path)
    if not owner:
        return (0, _empty_template_message(labels_path))
    if store is None:
        return (0, "Owner labels loaded but no outcome store available to score against.")
    auto = auto_labels_from_store(store, list(owner))
    if not auto:
        return (
            0,
            f"{len(owner)} owner labels loaded, but none of those sessions carry a "
            f"verified automatic label yet — nothing to compare.",
        )
    metrics = compute_metrics(owner, auto)
    text = format_metrics(metrics)
    if metrics.confusion.fp + metrics.confusion.tn == 0:
        # No owner-'bad' items were compared, so the FPR denominator is empty and the
        # false-positive bound cannot bite — the calibration set must include bad examples.
        text += (
            "\nWARNING: no owner-'bad' sessions were compared, so FPR is undefined and "
            "the pre-registered bound cannot flag a permissive labeler. Add owner-'bad' "
            "examples to the calibration set."
        )
    # Exit non-zero when flagged so the owner/CI notices a labeler that is not trustworthy.
    return (1 if metrics.flagged else 0, text)


def main() -> None:
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_LABELS
    store: _StoreLike | None
    try:
        from shunt.db.store import OutcomeStore

        store = OutcomeStore()
    except Exception as exc:  # noqa: BLE001 (a report tool must degrade, not crash)
        print(f"(outcome store unavailable: {exc})")
        store = None
    exit_code, text = run(store, labels_path)
    print(text)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
