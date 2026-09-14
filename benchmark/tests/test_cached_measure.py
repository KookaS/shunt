"""Stage-4 cached-token measurement: extraction and the archived-message backfill.

Records the provider-reported cached-prompt-token data this corpus already paid for:
``infer._sum_cached_in_tokens`` (presence-strict extraction from a cell's message
list) and the archived-message backfill that repopulates ``cached_in_tok`` on rows
collected before the extractor existed. Consumption of the resulting per-model
pooled rate into the cache-aware axis is deliberately a SEPARATE change (it rewrites
figure captions and is done with its own regen pass); this file locks the data half.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from benchmark.routing.scripts.backfill_cached_tokens import (
    backfill_summary,
    cached_from_messages,
    scan_message_lists,
)
from benchmark.runner import infer

_HEADER: Final[tuple[str, ...]] = (
    "challenge_id",
    "model",
    "reasoning",
    "pass",
    "cost",
    "in_tok",
    "out_tok",
    "calls",
    "cached_in_tok",
)


def _write_results(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    p = tmp_path / "results.csv"
    import csv

    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _HEADER})
    return p


# ── extraction: presence-strict, never a fabricated zero ───────────────────────


def _usage_message(cached: int | None) -> dict:
    usage: dict = {"prompt_tokens": 100, "completion_tokens": 10}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"extra": {"response": {"usage": usage}}}


def test_sum_cached_all_present_sums() -> None:
    msgs = [_usage_message(300), _usage_message(0), _usage_message(700)]
    assert infer._sum_cached_in_tokens(msgs) == 1000


def test_sum_cached_reports_zero_as_measured_zero() -> None:
    # A provider that reports the field with 0 cached tokens measured a 0% hit on that
    # call — that is data, not absence.
    assert infer._sum_cached_in_tokens([_usage_message(0)]) == 0


def test_sum_cached_missing_field_on_any_usage_returns_none() -> None:
    msgs = [_usage_message(300), _usage_message(None), _usage_message(700)]
    assert infer._sum_cached_in_tokens(msgs) is None


def test_sum_cached_no_usage_returns_none() -> None:
    assert infer._sum_cached_in_tokens([{"role": "user", "content": "hi"}]) is None


# ── the backfill: dry-run and write both derive the same pooled rate ───────────


def _message_file(root: Path, cid: str, model: str, arm: str, cached: int) -> Path:
    p = root / f"{cid}__{model}__{arm}.json"
    p.write_text(json.dumps({"messages": [_usage_message(cached)]}))
    return p


def test_backfill_dry_run_and_write_agree(tmp_path: Path) -> None:
    root = tmp_path / "message_lists"
    root.mkdir()
    _message_file(root, "org__repo-1", "deepseek-v4-pro", "high", 970000)
    _message_file(root, "org__repo-2", "deepseek-v4-pro", "high", 980000)
    results = _write_results(
        tmp_path,
        [
            {
                "challenge_id": "org__repo-1",
                "model": "deepseek-v4-pro",
                "reasoning": "high",
                "in_tok": "1000000",
                "cached_in_tok": "",
            },
            {
                "challenge_id": "org__repo-2",
                "model": "deepseek-v4-pro",
                "reasoning": "high",
                "in_tok": "1000000",
                "cached_in_tok": "",
            },
        ],
    )
    dry = backfill_summary(results, root, write=False)
    assert dry["matched_rows"] == 2
    assert dry["cells_to_update"] == 2
    assert dry["pooled_rate"] == pytest.approx(0.975)
    written = backfill_summary(results, root, write=True)
    assert written["cells_written"] == 2
    import csv

    with results.open() as f:
        updated = {r["challenge_id"]: r["cached_in_tok"] for r in csv.DictReader(f)}
    assert updated == {"org__repo-1": "970000", "org__repo-2": "980000"}


def test_backfill_skips_unmatched_and_usage_free_archives(tmp_path: Path) -> None:
    root = tmp_path / "message_lists"
    root.mkdir()
    _message_file(root, "org__repo-1", "deepseek-v4-pro", "high", 500)
    (root / "org__repo-ghost__deepseek-v4-pro__high.json").write_text(json.dumps({"messages": []}))
    results = _write_results(
        tmp_path,
        [
            {
                "challenge_id": "org__repo-1",
                "model": "deepseek-v4-pro",
                "reasoning": "high",
                "in_tok": "1000",
                "cached_in_tok": "",
            },
        ],
    )
    summary = backfill_summary(results, root, write=False)
    assert summary["matched_rows"] == 1
    assert summary["row_misses"] == 1  # the ghost archive has no results row
    assert summary["no_usage_archives"] == 0  # unmatched archives are not counted blanked
    assert summary["cells_to_update"] == 1
    # The ghost row is not in results.csv, so only repo-1 updates; rate over repo-1 only.
    assert summary["pooled_rate"] == pytest.approx(0.5)


def test_cached_from_messages_and_scan_parse_names_with_inner_separator(tmp_path: Path) -> None:
    # challenge ids contain "__" (org__repo); rsplit("__", 2) must still parse.
    assert infer._sum_cached_in_tokens([_usage_message(42)]) == 42
    assert cached_from_messages([_usage_message(42)]) == 42
    root = tmp_path / "ml"
    root.mkdir()
    _message_file(root, "astropy__astropy-12907", "deepseek-v4-pro", "high", 11)
    scanned = scan_message_lists(root)
    assert ("astropy__astropy-12907", "deepseek-v4-pro", "high") in scanned
    assert scanned[("astropy__astropy-12907", "deepseek-v4-pro", "high")] == 11
