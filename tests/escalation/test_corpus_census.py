"""The escalation corpus census is DERIVED, never hardcoded — the gate keeps it that way.

Proves `census()` reports the live corpus, moves when a file is planted, and that no committed
`.py`/`.md` under `benchmark/` or `docs/` states a stale corpus count.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from benchmark.escalation.corpus import LIVE_DIR, census

_ROOT = Path(__file__).resolve().parents[2]

# Stale corpus counts that used to be hardcoded across ~21 sites. A committed doc or string
# stating either one is a disagreement with the live corpus and must fail the gate.
_STALE_COUNTS = (
    "799",
    "29,422",
    "29 422",
    "29422",
    # Superseded 2026-09-04 when the 200 deepseek-v4-pro trajectories landed.
    "822",
    "30,541",
    "30 541",
    "30541",
)


def test_census_returns_the_live_corpus_counts() -> None:
    # The committed corpus, counted from disk at this commit. Asserting the tuple pins the
    # gate: when the corpus grows, this fails until the docs are updated to match.
    got = census()
    assert got.trajectories == 1022
    assert got.steps == 38_211
    # The manifest is the committed ledger; the census and the manifest must agree, or one of
    # them has rotted.
    manifest = json.loads((LIVE_DIR / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["trajectories"]
    assert got.trajectories == len(entries)
    assert got.steps == sum(v["n_steps"] for v in entries.values())


def test_census_moves_when_a_file_is_planted(tmp_path: Path) -> None:
    # The census must report the new count when the corpus grows. Copy a real trajectory
    # into a temp dir, then plant an extra copy — the census must count both.
    real = next(LIVE_DIR.glob("*.jsonl"))
    shutil.copy(real, tmp_path / "one.jsonl")
    one = census(tmp_path)
    shutil.copy(real, tmp_path / "two.jsonl")
    planted = census(tmp_path)
    assert planted.trajectories == one.trajectories + 1
    assert planted.steps == one.steps * 2  # two identical files, twice the steps


def test_census_zero_length_file_contributes_zero_trajectories(tmp_path: Path) -> None:
    # A 0-byte .jsonl used to count as trajectories=1 (the counter incremented before the
    # no-records guard). A file with no parseable records is not a trajectory.
    (tmp_path / "empty.jsonl").write_text("")
    got = census(tmp_path)
    assert got.trajectories == 0
    assert got.steps == 0


def test_census_corrupt_first_line_is_skipped_with_a_warning(tmp_path: Path) -> None:
    # A first line that is not valid JSON used to crash the census with a JSONDecodeError.
    # A corrupt committed file should not blow up the count — it is skipped with a warning
    # and contributes 0 trajectories / 0 steps.
    (tmp_path / "corrupt.jsonl").write_text("not-json-line\n")
    with pytest.warns(UserWarning, match="not valid JSON"):
        got = census(tmp_path)
    assert got.trajectories == 0
    assert got.steps == 0


def test_census_non_object_header_is_skipped_with_a_warning(tmp_path: Path) -> None:
    # A header that parses as JSON but is not an object (e.g. a bare number) would crash on
    # `.get`; it is skipped like any other corrupt file.
    (tmp_path / "notobject.jsonl").write_text('"just a string"\n')
    with pytest.warns(UserWarning, match="not a JSON object"):
        got = census(tmp_path)
    assert got.trajectories == 0
    assert got.steps == 0


@pytest.mark.parametrize("stale", _STALE_COUNTS)
def test_no_committed_source_states_a_stale_corpus_count(stale: str) -> None:
    # No committed .py/.md under benchmark/ or docs/ may hardcode the old corpus count.
    # This is the machine form of the story's grep gate — a stale literal is a disagreement
    # with the live corpus, and it fails here before it can be committed.
    #
    # MATCHED ON A NUMBER BOUNDARY, NOT AS A RAW SUBSTRING (fixed 2026-08-21). The plain
    # `stale in text` form fired on any longer number that happened to contain the digits:
    # a measured inter-arrival gap of `47994.854` and a measured cost of `0.0033831679999…`
    # both contain "799" and neither states a corpus count. That is a false positive on
    # MEASURED DATA, and the only two ways out of it are to fix the matcher or to falsify the
    # data — so the matcher is fixed here. `\b` still catches every real usage (`799`,
    # `n=799`, `799 trajectories`, `(799)`) because a corpus count is always its own token.
    pattern = re.compile(rf"(?<!\d){re.escape(stale)}(?!\d)")
    offenders: list[str] = []
    for tree in (_ROOT / "benchmark", _ROOT / "docs"):
        for path in tree.rglob("*"):
            if path.suffix not in (".py", ".md"):
                continue
            if "data/live" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, (
        f"{stale!r} still appears in committed sources (would rot when the corpus grows): "
        f"{offenders}"
    )
