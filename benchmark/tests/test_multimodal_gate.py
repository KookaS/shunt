"""The multimodal eligibility gate: Verified text coverage decides who may be scheduled.

A model that has not collected the whole Verified text corpus is refused a multimodal cell
by name at scheduling time — never run and left to fail inside the harness. The gate is a
pure function over the committed CSVs, so these tests use monkeypatched paths and fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import config, model_coverage
from benchmark.runner import run_matrix, swebench_multimodal_specs


def _write_specs(directory: Path, ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for cid in ids:
        (directory / f"{cid}.json").write_text(json.dumps({"instance_id": cid}))


def _write_results(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["challenge_id,model,reasoning,pass"]
    lines += [f"{cid},{model},high,True" for cid, model in rows]
    path.write_text("\n".join(lines) + "\n")


def _write_work_results(path: Path, rows: list[tuple[str, str, int, float]]) -> None:
    """Rows carrying the work columns `impute.is_zero_work` reads."""
    lines = ["challenge_id,model,reasoning,pass,calls,real_cost"]
    lines += [f"{cid},{model},high,True,{calls},{cost}" for cid, model, calls, cost in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A 4-challenge Verified store plus the two result CSVs the gate reads."""
    specs = tmp_path / "swebench_verified"
    _write_specs(specs, ["c1", "c2", "c3", "c4"])
    paid = tmp_path / "results.csv"
    free = tmp_path / "results_free.csv"
    monkeypatch.setattr(config, "challenge_dir", lambda source=None: specs)
    monkeypatch.setattr(config, "results_csv_path", lambda: paid)
    monkeypatch.setattr(config, "free_results_csv_path", lambda: free)
    return paid, free


def test_verified_coverage_counts_distinct_verified_ids_only(corpus: tuple[Path, Path]) -> None:
    paid, _ = corpus
    _write_results(
        paid,
        [
            ("c1", "full"),
            ("c2", "full"),
            ("c3", "full"),
            ("c4", "full"),
            ("c1", "full"),  # a second row for one challenge must not double-count
            ("c1", "near"),
            ("c2", "near"),
            ("c3", "near"),
            ("not-verified", "near"),  # outside the Verified store: never counts
        ],
    )

    assert model_coverage.verified_coverage("full") == 1.0
    assert model_coverage.verified_coverage("near") == 0.75
    assert model_coverage.verified_coverage("absent") == 0.0
    counts, total = model_coverage.verified_coverage_table()
    assert total == 4
    assert counts == {"full": 4, "near": 3}


def test_free_corpus_counts_and_a_missing_free_file_is_graceful(
    corpus: tuple[Path, Path],
) -> None:
    paid, free = corpus
    _write_results(paid, [("c1", "paid-only")])
    _write_results(
        free,
        [("c1", "free-only"), ("c2", "free-only"), ("c3", "free-only"), ("c4", "free-only")],
    )

    assert model_coverage.verified_coverage("free-only") == 1.0
    assert model_coverage.verified_coverage("paid-only") == 0.25

    free.unlink()  # a fresh checkout has no free corpus; that is not an error
    assert model_coverage.verified_coverage("free-only") == 0.0
    assert model_coverage.verified_coverage("paid-only") == 0.25


def test_zero_work_rows_do_not_count_as_coverage(corpus: tuple[Path, Path]) -> None:
    """Aborted-collection residue (calls=0, real_cost=0) never executed: not a real row."""
    paid, _ = corpus
    _write_work_results(
        paid,
        [
            ("c1", "residue", 0, 0.0),
            ("c2", "residue", 0, 0.0),
            ("c3", "residue", 0, 0.0),
            ("c4", "residue", 0, 0.0),
        ],
    )

    assert model_coverage.verified_coverage("residue") == 0.0
    assert model_coverage.multimodal_eligible("residue") is False
    counts, total = model_coverage.verified_coverage_table()
    assert "residue" not in counts
    assert total == 4  # the denominator stays the spec-file count


def test_a_model_with_real_rows_scores_normally(corpus: tuple[Path, Path]) -> None:
    """One real row per challenge scores full; a mix of zero-work and one real scores 0.25."""
    paid, _ = corpus
    _write_work_results(
        paid,
        [
            ("c1", "real", 7, 0.02),
            ("c2", "real", 7, 0.02),
            ("c3", "real", 7, 0.02),
            ("c4", "real", 7, 0.02),
            ("c1", "mixed", 0, 0.0),
            ("c2", "mixed", 5, 0.01),  # one real row
        ],
    )

    assert model_coverage.verified_coverage("real") == 1.0
    assert model_coverage.multimodal_eligible("real") is True
    assert model_coverage.verified_coverage("mixed") == 0.25
    assert model_coverage.multimodal_eligible("mixed") is False


def test_free_corpus_zero_work_rows_do_not_count_and_real_rows_do(
    corpus: tuple[Path, Path],
) -> None:
    """The exclusion applies to the free corpus too; real free-only rows still count."""
    paid, free = corpus
    _write_results(paid, [])
    _write_work_results(
        free,
        [
            ("c1", "free-residue", 0, 0.0),
            ("c2", "free-residue", 0, 0.0),
            ("c3", "free-real", 9, 0.0),
            ("c4", "free-real", 9, 0.0),
        ],
    )

    assert model_coverage.verified_coverage("free-residue") == 0.0
    assert model_coverage.verified_coverage("free-real") == 0.5
    assert model_coverage.multimodal_eligible("free-residue") is False


def test_multimodal_eligible_is_the_strict_full_corpus_reading(corpus: tuple[Path, Path]) -> None:
    paid, _ = corpus
    _write_results(
        paid,
        [
            ("c1", "all"),
            ("c2", "all"),
            ("c3", "all"),
            ("c4", "all"),
            ("c1", "most"),
            ("c2", "most"),
            ("c3", "most"),
        ],
    )

    assert model_coverage.multimodal_eligible("all") is True
    assert model_coverage.multimodal_eligible("most") is False


def test_collector_refuses_below_gate_and_permits_at_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_coverage, "verified_coverage_table", lambda: ({"done": 5, "short": 2}, 5)
    )

    allowed, refusals = run_matrix._apply_multimodal_gate(
        swebench_multimodal_specs.SOURCE, ["done", "short"]
    )

    assert allowed == ["done"]
    assert list(refusals) == ["short"]
    assert "2/5" in refusals["short"]
    assert "multimodal gate" in refusals["short"]


def test_collector_gate_is_a_noop_for_a_text_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden() -> tuple[dict[str, int], int]:
        raise AssertionError("the gate must not read coverage for a text source")

    monkeypatch.setattr(model_coverage, "verified_coverage_table", _forbidden)

    allowed, refusals = run_matrix._apply_multimodal_gate("swebench_verified", ["any", "model"])

    assert allowed == ["any", "model"]
    assert refusals == {}
