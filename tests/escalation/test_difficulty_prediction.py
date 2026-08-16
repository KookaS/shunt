"""The predicted-difficulty go/no-go: stage order, the shared gate, and derived verdicts.

Each stage's acceptance is enforced: stage-order, gate (no Stage-2 number before the shared
admissibility gate clears), derived-verdict, real-only — plus the R0 positive+shuffled-null pair.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmark.admissibility import admissibility_verdict
from benchmark.escalation import difficulty_prediction as dp
from benchmark.escalation import schema
from benchmark.escalation.difficulty_prediction import (
    Challenge,
    LabelRecord,
    Stage1Record,
)
from tests.escalation.factories import make_step, make_trajectory
from tests.fake_embedder import FakeEmbedder

_REPOS = tuple(f"repo-{i}" for i in range(12))


def _challenges(n: int = 48) -> list[Challenge]:
    return [
        Challenge(
            instance_id=f"challenge-{i}",
            difficulty=dp._CLASSES[i % 3],
            repo=_REPOS[i % len(_REPOS)],
            # The difficulty's marker is prepended, so the MarkerAwareEmbedder (and only it)
            # recovers annotated difficulty from the text — the hermetic analogue of the real
            # embedder recovering the planted marker in the positive control.
            problem_statement=(
                f"{dp.PLANTED_MARKERS[dp._CLASSES[i % 3]][0]}\n"
                f"Instance {i} description with some code.\n"
                f"```python\nprint({i})\n```\n"
                f"see tests/test_mod_{i}.py"
            ),
        )
        for i in range(n)
    ]


def _labels(n: int = 18, *, label: bool | None = None) -> list[LabelRecord]:
    return [
        LabelRecord(
            instance_id=f"challenge-{i}",
            difficulty=dp._CLASSES[i % 3],
            repo=_REPOS[i % len(_REPOS)],
            problem_statement=f"statement {i}",
            label=(i % 3 == 2) if label is None else label,
        )
        for i in range(n)
    ]


def _plain_challenges(n: int = 96) -> list[Challenge]:
    """Challenges whose statements carry NO marker, so the real difficulty is not learnable.

    With MarkerAwareEmbedder the gate clears (its planted corpus gets a marker) while Stage 1
    cannot recover annotated difficulty — the pre-registered stage-1 stop.
    """
    return [
        Challenge(
            instance_id=f"challenge-{i}",
            difficulty=dp._CLASSES[i % 3],
            repo=_REPOS[i % len(_REPOS)],
            problem_statement=f"Plain instance {i} body, no marker.",
        )
        for i in range(n)
    ]


class MarkerAwareEmbedder(FakeEmbedder):
    """A hermetic embedder on which the PLANTED marker is learnable.

    The stock FakeEmbedder yields a random vector per text; this subclass folds the marker
    class into a few dimensions so a linear head recovers it on unit-scale fixtures. Tests only.
    """

    def __init__(self, *, dim: int = 64) -> None:
        super().__init__(dim=dim)

    def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        vectors = list(super().embed_batch(texts))
        for vector, text in zip(vectors, texts, strict=True):
            for index, (marker, _reps) in enumerate(dp.PLANTED_MARKERS.values()):
                if marker in text:
                    vector[index * 8] += 6.0
        return vectors


def _write_traj(live_dir: Path, instance_id: str, model: str, arm: str, resolved: bool) -> None:
    traj = make_trajectory(
        [make_step(step_index=0, success=resolved)],
        trajectory_id=f"{instance_id}__{model}__{arm}",
        terminal_resolved=resolved,
    )
    schema.dump_jsonl(traj, live_dir / f"{instance_id}__{model}__{arm}.jsonl")


# ── Stage 0: free surface features, no model ──


def test_surface_features_are_exactly_the_four_defined_ones() -> None:
    statement = (
        "A failing test with a traceback:\n"
        "```python\nTraceback (most recent call last):\n```\n"
        "see lib/parse.py and tests/test_run.py"
    )
    features = dp.surface_features(statement)
    assert features.length == len(statement)
    assert features.code_blocks == 1
    assert features.has_traceback is True
    assert features.files_mentioned >= 2


def test_stage0_reports_scores_even_when_null() -> None:
    # Labels with no systematic link to any surface feature: the record must still carry every
    # surface score against BOTH outcomes and report no stage-0 go — a null Stage 0 is a
    # result, not an absence.
    labels = _labels(n=18)
    record = dp.stage0(_challenges(), labels)
    assert record.n_instances == 18
    assert record.n_positives == 6
    assert record.piv3_clear is False
    assert len(record.scores) == 8  # 4 features x {learning_to_defer, annotated_difficulty}
    for score in record.scores:
        assert score.outcome in (dp.STAGE0_OUTCOME_LABEL, dp.STAGE0_OUTCOME_ANNOTATED)
        assert all(math.isfinite(v) for v in score.boot_ci)
    assert sum(s.outcome == dp.STAGE0_OUTCOME_LABEL for s in record.scores) == 4
    assert sum(s.outcome == dp.STAGE0_OUTCOME_ANNOTATED for s in record.scores) == 4
    emitted = record.to_dict()
    assert emitted["n_instances"] == 18
    assert len(emitted["scores"]) == 8


# ── Stage order and gate gating ──


def test_stage1_refuses_without_a_stage0_record() -> None:
    with pytest.raises(dp.Stage0MissingError):
        dp.stage1(MarkerAwareEmbedder(), _challenges(), None)


def test_no_stage2_number_without_an_admissible_gate_record() -> None:
    inadmissible = admissibility_verdict(
        0.51,
        0.5,
        chance_level=0.5,
        chance_band=0.02,  # positive at chance
    )
    record = Stage1Record(
        predictions={f"challenge-{i}": i % 3 for i in range(18)},
        expected={f"challenge-{i}": float(i % 3) for i in range(18)},
        recovery_rho=0.0,
        recovery_perm_p=1.0,
        beats_null=False,
        n_challenges=18,
    )
    with pytest.raises(dp.Stage2NotAllowedError):
        dp.stage2(record, _labels(), inadmissible)


def test_a_whole_pipeline_that_cannot_learn_its_control_reports_coverage_gap(
    tmp_path: Path,
) -> None:
    # The stock FakeEmbedder cannot recover the planted marker (random vectors), so the
    # assembled instrument fails its positive control and run_pipeline must emit COVERAGE-GAP
    # with NO auroc anywhere — a coverage gap, never a falsification.
    payload = dp.run_pipeline(tmp_path / "out.json", FakeEmbedder(), _challenges(), _labels())
    assert payload["decision_rule_outcome"] == "COVERAGE-GAP"
    assert payload["pivot"] == "INSTRUMENT_FAILED"
    assert payload["stage2"] is None
    # Stage 0 legitimately reports surface-feature AUROCs; the STAGE-2 number must be absent.
    assert "auroc" not in json.dumps(payload.get("stage2"))
    assert payload["admissibility"]["admissible"] is False


# ── the instrument-validity gate (R0) — the SHARED adjudicator ──


def test_the_instrument_clears_the_shared_admissibility_gate() -> None:
    # Positive: with a marker the head can learn, the assembled pipeline recovers the planted
    # label. Destroyed-signal: shuffle the label, the same scores collapse to chance. The
    # adjudication is `benchmark.admissibility` — the pinned shared gate, not a re-derivation.
    gate = dp.admissibility_gate(MarkerAwareEmbedder(), _challenges(96))
    assert gate.positive_passed
    assert gate.null_at_chance
    assert gate.admissible


def test_the_positive_control_score_is_well_above_chance() -> None:
    gate = dp.admissibility_gate(MarkerAwareEmbedder(), _challenges(96))
    assert gate.positive_score > 0.65  # clears the empirical chance band by a wide margin
    assert gate.positive_score > 0.5 + gate.chance_band


# ── the derived decision rule ──


def test_decision_rule_is_derived_from_numbers_not_hand_written() -> None:
    cases = [
        (0.70, (0.56, 0.85), "PROCEED"),  # >= 0.65, interval excludes 0.5
        (0.66, (0.51, 0.82), "PROCEED"),  # just above 0.65, still excludes 0.5
        (0.65, (0.52, 0.80), "PROCEED"),  # exactly the validates threshold
        (0.55, (0.40, 0.70), "CLOSE"),  # < 0.60 (and interval spans)
        (0.72, (0.48, 0.90), "CLOSE"),  # >= 0.65 but interval spans 0.5
        (0.62, (0.45, 0.78), "CLOSE"),  # interval spans 0.5
        (0.62, (0.51, 0.73), "UNDERPOWERED"),  # in 0.60-0.65, interval excludes 0.5
        (0.60, (0.60, 0.80), "UNDERPOWERED"),  # exactly 0.60 is not < 0.60
    ]
    for auroc, ci, expected in cases:
        assert dp.decision_rule(auroc, ci) == expected, (auroc, ci)


def test_stage2_emitted_json_carries_the_required_fields_and_a_derived_outcome() -> None:
    admissible = admissibility_verdict(0.8, 0.5, chance_level=0.5, chance_band=0.02)
    labels = _labels(n=18)
    # Predicted difficulty ordered so the label correlates: positive labels get hard (2).
    predictions = {f"challenge-{i}": (2 if i < 6 else 0) for i in range(18)}
    record = Stage1Record(
        predictions=predictions,
        expected={k: float(v) for k, v in predictions.items()},
        recovery_rho=0.5,
        recovery_perm_p=0.001,
        beats_null=True,
        n_challenges=18,
    )
    result = dp.stage2(record, labels, admissible)
    emitted = result.to_dict()
    for key in ("auroc", "perm_p", "boot_ci", "n", "n_positives", "decision_rule_outcome"):
        assert key in emitted, key
    assert emitted["decision_rule_outcome"] == dp.decision_rule(result.auroc, result.boot_ci)
    assert emitted["n"] == 18


def test_a_pipeline_with_an_admissible_gate_and_learnable_head_reaches_stage2(
    tmp_path: Path,
) -> None:
    challenges = _challenges(96)
    labels = _labels(n=18)
    payload = dp.run_pipeline(tmp_path / "out.json", MarkerAwareEmbedder(), challenges, labels)
    assert payload["stage0"] is not None
    assert payload["admissibility"] is not None
    assert payload["stage1"] is not None
    assert payload["stage2"] is not None
    assert "auroc" in payload["stage2"]
    assert payload["decision_rule_outcome"] in {
        "PROCEED",
        "CLOSE",
        "UNDERPOWERED",
    }
    # Stage 2's outcome is the same pure function applied to its own numbers.
    s2 = payload["stage2"]
    assert s2["decision_rule_outcome"] == dp.decision_rule(s2["auroc"], tuple(s2["boot_ci"]))


def test_a_pipeline_that_passes_the_gate_but_cannot_recover_difficulty_stops_at_stage1(
    tmp_path: Path,
) -> None:
    # The gate clears (its planted marker is learnable) but Stage 1 cannot recover the real
    # annotated difficulty (plain statements, no learnable signal): run_pipeline must STOP at
    # Stage 1 with no stage2 — the absence of a Stage-2 number IS the finding.
    payload = dp.run_pipeline(
        tmp_path / "out.json", MarkerAwareEmbedder(), _plain_challenges(), _labels()
    )
    assert payload["admissibility"]["admissible"] is True
    assert payload["pivot"] == "STAGE1_NULL"
    assert payload["decision_rule_outcome"] == "STOPPED_STAGE1"
    assert payload["stage2"] is None
    assert payload["stage1"]["beats_null"] is False


def test_a_stage0_feature_that_clears_the_bar_stops_before_any_embedder(
    tmp_path: Path,
) -> None:
    # Statement length is made PERFECTLY predictive of the label, so a free surface feature
    # clears the validates-if bar: run_pipeline must ship the one-line rule and never touch an
    # embedder. The poison embedder proves it — any embed call would raise.
    class _Poison:
        def embed_batch(self, texts):  # type: ignore[no-untyped-def]
            raise AssertionError("stage-0 go must not embed")

    labels = _labels(n=18, label=None)
    labels = [
        dp.LabelRecord(
            instance_id=r.instance_id,
            difficulty=r.difficulty,
            repo=r.repo,
            problem_statement=f"statement {i}",
            label=i >= 9,
        )
        for i, r in enumerate(labels)
    ]
    payload = dp.run_pipeline(tmp_path / "out.json", _Poison(), _plain_challenges(), labels)
    assert payload["decision_rule_outcome"] == "PROCEED_STAGE0"
    assert payload["pivot"] == "STAGE0_CLEAR"
    assert payload["stage1"] is None
    assert payload["stage2"] is None
    assert "admissibility" not in payload


# ── the label and its derivation ──


def test_learning_to_defer_label_derives_from_both_terminal_outcomes(tmp_path: Path) -> None:
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    # cheap fails AND strong resolves -> label 1
    _write_traj(live_dir, "challenge-0", dp.CHEAP_MODEL, dp.CHEAP_ARM, resolved=False)
    _write_traj(live_dir, "challenge-0", dp.STRONG_MODEL, dp.STRONG_ARM, resolved=True)
    # cheap resolves AND strong resolves -> label 0
    _write_traj(live_dir, "challenge-1", dp.CHEAP_MODEL, dp.CHEAP_ARM, resolved=True)
    _write_traj(live_dir, "challenge-1", dp.STRONG_MODEL, dp.STRONG_ARM, resolved=True)
    # cheap fails AND strong fails -> label 0
    _write_traj(live_dir, "challenge-2", dp.CHEAP_MODEL, dp.CHEAP_ARM, resolved=False)
    _write_traj(live_dir, "challenge-2", dp.STRONG_MODEL, dp.STRONG_ARM, resolved=False)
    # only the cheap arm ran -> excluded (no strong arm on the instance)
    _write_traj(live_dir, "challenge-3", dp.CHEAP_MODEL, dp.CHEAP_ARM, resolved=False)
    records = dp.learning_to_defer_records(_challenges(), live_dir=live_dir)
    by_id = {r.instance_id: r.label for r in records}
    assert by_id["challenge-0"] is True
    assert by_id["challenge-1"] is False
    assert by_id["challenge-2"] is False
    assert "challenge-3" not in by_id


def test_repo_folds_require_at_least_two_repos() -> None:
    # A single-repo corpus cannot hold a repo out, so grouped CV is undefined; the error must
    # say what is wrong rather than leak a raw sklearn ValueError.
    with pytest.raises(ValueError, match="at least two repos"):
        dp._repo_folds(["repo-a"] * 20, 5)
    assert dp._repo_folds([], 5) == []


def test_a_fold_missing_a_class_maps_columns_by_class_not_position() -> None:
    # The linear head only emits columns for the classes its TRAIN fold saw. The old code
    # assumed column order == ordinal order, so a fold whose train lacked 'easy' shifted
    # every held-out 'medium'/'hard' prediction down a class. Columns must be mapped back
    # through `head.classes_`, and the expected-score arithmetic must not crash on a
    # 2-column proba.
    challenges = [
        Challenge(f"c{i}", "medium" if i % 2 else "hard", "repo-a", "stmt") for i in range(8)
    ]
    # Train holds only medium(1)+hard(2) — 'easy'(0) is absent from this fold's fit.
    matrix = np.array([[float(i)] for i in range(8)], dtype=np.float32)
    train_idx, test_idx = [0, 1, 2, 3], [4, 5, 6, 7]
    predicted, expected = dp._fit_and_predict_matrix(matrix, challenges, (train_idx, test_idx))
    assert all(p >= 1 for p in predicted)  # no held-out row may map to the absent 'easy' (0)
    assert len(predicted) == len(test_idx) == len(expected)


# ── real-only: no proxy vectorizer in the import graph ──


def test_no_proxy_vectorizer_in_the_experiment_import_graph() -> None:
    module_path = Path(dp.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    banned = ("TfidfVectorizer", "HashingVectorizer", "CountVectorizer")
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            hits.append(node.module)
            hits.extend(a.name for a in node.names)
    assert not any("feature_extraction" in name for name in hits)
    assert not any("fastembed" in name for name in hits)
    for name in hits:
        assert not any(b in name for b in banned), name
