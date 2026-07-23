"""Tests for the auto-labeler validation metrics + the empty-template path."""

from __future__ import annotations

from pathlib import Path

from benchmark.calibration import eval_labeler
from benchmark.calibration.labeler_metrics import (
    PREREGISTERED_FPR_BOUND,
    compute_metrics,
    confusion_matrix,
)


def _template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "benchmark/calibration/labels.template.yaml"


class TestConfusionAndMetrics:
    def test_hand_built_confusion_matrix(self):
        # owner:  s1 good, s2 good, s3 bad,  s4 bad,  s5 good
        # auto :  s1 good, s2 bad,  s3 good, s4 bad,  s5 good
        owner = {"s1": "good", "s2": "good", "s3": "bad", "s4": "bad", "s5": "good"}
        auto = {"s1": "good", "s2": "bad", "s3": "good", "s4": "bad", "s5": "good"}
        cm = confusion_matrix(owner, auto)
        assert (cm.tp, cm.fp, cm.fn, cm.tn) == (2, 1, 1, 1)
        assert cm.n == 5

    def test_precision_recall_f1_fpr_are_correct(self):
        owner = {"s1": "good", "s2": "good", "s3": "bad", "s4": "bad", "s5": "good"}
        auto = {"s1": "good", "s2": "bad", "s3": "good", "s4": "bad", "s5": "good"}
        m = compute_metrics(owner, auto)
        assert m.precision == 2 / 3  # tp/(tp+fp) = 2/3
        assert m.recall == 2 / 3  # tp/(tp+fn) = 2/3
        assert abs(m.f1 - 2 / 3) < 1e-9
        assert m.fpr == 0.5  # fp/(fp+tn) = 1/2

    def test_perfect_agreement_kappa_is_one(self):
        owner = {"a": "good", "b": "bad", "c": "good", "d": "bad"}
        m = compute_metrics(owner, dict(owner))
        assert abs(m.cohen_kappa - 1.0) < 1e-9
        assert not m.flagged

    def test_chance_level_kappa_near_zero(self):
        # Labeler ignores truth and says 'good' half the time at random-ish -> kappa ~ 0.
        owner = {f"s{i}": ("good" if i % 2 == 0 else "bad") for i in range(100)}
        auto = {f"s{i}": ("good" if i % 3 == 0 else "bad") for i in range(100)}
        m = compute_metrics(owner, auto)
        assert abs(m.cohen_kappa) < 0.25

    def test_only_shared_valid_sessions_are_compared(self):
        owner = {"s1": "good", "s2": "bad", "s3": "good"}
        auto = {"s1": "good", "s2": "bad"}  # s3 missing from auto
        m = compute_metrics(owner, auto)
        assert m.n_compared == 2


class TestFprFlag:
    def test_labeler_over_bound_is_flagged(self):
        # 10 owner-bad sessions, the labeler calls 3 of them good -> FPR 0.3 > 0.10 bound.
        owner = {f"b{i}": "bad" for i in range(10)}
        owner.update({f"g{i}": "good" for i in range(10)})
        auto = {f"b{i}": ("good" if i < 3 else "bad") for i in range(10)}
        auto.update({f"g{i}": "good" for i in range(10)})
        m = compute_metrics(owner, auto)
        assert m.fpr == 0.3
        assert m.flagged
        assert m.fpr_bound == PREREGISTERED_FPR_BOUND

    def test_labeler_within_bound_not_flagged(self):
        owner = {f"b{i}": "bad" for i in range(20)}
        owner.update({f"g{i}": "good" for i in range(20)})
        auto = {f"b{i}": ("good" if i < 1 else "bad") for i in range(20)}  # FPR 0.05
        auto.update({f"g{i}": "good" for i in range(20)})
        m = compute_metrics(owner, auto)
        assert m.fpr == 0.05
        assert not m.flagged


class TestEmptyTemplateAndScript:
    def test_load_owner_labels_ignores_null_placeholders(self, tmp_path: Path):
        # The committed template has only null placeholders -> zero valid labels.
        template = _template_path()
        assert eval_labeler.load_owner_labels(template) == {}

    def test_run_gives_guidance_when_template_empty(self, tmp_path: Path):
        template = _template_path()
        code, text = eval_labeler.run(store=None, labels_path=template)
        assert code == 0
        assert "No owner labels" in text

    def test_load_owner_labels_keeps_only_valid(self, tmp_path: Path):
        p = tmp_path / "labels.yaml"
        p.write_text("labels:\n  s1: good\n  s2: bad\n  s3: maybe\n  s4: null\n")
        assert eval_labeler.load_owner_labels(p) == {"s1": "good", "s2": "bad"}

    def test_run_scores_against_a_fake_store(self, tmp_path: Path):
        p = tmp_path / "labels.yaml"
        p.write_text("labels:\n  s1: good\n  s2: bad\n")

        class FakeStore:
            def get_outcome(self, session_id: str):
                return {"tier2_outcome": "success" if session_id == "s1" else "failure"}

        code, text = eval_labeler.run(FakeStore(), p)
        assert code == 0  # labeler perfectly agrees -> not flagged
        assert "Precision" in text
