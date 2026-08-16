"""Predicted-difficulty go/no-go: can the problem statement a router sees carry escalation signal?

Three pre-registered stages (free features -> shipped-embedder difficulty head -> substituted
AUROC) on the learning-to-defer label, gated by the shared admissibility adjudicator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.escalation import corpus, metrics, schema
from shunt.router.embedder import Embedder, EmbedderUnavailableError

if TYPE_CHECKING:
    import numpy.typing as npt

# The challenge specs carry the annotated 3-level difficulty and the backfilled problem
# statement; the latter is what a router actually sees at decision time.
CHALLENGES_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "routing" / "data" / "challenges.json"
)
LIVE_DIR: Final[Path] = corpus.LIVE_DIR

# The escalation arm pair the learning-to-defer label is defined on: the cheapest arm
# (deepseek-v4-flash, run at `high`) and the strongest arm (kimi-k3, run at `max`).
CHEAP_MODEL: Final[str] = "deepseek-v4-flash"
STRONG_MODEL: Final[str] = "kimi-k3"
CHEAP_ARM: Final[str] = "high"
STRONG_ARM: Final[str] = "max"

# Annotated difficulty is a 3-level ordinal; the head's prediction substitutes for it.
ORDINAL: Final[dict[str, int]] = {"easy": 0, "medium": 1, "hard": 2}
_CLASSES: Final[tuple[str, ...]] = ("easy", "medium", "hard")

# The pre-registered inference budget: same null and bootstrap as the §5 evaluation.
N_PERMUTATIONS: Final[int] = 2000
N_RESAMPLES: Final[int] = 2000
SEED: Final[int] = 0
# Grouped CV by repo: an entire repository is held out per fold.
CV_FOLDS: Final[int] = 5
# The decision rule's thresholds — PROCEED >= 0.65, CLOSE < 0.60 or interval spanning 0.5.
VALIDATES_AUROC: Final[float] = 0.65
FALSIFIES_AUROC: Final[float] = 0.60

_TRACEBACK_RE: Final[re.Pattern[str]] = re.compile(r"Traceback \(most recent call last\)")
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```")
_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"`?[\w./-]+\.(?:py|md|rst|txt|yml|yaml|json|toml|cfg|ini|sh|html)`?"
)

# The planted-signal markers: one distinctive phrase per difficulty class, repeated so it
# dominates the statement embedding. A linear head must be able to learn the class from the
# marker, which is the definition of a known-learnable positive control.
PLANTED_MARKERS: Final[dict[str, tuple[str, int]]] = {
    "easy": ("albatross embedded driver sensor", 24),
    "medium": ("harbor node compiler relay", 24),
    "hard": ("quasar kernel vector plexus", 24),
}

STAGE0_OUTCOME_LABEL = "learning_to_defer"
STAGE0_OUTCOME_ANNOTATED = "annotated_difficulty"

# Label shuffles behind the destroyed-signal null in `admissibility_gate`. The null's median
# is the judged score and its empirical |spread| the chance band; 100 draws make the band a
# stable property of the distribution rather than of one draw.
_NULL_DRAWS: Final[int] = 100


class Stage0MissingError(RuntimeError):
    """The Stage-1 entry point requires a Stage-0 record; there is none."""


class Stage2NotAllowedError(RuntimeError):
    """A Stage-2 number may not be quoted before the instrument cleared its gate."""


@dataclass(frozen=True)
class Challenge:
    """One SWE-bench Verified challenge spec, joined with its backfilled statement."""

    instance_id: str
    difficulty: str
    repo: str
    problem_statement: str


@dataclass(frozen=True)
class SurfaceFeatures:
    """Free surface features of a problem statement — Stage 0, no model at all."""

    length: int
    code_blocks: int
    has_traceback: bool
    files_mentioned: int

    def as_vector(self) -> tuple[float, float, float, float]:
        """The four features as numbers, traceback as 0/1, for ranking statistics."""
        return (
            float(self.length),
            float(self.code_blocks),
            float(self.has_traceback),
            float(self.files_mentioned),
        )


@dataclass(frozen=True)
class LabelRecord:
    """One learning-to-defer instance: the label plus what the predictor may see."""

    instance_id: str
    difficulty: str
    repo: str
    problem_statement: str
    label: bool


@dataclass(frozen=True)
class SurfaceScore:
    """One Stage-0 surface feature scored against one outcome."""

    feature: str
    outcome: str
    auroc: float
    n: int
    n_positives: int
    boot_ci: tuple[float, float]

    def clears_validates(self) -> bool:
        """The stage-0 stop: this free feature already clears the validates-if bar."""
        return self.auroc >= VALIDATES_AUROC and not (self.boot_ci[0] <= 0.5 <= self.boot_ci[1])


@dataclass(frozen=True)
class Stage0Record:
    """Stage-0 result — reported even when null, and REQUIRED before Stage 1 runs."""

    scores: tuple[SurfaceScore, ...]
    n_instances: int
    n_positives: int
    piv3_clear: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "n_instances": self.n_instances,
            "n_positives": self.n_positives,
            "piv3_clear": self.piv3_clear,
            "scores": [
                {
                    "feature": s.feature,
                    "outcome": s.outcome,
                    "auroc": round(s.auroc, 4),
                    "n": s.n,
                    "n_positives": s.n_positives,
                    "boot_ci": [round(s.boot_ci[0], 4), round(s.boot_ci[1], 4)],
                    "clears_validates_bar": s.clears_validates(),
                }
                for s in self.scores
            ],
        }


@dataclass(frozen=True)
class Stage1Record:
    """Out-of-fold predicted difficulty for every challenge, plus its recovery check."""

    predictions: dict[str, int]
    expected: dict[str, float]
    recovery_rho: float
    recovery_perm_p: float
    beats_null: bool
    n_challenges: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_challenges": self.n_challenges,
            "recovery_rho": round(self.recovery_rho, 4),
            "recovery_perm_p": round(self.recovery_perm_p, 4),
            "beats_null": self.beats_null,
            "n_predictions": len(self.predictions),
        }


@dataclass(frozen=True)
class Stage2Record:
    """Predicted difficulty vs the learning-to-defer label, next to the thresholds."""

    auroc: float
    perm_p: float
    boot_ci: tuple[float, float]
    n: int
    n_positives: int
    decision_rule_outcome: str
    admissibility: AdmissibilityResult

    def to_dict(self) -> dict[str, object]:
        return {
            "auroc": round(self.auroc, 4),
            "perm_p": round(self.perm_p, 4),
            "boot_ci": [round(self.boot_ci[0], 4), round(self.boot_ci[1], 4)],
            "n": self.n,
            "n_positives": self.n_positives,
            "decision_rule_outcome": self.decision_rule_outcome,
            "validates_threshold": VALIDATES_AUROC,
            "falsifies_threshold": FALSIFIES_AUROC,
            "admissibility": self.admissibility.to_dict(),
        }


def load_challenges(path: Path = CHALLENGES_PATH) -> list[Challenge]:
    """Read the challenge specs joined with their backfilled problem statements."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks: dict[str, dict[str, object]] = data["tasks"]
    out: list[Challenge] = []
    for entry in data["challenges"]:
        instance_id = str(entry["id"])
        task = tasks[instance_id]
        statement = str(task["problem_statement"])
        if not statement.strip():
            raise ValueError(f"challenge {instance_id} has an empty problem_statement")
        out.append(
            Challenge(
                instance_id=instance_id,
                difficulty=str(entry["difficulty"]),
                repo=str(task["repo"]),
                problem_statement=statement,
            )
        )
    return out


def surface_features(statement: str) -> SurfaceFeatures:
    """The four free surface features of one statement — defined so they are testable."""
    fences = len(_FENCE_RE.findall(statement))
    return SurfaceFeatures(
        length=len(statement),
        code_blocks=fences // 2,
        has_traceback=bool(_TRACEBACK_RE.search(statement)),
        files_mentioned=len(set(_FILE_RE.findall(statement))),
    )


def _parse_live_trajectories(live_dir: Path) -> dict[str, dict[str, dict[str, bool]]]:
    """instance -> model -> arm -> terminal_resolved, read off the committed corpus."""
    rows: dict[str, dict[str, dict[str, bool]]] = {}
    for path in sorted(live_dir.glob("*.jsonl")):
        parts = path.stem.split("__")
        if len(parts) < 3:
            continue
        instance_id = "__".join(parts[:-2])
        model = parts[-2]
        arm = parts[-1]
        traj = schema.load_jsonl(path)
        rows.setdefault(instance_id, {}).setdefault(model, {})[arm] = bool(
            traj.header.terminal_resolved
        )
    return rows


def learning_to_defer_records(
    challenges: Sequence[Challenge], live_dir: Path = LIVE_DIR
) -> list[LabelRecord]:
    """The label: cheap fails AND strong resolves, on instances where both arms ran.

    Instances are the unit (one row each); the label is a pure function of two committed
    terminal outcomes, exposing only what a router holds at t=0 — the statement.
    """
    rows = _parse_live_trajectories(live_dir)
    by_id = {c.instance_id: c for c in challenges}
    out: list[LabelRecord] = []
    for instance_id in sorted(rows):
        models = rows[instance_id]
        cheap = models.get(CHEAP_MODEL, {})
        strong = models.get(STRONG_MODEL, {})
        if CHEAP_ARM not in cheap or STRONG_ARM not in strong:
            continue
        challenge = by_id.get(instance_id)
        if challenge is None:
            continue
        label = (not cheap[CHEAP_ARM]) and strong[STRONG_ARM]
        out.append(
            LabelRecord(
                instance_id=instance_id,
                difficulty=challenge.difficulty,
                repo=challenge.repo,
                problem_statement=challenge.problem_statement,
                label=label,
            )
        )
    return out


def _auroc_boot(
    scores: Sequence[float], labels: Sequence[bool], groups: Sequence[str]
) -> tuple[float, tuple[float, float]]:
    """AUROC plus its 2000-resample group bootstrap (groups = instances), seed 0."""
    observed = metrics.auroc(scores, labels)
    if sum(labels) == 0 or all(labels):
        # A single-class sample has no ranking to estimate — auroc() itself returns 0.5 for
        # it, so the interval must say "chance, no information" rather than NaN.
        return observed, (0.5, 0.5)
    ci = metrics.grouped_bootstrap_ci(
        list(scores),
        list(labels),
        list(groups),
        metrics.auroc,
        n_resamples=N_RESAMPLES,
        seed=SEED,
    )
    return observed, ci


def _score_one_feature(
    feature: str,
    values: Sequence[float],
    labels: Sequence[bool],
    groups: Sequence[str],
    outcome: str,
) -> SurfaceScore:
    observed, ci = _auroc_boot(values, labels, groups)
    return SurfaceScore(
        feature=feature,
        outcome=outcome,
        auroc=observed,
        n=len(labels),
        n_positives=sum(labels),
        boot_ci=ci,
    )


_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "statement_length",
    "code_block_count",
    "traceback_presence",
    "files_mentioned",
)


def _feature_vectors(
    statements: Sequence[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """The four free surface features as numbers, in the fixed emitted row order."""
    features = [surface_features(s) for s in statements]
    return (
        [float(f.length) for f in features],
        [float(f.code_blocks) for f in features],
        [1.0 if f.has_traceback else 0.0 for f in features],
        [float(f.files_mentioned) for f in features],
    )


def stage0(challenges: Sequence[Challenge], labels: Sequence[LabelRecord]) -> Stage0Record:
    """Score the four surface features against BOTH outcomes (stage-0 stop).

    The learning-to-defer leg is the decision leg (a clearing free feature stops the
    experiment); the annotated-difficulty leg is a diagnostic on every challenge.
    """
    label_vectors = _feature_vectors([rec.problem_statement for rec in labels])
    label_vec = [rec.label for rec in labels]
    label_groups = [rec.instance_id for rec in labels]
    label_scores = tuple(
        _score_one_feature(name, values, label_vec, label_groups, STAGE0_OUTCOME_LABEL)
        for name, values in zip(_FEATURE_NAMES, label_vectors, strict=True)
    )
    annotated_vectors = _feature_vectors([c.problem_statement for c in challenges])
    annotated_labels = [c.difficulty == "hard" for c in challenges]
    annotated_groups = [c.instance_id for c in challenges]
    annotated_scores = tuple(
        _score_one_feature(
            name, values, annotated_labels, annotated_groups, STAGE0_OUTCOME_ANNOTATED
        )
        for name, values in zip(_FEATURE_NAMES, annotated_vectors, strict=True)
    )
    # Only the label leg can clear the validates-if bar: the one-line rule must be judged
    # against the escalation outcome, not against the annotation it would replace.
    piv3_clear = any(s.clears_validates() for s in label_scores)
    return Stage0Record(
        scores=label_scores + annotated_scores,
        n_instances=len(labels),
        n_positives=sum(label_vec),
        piv3_clear=piv3_clear,
    )


def _embed_matrix(embedder: Embedder, statements: Sequence[str]) -> npt.NDArray[np.float32]:
    """Embed every statement through the shipped Embedder, stacked, in memory-bounded chunks.

    Chunking bounds peak allocation so a long corpus cannot OOM the run (measured in the
    dev container: 25 x 4k-char statements peak ~7.4GB RSS, 10-statements ~2.9GB).
    """
    chunk = 10
    rows: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(statements), chunk):
        part = list(statements[start : start + chunk])
        try:
            vectors = embedder.embed_batch(part)
        except EmbedderUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001  # any model failure is a load/embed failure
            raise EmbedderUnavailableError(
                f"embedding failed after model load ({type(exc).__name__}); "
                "the pipeline must run against the real embedder, never a stand-in"
            ) from exc
        rows.extend(v.astype(np.float32) for v in vectors)
        del part, vectors
        gc.collect()
    return np.vstack(rows)


def _repo_folds(repos: Sequence[str], n_splits: int) -> list[tuple[list[int], list[int]]]:
    """Grouped CV index pairs: no repository appears in both train and test of a fold."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(repos))
    if n_groups == 0:
        return []
    if n_groups == 1:
        # GroupKFold needs at least two groups; a single-repo corpus cannot hold a repo out.
        # Fail with a name that says what is wrong rather than a sklearn ValueError.
        raise ValueError(
            "grouped CV by repo requires at least two repos, got 1 — a single-repo corpus "
            "cannot hold an entire repo out, so no Stage-1 recovery is estimable on it"
        )
    # GroupKFold needs n_splits <= n_groups; clamp so a repo-poor sample still folds.
    splits = max(2, min(n_splits, n_groups))
    indices = list(range(len(repos)))
    folds: list[tuple[list[int], list[int]]] = []
    for train, test in GroupKFold(n_splits=splits).split(indices, groups=repos):
        folds.append((train.tolist(), test.tolist()))
    return folds


def _fit_and_predict_matrix(
    matrix: npt.NDArray[np.float32],
    challenges: Sequence[Challenge],
    fold: tuple[list[int], list[int]],
) -> tuple[list[float], list[float]]:
    """One grouped fold: fit a 3-class linear head, predict the held-out repos."""
    from sklearn.linear_model import LogisticRegression

    train_idx, test_idx = fold
    y = np.array([ORDINAL[c.difficulty] for c in challenges], dtype=np.int64)
    train_classes = set(y[train_idx].tolist())
    if len(train_classes) < 2:
        # LogisticRegression refuses to fit a single-class train set; a grouped fold that
        # leaves only one difficulty class on its train side has no 3-class head to speak of,
        # and silently emitting one class would fabricate recovery. Name it rather than let
        # sklearn's bare ValueError escape.
        raise ValueError(
            f"fold train set has only one difficulty class {sorted(train_classes)} — a "
            "grouped-CV head needs at least two classes in train, so this fold's predictions "
            "are undefined"
        )
    head = LogisticRegression(max_iter=2000, C=1.0)
    head.fit(matrix[train_idx], y[train_idx])
    probs = head.predict_proba(matrix[test_idx])
    # The fitted head only emits columns for the classes its TRAIN fold saw. Map each column
    # back to the ordinal it stands for via `head.classes_` — never assume column order equals
    # the full 0..2 ordinal range, or a fold whose train lacked a class silently shifts every
    # held-out prediction to a wrong class and the expected-score arithmetic crashes.
    column_ordinal = np.array([int(cls) for cls in head.classes_.tolist()], dtype=np.float64)
    predicted_class = column_ordinal[np.argmax(probs, axis=1)].astype(np.int64)
    expected = (probs * column_ordinal).sum(axis=1)
    return predicted_class.tolist(), expected.tolist()


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, ties broken by average rank — a 3-class comparison."""
    from scipy.stats import rankdata

    rx = rankdata(list(x))
    ry = rankdata(list(y))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def _out_of_fold_predictions(
    matrix: npt.NDArray[np.float32], challenges: Sequence[Challenge]
) -> tuple[dict[str, int], dict[str, float]]:
    """Grouped-CV predictions for every challenge — embedded once, reused by every fold."""
    repos = [c.repo for c in challenges]
    predictions: dict[str, int] = {}
    expected: dict[str, float] = {}
    for fold in _repo_folds(repos, CV_FOLDS):
        _, test_idx = fold
        pred, exp = _fit_and_predict_matrix(matrix, challenges, fold)
        for idx, class_idx, exp_val in zip(test_idx, pred, exp, strict=True):
            predictions[challenges[idx].instance_id] = int(class_idx)
            expected[challenges[idx].instance_id] = float(exp_val)
    return predictions, expected


def stage1(
    embedder: Embedder,
    challenges: Sequence[Challenge],
    stage0: Stage0Record | None,
    matrix: npt.NDArray[np.float32] | None = None,
) -> Stage1Record:
    """Fit the 3-class difficulty head with grouped CV by repo; Stage 0 must have run."""
    if stage0 is None:
        raise Stage0MissingError(
            "Stage 1 refuses to run without a Stage-0 record: Stage 0 must run first and its "
            "result must be present."
        )
    if matrix is None:
        matrix = _embed_matrix(embedder, [c.problem_statement for c in challenges])
    predictions, expected = _out_of_fold_predictions(matrix, challenges)
    annotated = [ORDINAL[c.difficulty] for c in challenges]
    # GroupKFold places every sample in exactly one test fold, so every challenge has a
    # prediction — but the failure mode of a future fold change (a dropped group) should be
    # a named error, not a bare KeyError mid-permutation.
    missing = [c.instance_id for c in challenges if c.instance_id not in predictions]
    if missing:
        raise ValueError(
            f"stage-1 fold produced no prediction for {len(missing)} challenge(s): "
            f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    predicted_ord = [predictions[c.instance_id] for c in challenges]
    observed = _spearman([float(p) for p in predicted_ord], [float(a) for a in annotated])
    rng = random.Random(SEED)
    permuted = list(annotated)
    draws: list[float] = []
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(permuted)
        draws.append(_spearman([float(p) for p in predicted_ord], [float(v) for v in permuted]))
    ordered = sorted(draws)
    p_value = (sum(1 for d in draws if d >= observed) + 1) / (N_PERMUTATIONS + 1)
    beats = observed > ordered[int(0.975 * (N_PERMUTATIONS - 1))]
    return Stage1Record(
        predictions=predictions,
        expected=expected,
        recovery_rho=observed,
        recovery_perm_p=p_value,
        beats_null=beats,
        n_challenges=len(challenges),
    )


def decision_rule(auroc: float, boot_ci: tuple[float, float]) -> str:
    """The pre-registered verdict, DERIVED from the numbers — never hand-written.

    PROCEED iff AUROC >= 0.65 and the bootstrap interval excludes 0.5. CLOSE iff AUROC < 0.60
    or the interval spans 0.5. Otherwise UNDERPOWERED (0.60-0.65, or an inconclusive interval).
    """
    low, high = boot_ci
    spans = low <= 0.5 <= high
    if auroc >= VALIDATES_AUROC and not spans:
        return "PROCEED"
    if auroc < FALSIFIES_AUROC or spans:
        return "CLOSE"
    return "UNDERPOWERED"


def _planted_corpus(challenges: Sequence[Challenge]) -> list[LabelRecord]:
    """A fixture with a KNOWN-learnable signal: per-class markers make difficulty learnable.

    Deterministic round-robin planting, marker blocks prepended per class; only the label
    link is destroyed in the shuffled variant — the marker and the head are untouched.
    """
    markers = PLANTED_MARKERS
    out: list[LabelRecord] = []
    for index, challenge in enumerate(challenges):
        planted = _CLASSES[index % len(_CLASSES)]
        marker, reps = markers[planted]
        statement = " ".join([marker] * reps) + "\n" + challenge.problem_statement
        label = planted == "hard"
        out.append(
            LabelRecord(
                instance_id=challenge.instance_id,
                difficulty=planted,
                repo=challenge.repo,
                problem_statement=statement,
                label=label,
            )
        )
    return out


def _planted_scores(matrix: npt.NDArray[np.float32], records: Sequence[LabelRecord]) -> list[float]:
    """Predicted difficulty for the planted corpus — the head is fit ONCE and reused.

    The head never sees the label, so predicted scores are identical across every label
    permutation: the controls differ ONLY in the label link.
    """
    challenges = [
        Challenge(
            instance_id=r.instance_id,
            difficulty=r.difficulty,
            repo=r.repo,
            problem_statement=r.problem_statement,
        )
        for r in records
    ]
    predictions, _ = _out_of_fold_predictions(matrix, challenges)
    return [float(predictions[r.instance_id]) for r in records]


def _permuted(labels: Sequence[bool], seed: int) -> list[bool]:
    """A label permutation from a fixed seed — the destroyed-signal control's raw material."""
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def admissibility_gate(
    embedder: Embedder, challenges: Sequence[Challenge], seed: int = SEED
) -> AdmissibilityResult:
    """Run the two R0 controls through the SHARED adjudicator (positive + shuffled null).

    Positive: the assembled pipeline recovers a planted signal; null: the same pipeline
    collapses to chance on the label-shuffled corpus, via the pinned shared gate.
    """
    sample = challenges[::5] if len(challenges) >= 60 else list(challenges)
    planted = _planted_corpus(sample)
    matrix = _embed_matrix(embedder, [r.problem_statement for r in planted])
    scores = _planted_scores(matrix, planted)
    labels = [r.label for r in planted]
    positive_score = metrics.auroc(scores, labels)
    # The destroyed-signal distribution: label shuffles from `_NULL_DRAWS` independent seeds.
    # Its MEDIAN is the judged null score. The chance band is the null's SPREAD AROUND ITS OWN
    # CENTRE, and the null is judged at chance only if its centre is within that spread of the
    # TRUE chance level (0.5). A band measured around the chance level would absorb a uniformly
    # shifted null — an instrument that manufactures a constant +0.1 AUROC off chance would sit
    # "at chance" against a band grown from that same shift, which is the exact leakage the null
    # leg exists to catch.
    shuffled_scores = [
        metrics.auroc(scores, _permuted(labels, seed + 1000 + draw)) for draw in range(_NULL_DRAWS)
    ]
    shuffled_score = statistics.median(shuffled_scores)
    band = max(
        0.0,
        sorted(abs(s - shuffled_score) for s in shuffled_scores)[int(0.975 * (_NULL_DRAWS - 1))],
    )
    return admissibility_verdict(positive_score, shuffled_score, chance_level=0.5, chance_band=band)


def stage2(
    stage1_record: Stage1Record,
    labels: Sequence[LabelRecord],
    admissibility: AdmissibilityResult,
) -> Stage2Record:
    """Substitute predicted difficulty for annotated difficulty; the gate MUST have cleared."""
    if not admissibility.admissible:
        raise Stage2NotAllowedError(
            "Stage-2 number refused: the instrument has NOT cleared its admissibility gate "
            f"({admissibility.headline}). A negative result from an instrument never shown to "
            "detect a positive is a coverage gap, not a falsification."
        )
    scores = [float(stage1_record.predictions[r.instance_id]) for r in labels]
    label_vec = [r.label for r in labels]
    groups = [r.instance_id for r in labels]
    observed, ci = _auroc_boot(scores, label_vec, groups)
    null = metrics.permute_statistic(
        scores, label_vec, metrics.auroc, n_permutations=N_PERMUTATIONS, seed=SEED
    )
    return Stage2Record(
        auroc=observed,
        perm_p=null.p_value,
        boot_ci=ci,
        n=len(labels),
        n_positives=sum(label_vec),
        decision_rule_outcome=decision_rule(observed, ci),
        admissibility=admissibility,
    )


def run_pipeline(
    out: Path,
    embedder: Embedder,
    challenges: Sequence[Challenge] | None = None,
    labels: Sequence[LabelRecord] | None = None,
) -> dict[str, object]:
    """Run Stage 0 -> gate -> Stage 1 -> Stage 2 and emit the verdict-bearing JSON."""
    if challenges is None:
        challenges = load_challenges()
    if labels is None:
        labels = learning_to_defer_records(challenges)
    stage0_record = stage0(challenges, labels)
    payload: dict[str, object] = {"stage0": stage0_record.to_dict()}
    if stage0_record.piv3_clear:
        # The stage-0 stop: a free surface feature already clears the validates-if bar —
        # ship the one-line rule; do not run an embedder to refine an answer already in hand.
        payload["decision_rule_outcome"] = "PROCEED_STAGE0"
        payload["stage1"] = None
        payload["stage2"] = None
        payload["pivot"] = "STAGE0_CLEAR"
        _write_json(out, payload)
        return payload
    gate = admissibility_gate(embedder, challenges)
    payload["admissibility"] = gate.to_dict()
    if not gate.admissible:
        # A failed instrument reports COVERAGE-GAP, never a falsification.
        payload["decision_rule_outcome"] = "COVERAGE-GAP"
        payload["stage1"] = None
        payload["stage2"] = None
        payload["pivot"] = "INSTRUMENT_FAILED"
        _write_json(out, payload)
        return payload
    stage1_record = stage1(embedder, challenges, stage0_record)
    payload["stage1"] = stage1_record.to_dict()
    if not stage1_record.beats_null:
        # Stage 1 cannot recover annotated difficulty at well above chance — stop at
        # Stage 1; Stage 2 is not runnable and its absence IS the finding.
        payload["decision_rule_outcome"] = "STOPPED_STAGE1"
        payload["stage2"] = None
        payload["pivot"] = "STAGE1_NULL"
        _write_json(out, payload)
        return payload
    stage2_record = stage2(stage1_record, labels, gate)
    payload["stage2"] = stage2_record.to_dict()
    payload["decision_rule_outcome"] = stage2_record.decision_rule_outcome
    payload["pivot"] = None
    _write_json(out, payload)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("difficulty_go_no_go.json"))
    ap.add_argument(
        "--embedder-model",
        default=None,
        help="the shipped embedder model key/repo (default: the active embedding.yaml model)",
    )
    args = ap.parse_args(argv)
    embedder = Embedder(model_name=args.embedder_model) if args.embedder_model else Embedder()
    payload = run_pipeline(args.out, embedder)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CHEAP_ARM",
    "CHEAP_MODEL",
    "CHALLENGES_PATH",
    "CV_FOLDS",
    "FALSIFIES_AUROC",
    "LIVE_DIR",
    "N_PERMUTATIONS",
    "N_RESAMPLES",
    "ORDINAL",
    "PLANTED_MARKERS",
    "SEED",
    "STRONG_ARM",
    "STRONG_MODEL",
    "VALIDATES_AUROC",
    "Challenge",
    "LabelRecord",
    "Stage0MissingError",
    "Stage0Record",
    "Stage1Record",
    "Stage2NotAllowedError",
    "Stage2Record",
    "SurfaceFeatures",
    "SurfaceScore",
    "admissibility_gate",
    "decision_rule",
    "learning_to_defer_records",
    "load_challenges",
    "run_pipeline",
    "stage0",
    "stage1",
    "stage2",
    "surface_features",
]


if __name__ == "__main__":
    raise SystemExit(_main())
