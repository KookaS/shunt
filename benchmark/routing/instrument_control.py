"""The routing instrument's two-sided validity control — run BEFORE any routing verdict is quoted.

Plants a known-learnable signal in the task text the router actually embeds and asks the
ASSEMBLED pipeline to recover it; then destroys the signal and asks it to collapse to chance.
"""

# TWO INSTRUMENTS SHARE THIS ONE PLANTED CORPUS, AND THAT IS DELIBERATE.
#
#   run_control()           -> knn_nulls.select_from_rates   (the ANALYSIS rule behind the
#                                                             transfer / cross-repo figures)
#   run_strategy_control()  -> kNNStrategy -> RouterEngine.decide -> SelectionRule
#                                                            (the SHIPPED rule behind
#                                                             routing/reports/strategy_summary.csv)
#
# `select_from_rates` documents three named divergences from `SelectionRule` (weighting,
# min_samples, the fallback branch), so clearing the gate on one certifies nothing about the
# other — and the strategy table, not the figures, is what the kill-gate comparison reads. Both
# legs therefore run, over the SAME corpus and the SAME two-sided adjudication, so a divergence
# between them is a diagnosis rather than a blind spot. There is exactly one control
# construction here; a second parallel one is how the first goes stale.

# WHERE THE CONTROL ENTERS, AND WHY THAT IS THE WHOLE POINT. The routing instrument is
#
#     matrix["tasks"][tid]  ->  routing_text()  ->  Embedder (real ONNX)  ->  cosine sims
#         ->  neighbourhood_rates()  ->  select_from_rates()  ->  scored against pass/fail
#
# and the verdicts it emits ("the router is ABOVE / INSIDE the shuffled-outcome null band") are
# read off the last stage. A control that starts at `sims` or at the pass matrix therefore proves
# the BACK HALF works and certifies nothing about `routing_text` and the embedder — which is where
# the serious defects live. This control enters at `matrix["tasks"]`, the FRONT: it hands the
# pipeline task metadata of the same shape the real corpus has and never touches an intermediate.
# Everything from `routing_text` onward is the shipped code path.
#
# A FAILURE HERE IMPLICATES: which text field `routing_text` selects; the embedder (model load,
# batching, ordering, normalisation); the similarity and neighbour ranking; the selection rule; and
# the scoring. It does NOT implicate the real corpus (whether SWE-bench-style task text carries
# routable signal is the question the instrument exists to ask, not something this control answers),
# the imputation/completion layer, cost accounting, the bootstrap intervals, or the served router.
#
# THE PLANTED SIGNAL IS ORTHOGONAL TO REPOSITORY IDENTITY, DELIBERATELY. `routing_text` now
# resolves to the real SWE-bench `problem_statement` on every committed task (median 1185 chars);
# it resolved to the ~106-character `<repo>@<commit12> - resolve <test-node-id>` label until the
# corpus was rebuilt, and the diagnostic numbers quoted below were measured on that older text.
# The control's planted text is padded to the corpus median so a positive control
# is exercised at the SAME input length as the real signal — a ~9x-shorter control would certify
# the embedder only for a regime the corpus never uses.
# Either way the repository name is present in the string the embedder is handed, and the real
# front end demonstrably propagates it:
# this module's own `repo_identity_positive_score` reads 0.7375 on the ANALYSIS leg against a
# chance level of 0.5. The STRATEGY leg reads 0.4125 on the same diagnostic over the same corpus,
# and both numbers are stated here on purpose: repository identity is propagated by the EMBEDDER,
# and the two decision rules exploit it differently, so quoting only the leg that supports the
# construction would be the selective citation this comment block exists to stop. One leg
# recovering the repository is enough to require the orthogonality, because a control is only as
# strong as the leg it is weakest against. A control that planted its signal wherever it liked
# would therefore be satisfied by a pipeline that recovers nothing but the repository name, which
# is trivially recoverable from a string containing the repository name. Passing such a control
# certifies the instrument for a question nobody asks. So it balances every repository across
# the two outcome classes: repository identity carries EXACTLY ZERO information about the label,
# and the signal lives only in the non-repo part of the text. `repo_identity_positive_score`
# re-runs the same corpus with the labels re-aligned onto repository identity and is reported as a
# DIAGNOSTIC CONTRAST, never as part of the verdict: if the orthogonal leg fails while that one
# passes, the front end propagates repository identity and nothing finer, which is a precise
# diagnosis rather than a bare red.
#
# THE REASON IS RECOVERABILITY, NOT A LOPSIDED REPOSITORY MIX — AN EARLIER VERSION OF THIS COMMENT
# SAID OTHERWISE. It claimed one repository supplies nearly half the rows. That is the 500-spec
# MANIFEST (`benchmark/challenges/swebench_verified/`: django/django is 231 of 500, 46.2%), and no
# routing verdict is computed on it. On the 177 SCORED tasks the analysis actually reads, the mix
# is nearly flat across 12 repositories — largest matplotlib/matplotlib at 23 (13.0%), django at
# 22 (12.4%) — so no repository dominates and the orthogonality construction is not motivated by
# one that does. It is motivated by the measured 0.7375 above: the label makes the repository
# recoverable at ANY mix, and a control that let its signal ride on that would certify nothing.
#
# THE CHANCE LEVEL IS ANALYTIC, NOT BORROWED FROM THE PIPELINE'S OWN NULL. Each of the two control
# arms passes exactly half the tasks by construction, so every signal-free rule scores 0.5 in
# expectation whatever the threshold does. Centring the band on the permutation null's own MEAN
# instead would make the destroyed-signal leg nearly vacuous — a pipeline that leaks would move the
# observation and the band together, and the leg would still read "at chance". Only the band's
# HALF-WIDTH is empirical, which is what a finite-sample null is for.

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing.scripts import knn_nulls
from benchmark.routing.strategies import routing_text

# Every signal-free rule scores this on the planted corpus: each arm passes half the tasks, so
# neither the threshold branch nor the argmax fallback can do better without reading the text.
CHANCE_LEVEL: Final[float] = 0.5

# The ANALYSIS leg's two arms, price-ascending. Deliberately NOT registry models: that leg scores
# through `select_from_rates`, which reads a bare rate matrix, so borrowing real model names would
# couple it to pricing config that has nothing to do with whether text reaches the decision. The
# STRATEGY leg cannot make that choice — `RouterEngine` selects out of the shipped `ModelPool`, so
# its arms must be registry models (see `strategy_arms`).
CONTROL_MODELS: Final[tuple[str, str]] = ("control-cheap", "control-strong")

# Minimum arms a crossed control needs: one the cheap side solves, one the dear side solves.
_MIN_ARMS: Final[int] = 2

# Independent destroyed-signal draws summarised by their median (see `run_control`). Odd, small,
# and fixed: the point is to denoise the estimate, not to move the decision boundary.
_NULL_DRAWS: Final[int] = 5

# Eight repositories, each contributing equally to BOTH outcome classes (see the header). Eight
# rather than four because the corpus size sets the permutation band's width, and the band is what
# the positive leg has to clear: at 80 tasks the bar was 0.638, at 160 it is tighter, so the same
# control is strictly harder to satisfy for one more (cheap) batch of short texts.
_REPOS: Final[tuple[tuple[str, str], ...]] = (
    ("django/django", "django"),
    ("astropy/astropy", "astropy"),
    ("sympy/sympy", "sympy"),
    ("scikit-learn/scikit-learn", "sklearn"),
    ("pytest-dev/pytest", "_pytest"),
    ("pylint-dev/pylint", "pylint"),
    ("matplotlib/matplotlib", "matplotlib"),
    ("psf/requests", "requests"),
)

# The two lexical families that carry the planted signal. Distinct vocabulary, same surface shape,
# and no repository token in either — so recovering the family requires reading the part of the
# label that is not the repo name.
_FAMILY_A: Final[tuple[tuple[str, str], ...]] = (
    ("serialization/tests/test_json.py", "test_serialize_datetime_field_to_iso8601"),
    ("serialization/tests/test_json.py", "test_dumpdata_roundtrip_preserves_unicode"),
    ("serialization/tests/test_yaml.py", "test_yaml_serializer_emits_natural_keys"),
    ("serialization/tests/test_yaml.py", "test_deserialize_stream_handles_empty_document"),
    ("serialization/tests/test_encoders.py", "test_encoder_rejects_non_utf8_payload"),
    ("serialization/tests/test_encoders.py", "test_decimal_encoder_keeps_precision"),
    ("serialization/tests/test_natural_keys.py", "test_natural_key_lookup_is_stable"),
    ("serialization/tests/test_natural_keys.py", "test_serialize_deferred_attribute"),
    ("serialization/tests/test_xml.py", "test_xml_serializer_escapes_control_characters"),
    ("serialization/tests/test_xml.py", "test_deserialize_xml_preserves_field_order"),
)
_FAMILY_B: Final[tuple[tuple[str, str], ...]] = (
    ("migrations/tests/test_autodetector.py", "test_rename_field_is_detected_not_dropped"),
    ("migrations/tests/test_autodetector.py", "test_alter_field_generates_single_operation"),
    ("migrations/tests/test_operations.py", "test_run_sql_operation_is_reversible"),
    ("migrations/tests/test_operations.py", "test_add_index_operation_updates_state"),
    ("migrations/tests/test_graph.py", "test_dependency_graph_rejects_a_cycle"),
    ("migrations/tests/test_graph.py", "test_squashed_migration_replaces_ancestors"),
    ("migrations/tests/test_executor.py", "test_executor_applies_migrations_in_order"),
    ("migrations/tests/test_executor.py", "test_unapply_restores_previous_schema_state"),
    ("migrations/tests/test_loader.py", "test_loader_ignores_unmigrated_application"),
    ("migrations/tests/test_loader.py", "test_migration_plan_is_deterministic"),
)

EmbedTexts = Callable[[list[str]], np.ndarray]

# The committed corpus's routing channel. `routing_text` resolves to `problem_statement` on every
# committed task (median ~1180 chars) — NOT the ~106-char `<repo>@<commit12> - resolve <node-id>`
# `description` label the control used to mirror. A positive control planted on ~9x shorter inputs
# than the real signal exercises a different embedding regime, so the planted texts are padded to
# the corpus median below.
_DATA_PATH: Final[Path] = Path(__file__).resolve().parent / "data" / "challenges.json"


@lru_cache(maxsize=1)
def _corpus_problem_statements() -> tuple[str, ...]:
    """The committed corpus's `problem_statement` channel — real task text, never a fake."""
    with _DATA_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return tuple(
        str(t.get("problem_statement") or "").strip()
        for t in data.get("tasks", {}).values()
        if str(t.get("problem_statement") or "").strip()
    )


def corpus_median_chars() -> int:
    """The committed problem_statement channel's median length — the control's length target."""
    lengths = [len(s) for s in _corpus_problem_statements()]
    if not lengths:
        # An empty corpus is a loud, named error, not a silent NaN: `np.median([])` is NaN
        # and `int(NaN)` raises a raw ValueError that names neither the corpus nor the fix.
        raise ValueError(
            f"no problem statements found in the routing corpus ({_DATA_PATH}) — the "
            "control's length target needs task text to exist"
        )
    return int(np.median(lengths))


@lru_cache(maxsize=1)
def _control_filler() -> str:
    """Deterministic shared natural-language filler drawn from the committed corpus."""
    # SHARED across every control text on purpose: the planted family signal must stay the
    # dominant distinguishing content. A per-task-unique filler dominates the bag-of-words and
    # the kNN neighbourhoods collapse onto filler similarity — measured: the positive control
    # drops from 0.80 to 0.52 and FAILS. A constant filler leaves the family vocabulary as the
    # only discriminator, while keeping the corpus's LENGTH and REGISTER (the point of D12).
    rng = np.random.default_rng(0x5EED)
    statements = _corpus_problem_statements()
    picked = [statements[i] for i in sorted(rng.choice(len(statements), 8, replace=False))]
    return "\n\n".join(picked)


def _pad_to_corpus_median(prefix: str) -> str:
    """Scale a planted control text to the corpus's median problem-statement length."""
    target = corpus_median_chars()
    filler = _control_filler()
    sep = "\n\n"
    need = target - len(prefix) - len(sep)
    if need <= 0:
        return prefix
    block = filler + sep
    # `.rstrip()` keeps `routing_text`'s `.strip()` from seeing a trailing whitespace cut and
    # returning a value that differs from what `_assert_planted_text_is_embedded` stored.
    fill = ((block * (need // len(block) + 1))[:need]).rstrip()
    return prefix + sep + fill


def _default_embed_texts(texts: list[str]) -> np.ndarray:
    """The shipped embedder — the same callable the kNN figures embed the real corpus with."""
    # Imported lazily so building the control corpus (and every test that injects its own
    # embedder) never pulls in hnswlib or triggers an ONNX load.
    from benchmark.routing.strategies.knn import _embed_texts

    return _embed_texts(texts)


def build_control_matrix(
    seed: int = 0, models: Sequence[str] = CONTROL_MODELS
) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    """The planted corpus: ``(matrix, task_ids, family_pass, repo_pass)``.

    ``family_pass`` is the gate's outcome matrix (label ⟂ repository); ``repo_pass`` re-aligns
    the same texts onto repository identity and is the diagnostic contrast only.
    """
    # `models` is price-ascending and may be longer than two: the strategy leg has to hand the
    # shipped `RouterEngine` its whole enabled pool, or `SelectionRule._escalate` would return a
    # model the corpus has no cell for and the task would vanish into `unscorable`. Only the two
    # EXTREMES are crossed; every model between them fails everywhere, which leaves the analytic
    # chance level at 0.5 untouched (see `_pass_matrix`).
    if len(models) < _MIN_ARMS:
        raise ValueError(f"a crossed control needs at least {_MIN_ARMS} arms; got {list(models)}")
    rng = np.random.default_rng(seed)
    tasks: dict[str, dict] = {}
    task_ids: list[str] = []
    families: list[int] = []
    repo_ranks: list[int] = []

    for repo_rank, (repo, package) in enumerate(_REPOS):
        for family, descriptors in ((0, _FAMILY_A), (1, _FAMILY_B)):
            for path, test_name in descriptors:
                commit = f"{int(rng.integers(0, 16**12)):012x}"
                task_id = f"{repo.replace('/', '__')}-{path}-{test_name}"
                # The signal-bearing label, em dash included, padded with shared real
                # problem-statement prose to the corpus median length — so the control embeds
                # text of the same LENGTH AND REGISTER as the real corpus's routing channel.
                # The `{repo}@{commit}` prefix stays first so the repository name remains
                # extractable and the orthogonality construction is unchanged.
                description = f"{repo}@{commit} — resolve {package}/{path}::{test_name}"
                # ONLY `problem_statement` is set, because that is the field `routing_text`
                # resolves through on every task of the committed corpus (it was `description`
                # before the corpus was rebuilt to carry problem statements).
                # `_assert_planted_text_is_embedded` turns a change in that resolution into a
                # loud failure rather than a quiet one.
                tasks[task_id] = {
                    "problem_statement": _pad_to_corpus_median(description),
                    "repo": repo,
                }
                task_ids.append(task_id)
                families.append(family)
                repo_ranks.append(repo_rank)

    family_arr = np.array(families)
    # Half the repositories carry each diagnostic label, so the diagnostic corpus has the same
    # 50/50 marginals as the gate corpus and the two scores are directly comparable.
    repo_arr = (np.array(repo_ranks) >= len(_REPOS) // 2).astype(int)
    matrix = {
        "tasks": tasks,
        "results": _results_for(task_ids, family_arr, models),
        "models": {m: {} for m in models},
    }
    n_arms = len(models)
    return (
        matrix,
        task_ids,
        _pass_matrix(family_arr, n_arms),
        _pass_matrix(repo_arr, n_arms),
    )


def _pass_matrix(labels: np.ndarray, n_arms: int = _MIN_ARMS) -> np.ndarray:
    """``(n, n_arms)`` outcomes, price-ascending: label 0 only the cheapest arm solves, label 1
    only the dearest."""
    # Crossed on purpose. If the dear arm passed everything, a signal-free rule that always
    # escalates would already score 1.0 and the pass rate could not discriminate at all — the
    # control would be measuring cost, not recovery. Crossed, no constant policy beats 0.5.
    #
    # Middle arms (strategy leg only) fail everywhere, so they are never eligible and never the
    # cheapest-untested escalation target: the shipped rule falls through to the dearest arm on
    # no evidence, which is exactly the 0.5-scoring constant the cheap arm also is. Every task is
    # still solved by exactly one arm, so 0.5 stays the analytic chance level.
    mat = np.zeros((len(labels), n_arms), dtype=float)
    mat[labels == 0, 0] = 1.0
    mat[labels == 1, -1] = 1.0
    return mat


def _results_for(task_ids: Sequence[str], labels: np.ndarray, models: Sequence[str]) -> dict:
    """The matrix's own ``results`` block, so the corpus is a complete matrix, not a fragment."""
    mat = _pass_matrix(labels, len(models))
    return {
        tid: {
            model: {"pass": bool(mat[i, j]), "cost": 0.001 * (j + 1)}
            for j, model in enumerate(models)
        }
        for i, tid in enumerate(task_ids)
    }


def _assert_planted_text_is_embedded(matrix: dict, task_ids: Sequence[str]) -> None:
    """Fail loudly if ``routing_text`` no longer returns the text the signal was planted in."""
    for tid in task_ids:
        meta = matrix["tasks"][tid]
        resolved = routing_text(tid, meta)
        if resolved != meta["problem_statement"]:
            raise RuntimeError(
                "the routing instrument control is no longer planting into the field "
                f"routing_text reads: it returned {resolved!r} for {tid}. Re-shape the control "
                "corpus to match the field the real corpus resolves through before trusting "
                "any verdict from it."
            )


def _embeddings(matrix: dict, task_ids: Sequence[str], embed_texts: EmbedTexts) -> np.ndarray:
    """Unit-normalised embeddings of each task's routing text, aligned to ``task_ids``."""
    # Same three steps as viz_knn.build_task_embeddings, with the embedder injectable so tests can
    # drive the assembled pipeline without a 600MB ONNX load. Their parity is pinned by
    # tests/test_instrument_control.py::TestFrontEndParity.
    tasks = matrix.get("tasks", {})
    texts = [routing_text(tid, tasks.get(tid, {})) for tid in task_ids]
    emb = np.asarray(embed_texts(texts), dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


Score = Callable[[np.ndarray, np.ndarray, int, float], float]


def _loo_pass_rate(sims: np.ndarray, pass_mat: np.ndarray, k: int, threshold: float) -> float:
    """The statistic the transfer figure's verdict is read off: leave-one-task-out routed pass."""
    idx = np.arange(pass_mat.shape[0])
    return knn_nulls.routed_pass_rate(sims, pass_mat, idx, idx, k, threshold)


def run_control(  # noqa: PLR0913 (one knob per pipeline parameter the control must mirror)
    *,
    k: int,
    threshold: float,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
    seed: int = 0,
    embed_texts: EmbedTexts | None = None,
    score: Score | None = None,
) -> AdmissibilityResult:
    """Run both legs through the assembled pipeline and adjudicate.

    ``k`` and ``threshold`` MUST be the ones the figure being certified uses — a control run at
    other settings certifies an instrument nobody is quoting.
    """
    embed = _default_embed_texts if embed_texts is None else embed_texts
    scorer = _loo_pass_rate if score is None else score

    matrix, task_ids, family_pass, repo_pass = build_control_matrix(seed=seed)
    _assert_planted_text_is_embedded(matrix, task_ids)
    emb = _embeddings(matrix, task_ids, embed)
    sims = emb @ emb.T

    positive = scorer(sims, family_pass, k, threshold)
    # The band is the spread of the SAME statistic under outcome-row permutations — the empirical
    # answer to "how far from 0.5 does this pipeline land when the text says nothing?".
    rng = np.random.default_rng(seed + 1)
    draws = np.array(
        [
            scorer(sims, family_pass[rng.permutation(family_pass.shape[0])], k, threshold)
            for _ in range(n_perm)
        ]
    )
    band = knn_nulls.band_of(draws)
    half_width = (band.hi - band.lo) / 2.0

    # The destroyed-signal leg: the MEDIAN of a few fresh draws, from seeds disjoint from the
    # band's. This does not soften the leg — the decision boundary is still chance ± the
    # SINGLE-draw band, so anything a leak would push above that boundary is still caught. It only
    # removes sampling noise from the estimate: one draw sits outside a 95% band 5% of the time by
    # construction, which would make an honest instrument read INADMISSIBLE once in twenty runs.
    shuffle_rng = np.random.default_rng(seed + 10_007)
    shuffled = float(
        np.median(
            [
                scorer(
                    sims, family_pass[shuffle_rng.permutation(family_pass.shape[0])], k, threshold
                )
                for _ in range(_NULL_DRAWS)
            ]
        )
    )

    verdict = admissibility_verdict(
        positive, shuffled, chance_level=CHANCE_LEVEL, chance_band=half_width
    )
    return replace(
        verdict,
        numbers={
            **verdict.numbers,
            "n_tasks": len(task_ids),
            "k": k,
            "threshold": threshold,
            "n_perm": n_perm,
            "null_mean": band.mean,
            "null_sd": band.sd,
            "null_lo": band.lo,
            "null_hi": band.hi,
            # Diagnostic ONLY — see the module header. High here with a failing positive leg
            # means the front end propagates repository identity and nothing finer.
            "repo_identity_positive_score": scorer(sims, repo_pass, k, threshold),
        },
    )


@lru_cache(maxsize=8)
def routing_instrument_admissibility(
    *, k: int, threshold: float, n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS
) -> AdmissibilityResult:
    """``run_control`` at the shipped embedder, memoised so one report embeds the corpus once."""
    return run_control(k=k, threshold=threshold, n_perm=n_perm)


# ---------------------------------------------------------------------------
# The SHIPPED rule's leg: kNNStrategy -> RouterEngine.decide -> SelectionRule.
# ---------------------------------------------------------------------------


def strategy_arms() -> tuple[str, ...]:
    """The enabled pool in the engine's OWN price order — every arm ``SelectionRule`` may return."""
    # Taken from the pool object the strategy builds, not from a re-sort of `enabled_models()`:
    # `_escalate` walks `ranked_models()`, so an arm missing from THIS list is an arm the control
    # corpus would have no cell for.
    from benchmark.routing.strategies.knn import _benchmark_model_pool

    return tuple(m.name for m in _benchmark_model_pool().ranked_models())


def _memoised(embed: EmbedTexts) -> EmbedTexts:
    """Embed each distinct text once across every leg — the permutations reuse the geometry."""
    # The strategy leg builds a fresh kNNStrategy per draw (its index and engine are bound to one
    # outcome matrix), and each build re-embeds the whole corpus. Only the OUTCOMES differ between
    # draws, never the text, so without this the control would run ~200 real ONNX passes over the
    # same 160 strings. This changes no vector, only how often it is computed.
    cache: dict[str, np.ndarray] = {}

    def wrapped(texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            fresh = np.asarray(embed(missing))
            for text, vec in zip(missing, fresh, strict=True):
                cache[text] = vec
        return np.array([cache[t] for t in texts], dtype=np.float32)

    return wrapped


def _strategy_pass_rate(  # noqa: PLR0913 (one knob per shipped-rule parameter)
    matrix: dict,
    task_ids: Sequence[str],
    *,
    k: int,
    threshold: float,
    min_samples: int,
    embed_texts: EmbedTexts,
) -> float:
    """Routed pass rate over the planted corpus, decided by the SHIPPED ``SelectionRule``."""
    from benchmark.routing import summary
    from benchmark.routing.strategies.knn import kNNStrategy

    strategy = kNNStrategy(
        k=k,
        success_rate_threshold=threshold,
        min_samples=min_samples,
        embed_texts=embed_texts,
    )
    decisions, unscorable = summary.evaluate(strategy, matrix, list(task_ids))
    if unscorable:
        # The shipped rule landed on an arm the planted corpus has no cell for, so those tasks
        # would be dropped and the remaining score would no longer be centred on 0.5. Silently
        # scoring the remainder is the failure this control exists to make impossible.
        raise RuntimeError(
            f"the shipped selection rule chose an arm outside the planted corpus on "
            f"{len(unscorable)} task(s) (e.g. {sorted(unscorable)[:3]}). Rebuild the control "
            f"corpus over the engine's full ranked pool before trusting any verdict from it."
        )
    return float(np.mean([passed for _tid, _model, passed, _cost in decisions]))


def run_strategy_control(  # noqa: PLR0913 (one knob per pipeline parameter the control mirrors)
    *,
    k: int,
    threshold: float,
    min_samples: int,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
    seed: int = 0,
    embed_texts: EmbedTexts | None = None,
    arms: Sequence[str] | None = None,
) -> AdmissibilityResult:
    """The same two legs as ``run_control``, scored through ``kNNStrategy`` instead.

    ``k``/``threshold``/``min_samples`` MUST be the ones the strategy table was computed at.
    """
    embed = _memoised(_default_embed_texts if embed_texts is None else embed_texts)
    models = tuple(strategy_arms()) if arms is None else tuple(arms)

    matrix, task_ids, family_pass, repo_pass = build_control_matrix(seed=seed, models=models)
    _assert_planted_text_is_embedded(matrix, task_ids)

    def score(labels: np.ndarray) -> float:
        planted = {**matrix, "results": _results_for(task_ids, labels, models)}
        return _strategy_pass_rate(
            planted,
            task_ids,
            k=k,
            threshold=threshold,
            min_samples=min_samples,
            embed_texts=embed,
        )

    family = family_pass[:, -1].astype(int)
    positive = score(family)

    rng = np.random.default_rng(seed + 1)
    band = knn_nulls.band_of(np.array([score(rng.permutation(family)) for _ in range(n_perm)]))
    half_width = (band.hi - band.lo) / 2.0

    shuffle_rng = np.random.default_rng(seed + 10_007)
    shuffled = float(
        np.median([score(shuffle_rng.permutation(family)) for _ in range(_NULL_DRAWS)])
    )

    verdict = admissibility_verdict(
        positive, shuffled, chance_level=CHANCE_LEVEL, chance_band=half_width
    )
    return replace(
        verdict,
        numbers={
            **verdict.numbers,
            "n_tasks": len(task_ids),
            "k": k,
            "threshold": threshold,
            "min_samples": min_samples,
            "n_perm": n_perm,
            "arms": list(models),
            "null_mean": band.mean,
            "null_sd": band.sd,
            "null_lo": band.lo,
            "null_hi": band.hi,
            # Diagnostic ONLY, exactly as on the analysis leg.
            "repo_identity_positive_score": score(repo_pass[:, -1].astype(int)),
        },
    )


@lru_cache(maxsize=8)
def strategy_instrument_admissibility(
    *,
    k: int,
    threshold: float,
    min_samples: int,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
) -> AdmissibilityResult:
    """``run_strategy_control`` at the shipped embedder, memoised per report run."""
    return run_strategy_control(k=k, threshold=threshold, min_samples=min_samples, n_perm=n_perm)


def _print(verdict: AdmissibilityResult, label: str) -> None:
    print(f"[{label}] {verdict.reason}")
    for key, value in verdict.numbers.items():
        print(f"  {key}: {value}")


def main() -> int:
    """Run both legs at the configured routing parameters and print the verdicts."""
    from benchmark import config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--permutations", type=int, default=knn_nulls.DEFAULT_PERMUTATIONS)
    ap.add_argument(
        "--leg",
        choices=("analysis", "strategy", "both"),
        default="both",
        help="analysis = select_from_rates (figures); strategy = SelectionRule (summary CSV)",
    )
    args = ap.parse_args()
    config.load(args.config)
    params = config.knn_params()
    k = args.k if args.k is not None else int(params.get("k", 10))
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(params.get("success_rate_threshold", 0.6))
    )
    min_samples = (
        args.min_samples if args.min_samples is not None else int(params.get("min_samples", 3))
    )
    verdicts = []
    if args.leg in ("analysis", "both"):
        verdicts.append(
            ("analysis", run_control(k=k, threshold=threshold, n_perm=args.permutations))
        )
    if args.leg in ("strategy", "both"):
        verdicts.append(
            (
                "strategy",
                run_strategy_control(
                    k=k, threshold=threshold, min_samples=min_samples, n_perm=args.permutations
                ),
            )
        )
    for label, verdict in verdicts:
        _print(verdict, label)
    return 0 if all(v.admissible for _label, v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
