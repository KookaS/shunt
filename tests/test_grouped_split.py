"""The shared repo-grouped split: whole repos per fold, seeded, and consumed by both halves."""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from benchmark import config, grouped_split
from benchmark.escalation import features, prefix_eval
from benchmark.routing.scripts import threshold_sweep

CONFIG_PATH = str(Path(config.__file__).resolve().parent / "benchmark.yaml")


def _corpus(n_repos: int = 12, per_repo: int = 15) -> list[str]:
    """SWE-bench-shaped ids across `n_repos` repositories, `per_repo` tasks each."""
    return [f"org{r}__repo{r}-{k}" for r in range(n_repos) for k in range(per_repo)]


def test_no_repo_straddles_a_fold() -> None:
    repos = [grouped_split.repo_of(task) for task in _corpus()]
    fold_ids = grouped_split.repo_grouped_fold_ids(repos, 5)

    seen: dict[str, set[int]] = defaultdict(set)
    for repo, fold in zip(repos, fold_ids, strict=True):
        seen[repo].add(int(fold))
    assert all(len(folds) == 1 for folds in seen.values())

    # A held-out task never shares a repo with any task in its own training index.
    for test_fold in sorted(set(fold_ids.tolist())):
        train_repos = {repos[i] for i in range(len(repos)) if fold_ids[i] != test_fold}
        test_repos = {repos[i] for i in range(len(repos)) if fold_ids[i] == test_fold}
        assert not (train_repos & test_repos)


def test_same_seed_reproduces_the_same_folds() -> None:
    repos = [grouped_split.repo_of(task) for task in _corpus()]
    a = grouped_split.repo_grouped_fold_ids(repos, 5, seed=42).tolist()
    b = grouped_split.repo_grouped_fold_ids(repos, 5, seed=42).tolist()
    c = grouped_split.repo_grouped_fold_ids(repos, 5, seed=7).tolist()
    assert a == b
    assert a != c


def test_every_task_lands_in_exactly_one_fold() -> None:
    repos = [grouped_split.repo_of(task) for task in _corpus()]
    fold_ids = grouped_split.repo_grouped_fold_ids(repos, 5)
    assert len(fold_ids) == len(repos)
    assert set(fold_ids.tolist()) == set(range(5))
    counts = [int((fold_ids == f).sum()) for f in range(5)]
    assert sum(counts) == len(repos)
    # Balance is bounded by ONE REPO, not one task: groups are atomic, so a fold can differ from
    # another by at most the largest whole repo it did or did not receive.
    largest_repo = max(Counter(repos).values())
    assert max(counts) - min(counts) <= largest_repo


def test_single_repo_corpus_fails_loudly() -> None:
    with pytest.raises(grouped_split.DegenerateFoldError):
        grouped_split.repo_grouped_fold_ids(["org__repo-1"] * 10, 5)


def test_both_halves_use_the_one_shared_module() -> None:
    # Routing's sweep has no private fold assignment and builds its folds here...
    routing_src = inspect.getsource(threshold_sweep.run_sweep)
    assert "grouped_split.repo_grouped_fold_ids" in routing_src
    assert not hasattr(threshold_sweep, "fold_assignment")
    # ...and escalation's public entry point delegates to the same module rather than
    # reimplementing a second partition.
    labels = [i % 2 == 0 for i in range(60)]
    groups = [f"c{i // 5}" for i in range(60)]
    expected = grouped_split.stratified_grouped_splits(labels, groups, prefix_eval.N_SPLITS)
    actual = prefix_eval.grouped_splits(labels, groups)
    assert len(actual) == len(expected)
    for (a_train, a_test), (e_train, e_test) in zip(actual, expected, strict=True):
        assert np.array_equal(a_train, e_train)
        assert np.array_equal(a_test, e_test)


def test_escalation_composes_instance_and_repo_grouping() -> None:
    # Two instances share each repo. The fold partition must keep every row of one INSTANCE
    # together AND every task of one REPO together — the two constraints hold at once.
    groups = [f"org{r}__repo{r}-{inst}" for r in range(6) for inst in range(2) for _ in range(3)]
    labels = [i % 2 == 0 for i in range(len(groups))]
    repos = [grouped_split.repo_of(group) for group in groups]

    splits = grouped_split.stratified_grouped_splits(labels, repos, 3)
    assert len(splits) == 3
    by_instance: dict[str, set[int]] = defaultdict(set)
    by_repo: dict[str, set[int]] = defaultdict(set)
    for fold, (_train, test) in enumerate(splits):
        for index in test:
            by_instance[groups[index]].add(fold)
            by_repo[repos[index]].add(fold)
    assert all(len(folds) == 1 for folds in by_instance.values())
    assert all(len(folds) == 1 for folds in by_repo.values())


def test_prefix_prepare_derives_the_repo_from_the_instance(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # `EvalRow.group` is the instance id (features.group_of). `_prepare` must hand the SPLIT the
    # repository, so two instances of one repo stay in one fold. Isolated with stubbed
    # census/fit: the assertion is about the group key, not about fitting a real corpus.
    rows = [
        features.EvalRow(
            f"a{i}", group="astropy__astropy-12907", model="m", failed=i % 2 == 0, features=(0.0,)
        )
        for i in range(4)
    ] + [
        features.EvalRow(
            f"b{i}", group="django__django-11099", model="m", failed=i % 2 == 0, features=(0.0,)
        )
        for i in range(4)
    ]
    census = prefix_eval.CorpusCensus(rows=rows, n_unstamped=0, n_too_short=0, n_by_margin=0)
    monkeypatch.setattr(prefix_eval, "corpus_census", lambda _trajs, _depth: census)
    monkeypatch.setattr(prefix_eval, "MIN_ROWS", 2)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        prefix_eval,
        "_fit_once",
        lambda _rows, _base, _labels, group_list: captured.setdefault("groups", list(group_list)),
    )

    prepared = prefix_eval._prepare([], 5)
    assert prepared is not None
    assert set(prepared.groups) == {"astropy/astropy", "django/django"}
    assert prepared.groups == captured["groups"]


def test_repo_of_is_the_single_implementation() -> None:
    # The routing probe half imports the shared parser; a second copy would let the two halves
    # disagree about what a repository is.
    from benchmark.routing.scripts import knn_nulls

    assert knn_nulls.repo_of("scikit-learn__scikit-learn-10297") == "scikit-learn/scikit-learn"
    assert grouped_split.repo_of("astropy__astropy-12907") == "astropy/astropy"


def test_repo_of_task_prefers_the_manifest_repo() -> None:
    assert grouped_split.repo_of_task("weird-id", {"repo": "org/repo"}) == "org/repo"
    assert grouped_split.repo_of_task("astropy__astropy-12907") == "astropy/astropy"


def test_sweep_records_the_seed_and_scores_every_task_out_of_fold() -> None:
    # The observable half: the REPORTS are the outer-fold rows (every task scored exactly once)
    # under a recorded seed; the inner/outer nesting itself is the existing per-fold loop.
    config.load(CONFIG_PATH)
    models = {
        "cheap": {"input_price": 0.1, "output_price": 0.1},
        "dear": {"input_price": 5.0, "output_price": 5.0},
    }
    task_ids = [f"org{r}__repo{r}-{k}" for r in range(8) for k in range(3)]
    results = {t: {m: {"pass": True, "real_cost": 0.01} for m in models} for t in task_ids}
    matrix = {
        "models": models,
        "tasks": {t: {"repo": grouped_split.repo_of(t)} for t in task_ids},
    }
    feats = np.random.default_rng(0).normal(size=(len(task_ids), 4))
    grid = threshold_sweep.Grid((2,), (0.5,), (1,))

    res = threshold_sweep.run_sweep(task_ids, feats, results, matrix, grid, 4)
    assert res.seed == grouped_split.DEFAULT_SEED
    assert res.n_folds == 4
    assert sum(r["n_scored"] for r in res.fold_rows) == len(task_ids)
    assert sum(r["n_scored"] for r in res.nested) == len(task_ids)
