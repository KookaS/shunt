"""Repo-grouped, seeded fold assignment shared by the routing and escalation halves.

The grouping key is the REPOSITORY, never the task or the instance: an entire repo lands in
one fold, so no held-out task shares a repo with a task in its own training index. Routing's
`threshold_sweep` and escalation's `prefix_eval` both build their folds here, so the two
halves cannot drift into two different partitions.

Escalation carries many model-by-arm trajectories per instance; grouping by repo is coarser
than that and SUBSUMES it — every trajectory of an instance shares the instance's repo, so it
lands in the same fold — which is the composition this helper exists to make structural.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_SEED: Final[int] = 42
DEFAULT_FOLDS: Final[int] = 5
# A fold holding fewer than this many tasks cannot estimate a held-out rate. The split fails
# loudly rather than emit a degenerate fold. FORWARD-LOOKING: the current 177-task routing
# corpus spans 12 repos whose largest is 13%, so it clears this many times over; the guard is
# specified against a 500-task manifest's 46%-single-repo shape.
MIN_TASKS_PER_FOLD: Final[int] = 2


class DegenerateFoldError(ValueError):
    """Repo grouping cannot produce folds that hold enough held-out tasks to score."""


def repo_of(task_id: str) -> str:
    """Source repository carried in a SWE-bench task id (`org__repo-1234` -> `org/repo`)."""
    # Split on the LAST hyphen, not the first: repo names contain hyphens themselves
    # (`scikit-learn__scikit-learn-10297`, `pytest-dev__pytest-5495`), and splitting on the
    # first one silently collapses them into a bogus `scikit` bucket.
    head, sep, tail = task_id.rpartition("-")
    if not sep or not tail.isdigit():
        head = task_id
    return head.replace("__", "/") if "__" in head else head


def repo_of_task(task_id: str, task_meta: Mapping[str, object] | None = None) -> str:
    """The repository for a task, preferring the manifest's explicit ``repo`` field."""
    if task_meta is not None:
        repo = task_meta.get("repo")
        if isinstance(repo, str) and repo:
            return repo
    return repo_of(task_id)


def repo_grouped_fold_ids(
    repos: Sequence[str],
    n_folds: int = DEFAULT_FOLDS,
    *,
    seed: int = DEFAULT_SEED,
    min_tasks_per_fold: int = MIN_TASKS_PER_FOLD,
) -> np.ndarray:
    """Fold id per row. Whole repos never straddle folds; the same seed gives the same folds."""
    # Deterministic and greedy: the repo order is a seeded shuffle of the SORTED names, then each
    # repo joins the currently-emptiest fold. Sorting first makes the seed the only source of
    # variation, so `seed=42` reproduces a partition exactly across runs and machines.
    if n_folds < 2:  # noqa: PLR2004 (a one-way split is not cross-validation)
        raise DegenerateFoldError(f"repo-grouped CV needs at least 2 folds, got {n_folds}")
    by_repo: dict[str, list[int]] = {}
    for index, repo in enumerate(repos):
        by_repo.setdefault(repo, []).append(index)
    if len(by_repo) < n_folds:
        raise DegenerateFoldError(
            f"repo-grouped CV with {n_folds} folds needs at least {n_folds} repos, got "
            f"{len(by_repo)} — a whole repo cannot be held out, so no fold can be scored"
        )
    order = [str(name) for name in np.random.default_rng(seed).permutation(sorted(by_repo))]
    loads = [0] * n_folds
    fold_of: dict[str, int] = {}
    for name in order:
        fold = min(range(n_folds), key=lambda f: loads[f])
        fold_of[name] = fold
        loads[fold] += len(by_repo[name])
    fold_ids = np.array([fold_of[repo] for repo in repos], dtype=int)
    counts = [int((fold_ids == f).sum()) for f in range(n_folds)]
    thin = {f: count for f, count in enumerate(counts) if count < min_tasks_per_fold}
    if thin:
        raise DegenerateFoldError(
            f"repo grouping left fold(s) {thin} below the {min_tasks_per_fold}-task floor — "
            "too thin to estimate a held-out rate"
        )
    return fold_ids


def stratified_grouped_splits(
    labels: Sequence[bool], groups: Sequence[str], n_splits: int = DEFAULT_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Label-stratified, grouped CV folds — the one partition escalation's arms share.

    STRATIFIED, and that is not cosmetic: plain GroupKFold balances folds by SIZE only, so a
    size/label-coupled corpus gets fold base rates that turn the prior column into a fold-id
    proxy (a confirmed artifact — see `prefix_eval`'s header). Stratifying on the label
    collapses the spread at source. There is deliberately NO fallback to GroupKFold: that IS
    the artifact stratification removes, so a silent degrade would restore the bug unannounced.
    """
    y = np.asarray(labels, dtype=int)
    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:  # noqa: PLR2004 (below two groups there is no partition to build)
        return []
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros((len(y), 1)), y, list(groups)))
