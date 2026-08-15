---
title: Reproducing the escalation results
description: The exact steps, outputs, and expected numbers for re-running the offline escalation evaluation from a fresh clone — no API keys, no spend.
---

# Reproducing the escalation results

Every number and figure on the [escalation page](escalation.md) comes out of one
offline command run over a corpus committed to this repository. No API keys, no
network calls to a provider, no spend. This page is the checklist for reproducing
them yourself, and for telling a broken environment apart from a real disagreement.

## 1. Clone, and fetch the corpus

The trajectory corpus is stored in [Git LFS](https://git-lfs.com). A clone without
it leaves small pointer stubs where the data should be, so fetch them explicitly:

```bash
git clone https://github.com/KookaS/shunt.git
cd shunt
git lfs install && git lfs pull
```

Skipping this is the single most common way the reproduction fails. The eval
refuses to start on pointer stubs and names the command above; it never silently
scores a partial corpus.

## 2. Install the eval dependencies

```bash
uv sync --extra benchmark
```

The `benchmark` extra carries `matplotlib`, `swebench` and the rest of the eval
stack. The Make target below passes it too (`uv run --extra benchmark`), so a bare
`uv run` — which strips the extra — is never the right invocation.

## 3. Run the eval

```bash
make escalation-eval
```

It reads only committed data. One measured run took 4 min 57 s of wall time on a
four-core container; the work is CPU-bound in the permutation nulls, so a larger
machine finishes sooner.

## What it writes

| Path | What it is |
|------|-----------|
| `benchmark/escalation/reports/metrics.json` | Every number behind the figures, plus the run block (corpus digest, census, permutation count, status) |
| `benchmark/escalation/figures.json` | The figure manifest: per-figure goal, limitations, sample sizes, and the corpus digest each was drawn from |
| `docs/assets/figures/escalation/corpus_and_coverage.png` | Corpus census, per-model coverage, stratified AUROC, prefix admission |
| `docs/assets/figures/escalation/escalation_decision.png` | The two counting modes' ROC curves against the permutation null |
| `docs/assets/figures/escalation/operating_point.png` | The shipped operating point: confusion, precision with interval, budget |
| `docs/assets/figures/escalation/escalation_budget.png` | When the policy fires, and how much of the run is left after it does |
| `docs/assets/figures/escalation/policy_sweep.png` | The `escalate_after_n` × `stale_window` grid, both counting modes |
| `docs/assets/figures/escalation/session_value.png` | The same decision scored at session cadence |

The JSON report also goes to stdout, followed by the two summary tables.

## The numbers a correct run reproduces

The eval is deterministic. Every resampler and every label permutation runs off an
explicitly seeded `random.Random` (default seed `0`), the corpus is loaded in sorted
filename order, and the model fits are deterministic — there is no unseeded
randomness in the pipeline. So a correct run is **bit-identical**, not merely close,
and the check is blunt: after `make escalation-eval`, `git status` must report no
change to `benchmark/escalation/reports/metrics.json`.

The run must be scoring the published corpus, whose digest is
`370b4f954df89cdf` (`run.corpus_digest` in `metrics.json`, and `data_digest` on
every entry of `figures.json`). A different digest means a different corpus, and
nothing below applies.

| Field in `metrics.json` | Value |
|---|---|
| `run.status` | `OK_OFFLINE_ONLY` |
| `run.canonical_deployability.label` | `OFFLINE-ONLY UPPER BOUND` |
| `run.n_trajectories` · `run.n_stamped` | 822 · 723 |
| `run.n_permutations` | 2000 |
| `escalation_decision.png.base_rate` | 0.4177 |
| `escalation_decision.png.auroc_as_shipped` | 0.6003 |
| `escalation_decision.png.auroc_edit_gated` | 0.7782 |
| `escalation_decision.png.null.null_mean` · `.null_sd` · `.p_value` | 0.5103 · 0.0197 · 0.0005 |
| `operating_point.png.as_shipped.precision` (CI) | 0.4177 ([0.3426, 0.4835]) |
| `operating_point.png.as_shipped.n_escalated` | 723 of 723 |
| `operating_point.png.edit_gated.precision` (CI) | 0.5893 ([0.5084, 0.6554]) |
| `operating_point.png.edit_gated.n_escalated` · `lift` | 431 of 723 · 1.4109 |

The figures are bit-identical too under the locked dependency set (`uv.lock`): the
six PNGs a fresh run writes match the committed ones byte for byte. That is a
property of the pinned `matplotlib` and its fonts, not of the eval, so treat a PNG
byte difference as an environment difference and `metrics.json` as the contract.

## When it does not match

**The run stops with `git-LFS pointer file, not trajectory data`.** The corpus was
never fetched. Run `git lfs install && git lfs pull` and try again. Do not delete
the offending files: a smaller corpus produces different numbers, quietly.

**`no trajectories found`.** The corpus directory is empty or `--live-dir` points
somewhere else. The default is `benchmark/escalation/data/live`.

**The digest matches but the numbers do not.** That is an environment difference,
not a disagreement about the data. Confirm the dependency set is the locked one
(`uv sync --extra benchmark`, not a hand-assembled venv) and that the command went
through `uv run --extra benchmark`.

**The digest does not match.** You are scoring a corpus that is not the published
one — a stale checkout, a partial LFS fetch, or locally modified trajectories.
`git status` on `benchmark/escalation/data/` will usually say which.

**You want to re-derive the outcomes, not re-score them.** Re-scoring needs only
what is in git. Re-deriving each step's verified outcome replays the agent's work
in containers and needs inputs that cannot be committed — the SWE-bench instance
images and the gold dataset rows. `make replay-inputs` enumerates every input this
checkout still lacks and exits non-zero rather than half-running. The per-step
state capture *is* committed: `make state-import` restores it into the local
scratch, `make state-verify` proves the restore matches what its index binds, and
`make state-export` is how it was packed. See `benchmark/README.md` for what is and
is not reproducible offline.
