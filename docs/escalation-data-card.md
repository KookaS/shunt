---
title: Escalation corpus data card
description: What the escalation trajectory corpus is, where its labels come from, what it does not cover, and how to obtain and verify it.
---

# Escalation corpus data card

Every escalation result on this site is of the form "under *this* data, under these
assumptions, we measure this." This page is the *this data* half. It describes the
trajectory corpus the escalation detector is scored on — what is in it, how its labels
were produced, and, at length, what it cannot support.

For what the escalation detector *is* and what the numbers came out to, see
[Escalation](escalation.md). For the separate routing dataset, see
[Benchmark dataset](benchmark-data.md). The two are different corpora with different
provenance; nothing here describes the routing matrix.

## Provenance

The challenges are **SWE-bench Verified** instances, pinned to one HuggingFace dataset
revision (`princeton-nlp/SWE-bench_Verified` at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`,
declared once in `benchmark/runner/swebench_specs.py`). The same pin materialises the
per-instance specs under `benchmark/challenges/swebench_verified/` and supplies the gold
`patch` / `test_patch` rows the replay grades against, so a stamp can never be graded
against a different revision than the task was built from.

The traces are **agent sessions**, captured live and then normalized into one frozen
schema (`benchmark/escalation/schema.py`: a header record, then one `StepView` per
agent decision). Four normalizers exist — `mini_swe_agent`, `swe_agent`, `openhands`
and `swe_smith`, under `benchmark/escalation/normalize/` — but the committed corpus is
**entirely mini-swe-agent**: all 822 trajectories declare `framework="mini_swe_agent"`.
The other three parsers are an unexercised seam, not coverage. <!-- generated-by: benchmark.escalation.corpus:census -->

Labels are **not** taken from the agent's own reports. Two levels:

- **Terminal label** (`terminal_resolved`, on the header) — whether the session solved
  the instance, from SWE-bench's grading harness.
- **Per-step labels** — produced by *container replay*. `benchmark/runner/offline_replay.py`
  reads the per-step workspace diffs captured during the live run, and for each step
  starts the instance's own image, applies that step's diff to the base commit, applies
  the gold `test_patch`, and runs SWE-bench's own test command
  (`MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]`) over SWE-bench's own test
  directives (`get_test_directives`, delegated — never re-derived).

Adjudication of each replay run is `benchmark/runner/swebench_grading.py`, which is
deliberately **not** the test command's exit code. It calls SWE-bench's own log parser
(`MAP_REPO_TO_PARSER`), its own report builder (`get_eval_tests_report`) and its own
resolution rule (`get_resolution_status`). The exit code and the grade disagree on real
instances — a test in a patched file that SWE-bench put in neither list sets the exit
code but cannot change the grade — so reading the exit code would have stamped steps
against a rule the official grader does not use. `benchmark/runner/swebench_harness.py`
is the surrounding container plumbing.

## Label definition

`terminal_resolved` and every per-step stamp mean the same thing, and it is a
**whole-spec gate**:

> The step is **green** iff every test in `FAIL_TO_PASS` ∪ `PASS_TO_PASS` for that
> instance passes in the replayed workspace. It is **red** otherwise, and
> `failing_check_id` is the first test in that set still failing.

Three consequences follow, and they matter for reading any escalation number:

- The stamp is **state-contingent, never action-contingent**. A `git log`, a `cat` and
  a full `pytest` run all receive the same key while the target spec is red.
- Because the F2P set is fixed per instance, same-key recurrence is a **per-instance
  time-to-fix counter**, not a "the agent kept making the same mistake" signal.
- XFAIL counts as a pass and SKIPPED demotes to unmeasured, both because that is
  SWE-bench's own semantics; a skipped F2P test would otherwise divide 0 by 0 and be
  reported as a confident green.

## Census

Every count below is derived at test time from `benchmark.escalation.corpus.census()`
and the committed `benchmark/escalation/reports/metrics.json`; a test
(`tests/escalation/test_escalation_data_card_coherence.py`) fails if this page and those
sources disagree.

| | |
|---|---:|
| Trajectories (one per live session) | **822** |
| Steps | **30,541** |
| Distinct SWE-bench Verified instances | **166** |
| Upstream repositories | **10** |
| Models | **6** |
| Reasoning arms | **6** |
| Per-step stamped trajectories | **723** |
| Median steps per run (range) | **32** (8–247) |
<!-- generated-by: benchmark.escalation.corpus:census -->

Terminal failure rate — the share of sessions that did **not** resolve their instance —
is **0.459** over all 822 trajectories and **0.418** over the 723 stamped ones. Escalation
figures that score per-step data quote the second; the session-cadence figure quotes the
first. They are different denominators, not a discrepancy. <!-- generated-by: benchmark.escalation.run_eval -->

Per model:

| Model | Trajectories | Stamped | Unstamped | Stamped share |
|---|---:|---:|---:|---:|
| deepseek-v4-flash | 272 | 248 | 24 | 0.912 |
| gpt-5-mini | 284 | 266 | 18 | 0.937 |
| kimi-k2.5 | 111 | 94 | 17 | 0.847 |
| kimi-k3 | 59 | 52 | 7 | 0.881 |
| qwen3.7-plus | 65 | 47 | 18 | 0.723 |
| zai-glm-5.2 | 31 | 16 | 15 | 0.516 |
| **Total** | **822** | **723** | **99** | **0.880** |
<!-- generated-by: benchmark.escalation.run_eval -->

## Known defects and coverage gaps

This is the section the page exists for. Read it before quoting any escalation number.

### Stamping coverage is model-correlated

99 of 822 trajectories carry no per-step verified outcomes and are dropped from every
per-step figure. The drop is **not uniform**: it ranges from 6.3% of `gpt-5-mini` runs
to 48.4% of `zai-glm-5.2` runs (see the table above). Stamping coverage tracks capture
*date*, capture date tracks *model*, so model and coverage are confounded on this corpus
and cannot be separated from within it. Any per-step result is therefore measured on a
population whose composition differs from the corpus's by model. <!-- generated-by: benchmark.escalation.run_eval -->

What that confound does to the one surviving per-step result is measured rather than
argued: `benchmark/escalation/coverage_sensitivity.py` re-scores the canonical cell on
nested strata cut at the observed per-model shares, each against its own nulls
(`coverage_sensitivity.strata[]` in the committed metrics). The separation holds at every
rung, including the best-covered single model. The ladder and its power cost are published
with the [escalation claim](escalation-claim.md). <!-- generated-by: benchmark.escalation.run_eval -->

### ~6% of steps are green because their state could not be reconstructed

The per-step diff recorder originally captured `git diff` — unstaged changes to tracked
files only — so the moment an agent staged, committed or stashed its work, the capture
collapsed to empty while the work was still there. Replay then rebuilds that step as
"base commit + nothing", which is indistinguishable from a real capability failure and
is not one. `benchmark/runner/state_capture_audit.py` detects that class post hoc and
marks the affected steps **unmeasured**, written as the sentinel
`(success=True, confirmed=False, is_infra_failure=True)`.

**1,957 steps — 6.4% of the corpus — across 309 of the 822 runs carry that sentinel.**
They are green in the data and were never measured. The forward fix (`git diff HEAD`)
has landed, so runs collected after it cannot reproduce the staging half; a *commit*
still defeats it, because the recorder is not told the instance's base SHA. A second,
narrower tranche — a capture that is incomplete rather than empty — is documented in the
same module and falls entirely on one model's scaffold. <!-- generated-by: benchmark.runner.state_capture_audit -->

### The flake guard is never exercised

The shipped escalation rule drops any failure with `confirmed=False` — a failure that
did not reproduce on re-run. On this corpus that guard is satisfied **by construction**:
`normalize/mini_swe_agent.stamp_step` hardcodes `confirmed=True`, and
`offline_replay.replay_step` runs each step's test directives exactly once, so there is
no second execution a genuine `confirmed` could come from. The guard's effect is
**unmeasured, not measured-as-zero**. It is an assumption stored as data. `confirmed`
also doubles as the stamped-ness marker `features.is_stamped` reads, so the two cannot
be decoupled without a schema change.

### Fields that are empty or constant

Nine `StepView` fields are **0% populated** on every one of the 30,541 committed steps:
`test_passed`, `test_total`, `subgoal_progress`, `model`, `reasoning_effort`,
`rank_index`, `effort_index`, `real_cost`, `replay_rc`. Three more are present but
**constant**: `is_revert` (always `False`), `retry_count` (always `0`), `loop_signal`
(always `False`).
<!-- generated-by: benchmark.escalation.corpus:census -->

No feature may be built on any of them, and three consequences are worth stating out loud.
`real_cost` being empty means **this corpus cannot price an escalation** — every cost
claim about escalation comes from elsewhere. `model` and `reasoning_effort` being empty
means the per-model breakdown above is read off the trajectory *id*
(`<instance>__<model>__<effort>`, via `features.model_of`), not off a recorded field.
`replay_rc` being empty says this corpus predates that field: the replay harness's return
code was folded into `exit_code` when these runs were stamped.

Three surviving fields are also **one column under three names**: `stamp_step` writes
`success`, `failing_check_id` and `blocking` in a single assignment, so
`success == (failing_check_id is None) == (not blocking)` holds by construction. Only
`fail_rate` is kept as a feature.

### No multi-session trajectories

One committed trajectory is **one session**. Nothing in this corpus spans several
sessions on the same task under one agent run. The shipped router decides **once per
session**, so a session-cadence replay of the shipped rule is structurally impossible
here — not merely unrun. Every per-step escalation number is a per-step policy the
product does not run, which is why the eval publishes a `deployability` verdict of
**OFFLINE-ONLY UPPER BOUND** rather than a shipped result. The session-cadence figure in
[Escalation](escalation.md) works around this by comparing *separate* sessions on the
same instance, which makes it observational.

### Privacy and projection

`COMMITTABLE_FIELDS` in `schema.py` names the behaviour-only subset a projected export
*would* carry — but nothing enforces it on the write path, and the shipped corpus is
**unprojected**: every committed step carries `metadata`, `observation`, `action`, `args`
and `result` as the live agent wrote them. What does run is secret redaction on every
free-text field before the bytes are written. A credential sweep returned zero hits; the
residual disclosure is upstream repository content (container paths under `/root/`) and
one provider-issued `tool_call_id` per step.

## Access mechanics

The corpus is **git-LFS-tracked**. `.gitattributes` routes
`benchmark/escalation/data/**/*.jsonl` (and `*.parquet`) through LFS, so a clone without
`git lfs pull` gets pointer files and the evals cannot read anything. `manifest.json` is
deliberately kept **out** of LFS so the integrity ledger stays diffable in review.

The per-step workspace diffs that replay consumes are **gitignored** — they live under
`benchmark/runner/artifacts/step_snapshots/` on the collecting host. Two targets move
them:

```bash
make state-export   # pack the captures into escalation/data/live/state/ (one .tar.gz per run)
make state-import   # restore them byte-identically on a fresh checkout
make state-verify   # prove the committed capture restores to what its index binds
```

No export is committed in this repository yet: the only file under
`escalation/data/live/state*` is `state_capture.json`, the per-step capture-health audit,
which holds no diffs. Until `make state-export` is run on the collecting host and its output
committed, `state-import` and `state-verify` fail with `no committed state plane`.

A checkout missing them raises `SnapshotsMissingError` rather than replaying a partial
run, because filesystem absence cannot be told from "this run captured nothing".

Two inputs are outside git and always will be: the SWE-bench instance images (~100 GB)
and the gold `patch` / `test_patch` rows, fetched from HuggingFace at the pinned
revision. `make replay-inputs` enumerates every input this checkout still lacks and
**exits non-zero** — a partial reproduction that silently produces different numbers is
worse than a refusal.

So: a clone can **re-score** policies over the committed corpus unconditionally
(`make escalation-eval`, no containers, no spend). Re-**deriving** the labels needs
Docker, the images and HuggingFace access.

## Integrity, and the ceiling of the check

The scored corpus carries a short deterministic fingerprint, recorded as `corpus_digest`
in `benchmark/escalation/reports/metrics.json`. At the committed run it is
**`370b4f954df89cdf`**. It is a fingerprint of the scored population — trajectory count,
stamped count, challenge ids — with no timestamp and no git sha, so it moves when the
data moves and not otherwise. <!-- generated-by: benchmark.escalation.run_eval -->

`benchmark/escalation/authenticity.py` is the integrity check, and its own header states
plainly what it is: **a consistency check, not a tamper detector.** It recomputes the
content hash, `n_steps`, and every derivable field, and cross-checks each trajectory
against `manifest.json`. That catches corruption and careless editing, which is what it
was built for.

What it does **not** catch was measured by running each attack on copies of real
committed data:

- flip a step's `success`, rehash, regenerate the manifest, and fix the derived
  `blocking` → **zero errors**;
- rewrite every step to success, consistently → **zero errors**;
- append 500 fabricated failing steps and rehash → **zero errors**;
- flip `terminal_resolved`, the eval's own label, and regenerate the manifest → **zero
  errors**.

A forger who keeps the invariants passes completely. This is a ceiling of the design,
not a gap in the implementation: every value is recomputed from the same file that
declares it, so nothing here can testify that the file describes a run that actually
happened. Signing the manifest with a key the collector holds, and sampled re-execution,
are what would. Until then, treat the label binding as tamper-*evident* for edits made
outside the two collection writers, never as tamper-*proof*. One further honesty note in
that module: the `recompute_dedup_key` leg is currently a no-op and is named as such
rather than left to look like a fourth safeguard.

## Licensing

The challenges derive from **SWE-bench Verified** (`princeton-nlp/SWE-bench_Verified`)
at the pinned revision above; the task content itself is drawn from open-source Python
repositories — ten of them in this corpus — and each instance carries that upstream
project's own license.
Consult the upstream dataset card and the source repositories before redistributing
derived task text — this project pins and cites the dataset, it does not relicense it.

The trajectory schema reserves a per-trajectory `license` field
(`TrajectoryHeader.license`), and it is a gap worth naming: it is **null on all 822
committed trajectories**, as is `dataset_revision`. Provenance is therefore
pinned in code and in this page, not recorded per record. <!-- generated-by: benchmark.escalation.corpus:census -->

The captured trajectories themselves — the agent's actions and the replayed outcomes —
are produced by this project and ship under the repository's
[Apache-2.0](https://github.com/KookaS/shunt/blob/main/LICENSE) license.
