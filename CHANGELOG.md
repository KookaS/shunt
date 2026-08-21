---
title: Changelog
description: Release history for shunt-router, with what each release does and does not claim.
---

# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [PEP 440](https://peps.python.org/pep-0440/) (`0.1.0a1` is an alpha).

This file is the source for the GitHub release notes, so the two cannot disagree.

## [Unreleased]

### Added

- **`shunt inspect` — diagnostic figures over the live outcome store.** A read-only command
  that never spends: it prints the store's census (embedded / labeled / tier-2, seeded vs live,
  `live cost` / `seeded cost` / `cost unknown`) and draws the neighbourhood figure for a prompt.
  matplotlib stays out of the core wheel, so the drawing lives behind a new **`inspect` extra**
  (`pip install 'shunt-router[inspect]'`); without it the command exits with that instruction
  rather than an ImportError traceback.
- **The live router, measured — a new documentation half and seven new figures.**
  [`docs/inference.md`](inference.md) judges the shipped router on its own outcome store rather
  than on a replayed benchmark corpus: what live inference cost, whether the choice
  distribution is collapsing, whether near neighbours still predict outcomes, and the
  off-policy value of routing and escalation. The figures ship in the wheel
  (`shunt.inspect.inference`), are rendered by `make inference-figures`, and land under
  `docs/assets/figures/inference/`. Every panel that cannot be computed says so on the canvas —
  an empty panel is a result, not a gap, and the off-policy numbers are **refused**
  (`NOT_IDENTIFIED`) rather than estimated where the logs cannot identify them. The committed
  render comes from a deterministic seed-only store containing no live traffic, and the page
  says so before the first figure.
- **An illustrative render of those seven figures, on data that is invented.**
  [`docs/inference-demo.md`](inference-demo.md) draws the same family over
  `benchmark/routing/demo_corpus.py` — a synthetic corpus of 703 sessions (453 live + 250 seeded):
  300 drawn live sessions resampled from 40 measured atoms, 153 invented live sessions covering
  escalation scenarios, and 250 seeded sessions from benchmark models — so the panels the measured
  page leaves empty can be read at all. It is illustration, never evidence: nothing on it is a
  measurement, no result cites it, and every canvas is stamped `SYNTHETIC — NOT MEASURED` by the
  renderer rather than by the drawing code. Rendered by `make demo-figures` into its own
  `docs/assets/figures/demo/` half. The embeddings are the one real thing on the page — genuine
  vectors borrowed from the committed seed bundle, so the geometry is real while the association
  to a session is invented — and the off-policy panel still refuses, because the shipped logging
  policy does not randomize.
- **Off-policy evaluation and instrument admissibility now ship in the wheel** as
  `shunt.analysis.ope` (IPS / SNIPS / doubly-robust with cross-fit, ESS, and the
  `IDENTIFIED` / `NOT_IDENTIFIED` verdict) and `shunt.analysis.admissibility` (the
  positive-control + shuffled-label adjudicator). They used to live under `benchmark/`, which
  shipped figures may not import. `benchmark.escalation.ope` and `benchmark.admissibility` are
  now re-export shims over the single implementation — no mirrored copy that can drift and
  silently change a published number.
- **The figure frame, layout audit and plot palette ship in the wheel** as
  `shunt.inspect.plot_frame`, `shunt.inspect.plot_contract` and `shunt.inspect.plot_style`, so
  shipped figure code draws through the same contract without importing `benchmark`.
  `benchmark.plot_frame`, `benchmark.plot_contract` and `benchmark.routing.plot_style` are
  re-export shims and every existing import keeps working; `plot_style` additionally keeps its
  benchmark-corpus-typed helpers local, since they have no place in the wheel. Note that
  importing any of them now forces the matplotlib **Agg** backend, because the shipped frame
  does.
- **`escalation_hold_reason` provenance.** Every escalation HOLD now records *why* it held, as
  one of the fixed reason tokens, in the turn's provenance — previously a hold was
  indistinguishable from an escalation that was never considered, which made the hold rate
  unmeasurable after the fact.
- **New figure targets and a half-scoped freshness gate.** `make inference-figures` (redraw the
  seven inference PNGs; `OUT=/tmp/x` diverts both the PNGs and the manifest to a scratch dir,
  leaving the committed tree untouched) and `make check-inference-figures`. The pipeline's
  `--check-figures` takes a new **`--half {demo,escalation,inference,routing}`** so one half's
  staleness no longer decides another half's exit code; `--half` is check-only and is a hard
  `parser.error` anywhere else. `make benchmark-figures` (the pipeline's `figures` stage)
  remains the only target that **re-records** the freshness manifest — a bare
  `make inference-figures` or `make routing-report` redraws without certifying, and
  `--check-figures` stays red until the stage runs. The manifest digests the committed
  **outputs** (the PNGs and the derived artifacts beside them) as well as the inputs, so a
  figure edited or regenerated out of band is caught too — an input-only digest went green
  whenever the inputs were untouched, however stale the drawing had become.
- **Per-conversation session identity, and model reuse across a resume or a fork.** The router
  now keys a session on the tool's own conversation id when the request carries one —
  `X-Session-Id`, with `x-session-affinity` as an alias and `x-parent-session-id` naming a fork's
  origin — instead of on `(source_ip, user_agent)` alone. A new conversation id starts a new
  session and takes a fresh routing decision; a resumed or forked conversation re-serves the model
  already locked for it (`session_resume` / `fork_resume`) rather than re-deciding mid-task, which
  is what keeps a resumed conversation cache-safe. Clients that send no conversation id (Claude
  Code, aider, plain HTTP) keep the old `(source_ip, user_agent)` grouping — the fallback is
  unchanged, it simply no longer applies to tools that identify their conversations. Header values
  are sanitised and length-capped before they reach the store. See
  [routing](routing.md#session-identity).

### Changed

- **Benchmark-seeded sessions carry a fixed timestamp** (`2020-01-01T00:00:00+00:00`) instead
  of the wall clock at import, so two seed runs from the same bundle write the same rows.
  Intentional rig behaviour change: seeded rows now drop out of the loop-health
  `recent_choices` window and sort last in a session listing. That is the point — the
  routing-collapse alarm was reading the replayed benchmark matrix's model distribution as
  *recent router behaviour*, because seeding stamped every one of its rows later than every
  real session.
- **Database schema is now v4.** The `sessions` table gains a nullable `external_session_id`
  column plus an index on it, which is where the conversation id above is persisted so a resume
  can find its earlier model. The migration is additive and backward-compatible: an existing
  v1/v2/v3 store is migrated in place on the next boot, and rows written before the bump keep
  working with `external_session_id` NULL. No action is required, and there is no downgrade path —
  a v4 store opened by an older build will not see the column.

### Fixed

- **`shunt inspect` and `/admin/loop-health` no longer present benchmark spend as inference
  cost.** Every cost and recency aggregate the *live* router publishes now excludes seeded
  (`bench:`) rows: `loop_health`'s `cost_by_model` and `recent_choices`, and the census panel
  of `shunt inspect`, which replaces its single `total cost:` line with `live cost:`,
  `seeded cost:` and `cost unknown: N sessions`. On a rig holding the 792-row seed corpus the
  old line read `$118.9242` against `$0.2492` of real live spend. `get_stats()["total_cost"]`
  is unchanged — it is the whole-store stat — and gains `live_total_cost` /
  `seeded_total_cost` beside it. Sessions whose provider reported no `usage.cost` are now
  counted and published (`n_cost_unknown`) rather than silently dropped from the sum.
- **The kNN index is reproducible.** The HNSW build passed `hnswlib`'s default
  `num_threads=-1`, so among exact-tie vectors the returned neighbour *ids* varied between
  builds of the identical corpus (38 of 50 probes on the 792-row seed corpus; distances were
  always identical). Pinned to a single thread — 792 rows now index in ~92 ms instead of
  ~20 ms, which is not a cost worth a non-reproducible neighbourhood.
  The two embedding readers that feed the index now also pin `ORDER BY rowid`: SQLite
  guarantees no scan order without one, and the row order they return *is* the index's slot
  layout, so a new index or an engine version bump could have silently reordered it and changed
  a committed figure with no other symptom.

## [0.1.0a1] — unreleased

First published artifact. Everything below already worked in the repository; what is
new is that you can now install it.

### Added

- **Published distribution.** `pip install shunt-router` and
  `ghcr.io/kookas/shunt-router`. Prior to this tag neither existed, while the docs
  said otherwise.
- **`shunt doctor`** — a read-only, non-spending install diagnosis: which provider
  keys resolve (presence only, never the value), how many models are routable (and how
  many have an open circuit breaker), whether the embedding weights are cached, whether
  the bind address is free, whether the learning loop has any labelled outcomes, which
  config values are built-in defaults versus overridden, and whether escalation is
  *armed* or merely enabled and **inert**. That last distinction was previously visible
  only as a boot warning in the logs, so an install doing nothing looked identical to one
  that worked. Exits non-zero only when the router could not serve a request at all —
  the embedder cache is judged against the active `router.strategy`, so a fixed strategy
  that never embeds is not failed for an uncached model. `--json` has a stable shape:
  every check present, in a fixed order, each with an explicit `status`.
- `Documentation` and `Issues` links in the package metadata.

<!-- FEATURE entries land here ONLY once the change is in the tree. This section is
     unreleased, which makes it tempting to list what the release is planned to
     contain — do not. A changelog that names an unshipped feature is the same
     defect as a README that says the package is published: it is checked by
     readers, not by tests. Two such entries were removed before this file was
     first committed.
     The "Published distribution" line above is the exception and not a violation:
     publishing IS the release act this version number denotes, not a feature the
     version contains. It becomes true at the moment the tag exists, which is the
     same moment this section stops saying "unreleased". -->


### Fixed

- `mkdocs.yml` described kNN routing as "not yet live", contradicting `docs/index.md`.
  The router picks the session model on the first turn.
- The `router.yaml` model allow-list comment named `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY` and `GEMINI_API_KEY`. No shipped model reads any of them — ten of
  eleven registry models route via Requesty (`REQUESTY_API_KEY`) and one via DeepSeek.
- `SECURITY.md` did not state that no endpoint authenticates its caller, nor that
  `router.budget.max_spend_usd` is a per-session soft cap rather than a spending defence.
- `shunt start` now rejects an out-of-range `SHUNT_PORT` at startup with a message naming the
  variable, instead of passing it to uvicorn to fail on. Non-integer values still raise, with
  a clearer message.
- `docs/index.md` said "every request forwards to the cheap default" and
  `docs/architecture.md` omitted `escalate` from the CLI verb list — both stale.

### What this release does NOT claim

Stated here because a changelog is where a reader checks what a version is worth.

- **The make-or-break kill gate is `UNTESTED`, not passed.** The committed verdict
  artifact reads `UNTESTED` because the coverage floor was not met (0.516 against a
  required 0.9), and the quality delta it did compute is negative: the kNN router is
  worse than always-frontier by more than the pre-registered 5pp margin on all three
  evidence bases.
- **The learned router contributes nothing.** The router's neighbourhood estimate sits
  inside the shuffled-outcome null, while a three-level human difficulty tag clears that
  null on the same pipeline and the same n — a working positive control beside a negative
  result. `docs/results.md` calls that a falsification of *this* embedding signal — on this
  corpus, at this n, with this encoder — and not a claim that no routing signal exists
  anywhere; the detection floor that bounds it is stated there too.
- **The escalation trigger, as shipped, is a null detector.** It fires on 723 of 723
  replayed trajectories and lands exactly on the base rate. It ships enabled as a
  design choice — one decision per session, cache-safe, bounded spend — not as a
  measured win.
- **The measured saving comes from the mechanism, not from prediction.** The
  session-cadence cascade — the one that ships enabled — costs **~24%** less than
  always-frontier at *indistinguishable* quality on the 74-task fully-measured set
  ($28.76 vs $37.63). On the 184-task set the figure is ~70%, but 35% of those cells are <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
  monotone-imputed as passes; the smaller number is the honest one. The adjacent 28% in
  `docs/results.md` belongs to `Price-Cascade`, which needs a verified outcome mid-session,
  breaks cache-safety, and is **rejected at boot** — it is not a number you can buy.

Full numbers, methods and caveats: [`docs/results.md`](docs/results.md).

[Unreleased]: https://github.com/KookaS/shunt/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/KookaS/shunt/releases/tag/v0.1.0a1
