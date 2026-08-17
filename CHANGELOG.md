---
title: Changelog
description: Release history for shunt-router, with what each release does and does not claim.
---

# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [PEP 440](https://peps.python.org/pep-0440/) (`0.1.0a1` is an alpha).

This file is the source for the GitHub release notes, so the two cannot disagree.

## [Unreleased]

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
