# Shunt — common developer commands.
#
#   make docs           Serve the docs site locally with live reload (http://127.0.0.1:8000)
#   make docs-build     Build the docs the way CI does (strict — broken links fail)
#   make stop           Stop all Shunt services started from this repo
#   make benchmark      Full pipeline: collect -> stamp -> evaluate -> report -> summary
#   make benchmark-live Run the live outcome matrix (spends real budget — supervise it)
#   make offline-replay Derive real per-step outcomes from captured diffs (no spend)
#   make escalation-eval Score the escalation detector offline (no spend)
#   make model-coverage Flag enabled models the live collection has not covered (no spend)
#   make state-capture-check Prove no corpus step is stamped from a state that was never captured
#   make state-capture-mark  Mark those steps unmeasured + write the audit record (rewrites the corpus)
#   make state-export   Commit the per-step state capture (~1.7 MB) so a clone can re-derive
#   make state-import   Restore that capture into the local scratch on a fresh clone
#   make state-verify   Prove the committed state capture restores to what its index binds
#   make replay-inputs  List every replay input this checkout still lacks (fails if any)
#   make routing-report Regenerate the routing backtest plots/report (no spend)
#   make benchmark-figures Regenerate the standalone routing figures + their manifest (no spend)
#   make check-figures  Prove the committed standalone figures are not stale (seconds, no spend)
#   make reconcile-cost Reconcile tracked benchmark cost against the billed bill (no spend)
#
# Docs deps are pulled ephemerally with `uv run --with-requirements`, so nothing
# is written into the project venv. mkdocs is the same version CI uses.
#
# The benchmark targets ALWAYS go through `uv run --extra benchmark` so the eval
# extras (mini-swe-agent, swebench, matplotlib, …) are present. A bare `uv run`
# auto-syncs the venv to the default locked deps and STRIPS that extra — a live
# scaffold imports `minisweagent` lazily per cell, so a stripped venv fails every
# cell with `No module named 'minisweagent'`. Pass extra flags via ARGS=…, e.g.
# `make benchmark-live ARGS="--live --max-cost 2"`.

.PHONY: docs docs-build stop help benchmark benchmark-live offline-replay escalation-eval model-coverage state-capture-check state-capture-mark state-export state-import state-verify replay-inputs routing-report benchmark-figures check-figures reconcile-cost
.DEFAULT_GOAL := help

DOCS_REQS := docs/requirements.txt
MKDOCS := uv run --with-requirements $(DOCS_REQS) mkdocs
BENCH := uv run --extra benchmark python -m
ARGS ?=

help:
	@echo "make docs            Serve docs locally with live reload (http://127.0.0.1:8000)"
	@echo "make docs-build      Build docs strictly (what CI runs before gh-pages deploy)"
	@echo "make stop            Stop all Shunt services started from this repo"
	@echo "make benchmark       Full pipeline collect->stamp->evaluate->report->summary (ARGS=\"--from report\")"
	@echo "make benchmark-live  Run the live outcome matrix (ARGS=\"--live --max-cost 2\")"
	@echo "make offline-replay  Derive real per-step outcomes from captured diffs (no spend)"
	@echo "make escalation-eval Score the escalation detector offline (no spend)"
	@echo "make model-coverage  Flag enabled models the live collection has not covered"
	@echo "make state-capture-check Prove no step is stamped from a state that was never captured"
	@echo "make state-capture-mark  Mark those steps unmeasured + write the audit record"
	@echo "make state-export    Commit the per-step state capture (~1.7 MB) for off-host replay"
	@echo "make state-import    Restore that capture into the local scratch (fresh clone)"
	@echo "make state-verify    Prove the committed capture restores to what its index binds"
	@echo "make replay-inputs   List every replay input this checkout lacks (fails if any)"
	@echo "make routing-report  Regenerate the routing backtest report (no spend)"
	@echo "make benchmark-figures Regenerate the standalone routing figures + manifest (no spend)"
	@echo "make check-figures   Verify the committed standalone figures are current (seconds)"
	@echo "make reconcile-cost  Reconcile tracked cost vs the real bill (ARGS=\"--billed 35 --timestamp 2026-07-27T00:00:00\")"

# Live-reload preview. Ctrl-C to stop. This is the same config gh-pages ships.
docs:
	$(MKDOCS) serve

# Strict build — mirrors .github/workflows/docs.yml. Output lands in ./site.
docs-build:
	$(MKDOCS) build --strict

# Stop only what THIS repo starts: the docker-compose stack (project "shunt") and
# any local `mkdocs serve`. The wrapper's `shunt-local` rig is a different compose
# project and is deliberately left untouched.
stop:
	-docker compose -f docker-compose.yml down
	-pkill -f "mkdocs serve" 2>/dev/null || true
	@echo "Stopped Shunt services from this repo (mkdocs serve + docker-compose stack)."

# Benchmark entrypoints — all forced through `uv run --extra benchmark` (see header).
# Every `--live` run spends real budget: launch it supervised and tee'd to a log.

# The one-command lifecycle: collect -> stamp -> evaluate -> report -> summary. The
# underlying modules below stay runnable on their own for debugging. Compute-only by
# default; add ARGS="--live --max-cost 2" to actually collect (supervise the spend).
benchmark:
	$(BENCH) benchmark.pipeline $(ARGS)

benchmark-live:
	$(BENCH) benchmark.runner.run_matrix $(ARGS)

offline-replay:
	$(BENCH) benchmark.runner.offline_replay $(ARGS)

escalation-eval:
	$(BENCH) benchmark.escalation.run_eval $(ARGS)

# Reads the committed corpora only. Exits nonzero when a model listed in `models:` has too little
# collected data to be evaluated — the signal that a live collection is incomplete, not finished.
model-coverage:
	$(BENCH) benchmark.model_coverage $(ARGS)

# The state-capture gate. The 2026-07 corpus was captured with a bare `git diff`, which cannot see
# staged/stashed/committed work: a step whose capture collapsed to 0 bytes replays against a
# pristine base, and one that merely LOST its staged half replays against a tree the agent never
# had — both stamped a failure that was never measured. The gate marks both classes
# (`empty_after_nonempty`, `partial_stage_loss`). `-check` is read-only and exits non-zero on any
# such step; `-mark` renders them unmeasured and writes the committed audit record. The forward fix
# for the staging half is `step_snapshots.DIFF_COMMAND` = `git diff HEAD`. No Docker, no spend.
state-capture-check:
	$(BENCH) benchmark.runner.state_capture_audit $(ARGS)

state-capture-mark:
	$(BENCH) benchmark.runner.state_capture_audit --apply $(ARGS)

# The committed state plane. `offline_replay` derives every per-step outcome from the gitignored
# per-step diffs, so without these a clone can re-score a policy but cannot re-derive an outcome
# (`SnapshotsMissingError`). Measured: 34 105 506 B of diffs -> 792 archives, 1 647 548 B on disk,
# ~1.5 MB in a packfile — plain git, deliberately NOT LFS (LFS stores each object whole, no delta).
# `-export` is idempotent on CONTENT, so a re-run on a host with a different zlib cannot churn the
# tree. `-verify` is the no-Docker, no-network guard. `replay-inputs` lists what is STILL missing —
# the instance images and the HF gold rows are not committable, and it says so rather than
# half-running.
state-export:
	$(BENCH) benchmark.runner.snapshot_archive export $(ARGS)

state-import:
	$(BENCH) benchmark.runner.snapshot_archive import $(ARGS)

state-verify:
	$(BENCH) benchmark.runner.snapshot_archive verify $(ARGS)

replay-inputs:
	$(BENCH) benchmark.runner.snapshot_archive requirements $(ARGS)

routing-report:
	$(BENCH) benchmark.routing.report $(ARGS)

# The 12 figures under benchmark/routing/scripts/. Heavy (real fastembed), so they are a
# deliberate target rather than a pre-commit hook — `make check-figures` is the cheap gate.
benchmark-figures:
	$(BENCH) benchmark.pipeline --from figures $(ARGS)

check-figures:
	$(BENCH) benchmark.pipeline --check-figures $(ARGS)

# Close the loop between tracked benchmark cost and the real provider bill (no spend).
# --billed is the owner-read Requesty dashboard figure; --timestamp is required.
reconcile-cost:
	$(BENCH) benchmark.cost_reconcile $(ARGS)
