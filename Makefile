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
#                       (every figure lands in docs/assets/figures/routing/)
#   make check-figures  Prove the committed standalone figures are not stale (seconds, no spend)
#   make inference-figures Regenerate the inference (live-router) figures + manifest (no spend)
#                       (OUT=/tmp/x renders a throwaway copy from the ambient store instead)
#   make check-inference-figures Prove the committed inference figures are not stale (seconds)
#   make seed-bundle    Build the LFS-tracked warm-start seed bundle from committed results.csv
#   make check-seed-bundle Prove the committed bundle is current (seconds, no spend)
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

.PHONY: e2e docs docs-build stop help benchmark benchmark-live offline-replay escalation-eval model-coverage state-capture-check state-capture-mark state-export state-import state-verify replay-inputs routing-report benchmark-figures check-figures inference-figures check-inference-figures demo-figures check-demo-figures seed-bundle check-seed-bundle reconcile-cost live-smoke
.DEFAULT_GOAL := help

DOCS_REQS := docs/requirements.txt
MKDOCS := uv run --with-requirements $(DOCS_REQS) mkdocs
BENCH := uv run --extra benchmark python -m
ARGS ?=

help:
	@echo "make e2e             Run the tool->Shunt handshake harness over Docker (no spend)"
	@echo "                       TOOL=curl  one tool     SCENARIO=escalation  one scenario"
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
	@echo "make inference-figures Regenerate the inference figures + manifest (OUT=/tmp/x for a scratch copy)"
	@echo "make check-inference-figures Verify the committed inference figures are current (seconds)"
	@echo "make demo-figures    Redraw the ILLUSTRATIVE demo figures (synthetic, watermarked, evidence of nothing)"
	@echo "make check-demo-figures Verify the committed demo figures are current (seconds)"
	@echo "make seed-bundle     Build the LFS-tracked warm-start seed bundle (real fastembed, embeds results.csv)"
	@echo "make check-seed-bundle Prove the committed seed bundle is current (seconds, no spend)"
	@echo "make reconcile-cost  Reconcile tracked cost vs the real bill (ARGS=\"--billed 35 --timestamp 2026-07-27T00:00:00\")"
	@echo "make live-smoke      One real, cheap session through Shunt against a real provider (ARGS=\"--live\")"

# The tool→Shunt handshake harness, over Docker, against a fake upstream: no key, no
# spend. With no TOOL it runs every declared (tool, scenario) leg and reports which
# failed; with TOOL=<name> it runs that tool alone, and SCENARIO= narrows further.
#
#   make e2e                                  every leg
#   make e2e TOOL=curl                        curl's wiring + escalation
#   make e2e TOOL=curl SCENARIO=escalation    one leg
#
# The `live` scenario is deliberately NOT reachable from here — it spends real money and
# is gated behind explicit env vars in run_scenario.sh.
TOOL ?=
SCENARIO ?=
e2e:
	@tests/integrations/e2e.sh $(TOOL) $(SCENARIO)

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
	SHUNT_PLOT_STRICT=1 $(BENCH) benchmark.escalation.run_eval $(ARGS)

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
	SHUNT_PLOT_STRICT=1 $(BENCH) benchmark.routing.report $(ARGS)

# The standalone figures under benchmark/routing/scripts/, written to docs/assets/figures/routing/ so
# the docs can link them relatively. Heavy (real fastembed), so they are
# a deliberate target rather than a pre-commit hook — `make check-figures` is the cheap gate.
# SHUNT_PLOT_STRICT makes benchmark/plot_contract.py refuse to write a figure with an
# overlapping or clipped artist, so a broken layout fails the regeneration instead of being
# committed and found later by whoever opens the PNG.
benchmark-figures:
	SHUNT_PLOT_STRICT=1 $(BENCH) benchmark.pipeline --from figures $(ARGS)

check-figures:
	$(BENCH) benchmark.pipeline --check-figures $(ARGS)

# THE FIGURE-STALENESS TRAP, for both families. A stage certifies only the jobs it
# REGENERATED, so a target that draws without re-recording leaves the job STALE:
#   * a bare `make routing-report` renders and never writes the manifest digest;
#   * `--from report` re-renders the escalation `run_eval` job through _escalation_status()
#     as a STATUS PROBE (pipeline.py:1004), which never calls write_figure_manifest — so it
#     burns ~25 min and the job is still stale.
# The minimal correct repair after touching figure inputs is `--from evaluate`
# (evaluate -> report -> figures):
#   $(BENCH) benchmark.pipeline --from evaluate
# Certification records BOTH digests — the job's inputs and the SHA-256 of each committed
# output it drew — so `--check-figures` reports four conditions, not one: STALE (inputs moved
# since the draw), MISSING (no PNG on disk), DRIFTED (the committed bytes are not the bytes
# certified for that job — a figure edited or redrawn outside the certifying stage) and
# UNCERTIFIED (the entry predates output digesting, so no output bytes are on record). An
# input-only gate called two genuinely drifted figures green; DRIFTED and UNCERTIFIED are why
# it no longer can. All four repair the same way — `--from evaluate`, never a hand edit to
# benchmark/routing/figure_inputs.json.
# `shunt/inspect/inference/**` is NOT in the benchmark digest closure (only plot_frame and
# plot_style are), so inference-figure edits do not re-stale the routing/escalation figures.

# The inference family: the LIVE ROUTER measured on its own outcome store, drawn by shipped
# code under src/shunt/inspect/inference/ (the rig container renders the same seven with
# `python -m shunt.inspect.inference`, having no benchmark/ of its own). Two modes, and the
# difference is the DATABASE, not the code:
#
#   make inference-figures                 the COMMITTED docs figures under
#                                          docs/assets/figures/inference/. Built from the
#                                          seed-only docs corpus — deterministic, no network,
#                                          no live rig — and the ONLY mode that may touch the
#                                          committed manifest.
#   make inference-figures OUT=/tmp/x      the same seven from whatever SHUNT_DATA_DIR points
#                                          at (seed + live rows), into /tmp/x, with the
#                                          manifest diverted to /tmp/figures.json. Never
#                                          touches the committed set.
#
# Exits non-zero when the OPE instrument is inadmissible: F7 refuses before drawing, so a
# zero here would ship a half-rendered family.
OUT ?=
inference-figures:
	SHUNT_PLOT_STRICT=1 $(BENCH) benchmark.routing.render_inference_figures \
		$(if $(OUT),--out-dir "$(OUT)",) $(ARGS)

# The cheap staleness gate for the inference half — seconds, draws nothing.
check-inference-figures:
	$(BENCH) benchmark.pipeline --check-figures --half inference $(ARGS)

# The DEMO half: the same seven drawings over `benchmark/routing/demo_corpus.py`, a synthetic
# corpus of 703 sessions (300 drawn from 40 measured atoms, 153 invented, 250 seeded). It exists
# so `docs/inference-demo.md` can show what a populated panel looks like; NOTHING it draws is a
# measurement, and every canvas is stamped `SYNTHETIC — NOT MEASURED` by the renderer rather than
# by the drawing code.
#
#   make demo-figures                the COMMITTED demo figures under docs/assets/figures/demo/
#   make demo-figures OUT=/tmp/x     a scratch copy, manifest diverted beside it
#
# SAME STALENESS TRAP as above: this target RENDERS and does not re-record. The demo job's
# stage is FIGURES, so `--from figures` (make benchmark-figures) is what certifies it.
demo-figures:
	SHUNT_PLOT_STRICT=1 $(BENCH) benchmark.demo.render_demo_figures \
		$(if $(OUT),--out-dir "$(OUT)",) $(ARGS)

check-demo-figures:
	$(BENCH) benchmark.pipeline --check-figures --half demo $(ARGS)

# The LFS-tracked warm-start seed bundle: precomputed per-embedding-fingerprint embeddings
# of the benchmark's MEASURED outcome cells, so a fresh deployment can warm the live kNN
# index without re-embedding. Heavy (real fastembed), so a deliberate target — NOT a
# pre-commit hook. `check-seed-bundle` is the cheap staleness gate (seconds, no spend).
seed-bundle:
	$(BENCH) benchmark.routing.build_seed_bundle $(ARGS)

check-seed-bundle:
	$(BENCH) benchmark.routing.build_seed_bundle --check $(ARGS)

# Close the loop between tracked benchmark cost and the real provider bill (no spend).
# --billed is the owner-read Requesty dashboard figure; --timestamp is required.
reconcile-cost:
	$(BENCH) benchmark.cost_reconcile $(ARGS)

# One real, cheap session through the shipped proxy against a real provider — the
# live smoke (docs/live-smoke-runbook.md). Gated IN THE SCRIPT, not here: it refuses
# without --live/--confirm AND an interactive TTY confirmation, so a bare `make
# live-smoke` can never spend. Run supervised and tee the output.
live-smoke:
	uv run python benchmark/runner/live_smoke.py $(ARGS)
