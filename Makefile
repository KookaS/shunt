# Shunt — common developer commands.
#
#   make docs           Serve the docs site locally with live reload (http://127.0.0.1:8000)
#   make docs-build     Build the docs the way CI does (strict — broken links fail)
#   make stop           Stop all Shunt services started from this repo
#   make benchmark      Full pipeline: collect -> stamp -> evaluate -> report -> summary
#   make benchmark-live Run the live outcome matrix (spends real budget — supervise it)
#   make offline-replay Derive real per-step outcomes from captured diffs (no spend)
#   make escalation-eval Score the escalation detector offline (no spend)
#   make routing-report Regenerate the routing backtest plots/report (no spend)
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

.PHONY: docs docs-build stop help benchmark benchmark-live offline-replay escalation-eval routing-report reconcile-cost
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
	@echo "make routing-report  Regenerate the routing backtest report (no spend)"
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

routing-report:
	$(BENCH) benchmark.routing.report $(ARGS)

# Close the loop between tracked benchmark cost and the real provider bill (no spend).
# --billed is the owner-read Requesty dashboard figure; --timestamp is required.
reconcile-cost:
	$(BENCH) benchmark.cost_reconcile $(ARGS)
