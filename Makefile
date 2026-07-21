# Shunt — common developer commands.
#
#   make docs        Serve the docs site locally with live reload (http://127.0.0.1:8000)
#   make docs-build  Build the docs the way CI does (strict — broken links fail)
#   make stop        Stop all Shunt services started from this repo
#
# Docs deps are pulled ephemerally with `uv run --with-requirements`, so nothing
# is written into the project venv. mkdocs is the same version CI uses.

.PHONY: docs docs-build stop help
.DEFAULT_GOAL := help

DOCS_REQS := docs/requirements.txt
MKDOCS := uv run --with-requirements $(DOCS_REQS) mkdocs

help:
	@echo "make docs        Serve docs locally with live reload (http://127.0.0.1:8000)"
	@echo "make docs-build  Build docs strictly (what CI runs before gh-pages deploy)"
	@echo "make stop        Stop all Shunt services started from this repo"

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
