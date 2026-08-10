#!/usr/bin/env bash
# Run one tool × one scenario of the handshake harness. The exit code IS the verdict.
#
#   tests/integrations/run_scenario.sh <tool> [wiring|escalation|live]
#
# Used by `make e2e`, by .github/workflows/integration-handshake.yml, and by hand.
# It exists because the escalation scenario needs THREE ordered steps that a single
# `docker compose up --exit-code-from` cannot express: bring the substrate up, run the
# tool's driver to completion, then run the verdict sidecar. `--exit-code-from` implies
# `--abort-on-container-exit`, which tears the run down the moment the driver exits.
set -euo pipefail

TOOL="${1:?usage: run_scenario.sh <tool> [wiring|escalation|live]}"
SCENARIO="${2:-wiring}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/examples/integrations/$TOOL"
SPEC="$DIR/handshake.yaml"
[ -f "$SPEC" ] || { echo "no such CI-eligible tool: $TOOL ($SPEC missing)" >&2; exit 2; }

# Read one dotted key out of the tool's handshake spec.
spec() {
  python3 - "$SPEC" "$1" <<'PY'
import sys, yaml
node = yaml.safe_load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    node = (node or {}).get(part)
print(node if node is not None else "")
PY
}

require_scenario() {
  local declared
  declared="$(spec scenarios)"
  case "$declared" in
    *"$1"*) ;;
    *) echo "$TOOL does not declare the '$1' scenario (scenarios: $declared)" >&2; exit 2 ;;
  esac
}

# ── the product image ─────────────────────────────────────────────────────────
#
# What this harness tests IS the product image, and the product changes — so reuse is
# conditional on FRESHNESS, not on mere existence. The previous "reuse if present" rule cost
# real signal: two escalation legs passed green against an image built before the fix they
# were supposed to prove, and `shunt explain` inside it still printed the pre-fix model. A
# green run against a stale image is worse than no run, because it is believed.
#
# The fast path survives — a rebuild is minutes of dependency compilation — but only while
# nothing the image is built FROM has changed: src/, pyproject.toml, uv.lock, Dockerfile.
IMAGE="shunt-router:handshake"

# Newest mtime under everything the image is built from. Python, not `find -newer`, because
# the comparison is against a timestamp (the image's creation), not against a file.
newest_source_mtime() {
  python3 - "$ROOT" <<'PY'
import os, sys
root = sys.argv[1]
newest = 0.0
for rel in ("pyproject.toml", "uv.lock", "Dockerfile"):
    path = os.path.join(root, rel)
    if os.path.exists(path):
        newest = max(newest, os.path.getmtime(path))
for dirpath, dirnames, filenames in os.walk(os.path.join(root, "src")):
    # Build outputs are not sources: __pycache__ and *.egg-info are rewritten by any local
    # import, which would report the image stale on every run and never reuse it.
    dirnames[:] = [
        d for d in dirnames
        if d != "__pycache__" and not d.endswith(".egg-info") and not d.startswith(".")
    ]
    for name in filenames:
        if name.endswith((".pyc", ".pyo")):
            continue
        newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
print(f"{newest:.0f}")
PY
}

# The image's creation time as an epoch. Non-zero exit ⇒ no such image.
image_created_epoch() {
  local created
  created="$(docker image inspect --format '{{.Created}}' "$IMAGE" 2>/dev/null)" || return 1
  [ -n "$created" ] || return 1
  python3 - "$created" <<'PY'
import datetime, re, sys
# Docker stamps RFC3339 with NANOsecond precision; fromisoformat takes at most 6 digits.
raw = re.sub(r"(\.\d{6})\d+", r"\1", sys.argv[1].strip()).replace("Z", "+00:00")
print(f"{datetime.datetime.fromisoformat(raw).timestamp():.0f}")
PY
}

human_time() { python3 -c 'import datetime,sys; print(datetime.datetime.fromtimestamp(int(sys.argv[1])).isoformat(sep=" ", timespec="seconds"))' "$1"; }

ensure_image() {
  local created newest
  created="$(image_created_epoch)" || created=""

  if [ "${SHUNT_E2E_REUSE_IMAGE:-}" = "1" ] && [ -n "$created" ]; then
    echo "==> IMAGE $IMAGE built $(human_time "$created") — REUSED UNCHECKED (SHUNT_E2E_REUSE_IMAGE=1); it may be stale"
    return 0
  fi

  # CI is the one place mtimes cannot answer the question. The workflow builds this image in
  # its `build-image` job and `docker load`s it here, but this job's checkout rewrites every
  # source mtime to ITS clone time — later than the build, always. Comparing them would
  # declare a provably-current image stale and rebuild it once per matrix leg. In CI the
  # image is built from THIS commit by construction, so provenance, not mtime, settles it.
  if [ -n "${CI:-}" ] && [ -n "$created" ]; then
    echo "==> IMAGE $IMAGE built $(human_time "$created") — reusing (CI: built from this commit by the build-image job; mtime comparison is meaningless after a fresh checkout)"
    return 0
  fi

  if [ -z "$created" ]; then
    echo "==> IMAGE $IMAGE absent — building"
  else
    newest="$(newest_source_mtime)"
    if [ "$newest" -le 0 ]; then
      # No source found at all — the freshness question is unanswerable, so never answer
      # "fresh". A zero here would otherwise predate every image and reuse forever.
      echo "==> IMAGE $IMAGE: found no sources under $ROOT to date it against — rebuilding"
    elif [ "$created" -ge "$newest" ]; then
      echo "==> IMAGE $IMAGE built $(human_time "$created") — fresh (newest source $(human_time "$newest")), reusing"
      return 0
    else
      echo "==> IMAGE $IMAGE built $(human_time "$created") is STALE (source changed $(human_time "$newest")) — rebuilding"
    fi
  fi

  docker build -t "$IMAGE" -f "$ROOT/Dockerfile" "$ROOT"
  created="$(image_created_epoch)" || { echo "build produced no $IMAGE" >&2; exit 2; }
  echo "==> IMAGE $IMAGE rebuilt $(human_time "$created")"
}

# Called per branch, AFTER that branch's guards: an unknown scenario or a live tier the
# operator did not consent to must be refused in milliseconds, never after a rebuild.
case "$SCENARIO" in
  wiring)
    ensure_image
    COMPOSE="$DIR/compose.yaml"
    SERVICE="$(spec service)"
    export SHUNT_ESC_TOOL="$TOOL"
    # Report the cwd measurement before teardown (the volume dies with `down -v`), and
    # never let it change the leg's verdict — it is an observation, not a check.
    trap 'docker compose -f "$COMPOSE" run --build --rm assert-escalation --cwd-only 2>/dev/null || true; docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
    docker compose -f "$COMPOSE" up --build --abort-on-container-exit --exit-code-from "$SERVICE"
    ;;

  escalation)
    require_scenario escalation
    ensure_image
    COMPOSE="$DIR/compose.escalation.yaml"
    DRIVER="$(spec escalation.service)"
    export SHUNT_ESC_TOOL="$TOOL"
    export SHUNT_ESC_MARKER="${SHUNT_ESC_MARKER:-SHUNT-ESC-$TOOL}"
    # One value for the driver AND the verdict: the sidecar asserts it observed at least as
    # many marker sessions as the driver drove prompts, so an escalation recorded without the
    # session boundaries this scenario is built on cannot pass. Passed with `run -e` because
    # the shared substrate's service block is fixed (examples/integrations/compose.base.yaml).
    export SHUNT_ESC_PROMPTS="${SHUNT_ESC_PROMPTS:-4}"
    # A fresh volume per run: the verdict must come from THIS run's decisions, never from
    # an escalation a previous run left behind in the store.
    docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true
    trap 'docker compose -f "$COMPOSE" logs shunt --tail 40 || true; docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
    docker compose -f "$COMPOSE" up --build -d shunt fake-upstream
    # `--build` on the sidecar too: it is built from the same Dockerfile but under its
    # OWN image name, so `up --build shunt` leaves it stale and an edit to the verdict
    # script would silently not run.
    docker compose -f "$COMPOSE" run --build --rm "$DRIVER"
    docker compose -f "$COMPOSE" run --build --rm \
      -e SHUNT_ESC_MIN_SESSIONS="$SHUNT_ESC_PROMPTS" assert-escalation
    ;;

  live)
    require_scenario live
    # Real provider, real spend. Never reachable from `pull_request` (see the workflow)
    # and never from a bare `make e2e`: the operator must say so out loud.
    [ "${SHUNT_LIVE_I_ACCEPT_SPEND:-}" = "yes" ] || {
      echo "refusing to run the live tier: set SHUNT_LIVE_I_ACCEPT_SPEND=yes and SHUNT_LIVE_API_KEY" >&2
      exit 2
    }
    : "${SHUNT_LIVE_API_KEY:?live tier needs SHUNT_LIVE_API_KEY}"
    ensure_image
    COMPOSE="$DIR/compose.live.yaml"
    DRIVER="$(spec live.service)"
    export SHUNT_ESC_PROMPTS="${SHUNT_ESC_PROMPTS:-4}"
    trap 'docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
    docker compose -f "$COMPOSE" up --build -d shunt
    docker compose -f "$COMPOSE" run --build --rm "$DRIVER"
    docker compose -f "$COMPOSE" run --build --rm \
      -e SHUNT_ESC_MIN_SESSIONS="$SHUNT_ESC_PROMPTS" assert-escalation
    ;;

  *)
    echo "unknown scenario: $SCENARIO (wiring|escalation|live)" >&2
    exit 2
    ;;
esac
