#!/usr/bin/env bash
# Fan `make e2e` out over the declared (tool, scenario) legs. One leg per line of output,
# and a non-zero exit if any REQUIRED leg failed.
#
#   tests/integrations/e2e.sh [tool] [scenario]
#
# Best-effort legs are reported and forgiven, exactly as the CI matrix forgives them via
# continue-on-error — the two must agree, or a green local run means nothing about CI.
# The `live` scenario is skipped here unconditionally: it spends real money and is run by
# hand through run_scenario.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WANT_TOOL="${1:-}"
WANT_SCENARIO="${2:-}"

legs="$(python3 - "$ROOT" "$WANT_TOOL" "$WANT_SCENARIO" <<'PY'
import glob, os, sys, yaml
root, want_tool, want_scenario = sys.argv[1], sys.argv[2], sys.argv[3]
for path in sorted(glob.glob(f"{root}/examples/integrations/*/handshake.yaml")):
    spec = yaml.safe_load(open(path))
    tool = os.path.basename(os.path.dirname(path))
    if want_tool and tool != want_tool:
        continue
    for scenario in spec.get("scenarios") or ["wiring"]:
        if scenario == "live" or (want_scenario and scenario != want_scenario):
            continue
        block = spec if scenario == "wiring" else (spec.get(scenario) or {})
        print(f"{tool} {scenario} {'best-effort' if block.get('best_effort') else 'required'}")
PY
)"

[ -n "$legs" ] || { echo "no legs match TOOL='$WANT_TOOL' SCENARIO='$WANT_SCENARIO'" >&2; exit 2; }

failed=0
while read -r tool scenario kind; do
  [ -n "$tool" ] || continue
  printf '\n===> %s / %s (%s)\n' "$tool" "$scenario" "$kind"
  if "$ROOT/tests/integrations/run_scenario.sh" "$tool" "$scenario"; then
    printf 'PASS  %s / %s\n' "$tool" "$scenario"
  elif [ "$kind" = "best-effort" ]; then
    printf 'FAIL (forgiven, best-effort)  %s / %s\n' "$tool" "$scenario"
  else
    printf 'FAIL (required)  %s / %s\n' "$tool" "$scenario"
    failed=1
  fi
done <<< "$legs"

exit "$failed"
