#!/usr/bin/env bash
# Context for the /work command (JSON on stdout). See work_context.py.
set -euo pipefail
SCRIPT_DIR="$(CDPATH="" cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/../python/work_context.py" "$@"
