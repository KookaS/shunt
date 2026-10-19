#!/usr/bin/env bash
# Context for the task-contract commands (JSON on stdout). See contract_context.py.
set -euo pipefail
SCRIPT_DIR="$(CDPATH="" cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/../python/contract_context.py" "$@"
