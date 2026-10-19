# Task contracts (spec-kit extension)

Adds the task-contract step to the spec-kit flow: specify → clarify → plan → **contract** →
tasks → analyze → implement → converge.

| Command | Claude Code | Codex | opencode |
|---|---|---|---|
| `speckit.contract.feature` (alias `speckit.contract`) | `/speckit-contract` | `$speckit-contract` | `/speckit.contract` |
| `speckit.contract.root` | `/speckit-contract-root` | `$speckit-contract-root` | `/speckit.contract.root` |

The feature command writes `specs/NNN-x/contract.yaml` from `spec.md` and `plan.md`, validates
it with the substrate's `scripts/contract.py`, writes the contract's tests and confirms they
fail. The root command writes a backlog item's `<backlog_dir>/contracts/<id>.yaml`.

Installed and kept current by the substrate's `speckit` project kit (`docs/kits.md`); the
schema is `docs/contracts.md` in the substrate. Do not edit the installed copy under
`.specify/extensions/contract/` — the next sync's update replaces it.
