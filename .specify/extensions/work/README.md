# Work intake (spec-kit extension)

One command from a plain request to an approval card, so the owner never types a spec-kit
step: backlog item → specify → clarify → plan → contract → checklist → tasks → analyze →
the completeness gate → the review card → STOP for approval. After approval,
`continue` runs implement → converge → the contract harness → the done lint.

| Command | Claude Code | Codex | opencode |
|---|---|---|---|
| `speckit.work.run` (alias `work`) | `/speckit-work` | `$speckit-work` | `/work` |

spec-kit 1.2.0 names every rendered SKILL `speckit-<name>`, so the alias reaches Claude and
Codex as `speckit-work`; opencode keeps it as written. From a session started at the host
root, the project's router runs it: `/<project>-speckit work <request>`.

The command follows each step's own installed command file (found by
`scripts/bash/work-context.sh --agent <driver>`). It never waits on a question: each open
question becomes an assumption (`- **A-n** (assumed) <question> → <default>. Affects: …` in
`spec.md`), the run continues on the recommended default to the card, and the questions are
listed once in a non-blocking `QUESTIONS:` batch before the last line,
`WORK-CARD: <contract.yaml> <card.svg>` (intake) or `WORK-DONE: <contract.yaml>`
(continue). Answers resume the unit, re-running only the steps each changed assumption
affects. A failing tool is logged, fixed in the same turn (never by weakening a check) and,
on a repeat, filed as a backlog item. It never commits the spec, plan, contract or tests:
the owner's approval does.

Installed and kept current by the substrate's `speckit` project kit (`docs/kits.md`); the
flow is `docs/work-intake.md` in the substrate. Do not edit the installed copy under
`.specify/extensions/work/` — the next sync's update replaces it.
