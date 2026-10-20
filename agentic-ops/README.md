# Agent ops

This directory is installed and kept current by the `agent-ops` project kit. It is
**managed content**: the source of truth is `kits/agent-ops/` in the substrate, and the
project's sync restores these files on every run. Edit the kit, not this copy.

## How the kit is selected

A project opts in through the host's kit selection:

```yaml
enable:
  project_kits:
    .: {kits: [project-base, speckit, agent-ops]}
```

The marker `.agentic-kit.json` records the selection in its existing `kits` list and the
installed version in `kit_versions`. The agent-ops kit contributes **no new marker key**
and no provider role: the flow files are ordinary configuration, so a project can add the
kit without changing what the spec gate, the commit check or the approval policy read.

Sync is the only writer of the installed surfaces. `python3 scripts/sync.py --target
<repo>` re-copies managed files, writes the seeded config when it is absent, and removes
managed files it recorded when the kit is deselected. Do not hand-edit an installed copy:
the next sync overwrites it.

## What it installs

| Path | Kind | Purpose |
|---|---|---|
| `agentic-ops/flow.yaml` | managed | Flow definitions and the shared step vocabulary |
| `agentic-ops/README.md` | managed | This page |
| `agentic-ops/ci/agent-ops.yml` | managed | A portable CI template: spec/contract lint, tests, step-record verifier placeholder |
| `agentic-ops/config.yaml` | seeded | Project and lane placeholders, written once and then the project's |

The seeded config holds no secrets. Point it at the project's central services and choose
the default lane; wire credentials through the host's existing secret mechanism.

## Flows and steps

`flow.yaml` defines `delivery` and the shorter `spike`, `incident`, `research`, `docs` and
`catalogue-change` flows. Every step carries:

| Field | Meaning |
|---|---|
| `id` | Stable step name, unique within its flow |
| `required` | `true` when the step must pass; `false` when it may be skipped |
| `skill` / `prompt` / `delegate` | Exactly one executor: a skill, a prompt, or a substrate delegate |
| `produces` | The artifact or evidence the step must leave behind |
| `verifier` | How the step is judged |
| `on_fail` | Where a failure returns |
| `skip_when` | The condition that excuses an optional step; `never` for a required one |

The status vocabulary is shared with `step-run/1` records: `pass`, `fail`, `partial`,
`not_run`, `stale`, `waived`, `skipped`, `abandoned`. An unreached step is `not_run`; an
optional step excused by `skip_when` is `skipped`; an owner-approved exception is
`waived`. Tokens and cost may be absent and never fail a step.

The `delivery` flow is the reference order. `design`, `test-review`, `arch-review` and
`metric-check` are optional; the retrospective is required in every flow.

## Relationship to the spec-kit work flow

The agent-ops kit **does not replace** the spec-kit work flow, the task contract, the
approval rules or the bridge. It names the existing steps so a run can be recorded
consistently:

| Agent-ops step | Existing surface |
|---|---|
| `idea` | Work intake (`work` extension) |
| `spec` | `speckit-specify` |
| `metric` | `speckit-clarify`, then the contract's metric rows |
| `tests` | `speckit-contract` (tests first, red run recorded) |
| `design` | `speckit-plan` |
| `goals` | `speckit-contract-root` |
| `execute` | `speckit-implement` |
| `test-pass`, `metric-check` | `contract_run` |
| `code-review` | `software-reviewer` |
| `test-review` | `software-auditor` |
| `arch-review` | `software-designer` |
| `spec-check` | `speckit-analyze` |
| `outcome` | `work_gate` and the review card |
| `retrospective` | The `retrospective` prompt |

Leaf auto-approval continues to the done line, a branch or epic still stops for the owner
card, and a root contract change still gets its veto window. The kit only describes the
steps; those policies stay where they already live.

## CI template

`ci/agent-ops.yml` is a portable starting point, not a host workflow: it names no host,
path or secret. Copy it into the project's CI directory and wire each job to the
project's own command. The three jobs are `spec-contract-lint`, `tests`, and
`step-record-verifier` (a placeholder until the step-record checker is installed).
