
## Task Contract

*GATE: the next step after this plan is the task contract (the `contract` command:
`/speckit-contract` in Claude Code, `$speckit-contract` in Codex, `/speckit.contract` in
opencode), then tasks. Tests are written first; this section is what the contract is built
from.*

**Contract file**: `specs/[###-feature-name]/contract.yaml` (written by the contract step,
validated by the project's contract CLI; the schema is the substrate's task contract,
schema 2). Not to be confused with `contracts/` above, which holds interface definitions.

**Level**: [leaf | branch — a branch feature has children with interfaces between them]

**Test runner and command**: [e.g. `pytest -q tests/` or NEEDS CLARIFICATION]

**Tests to write first** (one row per acceptance scenario / functional requirement; together
the rows cover EVERY `FR-###` and `SC-###` of the spec):

| Id | Level | Category | Path | Covers |
|----|-------|----------|------|--------|
| [UT-…] | [unit/property/contract/integration/acceptance] | [happy/edge/error/security/perf/regression/integration/acceptance] | [tests/…] | [FR-###, SC-### — the ids this test proves] |

**Metrics** (only from measurable success criteria `SC-###`; a missing number is never
invented — it is `THRESHOLD-UNSET` on a target, or a question for the owner). Each one is
PROPOSED with a suggested value and where it came from; the owner accepts it at approval.
Prefer a catalogue entry (`<backlog_dir>/metrics.yaml`, written as `ref:`); each metric
either feeds a parent metric (with how it combines) or is local to this feature:

| Id | From | Tier | Ref | Via (command that measures it) | Comparator | Suggested value | Source | Feeds + combine, or local |
|----|------|------|-----|--------------------------------|------------|-----------------|--------|---------------------------|
| [TH-…] | [SC-###] | [gate/threshold/target/guardrail] | [catalogue id or —] | [command] | ["<="] | [number or THRESHOLD-UNSET] | [default/catalog/sibling/baseline] | [`feeds: <parent metric>`, `combine: max` — or `local: <why>`] |

**Abort path**: a test that contradicts the spec stops the work and is escalated; neither
is edited to agree.
