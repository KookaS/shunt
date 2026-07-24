# Trajectory fixtures — provenance

Tiny per-framework fixtures used to validate the normalizers. The three `.traj` file
fixtures are trimmed slices of genuine published trajectories (whole steps removed; every
retained field is the upstream value). The `mini_swe_agent.json` fixture is a
schema-real in-memory `agent.messages` sample — its structure matches the installed
`minisweagent` message contract exactly, with illustrative field values (a live capture
is a paid run). Fixtures are test-only inputs; no fixture is committed as benchmark data.
All upstream sources are permissively licensed.

| Fixture | Framework | Upstream source | License |
|---------|-----------|-----------------|---------|
| `swe_agent.traj` | SWE-agent | `SWE-agent/SWE-agent` — `tests/test_data/trajectories/gpt4__swe-agent__test-repo__default_from_url__.../swe-agent__test-repo-i1.traj` (first 2 steps + info) | MIT |
| `swe_smith.traj` | SWE-smith | `SWE-bench/SWE-smith` — `tests/test_logs/trajectories/getmoto__moto.694ce1f4.pr_7331/...traj` (one real step) | MIT |
| `openhands.json` | OpenHands | `OpenHands/trajectory-visualizer` — `demo1.json` (first 8 real events) | MIT |
| `mini_swe_agent.json` | mini-swe-agent | `SWE-agent/mini-swe-agent` — an `agent.messages` list shaped exactly as `DefaultAgent.run()` leaves it (system + user + two assistant bash tool-call turns each paired with a `tool` observation + `exit`); values are illustrative-but-real-shaped, no synthesized schema | MIT |

Trimming only removed whole steps/events and the per-step nested LLM `messages` log
(not part of the StepView contract); every retained field is the upstream value.
