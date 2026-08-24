# Security Policy

Shunt sits in the request path and handles provider API keys.

## Reporting a vulnerability

Report privately via **GitHub Security Advisories** ("Report a vulnerability" on the
repo's Security tab). Include a description, reproduction steps, and impact.

Reports without a working reproduction (terminal recording is fine) will be closed
without review — AI tools generate plausible-looking but non-reproducible reports,
and triaging them wastes time.

## Scope

In scope: the Shunt router code. Out of scope: third-party providers Shunt proxies
to, and misconfigurations against documented defaults.

## Trust model — what the port protects

**Reaching the port is the whole access-control boundary. There is no
authentication on any endpoint, including the admin one.**

All five routes — `GET /health`, `GET /admin/loop-health`, `GET /v1/models`,
`POST /v1/chat/completions` and `POST /v1/messages` — authenticate nobody. Shunt is
safe by *placement*, not by permission: the default bind is `127.0.0.1`, so only
local processes can reach it.

**The primary exposure is the two POST routes, not the admin one.** They forward to
your provider using *your* key, so a reachable port is spendable credit for anyone who
can reach it.

Do not mistake `router.budget.max_spend_usd` for a defence. It is **per-session**, held
**in memory**, **unset by default**, and a *soft* ceiling checked at the next request
boundary — so the request that crosses it still completes. It resets with every new
session and with every restart, and it has no notion of caller identity. An
unauthenticated caller simply starts another session. It bounds one session's spend; it
does not bound your bill, attribute a charge, or prevent anything.

`GET /admin/loop-health` is the cheapest thing behind that port by comparison: it
returns aggregates only, never prompts or completions.

Two consequences an operator must plan for:

- **Do not bind Shunt to a routable interface without putting your own
  authentication in front of it.** Setting `SHUNT_HOST=0.0.0.0` on a host with a
  public interface hands your provider credit to anyone who can route to the port.
- **The container image sets `SHUNT_HOST=0.0.0.0` on purpose** — that is how a
  process is reachable from outside its own container, and it is correct there. The
  boundary then moves to the *publish* flag, which is why the documented command is
  `-p 127.0.0.1:8080:8080` and not `-p 8080:8080`. Publishing without the
  `127.0.0.1:` prefix exposes it on every host interface.

Authentication is future work, not a shipped feature. This section exists so that is a
documented boundary rather than a discovered one.

## Data at rest

Shunt keeps its state in a local SQLite database (`~/.local/share/shunt/outcomes.db` by
default; relocate with `SHUNT_DATA_DIR`). It is a plain file with no encryption at rest and no
per-row redaction; its protection is filesystem permissions, the same as the port's protection
is placement.

Two columns hold conversation text:

- `sessions.prompt_text` — the opening prompt of each session, always.
- `sessions.decision_provenance` — with `escalation.context_transfer: summary` enabled
  (**off by default**), this holds the frozen handover note shunt authored *and* the client's
  leading system blocks verbatim, under `context_transfer_prefix`. For a coding agent those
  system blocks routinely carry the working directory, git status and recent commit subjects.
  It is stored so a resumed session can restore the exact bytes it already sent; regenerating
  or dropping them costs a cache miss on every turn.

Under the default `context_transfer: full`, no conversation body beyond `prompt_text` is
written. The session-identity `prefix_digest` column is a one-way digest in every mode and
holds no text and no filesystem path.

If the machine's filesystem is not a trust boundary you are comfortable with, do not enable
`context_transfer: summary`, and treat the database as sensitive.

## Known non-issues

Attacks requiring the operator to explicitly disable a security default
(e.g. binding to `0.0.0.0` instead of `127.0.0.1`, or publishing the container port
on all interfaces) are not considered vulnerabilities.

## Supported versions

Only the latest release. Pre-1.0, no further guarantees.
