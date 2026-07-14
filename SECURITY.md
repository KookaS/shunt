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

## Known non-issues

Attacks requiring the operator to explicitly disable a security default
(e.g. binding to `0.0.0.0` instead of `127.0.0.1`) are not considered vulnerabilities.

## Supported versions

Only the latest release. Pre-1.0, no further guarantees.
