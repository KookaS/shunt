# windsurf → Shunt

> **Documented for completeness — not currently integrable.**
> [Windsurf](https://windsurf.com) is closed-source and exposes **no** OpenAI
> base-URL override, so there is no supported way to point it at a localhost proxy
> like Shunt today. This directory records that gap so the picture is honest, not
> aspirational.

**Docs-only — nothing to verify.** Windsurf is a GUI editor and can't be driven in
headless CI. There is no `compose.yaml` here.

## Status

Not integrable with Shunt at this time. Windsurf's model configuration does not
offer a custom base-URL / OpenAI-compatible endpoint field, so its requests can't
be redirected to a local router. Its coding surfaces run against Windsurf's own
backend end to end.

If a future Windsurf release adds a custom-endpoint or OpenAI base-URL override,
the setup would mirror the other OpenAI-wire tools — base URL
`http://127.0.0.1:8080/v1`, any non-empty placeholder key, a Shunt-routed model id
such as `auto`. Until then, prefer a tool with a base-URL override (see the sibling
directories) if you want Shunt to route your coding agent. See
[`../README.md`](../README.md) for the overall integration model.
