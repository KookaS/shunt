# zed → Shunt

The [Zed](https://zed.dev) editor's assistant, pointed at Shunt through an
OpenAI-compatible language-model provider. Shunt picks the model; Zed never learns
which one.

**Docs-only — verify manually.** Zed's assistant runs inside the editor (a GUI),
so it can't be driven in headless CI. There is no `compose.yaml` here. The
configuration below is verified; confirm it in your editor.

## Point Zed at Shunt

Add an OpenAI-compatible provider to Zed's `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "shunt": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          {
            "name": "auto",
            "display_name": "auto (Shunt-routed)",
            "max_tokens": 8192
          }
        ]
      }
    }
  }
}
```

The OpenAI wire `api_url` keeps its **`/v1` suffix**. Enter the API key through
Zed's provider UI when prompted — any non-empty placeholder works, since Shunt
holds the real provider keys and ignores this field. Select the `auto` model in
the assistant to let Shunt route.

## Status

Works. Zed's OpenAI-compatible provider speaks the wire Shunt serves, so routing
is transparent. The only limitation is the harness: it's GUI-only, so this
directory documents the setup rather than running a CI handshake. See
[`../README.md`](../README.md) for the overall integration model.
