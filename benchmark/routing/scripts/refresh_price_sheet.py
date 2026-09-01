"""Refresh benchmark/routing/data/price_sheet.json from the channels' live catalogues.

python -m benchmark.routing.scripts.refresh_price_sheet            # write the sheet
python -m benchmark.routing.scripts.refresh_price_sheet --dry-run  # print, write nothing
"""

# REAL PRICES ONLY. Every number this writes was fetched, this run, from a public
# machine-readable endpoint. A listing the endpoint does not carry is recorded as UNRESOLVED
# and its model simply has fewer quotes; a model no channel prices gets `canonical: null` and
# therefore no repriced cost anywhere. Nothing is estimated, interpolated, or carried forward
# from the previous sheet -- a stale price silently re-emitted under a fresh `as_of` would be
# the one failure this artifact exists to prevent.
#
# Which listings count as the same product is NOT decided here: it is the hand-authored map in
# `data/price_channels.yaml`, and this script resolves exactly the ids that map names.
#
# Refreshing the sheet re-digests it, which marks every routing figure STALE (the sheet is a
# declared figure input in benchmark/pipeline.py). That is the intended behaviour: a figure
# drawn at last month's prices is not a figure at this month's.

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from benchmark.routing.repricing import CHANNELS_PATH, RANK_RULE, SHEET_PATH

_TIMEOUT: Final[int] = 45
_UA: Final[str] = "shunt-price-refresh"

# The three public catalogues that publish prices without a key. `native` is in
# `repricing.CHANNELS` and absent here on purpose: no native provider serves a
# machine-readable price endpoint, and this script writes only what it fetched.
_ENDPOINTS: Final[dict[str, str]] = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "requesty": "https://router.requesty.ai/v1/models",
    "huggingface": "https://router.huggingface.co/v1/models",
}

# Each channel publishes prices in its own unit. OpenRouter quotes USD PER TOKEN as strings,
# Requesty USD per token as floats, HuggingFace USD per 1M as floats. Getting this wrong is a
# 1e6 error that still renders, so the conversion lives beside the endpoint that needs it.
_PER_TOKEN_TO_PER_1M: Final[float] = 1_000_000.0


def _fetch(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": _UA})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
        return json.loads(response.read())


def _openrouter_prices(payload: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for entry in payload.get("data", []):
        pricing = entry.get("pricing") or {}
        try:
            prompt = float(pricing["prompt"]) * _PER_TOKEN_TO_PER_1M
            completion = float(pricing["completion"]) * _PER_TOKEN_TO_PER_1M
        except (KeyError, TypeError, ValueError):
            continue
        out[str(entry.get("id", ""))] = (prompt, completion)
    return out


def _requesty_prices(payload: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for entry in payload.get("data", []):
        try:
            prompt = float(entry["input_price"]) * _PER_TOKEN_TO_PER_1M
            completion = float(entry["output_price"]) * _PER_TOKEN_TO_PER_1M
        except (KeyError, TypeError, ValueError):
            continue
        out[str(entry.get("id", ""))] = (prompt, completion)
    return out


def _huggingface_prices(payload: Any) -> dict[str, tuple[float, float]]:
    """Cheapest LIVE inference provider per HF model id; already USD per 1M."""
    # HF fans one model id out over several providers, each with its own price and status. A
    # provider that is not `live`, or is a free tier, is not a price this sheet may quote.
    out: dict[str, tuple[float, float]] = {}
    for entry in payload.get("data", []):
        best: tuple[float, float] | None = None
        for provider in entry.get("providers", []) or []:
            if provider.get("status") != "live" or provider.get("is_free"):
                continue
            pricing = provider.get("pricing") or {}
            try:
                candidate = (float(pricing["input"]), float(pricing["output"]))
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or sum(candidate) < sum(best):
                best = candidate
        if best is not None:
            out[str(entry.get("id", ""))] = best
    return out


_PARSERS: Final[dict[str, Any]] = {
    "openrouter": _openrouter_prices,
    "requesty": _requesty_prices,
    "huggingface": _huggingface_prices,
}


def _catalogues() -> tuple[dict[str, dict[str, tuple[float, float]]], dict[str, str]]:
    """Fetch every channel. A channel that fails is EMPTY and says so; it is never guessed."""
    prices: dict[str, dict[str, tuple[float, float]]] = {}
    status: dict[str, str] = {}
    for channel, url in _ENDPOINTS.items():
        try:
            prices[channel] = _PARSERS[channel](_fetch(url))
            status[channel] = f"ok ({len(prices[channel])} listings)"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            prices[channel] = {}
            status[channel] = f"UNREACHABLE: {type(exc).__name__}: {exc}"
    return prices, status


def _rank_key(quote: dict[str, Any]) -> tuple[float, float, str]:
    return (
        quote["input_cost_per_1m"] + quote["output_cost_per_1m"],
        quote["input_cost_per_1m"],
        f"{quote['channel']}:{quote['id']}",
    )


def build_sheet(
    channel_map: dict[str, dict[str, list[str]]],
    prices: dict[str, dict[str, tuple[float, float]]],
    status: dict[str, str],
    as_of: str,
) -> dict[str, Any]:
    """Assemble the sheet from resolved quotes only; unresolved ids are named, not filled."""
    models: dict[str, Any] = {}
    for model, channels in sorted(channel_map.items()):
        quotes: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for channel, ids in sorted((channels or {}).items()):
            for listing_id in ids or []:
                found = prices.get(channel, {}).get(listing_id)
                if found is None:
                    unresolved.append(f"{channel}:{listing_id}")
                    continue
                quotes.append(
                    {
                        "channel": channel,
                        "id": listing_id,
                        "input_cost_per_1m": round(found[0], 6),
                        "output_cost_per_1m": round(found[1], 6),
                        "source": _ENDPOINTS[channel],
                        "as_of": as_of,
                    }
                )
        quotes.sort(key=_rank_key)
        models[model] = {
            "canonical": quotes[0] if quotes else None,
            "quotes": quotes,
            "unresolved": sorted(unresolved),
        }
    return {
        "schema": 1,
        "as_of": as_of,
        "rank_rule": RANK_RULE,
        "channel_status": status,
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the sheet, write nothing")
    parser.add_argument("--out", type=Path, default=SHEET_PATH)
    args = parser.parse_args(argv)

    channel_map = yaml.safe_load(CHANNELS_PATH.read_text())["models"]
    prices, status = _catalogues()
    sheet = build_sheet(channel_map, prices, status, datetime.now(UTC).date().isoformat())

    for channel, line in sorted(status.items()):
        print(f"{channel:14s} {line}", file=sys.stderr)
    priced = sum(1 for entry in sheet["models"].values() if entry["canonical"] is not None)
    print(f"priced {priced}/{len(sheet['models'])} models", file=sys.stderr)

    text = json.dumps(sheet, indent=2, sort_keys=True) + "\n"
    if args.dry_run:
        print(text)
        return 0
    args.out.write_text(text)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
