#!/usr/bin/env python3
"""CI free-tier smoke: one real, genuinely-$0 completion through Shunt.

Replaces the owner smoke's interactive spend gates with a structural one: the
served model must be an OpenRouter :free model, or the run refuses to bill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import benchmark.runner.live_smoke as live_smoke
from benchmark.corpus_lock import atomic_write_text
from shunt.models.config import ModelConfig

FREE_TIER_CI_ENV: Final[str] = "SHUNT_FREE_TIER_CI"
FREE_MODEL_SUFFIX: Final[str] = ":free"
OPENROUTER_HOST: Final[str] = "openrouter.ai"
DEFAULT_MAX_COST_USD: Final[float] = 0.0
DEFAULT_CONFIG_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "configs" / "free-tier"
DEFAULT_RUN_ROOT: Final[Path] = (
    Path(__file__).resolve().parent.parent / "runner" / "artifacts" / "free-tier-smoke"
)


# ── gates (pure — unit-tested) ────────────────────────────────────────────────


def free_tier_refusal(model: ModelConfig) -> str | None:
    """Refuse unless *model* is verified $0 on OpenRouter, naming the failure.

    The auto-approval replaces the interactive confirmation only for a model
    that cannot bill — every billing-relevant fact is asserted from the registry.
    """
    model_id = model.model_id or model.name
    if not model_id.endswith(FREE_MODEL_SUFFIX):
        return (
            f"refusing to run: served model {model_id!r} does not end in "
            f"{FREE_MODEL_SUFFIX!r} — only OpenRouter :free models are verified $0"
        )
    if OPENROUTER_HOST not in model.base_url:
        return (
            f"refusing to run: provider base_url {model.base_url!r} is not "
            f"OpenRouter ({OPENROUTER_HOST}) — only OpenRouter :free models are verified $0"
        )
    pricing = model.pricing
    if pricing is None or pricing.input_cost_per_1m != 0.0 or pricing.output_cost_per_1m != 0.0:
        return (
            f"refusing to run: served model {model_id!r} is not priced at 0/0 per "
            "1M tokens — a positive price would bill"
        )
    return None


def ci_tty_refusal(isatty: Callable[[], bool], ci_env: str | None) -> str | None:
    """Non-TTY operation needs an explicit ``SHUNT_FREE_TIER_CI=1`` override."""
    if isatty():
        return None
    if ci_env == "1":
        return None
    return (
        f"refusing to run: stdin is not a TTY and {FREE_TIER_CI_ENV} is not set to 1; "
        "this CI smoke must be explicitly opted in"
    )


def missing_key_skip(key_env: str, key_value: str | None) -> bool:
    """True when the provider key is absent — CI skips (exit 0), never fails."""
    return not bool(key_value)


# ── run ───────────────────────────────────────────────────────────────────────


def _make_run_dir(run_root: Path) -> tuple[Path, Path, str]:
    """Create a stamped run dir; returns (run_dir, data_dir, stamp)."""
    run_root.mkdir(parents=True, exist_ok=True)
    # Microsecond resolution: a 1-second stamp let two invocations in the same second
    # collide on the same run dir, so a concurrent PASS and FAIL fought over one dir.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "data", stamp


def _write_verdict(run_dir: Path, stamp: str, verdict: dict) -> Path:
    """Persist the structured verdict JSON; returns its path."""
    verdict["run"] = stamp
    path = run_dir / "free-tier-smoke.json"
    # Atomic (temp + os.replace, the corpus writer pattern): a concurrent reader — or a
    # second smoke sharing the run dir — sees the old file or the new one, never a
    # truncated half. Without it a PASS verdict could be overwritten by a half-written
    # FAIL from a same-second sibling run.
    atomic_write_text(path, json.dumps(verdict, indent=2) + "\n")
    return path


def _verify(server, expected: ModelConfig, max_cost: float, report: Callable[[str], None]):
    """Boot, probe and verify one completion; returns the (problems, warnings, verdict).

    Reuses ``live_smoke``'s server boot and pure pass-criteria functions verbatim.
    """
    live = live_smoke
    if not live.wait_healthy(server, live.BOOT_TIMEOUT_S, report):
        return ["shunt server did not serve /health"], [], None
    report(f"server healthy on 127.0.0.1:{server.port}")
    if not live.model_listed(server.port, expected.name):
        return [f"expected model {expected.name} is not in GET /v1/models"], [], None

    report(f"sending one real completion (max_tokens=16, marker={live.PROMPT_MARKER!r})...")
    status, headers, body = live.post_completion(server.port, live.REQUEST_TIMEOUT_S)
    if status is None:
        return [f"could not reach shunt: {body}"], [], None
    report(f"completion returned HTTP {status}")

    decision = live.decision_from_headers(headers)
    served = decision[0] if decision is not None else expected.name
    reason = decision[1] if decision is not None else None
    session_id = headers.get("X-Shunt-Session-Id", "")

    problems = live.verify_headers(status, headers, expected.name, live.STRATEGY)
    problems.extend(live.content_problems(body))

    row = (
        live.read_session_row(server.data_dir, session_id, live.SESSION_WAIT_S)
        if session_id
        else None
    )
    capture_problems, warnings = live.verify_capture(
        row, expected.name, max_cost, live.PROMPT_MARKER
    )
    problems.extend(capture_problems)
    cost = float(row.get("cost") or 0.0) if row else 0.0

    verdict = {
        "status": "PASS" if not problems else "FAIL",
        "expected_model": expected.name,
        "served_model": served,
        "decision_reason": reason,
        "session_id": session_id,
        "recorded_cost_usd": cost,
        "spend_cap_usd": max_cost,
        "problems": problems,
        "warnings": warnings,
    }
    return problems, warnings, verdict


def main(argv: list[str] | None = None) -> int:
    """Run the free-tier smoke; 0 = pass, 1 = smoke failed, 2 = refused by a gate."""
    parser = argparse.ArgumentParser(
        prog="free_tier_smoke",
        description=(
            "One real, genuinely-$0 completion through Shunt against an OpenRouter "
            ":free model, verifying the decision headers and the captured session row. "
            "CI-oriented: non-TTY runs need SHUNT_FREE_TIER_CI=1; a missing "
            "OPENROUTER_API_KEY skips (exit 0) instead of failing."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory holding models.yaml + router.yaml (default {DEFAULT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help=(
            f"Hard cap on the recorded session cost in USD (default {DEFAULT_MAX_COST_USD:.2f}). "
            "A genuinely-$0 model must record 0."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=live_smoke.DEFAULT_PORT,
        help=f"Shunt listen port (default {live_smoke.DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"Parent directory for run artifacts (default {DEFAULT_RUN_ROOT}).",
    )
    args = parser.parse_args(argv)

    registry_path = args.config_dir / "models.yaml"
    policy_path = args.config_dir / "router.yaml"
    try:
        pool = live_smoke.build_pool(registry_path, policy_path)
        expected = live_smoke.cheapest_live_model(pool)
    except Exception as exc:  # noqa: BLE001 - a bad config must abort before any spend
        print(f"aborted: cannot resolve the free-tier model set: {exc}", file=sys.stderr)
        return 2

    refusal = free_tier_refusal(expected)
    if refusal is not None:
        print(f"aborted: {refusal}", file=sys.stderr)
        return 2
    key_env = expected.api_key_env_var
    key_value = os.environ.get(key_env)

    run_root = args.run_dir or DEFAULT_RUN_ROOT
    run_dir, data_dir, stamp = _make_run_dir(run_root)

    if missing_key_skip(key_env, key_value):
        message = (
            f"SKIP: ${key_env} is not set in the environment — nothing was sent and "
            "nothing will be billed. The CI runner injects this secret; an unset "
            "secret is a skip, not a failure."
        )
        print(message, file=sys.stderr)
        (run_dir / "free-tier-smoke.log").write_text(message + "\n", encoding="utf-8")
        _write_verdict(
            run_dir,
            stamp,
            {
                "status": "SKIP",
                "reason": f"${key_env} is not set in the environment",
                "expected_model": expected.name,
                "run_log": str(run_dir / "free-tier-smoke.log"),
                "data_dir": str(data_dir),
            },
        )
        return 0

    tty_refusal = ci_tty_refusal(sys.stdin.isatty, os.environ.get(FREE_TIER_CI_ENV))
    if tty_refusal is not None:
        print(f"aborted: {tty_refusal}", file=sys.stderr)
        return 2

    log_path = run_dir / "free-tier-smoke.log"

    def report(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")
        print(line)

    report(f"free-tier smoke starting: run={stamp}")
    report(f"  expected model: {expected.name} ({expected.model_id}) via {expected.provider}")
    report(f"  key env:        {key_env}")
    report(f"  spend cap:      ${args.max_cost:.2f} (genuinely-$0 model must record 0)")
    report(f"  run dir:        {run_dir}")

    problems: list[str] = []
    warnings: list[str] = []
    try:
        server = live_smoke.start_server(args.port, data_dir, run_dir, str(args.config_dir))
        try:
            problems, warnings, verdict = _verify(server, expected, args.max_cost, report)
        finally:
            server.stop()
    except Exception as exc:  # noqa: BLE001 - a server failure must surface as FAIL
        problems = [f"server error: {exc}"]
        verdict = None

    for warning in warnings:
        report(f"WARNING: {warning}")
    verdict = verdict or {
        "status": "FAIL",
        "expected_model": expected.name,
        "served_model": None,
        "decision_reason": None,
        "session_id": None,
        "recorded_cost_usd": 0.0,
        "spend_cap_usd": args.max_cost,
        "problems": problems,
        "warnings": warnings,
    }
    verdict["run"] = stamp
    verdict["run_log"] = str(log_path)
    verdict["data_dir"] = str(data_dir)
    _write_verdict(run_dir, stamp, verdict)

    if verdict["status"] == "FAIL":
        report("FAIL: " + "; ".join(problems))
        report(f"recorded in {run_dir} (keep the run log as the outcome record)")
        return 1
    report(
        f"PASS: one real $0 session routed through Shunt and captured. "
        f"model={verdict['served_model']} reason={verdict['decision_reason']} "
        f"cost=${verdict['recorded_cost_usd']:.6f} session={verdict['session_id']}"
    )
    report(f"recorded in {run_dir} (keep the run log as the outcome record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
