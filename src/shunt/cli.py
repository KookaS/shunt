"""CLI entry point for shunt-router."""

from __future__ import annotations

import argparse
import os
import sys

from shunt import __version__
from shunt.log_config import LEVELS, LOG_LEVEL_ENV


def _apply_router_flag_overrides(args: argparse.Namespace) -> None:
    """Translate `shunt start` routing flags into env vars (CLI > env > file > default).

    Only flags actually passed override; absent flags leave any existing env var intact.
    """
    strategy = getattr(args, "strategy", None)
    explore = getattr(args, "explore", None)
    budget = getattr(args, "explore_budget_frac", None)
    if strategy is not None:
        os.environ["SHUNT_ROUTER_STRATEGY"] = strategy
    if explore is not None:
        os.environ["SHUNT_EXPLORATION_ENABLED"] = "1" if explore else "0"
    if budget is not None:
        os.environ["SHUNT_EXPLORE_BUDGET_FRAC"] = str(budget)


def _add_start_flags(parser: argparse.ArgumentParser) -> None:
    """Register the routing-override flags on the `start` subcommand."""
    parser.add_argument(
        "--strategy",
        default=None,
        help="Active routing strategy (overrides router.yaml / env).",
    )
    parser.add_argument(
        "--explore",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable exploration (--explore / --no-explore).",
    )
    parser.add_argument(
        "--explore-budget-frac",
        type=float,
        default=None,
        help="Exploration budget fraction (~1.4x cost at 0.4).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=[level.lower() for level in LEVELS],
        help="Log verbosity (overrides SHUNT_LOG_LEVEL). Use debug to trace routing.",
    )


def _start(args: argparse.Namespace) -> None:
    from shunt.proxy.server import run
    from shunt.secrets import load_dotenv_file

    # Load a local .env (gitignored) if present so provider keys are available;
    # real env vars still win. Missing file is a no-op (env-only setups unaffected).
    load_dotenv_file()
    _apply_router_flag_overrides(args)
    # Export rather than pass through: `run()` configures logging for the whole process
    # (uvicorn included), and the env var is the same knob the container sets.
    if getattr(args, "log_level", None):
        os.environ[LOG_LEVEL_ENV] = args.log_level.upper()
    run()


def _version(args: argparse.Namespace) -> None:
    print(f"shunt-router {__version__}")


def _explain(args: argparse.Namespace) -> None:
    from shunt.db.store import OutcomeStore

    store = OutcomeStore()
    session = store.get_session(args.session_id)
    if session is None:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    provenance_raw = session.get("decision_provenance")
    if not provenance_raw:
        print(f"Session {args.session_id} has no decision provenance stored.")
        sys.exit(1)

    import json

    prov = json.loads(provenance_raw)

    print(f"Session:        {args.session_id}")
    print(f"Model chosen:   {prov.get('model_chosen', '?')}")
    print(f"Selection rule: {prov.get('selection_rule_used', '?')}")
    print(f"Fallback:       {'yes' if prov.get('fallback_chain_triggered') else 'no'}")
    es_reason = prov.get("tier_escalation_reason")
    if es_reason:
        print(f"Escalation:     {es_reason}")
    print(f"Router propensity: {prov.get('router_propensity', '?')}")
    print()

    candidate_scores = prov.get("candidate_model_scores", {})
    if candidate_scores:
        print("Candidate model scores:")
        for model, score in sorted(candidate_scores.items(), key=lambda x: -x[1]):
            marker = " ← SELECTED" if model == prov.get("model_chosen") else ""
            print(f"  {model}: {score:.4f}{marker}")
        print()

    neighbor_ids = prov.get("top_k_neighbor_ids", [])
    confidence_scores = prov.get("neighbor_confidence_scores", [])
    if neighbor_ids:
        print(f"Top-k neighbors ({len(neighbor_ids)}):")
        for sid, conf in zip(neighbor_ids, confidence_scores, strict=False):
            print(f"  {sid}  (confidence={conf:.3f})")


def _flag(args: argparse.Namespace) -> None:
    """Record a human-verified outcome for a routed session."""
    # This is the router's outcome write-back path. Until it is used, no outcome row exists,
    # every neighbourhood is empty, the engine stays in cold-start and routes to the cheap
    # default — so kNN and exploration are configured but inert. A human rating counts as a
    # Tier-2 (verified) label: it is a person confirming the task actually worked, which is
    # exactly the ground truth the routing is meant to learn from. Automatic Tier-2 capture
    # from a test/typecheck run is a separate, larger piece of work.
    from shunt.db.store import OutcomeStore

    store = OutcomeStore()
    if store.get_session(args.session_id) is None:
        # Fail loudly: silently accepting an unknown id would poison the corpus with labels
        # attached to nothing, and the router cannot tell a typo from a real session.
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    outcome = "success" if args.rating == "good" else "failure"
    store.store_outcome(
        session_id=args.session_id,
        tier1_outcome=outcome,
        tier1_confidence=1.0,
        tier2_outcome=outcome,
        tier2_confidence=1.0,
        aggregated_confidence=1.0,
        human_label=args.rating,
    )
    print(f"Flagged {args.session_id} as {args.rating} ({outcome}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shunt",
        description="Tool-agnostic, cache-safe LLM router.",
    )
    parser.set_defaults(func=_start)

    sub = parser.add_subparsers(title="commands")

    start = sub.add_parser("start", help="Start the proxy server (default)")
    _add_start_flags(start)
    start.set_defaults(func=_start)

    explain = sub.add_parser("explain", help="Explain a routing decision")
    explain.add_argument("session_id", help="Session ID to explain")
    explain.set_defaults(func=_explain)

    flag = sub.add_parser("flag", help="Flag a session outcome as good or bad")
    flag.add_argument("session_id", help="Session ID to flag")
    flag.add_argument("rating", choices=["good", "bad"], help="Outcome rating")
    flag.set_defaults(func=_flag)

    version = sub.add_parser("version", help="Print version")
    version.set_defaults(func=_version)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
