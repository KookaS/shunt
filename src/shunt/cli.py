"""CLI entry point for shunt-router."""

from __future__ import annotations

import argparse
import sys

from shunt import __version__


def _start(args: argparse.Namespace) -> None:
    from shunt.proxy.server import run
    from shunt.secrets import load_dotenv_file

    # Load a local .env (gitignored) if present so provider keys are available;
    # real env vars still win. Missing file is a no-op (env-only setups unaffected).
    load_dotenv_file()
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
    print("shunt flag: not yet implemented (planned)")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shunt",
        description="Tool-agnostic, cache-safe LLM router.",
    )
    parser.set_defaults(func=_start)

    sub = parser.add_subparsers(title="commands")

    start = sub.add_parser("start", help="Start the proxy server (default)")
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
