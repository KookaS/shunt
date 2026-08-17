"""CLI entry point for shunt-router."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from shunt import __version__
from shunt.log_config import LEVELS, LOG_LEVEL_ENV

if TYPE_CHECKING:
    from shunt.router.diagnostics import DoctorReport
    from shunt.router.inspection import EscalationReport


def _apply_router_flag_overrides(args: argparse.Namespace) -> None:
    """Translate `shunt start` routing flags into env vars (CLI > env > file > default).

    Only flags actually passed override; absent flags leave any existing env var intact.
    """
    strategy = getattr(args, "strategy", None)
    explore = getattr(args, "explore", None)
    budget = getattr(args, "explore_budget_frac", None)
    work_dir = getattr(args, "work_dir", None)
    if strategy is not None:
        os.environ["SHUNT_ROUTER_STRATEGY"] = strategy
    if work_dir is not None:
        # Same env var the file's single `work_dir` is overridden by, so the flag inherits
        # its precedence (map > flag/env > file > validated launch dir) with no second path.
        os.environ["SHUNT_WORK_DIR"] = work_dir
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
        "--work-dir",
        default=None,
        help=(
            "Repo whose tests are re-run off the wire to verify outcomes. Overrides "
            "capture.work_dir. Unset: the launch directory is used if it is a git repo with "
            "a test framework. WARNING: this runs that repo's own test code."
        ),
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
    es_reason = prov.get("rank_escalation_reason")
    if es_reason:
        print(f"Escalation:     {es_reason}")
    # An EFFORT escalation keeps the model and raises its reasoning arm, so `Model chosen`
    # alone cannot show it: without this line an effort-escalated session explains
    # identically to the un-escalated base pick it is no longer running.
    arm = prov.get("escalated_reasoning_arm")
    if arm:
        print(f"Reasoning arm:  {arm}  (escalated)")
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


def _escalate(args: argparse.Namespace) -> None:
    """Inspect the auto-escalation state for a repo — read-only, never mutating."""
    # READ-ONLY BY DESIGN, and the mutation the flags deliberately omit is the point: a running
    # server holds this state in memory and re-serializes the WHOLE snapshot on its own cadence,
    # so a `--force-rank`/`--suppress` writing `router_state` from this process would be silently
    # clobbered, or would restore a half-consistent ladder (a rank floor with no matching effort
    # arm). Escalation is also boundary-only by construction; a CLI that pushed a rung mid-flight
    # would be the one path that can change a model outside a decision boundary, which is the
    # cache-safety spine. Change the knobs in router.yaml and restart instead.
    from shunt.db.store import OutcomeStore
    from shunt.models import ModelPool
    from shunt.router.inspection import escalation_report, resolve_inspection_work_dir
    from shunt.router.policy import load_router_policy, resolved_policy_path

    policy_path = resolved_policy_path()
    policy = load_router_policy()
    work_dir, source = resolve_inspection_work_dir(policy, args.work_dir, os.getcwd())
    model_pool = ModelPool(config_path=os.environ.get("SHUNT_MODEL_CONFIG_PATH"))
    model_pool.restrict_to_live(policy.models)
    store = OutcomeStore()
    try:
        report = escalation_report(
            policy=policy,
            policy_path=policy_path,
            model_pool=model_pool,
            outcome_store=store,
            work_dir=work_dir,
            work_dir_source=source,
        )
    finally:
        store.close()

    if args.as_json:
        import dataclasses
        import json

        print(json.dumps(dataclasses.asdict(report), indent=2))
        return
    _print_escalation_report(report)


def _print_escalation_report(report: EscalationReport) -> None:
    """Render the escalation report as the CLI's plain stdout block."""
    print(f"Work dir:       {report.work_dir or '(unresolved)'}  [{report.work_dir_source}]")
    print(f"Task key:       {report.task_key or '(none)'}")
    print(f"Escalation:     {'enabled' if report.enabled else 'DISABLED'}")
    print(f"Policy file:    {report.policy_path}")
    print()
    print("Config:")
    for item in report.config:
        print(f"  {item.key:<20} {item.value!s:<16} ({item.source})")
    print()
    if report.work_dir is None:
        _print_inert(report)
        return
    _print_ladder(report)
    print()
    _print_window(report)
    print()
    _print_suppression(report)
    print()
    _print_next(report)


def _print_inert(report: EscalationReport) -> None:
    """Explain the inert state: no repo resolved means no verified-failure signal at all."""
    print(
        "No work_dir resolved, so escalation is INERT: its only signal is the off-wire "
        "test re-run of a repo, and there is none to run."
    )
    print(
        "Set capture.work_dir / SHUNT_WORK_DIR, launch shunt inside the repo, or pass --work-dir."
    )
    if report.mapped_work_dirs:
        print("Per-tool capture.work_dirs (pass one with --work-dir):")
        for identity, path in sorted(report.mapped_work_dirs.items()):
            print(f"  {identity}: {path}")


def _print_ladder(report: EscalationReport) -> None:
    """Print the task's rung: which model, which reasoning arm, how much headroom."""
    ladder = report.ladder
    floor = "none climbed yet" if ladder.rank_floor is None else str(ladder.rank_floor)
    print("Ladder position:")
    seen = "" if report.state_present else "  (no persisted state for this task)"
    print(f"  Decisions seen: {report.decision_index}{seen}")
    print(f"  Rank floor:     {floor}")
    print(f"  Model:          {ladder.model or '(none)'}  [{ladder.model_source}]")
    print(f"  Rank:           {ladder.rank_index} of 0..{ladder.max_rank_index}")
    arm = ladder.effort_arm or "(model has no reasoning arms)"
    print(f"  Reasoning arm:  {arm}  (rung {ladder.effort_index} of 0..{ladder.max_effort_index})")


def _print_window(report: EscalationReport) -> None:
    """Print the live failure log, per key then per event, with each event's counting verdict."""
    n = next((c.value for c in report.config if c.key == "escalate_after_n"), "?")
    print(f"Failure window (a key escalates at {n} counting failures):")
    if not report.keys:
        print("  (no failures in the window)")
    for key in report.keys:
        due = "  ← DUE" if key.due else ""
        print(f"  {key.dedup_key}: {key.countable} counting / {key.events} event(s){due}")
    if report.events:
        print("  events:")
    for event in report.events:
        stale = "" if event.in_window else "  [stale — outside the window]"
        print(f"    #{event.decision_index} {event.dedup_key}: {event.verdict}{stale}")


def _print_suppression(report: EscalationReport) -> None:
    """Print the collapse guard — the reported failure mode where escalation silently no-ops."""
    print("Suppression:")
    print(f"  Routing-collapse alarm: {'YES' if report.collapse_alarm else 'no'}")
    print(f"  Cold start active:      {'yes' if report.cold_start_active else 'no'}")
    if report.cold_start_active:
        print("    (the alarm is ignored while cold-start routes every session to the cheap model)")
    verdict = "YES — escalation will HOLD" if report.suppressed else "no"
    print(f"  Escalation suppressed:  {verdict}")


def _print_next(report: EscalationReport) -> None:
    """Print what the real decide_escalation returns against the real persisted state."""
    print("Next decision (from the real decide_escalation, on the state above):")
    print(f"  Action: {report.next_action}")
    print(f"  Reason: {report.next_reason}")
    if report.exploration_epsilon > 0:
        print(
            f"  Note:   exploration_epsilon={report.exploration_epsilon} — at a flagged "
            "checkpoint the escalation is withheld (HOLD) with that probability."
        )


def _doctor(args: argparse.Namespace) -> None:
    """Diagnose the install: what is live, what is inert, and why. Never spends, never mutates."""
    from shunt.router.diagnostics import doctor_report
    from shunt.secrets import load_dotenv_file

    # Same .env load `start` does, or doctor would report keys as MISSING that the server
    # will happily find — the single most confusing answer this command could give.
    load_dotenv_file()
    report = doctor_report(work_dir=args.work_dir, launch_dir=os.getcwd())

    if args.as_json:
        import dataclasses
        import json

        # `status` is the stable field a machine consumer keys on, and every name in
        # CHECK_ORDER is present in every branch — see diagnostics.CHECK_ORDER for why.
        payload = {
            "serviceable": report.serviceable,
            "checks": [
                {**dataclasses.asdict(check), "status": check.status} for check in report.checks
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_doctor_report(report)

    if not report.serviceable:
        sys.exit(1)


def _print_doctor_report(report: DoctorReport) -> None:
    """Render the diagnosis, one block per check, worst news legible at a glance."""
    markers = {"fail": "FAIL", "warn": "WARN", "ok": "ok", "skipped": "n/a"}
    for check in report.checks:
        marker = markers[check.status]
        head, _, rest = check.detail.partition("\n")
        print(f"[{marker:^4}] {check.name:<12} {head}")
        for line in rest.splitlines():
            print(f"{'':<21}{line}")
    print()
    if report.serviceable:
        print("Router is serviceable. WARN lines are degraded-but-working states, not errors.")
    else:
        print("Router CANNOT serve a request — fix the FAIL line(s) above.")


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


def _reindex(args: argparse.Namespace) -> None:
    """Re-embed the whole corpus into the active embedder's space (offline command)."""
    # Run with the server STOPPED: it opens the same SQLite DB + .hnsw2 index. The rewrite
    # is atomic (one txn for the blobs, temp-then-replace for the index, fingerprint last),
    # so an interrupted run leaves the OLD space intact and boot simply asks again.
    from shunt.db.store import OutcomeStore
    from shunt.router.embedder import Embedder
    from shunt.secrets import load_dotenv_file

    load_dotenv_file()
    print("shunt reindex: re-embedding the corpus (run with the server stopped)...")
    embedder = Embedder()
    store = OutcomeStore()
    try:
        summary = store.reindex_corpus(embedder)
    except Exception as exc:
        print(f"reindex FAILED: {exc}. The corpus is unchanged (old space intact).")
        sys.exit(1)
    finally:
        store.close()
    print(
        f"reindex OK: {summary['reindexed']} session(s) re-embedded into "
        f"{embedder.model_name}. Fingerprint {summary['old_fingerprint']} -> "
        f"{summary['new_fingerprint']}. Restart the server to pick up the new space."
    )


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

    escalate = sub.add_parser(
        "escalate",
        help="Inspect auto-escalation state for a repo (read-only)",
        description="Read-only: the effective escalation config and where each value came "
        "from, the task's rung on the ladder, the live failure window, whether the "
        "collapse guard is suppressing escalation, and what the next decision would do. "
        "Nothing is mutated — the running server owns this state.",
    )
    escalate.add_argument(
        "--work-dir",
        default=None,
        help="Repo whose escalation state to inspect (default: the router's own resolution).",
    )
    escalate.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the report as JSON instead of the text block.",
    )
    escalate.set_defaults(func=_escalate)

    doctor = sub.add_parser(
        "doctor",
        help="Diagnose the install: which keys resolve, what is armed, what is inert",
        description="Read-only and non-spending: which provider keys resolve (presence only, "
        "never the value), how many models the registry leaves routable, whether the embedding "
        "weights are cached, whether the bind address is free, and — the one a new install "
        "usually gets wrong — whether escalation is ARMED or merely enabled and inert. Exits "
        "non-zero only when the router could not serve a request at all.",
    )
    doctor.add_argument(
        "--work-dir",
        default=None,
        help="Repo to check escalation arming against (default: the router's own resolution).",
    )
    doctor.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the diagnosis as JSON instead of the text block.",
    )
    doctor.set_defaults(func=_doctor)

    flag = sub.add_parser("flag", help="Flag a session outcome as good or bad")
    flag.add_argument("session_id", help="Session ID to flag")
    flag.add_argument("rating", choices=["good", "bad"], help="Outcome rating")
    flag.set_defaults(func=_flag)

    reindex = sub.add_parser(
        "reindex",
        help="Re-embed the corpus into the active embedder's space (run offline)",
        description="Offline: re-embed every stored session into the embedding.yaml active "
        "model's space and advance the corpus fingerprint. Run with the server STOPPED.",
    )
    reindex.set_defaults(func=_reindex)

    version = sub.add_parser("version", help="Print version")
    version.set_defaults(func=_version)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
