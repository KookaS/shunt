"""The escalation scenario's verdict: exit 0 only if SHUNT escalated, per Shunt's own state."""

# Why not the tool's stdout? Because grepping a CLI's output for ``X-Shunt-Decision`` is
# what made the hand-driven recipe unreliable — most agentic CLIs never surface response
# headers, so a swallowed header is indistinguishable from a router that never escalated.
#
# Three Shunt-side signals were available; this reads the third, and uses the second as a
# receipt:
#
# * ``GET /admin/loop-health`` — aggregates only (label coverage, propensities, collapse
#   alarms). It carries no escalation field at all, so it cannot decide this verdict. The
#   *driver* still polls it, because ``verification.verified_outcomes`` is the one progress
#   counter a bare-HTTP client can reach. NOT ``label_coverage.verified_labeled``: that block
#   is kNN-corpus coverage, gated on the session carrying an embedding, and the shipped default
#   strategy never embeds.
# * ``shunt explain <session_id>`` — prints the decision's provenance, including
#   ``Escalation:``. It needs a session id, which is precisely what a header-swallowing
#   CLI denies us. Used below as the human-readable receipt once the id is known.
# * the sqlite outcome store — the substrate both of the above read. ``sessions.
#   decision_provenance`` holds ``auto_escalated`` per decision and is queryable *without*
#   a session id, by matching the marker the driver planted in the prompt. That marker
#   match is what keeps the verdict honest: it proves the escalated decision was served to
#   THIS tool's request, not to some other traffic against the same repo.

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shunt.router.escalation import next_rung_rank

# Two-or-more segment POSIX paths — `/repo/project`, `/work/src`. Two precision rules,
# both learned from real captures rather than guessed:
#   * a single leading segment ("/" alone, "/v1" from a URL) is not a directory claim;
#   * the lookbehind is load-bearing. Without it `path/to/filename.js` — boilerplate in
#     several CLIs' system prompts — matches at `/to/filename.js` and the tool is
#     recorded as announcing a working directory it never mentioned.
_ABS_PATH = re.compile(r"(?<![A-Za-z0-9._+-])/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)+")
# URLs are stripped BEFORE matching: `https://github.com/o/r/issues` otherwise yields the
# "absolute path" /github.com/o/r/issues, because the second slash of `//` is a valid left
# boundary. A link in a system prompt says nothing about a working directory.
_URL = re.compile(r"\bhttps?://\S+")
# A `~/`-relative path resolves under HOME, not the working directory. Recorded as
# evidence (a tool DID name a filesystem path) but not counted as a cwd announcement.
_HOME_RELATIVE = re.compile(r"~(?:/[A-Za-z0-9._+-]+)+")

# The ONLY reason string the verified-failure ladder stamps (src/shunt/router/escalation.py
# builds it as f"same_verified_failure_x{escalate_after_n}"). It is matched, not merely read,
# because `rank_escalation_reason` is also written on non-escalation paths — the router's
# fallback chain stamps `exploration_untested` / `safe_fallback` into the same field
# (src/shunt/router/engine.py). Accepting "any string" would accept those, and would accept
# a hand-written one. The captured integer is the ladder's threshold, and the failure count
# in the store must reach it.
_ESC_REASON = re.compile(r"same_verified_failure_x(\d+)")


def say(line: str) -> None:
    """The leg's report. This IS a container entrypoint — stdout is its whole interface."""
    print(line)  # noqa: T201 - container entrypoint


def warn(line: str) -> None:
    """Diagnostics on stderr, so a failing leg explains itself in the CI log."""
    print(line, file=sys.stderr)  # noqa: T201 - container entrypoint


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def poll_until[T](pred: Callable[[], T | None], *, timeout: float, what: str) -> T:
    """Poll *pred* until it returns a non-None value, bounded. Never sleeps on a guess."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = pred()
        if found is not None:
            return found
        time.sleep(0.25)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


# ── the outcome store ─────────────────────────────────────────────────────────


def _connect(db_path: Path) -> sqlite3.Connection:
    """Read-only connection — the sidecar observes the router's DB, it never migrates it."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _marker_sessions(db_path: Path, marker: str) -> list[dict[str, Any]]:
    """Every session whose prompt carries *marker*, oldest first, with parsed provenance."""
    if not db_path.exists():
        return []
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, model_chosen, decision_provenance FROM sessions "
                "WHERE prompt_text LIKE ? ORDER BY rowid",
                (f"%{marker}%",),
            ).fetchall()
    except sqlite3.DatabaseError:
        # The router may be mid-write on a fresh DB; the caller is polling, so just retry.
        return []
    return [
        {
            "session_id": row["session_id"],
            "model_chosen": row["model_chosen"],
            "provenance": json.loads(row["decision_provenance"] or "{}"),
        }
        for row in rows
    ]


def _verified_failures(db_path: Path, marker: str | None = None) -> int:
    """Count of rerun-confirmed Tier-2 failures — the input the ladder actually counts."""
    # `tombstoned` rows are excluded: a retracted event is not evidence. With *marker* the
    # count is restricted to THIS run's sessions, so a red left behind by other traffic
    # against the same store cannot stand in for one this run produced.
    if not db_path.exists():
        return 0
    sql = (
        "SELECT COUNT(*) AS n FROM outcome_events "
        "WHERE tier = 2 AND outcome = 'failure' AND COALESCE(tombstoned, 0) = 0"
    )
    params: tuple[str, ...] = ()
    if marker is not None:
        sql += " AND session_id IN (SELECT session_id FROM sessions WHERE prompt_text LIKE ?)"
        params = (f"%{marker}%",)
    try:
        with _connect(db_path) as conn:
            row = conn.execute(sql, params).fetchone()
    except sqlite3.DatabaseError:
        return 0
    return int(row["n"])


def _escalated(db_path: Path, marker: str) -> dict[str, Any] | None:
    """The first marker-carrying decision Shunt stamped as an auto-escalation, if any."""
    for session in _marker_sessions(db_path, marker):
        if session["provenance"].get("auto_escalated") is True:
            return session
    return None


# ── the ladder the DEPLOYMENT's own config declares ───────────────────────────
#
# "Escalation fired" and "escalation fired the rung the config says" are different claims, and
# only the second is a conformance check. The rung is therefore not hardcoded here: it is
# recomputed from the running deployment's registry + router policy through the SAME pure
# function the router uses (`shunt.router.escalation.next_rung_rank`), so this file cannot hold
# a second, drifting copy of the ladder arithmetic. That is also what makes the check portable
# across deployments — the hermetic fake registry and a real $0 free-tier registry declare
# different pools, and each is asserted against its own.


@dataclass(frozen=True)
class LadderSpec:
    """Rank order + the two escalation knobs, as the deployment's config declares them."""

    ranks: dict[str, int]
    max_rank_index: int
    escalate_after_n: int
    rank_shortlist: int

    def expected_rung(self, from_model: str) -> int | None:
        """The rank the ladder aims at from *from_model*; None when the model is off-registry."""
        base = self.ranks.get(from_model)
        if base is None:
            return None
        # CLAMPED to the top rank. `next_rung_rank` is pure arithmetic over the shortlist and
        # happily names a rank above the pool when the shortlist is wider than the pool — a
        # 2-model pool with rank_shortlist=3 aims at rank 2 from rank 1. The router resolves
        # that by `models_from_rank(target)` simply returning nothing above the top, so the
        # highest rung it can actually SERVE is max_rank_index. Measured on the $0 rig, where
        # the unclamped form reported a ladder violation against a rank that cannot exist.
        aim = next_rung_rank(base, self.max_rank_index, self.rank_shortlist)
        return min(aim, self.max_rank_index)


def load_ladder_spec() -> LadderSpec:
    """Read the ladder off the deployment's real config. Raises — never guesses a default.

    A guessed ladder would make the conformance check assert the shipped defaults rather than
    what this deployment runs, which is how a gate goes quietly green on a changed policy.
    """
    from shunt.models import ModelPool
    from shunt.router.policy import load_router_policy

    policy = load_router_policy()
    pool = ModelPool(config_path=os.environ.get("SHUNT_MODEL_CONFIG_PATH"))
    if policy.models:
        pool.restrict_to_live(policy.models)
    ranked = [m.name for m in pool.ranked_models()]
    if not ranked:
        raise RuntimeError("the resolved model pool is empty — no ladder to conform to")
    return LadderSpec(
        ranks={name: i for i, name in enumerate(ranked)},
        max_rank_index=len(ranked) - 1,
        escalate_after_n=policy.escalation.escalate_after_n,
        rank_shortlist=policy.escalation.rank_shortlist,
    )


# ── the verdict ───────────────────────────────────────────────────────────────
#
# Finding `auto_escalated: true` is where the check STARTS, not where it ends. That flag is
# one boolean in one JSON blob, and a store containing nothing but that boolean — no verified
# failures, one session, the cold-start model unchanged — satisfied the old pass condition and
# printed OK. So the flag is now treated as a CLAIM, and the four independent facts that must
# hold if the claim is true are each asserted against the store:
#
#   (a) the rerun-confirmed Tier-2 failures the ladder counts actually exist, on THIS run's
#       sessions, in at least the number the reason string names;
#   (b) the run really crossed the session boundaries it needs (>= the driver's prompt count),
#       so a single-session store cannot pass;
#   (c) what was SERVED changed — the escalated decision's model differs from the model the
#       preceding non-escalated marker sessions were served;
#   (d) the reason is the verified-failure reason, not any string.
#
# Each is a fact a forger would have to fabricate consistently across two tables; together they
# are what make the exit code mean "Shunt escalated" rather than "a boolean was true".
#
# (a)-(d) prove an escalation HAPPENED. Four more prove it happened AS SPECIFIED — which is the
# only claim the evidence supports, and the only one this gate is allowed to make (the escalate
# arm does NOT beat always-frontier on quality, so a live gate must never be built around a
# performance win):
#
#   (e) it fired on the CONFIGURED recurrence threshold — the reason's x<n> equals this
#       deployment's `escalate_after_n`, not merely some integer;
#   (f) it walked the rung the CONFIGURED ladder aims at, recomputed from the deployment's own
#       registry + policy through the router's own `next_rung_rank`;
#   (g) at most one step per recurrence — no two adjacent escalated sessions, which is what the
#       retirement of the consumed failure window guarantees;
#   (h) `shunt explain` names the ESCALATED model, not the base pick.
#   (i) a rung logged as `raise_effort` really did KEEP the model — the served model is
#       the pre-escalation one, since that is the whole definition of the cache-safe rung.
#
# NOT asserted, and deliberately not faked: "never mid-cached-turn". The store holds ONE row per
# session with ONE model, so a mid-session switch is not representable in it — there is no
# evidence here either way, and an assertion over a field that cannot disagree would be a gate
# that always passes. The property is enforced structurally (one decision per session, taken at
# the boundary) and the observable this harness DOES carry is (c)+(f): the change of served model
# coincides with a new session id.


def _baseline_models(sessions: list[dict[str, Any]], escalated_session_id: str) -> set[str]:
    """Models served on the non-escalated marker sessions BEFORE the escalated one."""
    models: set[str] = set()
    for session in sessions:
        if session["session_id"] == escalated_session_id:
            break
        if session["provenance"].get("auto_escalated") is not True:
            models.add(str(session["model_chosen"]))
    return models


def _preceding_model(sessions: list[dict[str, Any]], escalated_session_id: str) -> str | None:
    """Model served on the LAST non-escalated marker session before the escalated one."""
    previous: str | None = None
    for session in sessions:
        if session["session_id"] == escalated_session_id:
            return previous
        if session["provenance"].get("auto_escalated") is not True:
            previous = str(session["model_chosen"])
    return previous


def _ladder_problems(
    session: dict[str, Any], sessions: list[dict[str, Any]], ladder: LadderSpec
) -> list[str]:
    """(e) the threshold and (f) the rung must be the ones this deployment's config declares."""
    prov = session["provenance"]
    problems: list[str] = []

    reason = prov.get("rank_escalation_reason")
    match = _ESC_REASON.fullmatch(str(reason)) if isinstance(reason, str) else None
    if match is not None and int(match.group(1)) != ladder.escalate_after_n:
        problems.append(
            f"the escalation fired at x{match.group(1)} verified failures, but this "
            f"deployment's policy declares escalate_after_n={ladder.escalate_after_n} — the "
            "mechanism did not trigger on the condition the config specifies"
        )

    # An EFFORT rung keeps the model on purpose (cache-safe), so rank conformance does not
    # apply to it; assertion (c) below is the one that speaks to that case.
    if prov.get("escalated_reasoning_arm"):
        return problems

    base = _preceding_model(sessions, str(session["session_id"]))
    if base is None:  # (c) already reports the absent pre-escalation decision
        return problems
    served = str(session["model_chosen"])
    served_rank = ladder.ranks.get(served)
    expected = ladder.expected_rung(base)
    base_rank = ladder.ranks.get(base)
    if expected is not None and expected == base_rank:
        # Names the CAUSE. "(c) served model unchanged — nothing was escalated" is true here but
        # misattributes it: the ladder did fire, it simply had no rung left. Measured on the $0
        # free-tier rig, where OpenRouter 429s the cheapest :free model until it is marked
        # unhealthy and the pool collapses to one reachable model at the top rank.
        return [
            *problems,
            f"the pre-escalation model {base!r} is already at the pool's TOP rank "
            f"({base_rank} of {ladder.max_rank_index}), so no rank rung exists above it — this "
            "deployment cannot demonstrate a rank step at all, whatever the ladder does",
        ]
    if served_rank is None or expected is None:
        unknown = base if expected is None else served
        problems.append(
            f"the escalation stepped {base!r} -> {served!r}, but {unknown!r} is not in the "
            "deployment's live model pool — the served decision is not on the ladder the "
            "config declares"
        )
    elif served_rank != expected:
        problems.append(
            f"the escalation stepped {base!r} (rank {ladder.ranks[base]}) -> {served!r} "
            f"(rank {served_rank}), but the configured ladder "
            f"(rank_shortlist={ladder.rank_shortlist}, top rank {ladder.max_rank_index}) aims "
            f"at rank {expected} — the ladder did not walk the rung the config specifies. The "
            "one legitimate cause is a model-health gap forcing a different pick, which must "
            "be confirmed deliberately rather than assumed"
        )
    return problems


def _effort_rung_problems(session: dict[str, Any]) -> list[str]:
    """(i) a `raise_effort` rung must have kept the model — that IS what the rung means."""
    # The gap this closes: the tool read the provenance's own claim of success and never asked
    # the store what was SERVED. A live run where the effort rung failed on the wire and the
    # fallback chain served a DIFFERENT model exited 0 — the tool reported the router escalated
    # as designed while the cache-safe rung had silently become a rank jump. `raise_effort` is
    # defined as "same model, request-level reasoning params only"; a served model that differs
    # from the pre-escalation one falsifies the record, whatever the flags say.
    prov = session["provenance"]
    record = prov.get("escalation_exploration")
    if not isinstance(record, dict) or str(record.get("action")) != "raise_effort":
        return []
    served = str(session["model_chosen"])
    before = prov.get("pre_escalation_model")
    if not isinstance(before, str) or not before:
        return [
            "the decision logs an `escalation_exploration` action of 'raise_effort' but carries "
            "no `pre_escalation_model`, so the claim that the model was KEPT cannot be checked "
            "against the store at all"
        ]
    if served != before:
        return [
            f"the decision logs a 'raise_effort' rung — same model, request-level reasoning "
            f"params only — but the served model is {served!r} while the pre-escalation model "
            f"was {before!r}: the cache-safe effort rung became a jump onto a different model, "
            "and the provenance records it as having worked"
        ]
    return []


def _is_ladder_step(session: dict[str, Any]) -> bool:
    """A decision that STEPPED the ladder, as opposed to one merely held at the rung it owns."""
    # `escalation_floor` is also stamped `auto_escalated: true` — it is the router refusing to
    # walk BACK below a rank the task already climbed, on every later session. Counting those as
    # steps made the cadence check fail a perfectly healthy live run (measured on the $0 rig:
    # one real step followed by seven floor holds). Only the verified-failure reason is a step.
    reason = session["provenance"].get("rank_escalation_reason")
    return isinstance(reason, str) and _ESC_REASON.fullmatch(reason) is not None


def _cadence_problems(sessions: list[dict[str, Any]], ladder: LadderSpec) -> list[str]:
    """(g) at most one escalation per recurrence: the acted-on failure window is retired."""
    # `RouterEngine._retire_after_escalation` drops the failures an escalation consumed, so the
    # NEXT session starts the counter at zero and cannot reach escalate_after_n (>= 2) from the
    # single verified failure one session close produces. Two adjacent ladder STEPS therefore
    # mean the window was not retired — the ladder ran away from its own condition.
    if ladder.escalate_after_n < 2:
        return []
    flags = [_is_ladder_step(s) for s in sessions]
    adjacent = [
        f"{sessions[i - 1]['session_id'][:8]}->{sessions[i]['session_id'][:8]}"
        for i in range(1, len(flags))
        if flags[i] and flags[i - 1]
    ]
    if not adjacent:
        return []
    return [
        f"consecutive escalated sessions ({', '.join(adjacent)}) — with "
        f"escalate_after_n={ladder.escalate_after_n} the window an escalation consumed is "
        "retired, so a back-to-back second step fired without the recurrence it requires"
    ]


def explain_problems(explain_text: str, served_model: str) -> list[str]:
    """(h) the shipped read-out must name the ESCALATED model, not the base pick."""
    # A regression this exact check exists for: `shunt explain` once reported the base model
    # for an escalated decision, so the one surface an operator reads to confirm an escalation
    # denied it had happened. Asserted here because the sidecar is the only place that holds
    # both the store's truth and the CLI's rendering of it at the same time.
    for line in explain_text.splitlines():
        if not line.startswith("Model chosen:"):
            continue
        named = line.split(":", 1)[1].strip()
        if named != served_model:
            return [
                f"`shunt explain` reports 'Model chosen: {named}' for the escalated decision, "
                f"but the store says {served_model!r} was served — the read-out is naming a "
                "model other than the escalated one"
            ]
        return []
    return [
        "`shunt explain` printed no 'Model chosen:' line for the escalated decision, so the "
        f"served model cannot be confirmed from the shipped read-out: {explain_text!r}"
    ]


def _verdict_problems(
    db_path: Path,
    marker: str,
    session: dict[str, Any],
    sessions: list[dict[str, Any]],
    *,
    min_sessions: int,
    ladder: LadderSpec,
) -> list[str]:
    """Every way the claimed escalation can fail to be backed by the store. Empty == pass."""
    prov = session["provenance"]
    problems: list[str] = []
    problems.extend(_ladder_problems(session, sessions, ladder))
    problems.extend(_cadence_problems(sessions, ladder))
    problems.extend(_effort_rung_problems(session))

    # (d) the reason
    reason = prov.get("rank_escalation_reason")
    match = _ESC_REASON.fullmatch(str(reason)) if isinstance(reason, str) else None
    if match is None:
        problems.append(
            f"rank_escalation_reason is {reason!r}, not the verified-failure reason "
            "'same_verified_failure_x<n>' — the field is also written by the fallback chain "
            "('exploration_untested', 'safe_fallback'), so any other value means this decision "
            "was not driven by verified failures"
        )

    # (a) the failures the reason claims
    required = int(match.group(1)) if match else 1
    verified = _verified_failures(db_path, marker)
    if verified < required:
        problems.append(
            f"only {verified} rerun-confirmed Tier-2 failure(s) on {marker!r} sessions, "
            f"need >= {required} — the provenance claims failures the outcome_events log "
            "does not hold"
        )

    # (b) the session boundaries
    if len(sessions) < min_sessions:
        problems.append(
            f"only {len(sessions)} marker session(s) observed, need >= {min_sessions} — the "
            "driver drives that many prompts across that many sessions, and an escalation "
            "recorded without them was not produced by this scenario"
        )

    # (c) what was actually served
    #
    # In this hermetic harness the fake registry (tests/integrations/fake_registry.yaml)
    # declares no reasoning arms, so the ladder's cache-safe effort rung is unreachable and
    # RAISE_RANK — a different model — is the only escalation that can occur. If that registry
    # ever grows an effort ladder, an escalation could legitimately keep the same model and
    # this assertion must be revisited DELIBERATELY, which is exactly what a hard failure here
    # forces. `escalated_reasoning_arm` in the provenance is the tell.
    served = str(session["model_chosen"])
    baseline = _baseline_models(sessions, str(session["session_id"]))
    if not baseline:
        problems.append(
            "no non-escalated marker session precedes the escalated one — there is no "
            "pre-escalation model to compare the served model against"
        )
    elif served in baseline:
        problems.append(
            f"served model {served!r} is unchanged from the pre-escalation model(s) "
            f"{sorted(baseline)} — nothing was escalated"
            + (
                f" (provenance claims a cache-safe effort step to arm "
                f"{prov['escalated_reasoning_arm']!r}; the harness registry declares no "
                "reasoning arms, so that path is unreachable here)"
                if prov.get("escalated_reasoning_arm")
                else ""
            )
        )

    return problems


def _explain(session_id: str) -> str:
    """The shipped read-out for the found decision — the human-readable half of the receipt."""
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell, no wire input)
        ["shunt", "explain", session_id],  # noqa: S607 (on PATH inside the router image)
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or proc.stderr.strip()


# ── cwd announcement measurement ──────────────────────────────────────────────


def _system_text(body: dict[str, Any]) -> str:
    """The request's system block, across both wire shapes, flattened to text."""
    parts: list[str] = []
    system = body.get("system")  # Anthropic shape, when it reaches the upstream unflattened
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(str(block.get("text", "")) for block in system if isinstance(block, dict))
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return "\n".join(parts)


def _cwd_report(tool: str, record_dir: Path) -> dict[str, Any]:
    """Does *tool* announce an absolute working directory in its system block?"""
    log = record_dir / "requests.jsonl"
    systems: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines() if log.exists() else []:
        try:
            body = json.loads(line).get("body") or {}
        except json.JSONDecodeError:
            continue
        text = _system_text(body)
        if text:
            systems.append(text)
    home_relative: set[str] = set()
    absolute: set[str] = set()
    for text in systems:
        home_relative.update(_HOME_RELATIVE.findall(text))
        stripped = _URL.sub(" ", _HOME_RELATIVE.sub(" ", text))
        absolute.update(_ABS_PATH.findall(stripped))
    return {
        "tool": tool,
        "requests_recorded": sum(1 for _ in log.read_text().splitlines()) if log.exists() else 0,
        "system_blocks": len(systems),
        "absolute_paths": sorted(absolute),
        # Kept separate rather than merged: a HOME-relative config path proves the tool
        # talks about the filesystem, which is NOT the same claim as announcing the
        # directory it was launched in. Merging them would inflate the answer.
        "home_relative_paths": sorted(home_relative),
        "announces_cwd": bool(absolute),
    }


# ── entrypoint ────────────────────────────────────────────────────────────────


def _write_cwd_report(tool: str) -> None:
    """Emit the cwd artifact. Always runs — the measurement is independent of the verdict."""
    record_dir = os.environ.get("FAKE_UPSTREAM_RECORD_DIR")
    if not record_dir:
        return
    report = _cwd_report(tool, Path(record_dir))
    Path(record_dir, "cwd-report.json").write_text(json.dumps(report, indent=2) + "\n")
    say(f"CWD-REPORT {json.dumps(report)}")


def _dump_state(db_path: Path, marker: str) -> None:
    """The store's side of the story, in the terms that distinguish the causes."""
    sessions = _marker_sessions(db_path, marker)
    warn(f"  marker sessions: {len(sessions)}")
    for session in sessions:
        prov = session["provenance"]
        warn(
            f"    {session['session_id'][:8]} model={session['model_chosen']} "
            f"rule={prov.get('selection_rule_used')} escalated={prov.get('auto_escalated', False)} "
            f"reason={prov.get('rank_escalation_reason')}"
        )
    warn(
        f"  rerun-confirmed verified failures: {_verified_failures(db_path, marker)} on marker "
        f"sessions, {_verified_failures(db_path)} in the store"
    )
    if len(sessions) < 2:
        # The exact defect that made the hand-driven recipe report a false negative.
        warn(
            "  hint: fewer sessions than prompts means the driver reused one session "
            "(same tool identity) — vary the User-Agent per prompt."
        )


def _fail(db_path: Path, marker: str) -> int:
    """No decision in the store even claims an escalation."""
    warn(f"FAIL: no auto-escalated decision for marker {marker!r}")
    _dump_state(db_path, marker)
    return 1


def _fail_unsupported(db_path: Path, marker: str, problems: list[str]) -> int:
    """A decision CLAIMS an escalation the rest of the store does not support."""
    warn(
        f"FAIL: a decision for marker {marker!r} is stamped auto_escalated=true, but the store "
        "does not back the claim:"
    )
    for problem in problems:
        warn(f"  - {problem}")
    _dump_state(db_path, marker)
    return 1


def main() -> int:
    tool = os.environ.get("SHUNT_ESC_TOOL", "unknown")
    # --cwd-only: emit the working-directory measurement and stop. The wiring scenario
    # uses it, so the "does this tool announce its cwd?" question is answered for EVERY
    # CI-eligible tool, not only the three that drive an escalation.
    if "--cwd-only" in sys.argv:
        _write_cwd_report(tool)
        return 0

    marker = os.environ["SHUNT_ESC_MARKER"]
    db_path = Path(os.environ.get("SHUNT_DATA_DIR", ".")) / "outcomes.db"
    timeout = _env_int("SHUNT_ESC_TIMEOUT", 120)

    _write_cwd_report(tool)
    try:
        session = poll_until(
            lambda: _escalated(db_path, marker),
            timeout=timeout,
            what=f"an auto-escalated decision on a {marker!r} session",
        )
    except TimeoutError as exc:
        warn(str(exc))
        return _fail(db_path, marker)

    try:
        ladder = load_ladder_spec()
    except Exception as exc:  # noqa: BLE001 - an unreadable policy must FAIL, never skip (f)/(g)
        warn("FAIL: cannot read this deployment's ladder config, so conformance to it is")
        warn(f"      unverifiable — refusing to pass on the remaining checks: {exc}")
        return 1

    sessions = _marker_sessions(db_path, marker)
    explain_text = _explain(session["session_id"])
    problems = _verdict_problems(
        db_path,
        marker,
        session,
        sessions,
        # The driver's prompt count, passed through by run_scenario.sh so the two cannot
        # drift. Absent, the declared default of every escalation compose file (4).
        min_sessions=_env_int("SHUNT_ESC_MIN_SESSIONS", 4),
        ladder=ladder,
    )
    problems.extend(explain_problems(explain_text, str(session["model_chosen"])))
    if problems:
        return _fail_unsupported(db_path, marker, problems)

    prov = session["provenance"]
    baseline = sorted(_baseline_models(sessions, str(session["session_id"])))
    say(f"OK: {tool} escalation scenario passed")
    say(f"  session:   {session['session_id']}")
    say(f"  model:     {session['model_chosen']} (escalated from {', '.join(baseline)})")
    say(f"  reason:    auto_escalation ({prov.get('rank_escalation_reason')})")
    say(f"  threshold: escalate_after_n={ladder.escalate_after_n} (from the deployment policy)")
    say(f"  ladder:    rank_shortlist={ladder.rank_shortlist}, top rank {ladder.max_rank_index}")
    say(f"  marker sessions observed: {len(sessions)}")
    say(f"  verified failures behind it: {_verified_failures(db_path, marker)}")
    say(explain_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
