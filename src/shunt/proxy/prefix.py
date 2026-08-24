"""Prompt-prefix session identity — the resume key for clients that send no session id."""

# Claude Code, aider and plain HTTP clients never declare a conversation id, so before this
# tier the router grouped them by `(source_ip, user_agent)`: every conversation from one
# machine collapsed onto one session, and a restarted conversation could not find its own
# prior decision at all. The prefix digest is a per-conversation key derived from the request
# itself: the scheme tag, the client identity, a digest of the resolved repo and the normalised
# opening prefix, joined and hashed once (see `compute_prefix_digest`).
#
# WHY IT IS STABLE ACROSS A RESUME. Only the FIRST system block and the FIRST user block are
# hashed. A conversation grows by appending turns, and a resumed conversation is replayed from
# its beginning, so those two blocks are fixed for the life of the conversation by construction
# — where a whole-body hash would change on every single turn.
#
# WHY VOLATILE INJECTIONS ARE NORMALISED, AND ONLY IN THE SYSTEM HALF. Coding agents splice the
# working directory, the date and the current git state into the opening SYSTEM prompt, so the
# same conversation resumed an hour later, or after a commit, would otherwise hash differently.
# The USER turn gets none of that treatment: the client replays it byte-for-byte on a resume, so
# it is already stable, and substituting `<path>` into it destroys the only discrimination the
# digest has. Unanchored over a user's task text, the path rule collapsed
# "fix the failing test in src/shunt/router/policy.py" and "…/db/store.py" onto ONE digest, and
# two unrelated tasks in one repo then resumed each other's locked model, arm and frozen prefix.
#
# WHY THE PROJECT IS BOUND RATHER THAN STRIPPED. Normalisation removes the cwd, which would make
# two repos driven by the same CLI with the same opening question collide. The resolved repo is
# therefore re-entered as a one-way digest — bound, not carried, exactly as the escalation task
# key digests its work_dir at ingress. Nothing recoverable is stored IN THIS COLUMN: the
# `prefix_digest` column holds a digest, never text and never a path.
#
# THE SCOPE OF THAT CLAIM IS THIS COLUMN, NOT THE ROW. It is a statement about `prefix_digest`
# and it does not generalise to the session record around it. `sessions.prompt_text` holds the
# prompt, and with `escalation.context_transfer: summary` enabled (OFF by default)
# `sessions.decision_provenance` holds `context_transfer_prefix` — the authored handover note
# and the client's leading system blocks VERBATIM, which for a coding agent carry the working
# directory, git status and recent commit subjects. That is plaintext at rest in the local
# SQLite database, and it is required: a resume that cannot restore those exact bytes resends
# the client's original messages and pays a cache miss every turn. Documented for operators in
# SECURITY.md and docs/escalation.md; do not read the digest guarantee as a database-wide one.

from __future__ import annotations

import hashlib
import re
from typing import Any, Final

from shunt.router.engine import task_state_key

# Bounds the hashed text. A coding agent's opening system block runs tens of thousands of
# characters; the head is the stable template, and the tail is where injected context lives — so
# the system half is truncated to the head and the tail is deliberately not hashed.
#
# The USER half is bounded at the same width but NOT truncated away: past the cap it carries a
# fingerprint of the whole text (see `canonicalise_user_prefix`). Truncating it outright merged
# every conversation whose task text differs only after 4 KB, which is routine the moment an
# agent pastes a file or a stack trace into its opening message.
_MAX_PREFIX_CHARS: Final[int] = 4096

_DIGEST_SCHEME: Final[str] = "shunt-prefix-v1"

# Whole lines the host injects fresh on every launch. Matched on the stripped line's head,
# case-insensitively, and dropped entirely — their VALUES are volatile and their presence is
# not informative.
_VOLATILE_LINE_PREFIXES: Final[tuple[str, ...]] = (
    "working directory:",
    "is directory a git repo:",
    "platform:",
    "os version:",
    "today's date",
    "current date",
    "current branch:",
    "main branch",
)

# Volatile lines that introduce a BLOCK: the header AND every line under it, up to the next
# blank line, is dropped. A git-status or commit listing changes on every commit, and dropping
# only its header would leave the churn in the hash.
_VOLATILE_BLOCK_PREFIXES: Final[tuple[str, ...]] = (
    "recent commits:",
    "gitstatus:",
    "git status:",
    "status:",
)

# Applied to what survives the line filter. Order matters: paths before shas, so a path
# component that looks like a sha is consumed as part of the path.
_NORMALISERS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Absolute POSIX paths (the cwd, worktree roots, file lists in a status block).
    (re.compile(r"/(?:[\w.+@~-]+/)*[\w.+@~-]+"), "<path>"),
    # Windows paths.
    (re.compile(r"[A-Za-z]:\\(?:[\w.+@~ -]+\\)*[\w.+@~-]+"), "<path>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"), "<time>"),
    # Commit ids in an injected git-status block.
    (re.compile(r"\b[0-9a-f]{7,40}\b"), "<sha>"),
)

_WHITESPACE = re.compile(r"\s+")


def _first_block_text(content: object) -> str:
    """The text of the FIRST content block of *content* (str, block list, or block dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "") or "")
    if isinstance(content, list):
        for block in content:
            text = _first_block_text(block)
            if text:
                return text
    return ""


def _first_message_text(messages: object, role: str) -> str:
    """The first block of the first *role* message, in wire order."""
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            text = _first_block_text(message.get("content", ""))
            if text:
                return text
    return ""


def extract_prefix(body: dict[str, Any]) -> tuple[str, str]:
    """The opening `(system, user)` prefix of a request body, in either wire format."""
    # Anthropic carries the system prompt in a top-level `system` field; OpenAI carries it as
    # the first `system` message. Either may be a string or a block list, first block only.
    system_text = _first_block_text(body.get("system", "")) or _first_message_text(
        body.get("messages"), "system"
    )
    user_text = _first_message_text(body.get("messages"), "user")
    return system_text, user_text


# Roles that carry a turn of the conversation. `system` is excluded because it is scaffolding
# the client re-sends on every request, opening turn included — counting it would make a fresh
# single-turn OpenAI request (`[system, user]`) look like a replay.
_TURN_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant", "tool"})

# A conversation being REPLAYED carries at least the earlier user turn and the reply it drew;
# an OPENING request carries exactly one turn, the user's first message. Two is therefore the
# lowest count that cannot be an opening, and the boundary between the two cases is exact
# rather than heuristic — which is what the resume tier needs, since resolving an opening
# request would merge two unrelated conversations that happen to start with the same question.
_REPLAY_MIN_TURNS: Final[int] = 2


def is_replayed_conversation(body: dict[str, Any]) -> bool:
    """Whether *body* replays an existing conversation, rather than opening a new one."""
    # THE UNIQUENESS GAP THIS CLOSES. The digest is stable across a resume by construction, but
    # it is NOT unique to a conversation: a brand-new conversation opening with the same first
    # user message, in the same repo, from the same client, hashes identically to an unrelated
    # earlier one. Resolving it would silently hand the new conversation the old one's locked
    # model and attribute its verified outcomes to the old one's task state — the exact
    # mislabelling the ambiguity guard exists to prevent, arriving through a door that guard
    # cannot watch (two conversations that routed to the SAME model expose no distinct value
    # for it to refuse on). Wire shape separates them exactly, so the tier is asked only about
    # requests that CAN be a resume.
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") in _TURN_ROLES)
    return turns >= _REPLAY_MIN_TURNS


def normalise_prefix(text: str) -> str:
    """Strip the host's per-launch injections out of a SYSTEM prefix so a resume hashes alike."""
    # SYSTEM ONLY. The volatile-token substitutions are unanchored, so applying them to a user's
    # task text erases the file paths, dates and hashes that distinguish one task from another.
    # `canonicalise_user_prefix` is the user half's counterpart.
    kept: list[str] = []
    in_block = False
    for line in text.splitlines():
        head = line.strip().lower()
        if in_block:
            # A blank line closes the injected block; anything inside it is dropped.
            in_block = bool(head)
            if in_block:
                continue
        if head.startswith(_VOLATILE_BLOCK_PREFIXES):
            in_block = True
            continue
        if head.startswith(_VOLATILE_LINE_PREFIXES):
            continue
        kept.append(line)
    normalised = "\n".join(kept)
    for pattern, placeholder in _NORMALISERS:
        normalised = pattern.sub(placeholder, normalised)
    return _WHITESPACE.sub(" ", normalised).strip()[:_MAX_PREFIX_CHARS]


def canonicalise_user_prefix(text: str) -> str:
    """Bound a USER prefix for hashing — whitespace collapsed, volatile tokens left INTACT."""
    # No substitution: see the module header. Whitespace is collapsed because a client may
    # re-wrap the same text, and that is the only rewrite a replay is known to perform.
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if len(collapsed) <= _MAX_PREFIX_CHARS:
        return collapsed
    # Past the cap the WHOLE text still contributes, as a length plus a digest of itself. The
    # result stays bounded, and two conversations that agree for 4 KB and then diverge no longer
    # hash alike. Deterministic over the replayed bytes, so a resume is unaffected.
    whole = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()
    return f"{collapsed[:_MAX_PREFIX_CHARS]}\x1f{len(collapsed)}\x1f{whole}"


def compute_prefix_digest(
    body: dict[str, Any],
    tool_identity: str,
    work_dir: str | None,
) -> str | None:
    """The conversation key for *body*, or None when there is nothing stable to key on."""
    # None (rather than a digest of "") when the opening user block is empty: a body with no
    # task text carries no conversation identity, and hashing the empty string would collapse
    # every such request onto one shared key — the very failure this tier exists to remove.
    system_text, user_text = extract_prefix(body)
    if not user_text.strip():
        return None
    parts = (
        _DIGEST_SCHEME,
        tool_identity,
        # The repo is bound as a digest so no raw path reaches session state or the DB.
        task_state_key(work_dir) if work_dir else "",
        normalise_prefix(system_text),
        canonicalise_user_prefix(user_text),
    )
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
