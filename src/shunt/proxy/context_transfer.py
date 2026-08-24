"""Context transfer — what an ESCALATED model receives of the conversation that ran on the
cheaper one. `full` forwards the client's messages untouched; `summary` replaces the prior
conversation, once, with a compaction Shunt authors."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)

CONTEXT_TRANSFER_FULL: Final[str] = "full"
CONTEXT_TRANSFER_SUMMARY: Final[str] = "summary"
# No `none`. Shunt can drop context from the request it forwards, but it cannot make the CLI
# forget: the client resends its whole conversation every turn, so `none` would have to strip
# on EVERY turn and the strong model could never accumulate state. That is a broken router,
# not a transfer mode, so it is not offered and the schema names it as refused.
CONTEXT_TRANSFER_MODES: Final[tuple[str, ...]] = (CONTEXT_TRANSFER_FULL, CONTEXT_TRANSFER_SUMMARY)

# The two reason tokens that mean "this session is serving an escalated rung". Any other
# `model_source` is a base pick, where there is nothing to transfer FROM.
ESCALATED_SOURCES: Final[frozenset[str]] = frozenset({"auto_escalation", "escalation_floor"})

# One attempt, hard-bounded. The summariser runs INSIDE the user's turn, so a hung call would
# hold the response open; a timeout degrades to `full` exactly like any other failure.
SUMMARY_TIMEOUT_SECONDS: Final[float] = 60.0

# Crude on purpose: a per-provider tokenizer is not worth a dependency for a ceiling whose only
# job is to catch a summariser that ignored the budget it was given.
_CHARS_PER_TOKEN: Final[int] = 4
# Slack before an answer counts as over-budget — the estimate above is approximate in both
# directions and a hair over the ceiling is not a reason to throw the summary away.
_BUDGET_SLACK: Final[float] = 1.25

# Prepended to the authored summary in the outbound request. The disclosure surfaces tell the
# OPERATOR; this line tells the MODEL, so it cannot mistake a compaction for a transcript and
# quote it back as something the user said.
SUMMARY_PREAMBLE: Final[str] = (
    "[shunt context transfer] The earlier part of this conversation was handled by a different "
    "model and has been REPLACED with the summary below. This is a summary written by another "
    "model, not the original transcript; do not quote it as the user's words.\n\n"
)

# What the summariser is asked for. Behaviour-only and instruction-shaped: it must not invent
# state, because everything it drops is state the escalated model will never see again.
_SUMMARY_INSTRUCTION: Final[str] = (
    "You are compacting a coding-assistant conversation so a different model can continue it. "
    "Write a factual handover note covering: what the user asked for, what was already done "
    "(files created or edited, commands run and their results), what is still failing, and what "
    "the next step is. Preserve exact file paths, identifiers, error text and decisions. Invent "
    "nothing, and do not address the user. Stay under {budget} tokens."
)

# A summariser call: (model, messages, max_tokens) -> completion text.
Summariser = Callable[[str, list[dict[str, Any]], int | None], Awaitable[str]]


@dataclass(frozen=True)
class ContextTransferPolicy:
    """The resolved `escalation.context_transfer*` knobs the proxy reads."""

    mode: str = CONTEXT_TRANSFER_FULL
    max_tokens: int | None = 2000
    model: str | None = None


@dataclass(frozen=True)
class ContextTransfer:
    """One session's transfer decision — taken ONCE, on the first escalated turn."""

    mode: str
    # The frozen compacted prefix. Every later turn resends these exact objects, so the
    # provider sees byte-identical leading bytes and the prefix cache HITS. A summary
    # regenerated per turn is a cache miss per turn, which costs more than `full`.
    prefix: tuple[dict[str, Any], ...] | None = None
    # How many leading messages of the ORIGINAL request the prefix stands in for. Later turns
    # send `prefix + messages[consumed:]`, so everything since the transfer is kept verbatim.
    consumed: int = 0
    summariser: str | None = None
    degraded_reason: str | None = None

    def restorable(self) -> dict[str, Any] | None:
        """The projection a RESUME rebuilds this decision from, or None when there is none."""
        # `provenance()` is the human read-out (`shunt explain`) and deliberately carries counts,
        # not content. Restoring the frozen prefix needs the content itself: the bytes are the
        # decision. A session evicted by the idle timeout and resumed without them resends the
        # client's original messages, which is a cache MISS every turn — the exact cost `summary`
        # exists to avoid.
        if self.mode != CONTEXT_TRANSFER_SUMMARY or self.prefix is None:
            return None
        return {
            "mode": self.mode,
            "prefix": [dict(m) for m in self.prefix],
            "consumed": self.consumed,
            "summariser": self.summariser,
        }

    def provenance(self) -> dict[str, Any]:
        """The `decision_provenance["context_transfer"]` record — what `shunt explain` prints."""
        if self.mode == CONTEXT_TRANSFER_SUMMARY:
            return {
                "mode": CONTEXT_TRANSFER_SUMMARY,
                "summariser": self.summariser,
                "replaced_messages": self.consumed,
                "prefix_messages": len(self.prefix or ()),
            }
        return {
            "mode": CONTEXT_TRANSFER_FULL,
            "requested": CONTEXT_TRANSFER_SUMMARY,
            "degraded_reason": self.degraded_reason,
        }


def from_restorable(record: object) -> ContextTransfer | None:
    """Rebuild a frozen `summary` transfer from its persisted projection, or None if unusable."""
    # Strict: a half-formed record must restore NOTHING rather than a prefix that misaligns with
    # `consumed` — a wrong boundary silently drops or duplicates turns of the conversation.
    if not isinstance(record, dict) or record.get("mode") != CONTEXT_TRANSFER_SUMMARY:
        return None
    prefix = record.get("prefix")
    consumed = record.get("consumed")
    if not isinstance(prefix, list) or not prefix or not isinstance(consumed, int) or consumed < 1:
        return None
    if not all(isinstance(m, dict) for m in prefix):
        return None
    summariser = record.get("summariser")
    return ContextTransfer(
        mode=CONTEXT_TRANSFER_SUMMARY,
        prefix=tuple(dict(m) for m in prefix),
        consumed=consumed,
        summariser=summariser if isinstance(summariser, str) else None,
    )


def estimate_tokens(text: str) -> int:
    """Approximate token count of *text* — a ceiling check, never an accounting number."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def split_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split into (leading system blocks, history to compact, the trailing turn)."""
    if len(messages) < 2:
        raise ValueError("nothing to compact: the request carries no history")
    head = 0
    while head < len(messages) - 1 and messages[head].get("role") == "system":
        head += 1
    systems = [dict(m) for m in messages[:head]]
    history = [dict(m) for m in messages[head:-1]]
    if not history:
        raise ValueError("nothing to compact: the request is system blocks plus one turn")
    return systems, history, dict(messages[-1])


def compact(
    messages: Sequence[dict[str, Any]], summary_text: str
) -> tuple[list[dict[str, Any]], int]:
    """Fold everything before the trailing turn into one authored user message.

    Returns the frozen prefix and how many original messages it replaces.
    """
    # The system blocks survive VERBATIM: they are the client's own operating instructions
    # (tool contracts, output format), and summarising them would change what the agent is
    # allowed to do rather than what it knows.
    systems, _history, _trailing = split_messages(messages)
    prefix = [*systems, {"role": "user", "content": SUMMARY_PREAMBLE + summary_text}]
    return prefix, len(messages) - 1


def summarisation_request(
    history: Sequence[dict[str, Any]], max_tokens: int | None
) -> list[dict[str, Any]]:
    """The message list sent to the summariser — the history, plus one instruction turn."""
    budget = max_tokens if max_tokens is not None else "roughly 2000"
    return [
        *(dict(m) for m in history),
        {"role": "user", "content": _SUMMARY_INSTRUCTION.format(budget=budget)},
    ]


def apply(transfer: ContextTransfer, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-apply the frozen prefix to *messages*, or return them unchanged.

    Byte-identical by construction: the same prefix objects are resent every turn.
    """
    if transfer.mode != CONTEXT_TRANSFER_SUMMARY or transfer.prefix is None:
        return list(messages)
    if len(messages) <= transfer.consumed:
        # The client rewrote history shorter than the boundary the prefix stands in for
        # (a rewind, a fork, a compaction of its own). Aligning anyway would silently drop
        # or duplicate turns, so this turn passes through in full instead.
        logger.warning(
            "context transfer: request carries %d message(s), fewer than the %d the frozen "
            "prefix replaces — forwarding the full conversation for this turn",
            len(messages),
            transfer.consumed,
        )
        return list(messages)
    return [*transfer.prefix, *messages[transfer.consumed :]]


async def resolve(
    messages: Sequence[dict[str, Any]],
    policy: ContextTransferPolicy,
    summariser_model: str,
    summarise: Summariser,
) -> ContextTransfer:
    """Take the session's once-only transfer decision. Never raises; degrades to `full`."""
    # ONE attempt, no retry loop. The fallback is not "try again", it is `full` — which is
    # correct behaviour, merely more expensive, so a retry buys nothing a user would notice
    # except a longer stall on the turn that already went wrong.
    try:
        _systems, history, _trailing = split_messages(messages)
    except ValueError as exc:
        return _degraded(str(exc))
    try:
        text = await asyncio.wait_for(
            summarise(
                summariser_model,
                summarisation_request(history, policy.max_tokens),
                policy.max_tokens,
            ),
            timeout=SUMMARY_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return _degraded(f"summariser timed out after {SUMMARY_TIMEOUT_SECONDS:.0f}s")
    except Exception as exc:  # any upstream failure, including a bad model id
        return _degraded(f"summariser failed: {type(exc).__name__}")
    if not isinstance(text, str) or not text.strip():
        return _degraded("summariser returned no text")
    if policy.max_tokens is not None:
        estimated = estimate_tokens(text)
        if estimated > policy.max_tokens * _BUDGET_SLACK:
            return _degraded(f"summary is over budget (~{estimated} tokens > {policy.max_tokens})")
    prefix, consumed = compact(messages, text.strip())
    return ContextTransfer(
        mode=CONTEXT_TRANSFER_SUMMARY,
        prefix=tuple(prefix),
        consumed=consumed,
        summariser=summariser_model,
    )


def _degraded(reason: str) -> ContextTransfer:
    """Degrade to `full`: the whole message list is forwarded, and the reason is recorded."""
    # NEVER a silent context drop. The expensive-but-correct path is the fallback, and the
    # WARNING plus the stamped `degraded_reason` are how an operator learns the cheap path
    # is not the one running.
    logger.warning(
        "context transfer requested `summary` but degraded to `full` (%s) — the escalated "
        "model receives the whole conversation, uncached, at full input price",
        reason,
    )
    return ContextTransfer(mode=CONTEXT_TRANSFER_FULL, degraded_reason=reason)
