"""The prompt-prefix conversation key: stable across a resume, distinct across projects."""

# This is the tier that makes a client sending no session id (Claude Code) resumable at all,
# so the two properties it must have are pinned here directly: appending turns must not move
# the digest, and two repos driven by the same CLI with the same opening question must not
# share one.

from __future__ import annotations

from typing import Any

from shunt.proxy.prefix import compute_prefix_digest, extract_prefix, normalise_prefix

_IDENTITY = "a" * 64
_REPO_A = "/home/dev/project-a"
_REPO_B = "/home/dev/project-b"

_SYSTEM_TEMPLATE = """You are Claude Code, Anthropic's official CLI for Claude.

<env>
Working directory: /home/dev/project-a
Is directory a git repo: Yes
Platform: linux
Today's date is 2026-08-22.
</env>
gitStatus: M src/shunt/proxy/server.py
Current branch: main
Recent commits:
8e30457 feat: ship session_cascade
"""

_SYSTEM_TEMPLATE_LATER = """You are Claude Code, Anthropic's official CLI for Claude.

<env>
Working directory: /home/dev/project-b
Is directory a git repo: Yes
Platform: linux
Today's date is 2026-09-01.
</env>
gitStatus: M docs/routing.md
Current branch: session/other
Recent commits:
c9c788d harness: fix six defects
"""


def _body(system: str, opening: str, turns: int = 0) -> dict[str, Any]:
    """An OpenAI-format body: the opening prefix plus *turns* appended exchanges."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": opening},
    ]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"answer {i}"})
        messages.append({"role": "user", "content": f"follow-up {i}"})
    return {"messages": messages, "stream": False}


def test_appending_twenty_turns_leaves_the_digest_unchanged() -> None:
    first = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Fix the router"), _IDENTITY, _REPO_A)
    later = compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE, "Fix the router", turns=20), _IDENTITY, _REPO_A
    )
    assert first is not None
    assert first == later


def test_volatile_injections_do_not_move_the_digest() -> None:
    # Same conversation, resumed on another day, from another checkout, after commits: the
    # cwd/date/git-status lines the host splices in must not re-key it.
    same = compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE_LATER, "Fix the router"), _IDENTITY, _REPO_A
    )
    assert same == compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE, "Fix the router"), _IDENTITY, _REPO_A
    )


def test_two_repos_with_the_same_opening_do_not_collide() -> None:
    # Same CLI, same template system prompt, same opening question — different project.
    a = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, _REPO_A)
    b = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, _REPO_B)
    assert a is not None and b is not None
    assert a != b


def test_different_openings_in_one_repo_do_not_collide() -> None:
    a = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, _REPO_A)
    b = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Delete a test"), _IDENTITY, _REPO_A)
    assert a != b


def test_different_clients_do_not_collide() -> None:
    a = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, _REPO_A)
    b = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), "b" * 64, _REPO_A)
    assert a != b


def test_a_body_with_no_task_text_has_no_key() -> None:
    assert compute_prefix_digest({"messages": []}, _IDENTITY, _REPO_A) is None
    assert (
        compute_prefix_digest(
            {"messages": [{"role": "system", "content": _SYSTEM_TEMPLATE}]}, _IDENTITY, _REPO_A
        )
        is None
    )
    assert (
        compute_prefix_digest(
            {"messages": [{"role": "user", "content": "   "}]}, _IDENTITY, _REPO_A
        )
        is None
    )


def test_an_unresolved_repo_still_keys_on_the_prefix() -> None:
    # No configured work_dir: the project cannot be bound, but the conversation is still
    # keyed far more tightly than by (ip, user_agent). Collisions are caught downstream by
    # the ambiguity guard, not by refusing to key at all.
    digest = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, None)
    assert digest is not None
    bound = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, "Add a test"), _IDENTITY, _REPO_A)
    assert digest != bound


def test_anthropic_and_openai_prefixes_are_both_read() -> None:
    anthropic = {
        "system": [{"type": "text", "text": _SYSTEM_TEMPLATE}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Fix the router"}]}],
    }
    system_text, user_text = extract_prefix(anthropic)
    assert system_text == _SYSTEM_TEMPLATE
    assert user_text == "Fix the router"
    openai_system, openai_user = extract_prefix(_body(_SYSTEM_TEMPLATE, "Fix the router"))
    assert openai_system == _SYSTEM_TEMPLATE
    assert openai_user == "Fix the router"


def test_only_the_first_block_of_the_opening_user_message_is_read() -> None:
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fix the router"},
                    {"type": "text", "text": "<attached-file>volatile</attached-file>"},
                ],
            }
        ]
    }
    other = {"messages": [{"role": "user", "content": "Fix the router"}]}
    assert compute_prefix_digest(body, _IDENTITY, _REPO_A) == compute_prefix_digest(
        other, _IDENTITY, _REPO_A
    )


def test_normalisation_replaces_paths_dates_times_and_shas() -> None:
    out = normalise_prefix("edit /home/dev/x/y.py at 2026-08-22 14:03 in 8e30457")
    assert "/home/dev" not in out
    assert "2026-08-22" not in out
    assert "8e30457" not in out
    assert out == "edit <path> at <date> <time> in <sha>"


def test_two_files_in_one_repo_are_two_conversations() -> None:
    """The normalisers must not run over the USER turn — they erase the task's identity."""
    # The regression: `_NORMALISERS`' path rule is unanchored, so applied to a user's own text it
    # rewrote both of these to "fix the failing test in src<path>" and one digest served two
    # unrelated tasks — the later one inheriting the earlier's locked model, arm and frozen
    # prefix. The user turn is replayed verbatim on a resume, so it needs no normalisation.
    a = compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE, "fix the failing test in src/shunt/router/policy.py"),
        _IDENTITY,
        _REPO_A,
    )
    b = compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE, "fix the failing test in src/shunt/db/store.py"),
        _IDENTITY,
        _REPO_A,
    )
    assert a is not None and b is not None
    assert a != b


def test_user_text_diverging_after_the_cap_does_not_collide() -> None:
    """A pasted file or stack trace pushes the discriminating text past 4 KB."""
    shared = "Here is the failing module:\n" + ("filler line\n" * 600)
    a = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, shared + "\nfix the parser"), _IDENTITY, None)
    b = compute_prefix_digest(_body(_SYSTEM_TEMPLATE, shared + "\nfix the writer"), _IDENTITY, None)
    assert len(shared) > 4096  # the divergence really is past the cap
    assert a is not None and b is not None
    assert a != b
    # And the whole-text fingerprint is deterministic, so the resume property still holds.
    assert a == compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE, shared + "\nfix the parser", turns=20), _IDENTITY, None
    )


def test_a_user_turn_carrying_a_date_still_resumes() -> None:
    """Bounded risk of not normalising the user half: the client replays it byte-for-byte."""
    body = _body(_SYSTEM_TEMPLATE, "ship the 2026-08-22 report")
    assert compute_prefix_digest(body, _IDENTITY, _REPO_A) == compute_prefix_digest(
        _body(_SYSTEM_TEMPLATE_LATER, "ship the 2026-08-22 report", turns=4), _IDENTITY, _REPO_A
    )
