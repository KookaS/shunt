"""The cwd-announcement measurement must not report a path the tool never announced."""

# This is a measurement, and a measurement that over-reports is worse than none: the
# question it answers ("which agentic CLIs tell the model their working directory?") would
# be settled wrongly, in writing, by a regex artifact. The false positive below is real —
# it came out of a live aider capture.

from __future__ import annotations

import json
from pathlib import Path

from tests.integrations.assert_escalation import _cwd_report, _system_text


def _write(record_dir: Path, bodies: list[dict[str, object]]) -> None:
    lines = [json.dumps({"path": "/v1/chat/completions", "body": body}) for body in bodies]
    (record_dir / "requests.jsonl").write_text("\n".join(lines) + "\n")


def test_illustrative_path_in_boilerplate_is_not_an_announcement(tmp_path: Path) -> None:
    # aider's system prompt says "path/to/filename.js" as an EXAMPLE. A matcher without a
    # left boundary reads that as the absolute path /to/filename.js and reports a cwd.
    _write(
        tmp_path,
        [{"messages": [{"role": "system", "content": "Return edits as path/to/filename.js"}]}],
    )
    report = _cwd_report("aider", tmp_path)
    assert report["system_blocks"] == 1
    assert report["absolute_paths"] == []
    assert report["announces_cwd"] is False


def test_a_real_working_directory_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [{"messages": [{"role": "system", "content": "Working directory: /home/dev/project"}]}],
    )
    report = _cwd_report("some-cli", tmp_path)
    assert report["absolute_paths"] == ["/home/dev/project"]
    assert report["announces_cwd"] is True


def test_a_link_is_not_a_working_directory(tmp_path: Path) -> None:
    # From a live opencode capture: the second slash of `//` is a valid left boundary, so
    # an unstripped URL reports the "absolute path" /github.com/anomalyco/opencode/issues.
    _write(
        tmp_path,
        [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Report bugs at https://github.com/anomalyco/opencode/issues",
                    }
                ]
            }
        ],
    )
    report = _cwd_report("opencode", tmp_path)
    assert report["absolute_paths"] == []
    assert report["announces_cwd"] is False


def test_home_relative_config_path_is_evidence_but_not_a_cwd_claim(tmp_path: Path) -> None:
    # ~/.config/opencode resolves under HOME. The tool named a path; it did not name the
    # directory it was launched in, and the report must not conflate the two.
    _write(
        tmp_path,
        [{"messages": [{"role": "system", "content": "Config lives in ~/.config/opencode"}]}],
    )
    report = _cwd_report("opencode", tmp_path)
    assert report["home_relative_paths"] == ["~/.config/opencode"]
    assert report["absolute_paths"] == []
    assert report["announces_cwd"] is False


def test_no_system_block_means_no_claim(tmp_path: Path) -> None:
    _write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}]}])
    report = _cwd_report("curl", tmp_path)
    assert report["system_blocks"] == 0
    assert report["announces_cwd"] is False


def test_missing_log_is_an_empty_measurement_not_a_crash(tmp_path: Path) -> None:
    # A leg that never reached the upstream must still produce a readable artifact.
    report = _cwd_report("never-ran", tmp_path)
    assert report == {
        "tool": "never-ran",
        "requests_recorded": 0,
        "system_blocks": 0,
        "absolute_paths": [],
        "home_relative_paths": [],
        "announces_cwd": False,
    }


def test_system_text_reads_both_wire_shapes() -> None:
    anthropic = {"system": [{"type": "text", "text": "cwd is /srv/app"}]}
    block = [{"type": "text", "text": "in /srv/app"}]
    openai = {"messages": [{"role": "system", "content": block}]}
    assert "/srv/app" in _system_text(anthropic)
    assert "/srv/app" in _system_text(openai)
