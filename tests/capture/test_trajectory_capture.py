"""KILL-RISK kill-gate 2: no free-text field and no seeded secret survives redact ->
committable projection; the live plane writes only encrypted bytes; capture is default-off and
observe-only. Also pins the committable whitelist at parity with the offline evaluator's schema.
"""

from __future__ import annotations

import pytest

from shunt.capture.trajectory import (
    COMMITTABLE_FIELDS,
    StepRecord,
    TrajectoryRecorder,
    redact_record,
)
from shunt.capture.trajectory_store import LiveTrajectorySink, load_key, resolve_live_dir

_SECRET = "sk-ant-SECRETTOKEN0123456789"


def _record_with_secret() -> StepRecord:
    return StepRecord(
        step_index=1,
        decision_index=1,
        metadata={"note": f"auth {_SECRET}"},
        observation=f"server said {_SECRET}",
        action=f"curl -H 'Authorization: Bearer {_SECRET}'",
        args=f"--token {_SECRET}",
        result=f"leaked {_SECRET}",
        failing_check_id="tests/x.py::t",
        exit_code=1,
        blocking=True,
        is_infra_failure=False,
        confirmed=True,
        success=False,
    )


def _fernet_key() -> bytes:
    from cryptography.fernet import Fernet

    return Fernet.generate_key()


def test_committable_projection_carries_no_free_text_or_secret() -> None:
    projected = _record_with_secret().committable()
    for field in ("metadata", "observation", "action", "args", "result"):
        assert field not in projected
    assert not any(_SECRET in str(v) for v in projected.values())
    assert set(projected) == set(COMMITTABLE_FIELDS)


def test_committable_projection_redacts_a_secret_bearing_failing_check_id() -> None:
    # `failing_check_id` is the one committable free-form string; a parametrized test id can carry
    # a secret, so the projection scrubs it like every other free-text field.
    record = StepRecord(
        step_index=1,
        decision_index=1,
        metadata={},
        observation="",
        action="",
        args=None,
        result="",
        failing_check_id=f"tests/x.py::test_login[{_SECRET}]",
        exit_code=1,
        blocking=True,
        is_infra_failure=False,
        confirmed=True,
        success=False,
    )
    assert _SECRET not in str(record.committable()["failing_check_id"])


def test_redaction_scrubs_secret_from_every_free_text_field() -> None:
    redacted = redact_record(_record_with_secret())
    for text in (redacted.observation, redacted.action, redacted.args or "", redacted.result):
        assert _SECRET not in text
    assert _SECRET not in redacted.metadata["note"]


def test_live_plane_writes_only_encrypted_bytes_and_round_trips(tmp_path) -> None:
    pytest.importorskip("cryptography")
    key = _fernet_key()
    sink = LiveTrajectorySink(tmp_path, key)
    sink.write([_record_with_secret()])
    written = next(tmp_path.glob("*.traj.enc")).read_bytes()
    # ciphertext only — no plaintext secret, no plaintext free-text prose on disk
    assert _SECRET.encode() not in written
    assert b"Authorization" not in written
    assert b"leaked" not in written
    # the encrypt -> redact -> decrypt round-trip: every ciphertext line decrypts to the
    # REDACTED record (defense in depth), never the raw one
    import json

    from cryptography.fernet import Fernet

    decrypt = Fernet(key)
    payloads = [decrypt.decrypt(line) for line in written.splitlines() if line]
    assert payloads
    for payload in payloads:
        assert _SECRET.encode() not in payload
        record = json.loads(payload.decode("utf-8"))
        # redaction ran before encryption: the free-text fields survive as keys but carry no
        # secret, and the behaviour fields round-trip unchanged
        assert record["exit_code"] == 1
        assert record["success"] is False
        assert record["blocking"] is True
        assert _SECRET not in record["observation"]
        assert _SECRET not in record["action"]
        assert _SECRET not in (record["args"] or "")
        assert _SECRET not in record["result"]


def test_encrypted_live_plane_restricts_dir_and_file_permissions(tmp_path) -> None:
    pytest.importorskip("cryptography")
    import stat

    store = tmp_path / "store"
    sink = LiveTrajectorySink(store, _fernet_key())
    sink.write([_record_with_secret()])
    enc = next(store.glob("*.traj.enc"))
    # private plane may hold private code/secrets at rest → owner-only dir + files.
    assert stat.S_IMODE(store.stat().st_mode) == 0o700
    assert stat.S_IMODE(enc.stat().st_mode) == 0o600


def test_capture_is_default_off_and_observe_only() -> None:
    written: list[list[StepRecord]] = []

    class _Spy:
        def write(self, records: list[StepRecord]) -> None:
            written.append(records)

    recorder = TrajectoryRecorder(_Spy())  # default enabled=False
    assert recorder.enabled is False
    recorder.record([_record_with_secret()])
    assert written == []  # inert when disabled — no persistence, no side effect


def test_enabled_recorder_persists_redacted_records() -> None:
    captured: list[list[StepRecord]] = []

    class _Spy:
        def write(self, records: list[StepRecord]) -> None:
            captured.append(records)

    TrajectoryRecorder(_Spy(), enabled=True).record([_record_with_secret()])
    assert _SECRET not in captured[0][0].observation  # redacted before the sink saw it


def test_load_key_errors_when_enabled_without_key(monkeypatch) -> None:
    monkeypatch.delenv("SHUNT_ESCALATION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SHUNT_ESCALATION_KEY"):
        load_key()


def test_live_dir_resolves_outside_repo(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SHUNT_HOME", str(tmp_path))
    assert resolve_live_dir(None) == tmp_path / "trajectories"
    assert resolve_live_dir("/custom/dir") == __import__("pathlib").Path("/custom/dir")
