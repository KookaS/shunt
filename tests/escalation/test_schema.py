"""Schema: JSONL round-trip lossless, committable_projection secret-safe, derivable
fields recomputable.
"""

from __future__ import annotations

import pytest

from benchmark.escalation import schema
from benchmark.escalation.authenticity import errors, verify_trajectory
from tests.escalation.factories import make_step, make_trajectory

_FREE_TEXT = ("metadata", "observation", "action", "args", "result")


def test_jsonl_round_trip_is_lossless(tmp_path) -> None:
    traj = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=True),
            make_step(
                step_index=1,
                decision_index=1,
                success=False,
                failing_check_id="tests/test_x.py::test_a",
                test_passed=3,
                test_total=5,
            ),
        ]
    )
    path = tmp_path / "traj.jsonl"
    schema.dump_jsonl(traj, path)
    loaded = schema.load_jsonl(path)
    assert loaded == traj


def test_committable_projection_excludes_all_free_text() -> None:
    step = make_step(
        success=False,
        failing_check_id="chk",
        observation="SECRET-TOKEN sk-abc123",
        action="curl https://internal",
        args="--header 'Authorization: Bearer sk-xyz'",
        result="leaked prose",
    )
    projection = schema.committable_projection(step)
    for key in _FREE_TEXT:
        assert key not in projection
    # no committed value carries the seeded secret text
    assert not any("sk-" in str(v) for v in projection.values())


def test_committable_projection_redacts_a_secret_bearing_failing_check_id() -> None:
    # `failing_check_id` is the ONE committable free-form string — a parametrized test id can
    # embed a secret. It must be scrubbed on the way into the committable projection.
    secret = "sk-ant-SECRETTOKEN0123456789"
    step = make_step(success=False, failing_check_id=f"tests/test_auth.py::test_login[{secret}]")
    projection = schema.committable_projection(step)
    assert secret not in str(projection["failing_check_id"])


def test_committable_projection_is_exactly_the_whitelist() -> None:
    step = make_step(success=False, failing_check_id="chk")
    assert set(schema.committable_projection(step)) == set(schema.COMMITTABLE_FIELDS)


def test_derivable_fields_recompute_equal() -> None:
    step = make_step(
        success=False,
        is_infra_failure=False,
        failing_check_id="tests/x.py::t",
        test_passed=2,
        test_total=4,
    )
    assert schema.recompute_dedup_key(step) == step.dedup_key
    assert schema.recompute_blocking(step) == step.blocking


def test_dedup_key_recompute_contract_is_idempotent_and_authentic() -> None:
    # Pin the Layer-1 contract (S1): dedup_key == normalize_dedup_key(failing_check_id) and the
    # normalizer is idempotent, so the authenticity recompute cross-check cannot false-positive on
    # real committed data. A future non-idempotent normalizer breaks THIS test — not silently CI.
    fid = "tests/test_x.py::test_a"
    normalized = schema.normalize_dedup_key(fid)
    assert schema.normalize_dedup_key(normalized) == normalized  # idempotent
    step = make_step(success=False, failing_check_id=fid)
    assert step.dedup_key == schema.normalize_dedup_key(step.failing_check_id)  # contract holds
    assert schema.recompute_dedup_key(step) == step.dedup_key  # no false mismatch
    assert schema.normalize_dedup_key(None) is None


def test_dump_jsonl_scrubs_a_secret_from_free_text(tmp_path) -> None:
    # SEC-MED: a future live collect must not be able to land an unredacted secret in a committed
    # dump — dump_jsonl scrubs free-text structurally, not on caller trust.
    secret = "sk-ant-SECRETTOKEN0123456789"
    traj = make_trajectory(
        [
            make_step(
                success=False,
                failing_check_id="chk",
                observation=f"server said {secret}",
                action=f"curl -H 'Authorization: Bearer {secret}'",
                args=f"--token {secret}",
                result=f"leaked {secret}",
            )
        ]
    )
    path = tmp_path / "t.jsonl"
    schema.dump_jsonl(traj, path)
    assert secret not in path.read_text(encoding="utf-8")
    # SEC-MED regression: the persisted (redacted) trajectory must pass its OWN Layer-1 check —
    # the header hash has to commit to the scrubbed bytes on disk, not the pre-scrub payload.
    reloaded = schema.load_jsonl(path)
    assert errors(verify_trajectory(reloaded)) == []
    assert reloaded.header.redacted is True


_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:1a2b3c4d5e6f70819293a4b5c6d7e8f90112233445566778899aabbccddeeff0\n"
    "size 12345\n"
)


def test_load_jsonl_on_an_lfs_pointer_names_the_remedy(tmp_path) -> None:
    # A clone without `git lfs pull` gets pointer stubs, not trajectories. The old failure was an
    # opaque JSON parse error naming nothing; the fix must name the command that fixes it.
    path = tmp_path / "astropy__astropy-12907__model__high.jsonl"
    path.write_text(_POINTER, encoding="utf-8")
    with pytest.raises(schema.LfsPointerError) as excinfo:
        schema.load_jsonl(path)
    assert "git lfs pull" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_preflight_lfs_raises_on_a_pointer_and_passes_on_real_data(tmp_path) -> None:
    # The CLI preflight must fail the whole run, never skip the stub — a smaller corpus would
    # report different numbers, which is worse than a crash.
    real = tmp_path / "real.jsonl"
    schema.dump_jsonl(make_trajectory([make_step(success=True)]), real)
    schema.preflight_lfs([real])  # clean corpus: no raise
    pointer = tmp_path / "pointer.jsonl"
    pointer.write_text(_POINTER, encoding="utf-8")
    with pytest.raises(schema.LfsPointerError, match="git lfs pull"):
        schema.preflight_lfs([real, pointer])
    assert schema.is_lfs_pointer(pointer) is True
    assert schema.is_lfs_pointer(real) is False


def test_content_hash_changes_when_a_label_is_mutated(tmp_path) -> None:
    traj = make_trajectory([make_step(success=False, failing_check_id="chk")])
    original = traj.header.content_sha256
    tampered = make_trajectory([make_step(success=True, failing_check_id="chk")])
    assert tampered.header.content_sha256 != original
