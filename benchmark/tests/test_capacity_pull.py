"""Offline tests for GHCR-first pulling and 429 backoff."""

from __future__ import annotations

from benchmark.runner.capacity.pull import pull_image
from benchmark.runner.capacity.sizes import ghcr_ref, instance_id_from_ref

REF = "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
MIRROR = ghcr_ref(instance_id_from_ref(REF))


class _Runner:
    """Records every argv and replays a scripted (rc, stdout) per command."""

    def __init__(self, responses: dict[str, tuple[int, str]] | None = None, default=(0, "")):
        self.calls: list[list[str]] = []
        self._responses = responses or {}
        self._default = default

    def __call__(self, argv: list[str]) -> tuple[int, str]:
        self.calls.append(argv)
        key = argv[-1]
        return self._responses.get(key, self._default)


def test_skip_when_image_is_local() -> None:
    runner = _Runner()
    assert pull_image(REF, local_fn=lambda ref: 123, run=runner) is True
    assert runner.calls == []


def test_ghcr_first_then_tag_back_to_runner_ref() -> None:
    runner = _Runner()
    assert pull_image(REF, local_fn=lambda ref: None, run=runner) is True
    assert runner.calls[0] == ["docker", "pull", MIRROR]
    assert runner.calls[1] == ["docker", "tag", MIRROR, REF]


def test_falls_back_to_docker_hub_when_ghcr_fails() -> None:
    runner = _Runner(responses={MIRROR: (1, "manifest unknown"), REF: (0, "")})
    assert pull_image(REF, local_fn=lambda ref: None, run=runner) is True
    assert runner.calls == [["docker", "pull", MIRROR], ["docker", "pull", REF]]


def test_retries_on_429_then_succeeds() -> None:
    runner = _Runner(default=("1", "toomanyrequests"))

    def once_then_ok(argv: list[str]) -> tuple[int, str]:
        runner.calls.append(argv)
        return (0, "") if len(runner.calls) > 1 else (1, "429 Too Many Requests")

    slept: list[float] = []
    assert pull_image(REF, local_fn=lambda ref: None, run=once_then_ok, sleep=slept.append) is True
    assert slept == [300.0]


def test_exhausted_429_retries_returns_false() -> None:
    runner = _Runner(default=(1, "429"))
    slept: list[float] = []
    assert (
        pull_image(
            REF,
            local_fn=lambda ref: None,
            run=runner,
            sleep=slept.append,
            max_429_retries=1,
        )
        is False
    )
    assert len(slept) == 2
