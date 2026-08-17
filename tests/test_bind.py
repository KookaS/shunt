"""`shunt.bind.resolve_bind` — the ONE definition of the address the proxy binds."""

# It exists precisely so `shunt start` and `shunt doctor` cannot disagree about the port, and it
# shipped with no test of its own: the range guard, the junk-value message and the server's
# fail-before-uvicorn path were all unverified. A drift here is silent and user-visible — doctor
# confidently reports the wrong port as free.

from __future__ import annotations

import pytest

from shunt.bind import DEFAULT_HOST, DEFAULT_PORT, HOST_ENV, PORT_ENV, resolve_bind


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (HOST_ENV, PORT_ENV):
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_loopback_when_nothing_is_set() -> None:
    assert resolve_bind() == (DEFAULT_HOST, DEFAULT_PORT)


def test_env_overrides_both_halves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOST_ENV, "0.0.0.0")  # noqa: S104 (a value under test, not a bind)
    monkeypatch.setenv(PORT_ENV, "9001")
    assert resolve_bind() == ("0.0.0.0", 9001)


def test_port_is_returned_as_an_int_not_the_raw_string(monkeypatch: pytest.MonkeyPatch) -> None:
    # uvicorn takes an int; returning the string would fail deep inside the server instead.
    monkeypatch.setenv(PORT_ENV, "8081")
    assert resolve_bind()[1] == 8081


@pytest.mark.parametrize("raw", ["not-a-number", "80 80", "", "8080.5"])
def test_a_non_integer_port_names_the_variable_and_the_value(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(PORT_ENV, raw)
    with pytest.raises(ValueError, match=PORT_ENV) as exc:
        resolve_bind()
    assert repr(raw) in str(exc.value)


@pytest.mark.parametrize("raw", ["65536", "-1", "99999"])
def test_an_out_of_range_port_is_rejected_here_not_at_the_socket(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    # connect_ex raises OverflowError for these, which reached the user as a crash rather than
    # as a diagnosis — the whole reason the range check lives in this function.
    monkeypatch.setenv(PORT_ENV, raw)
    with pytest.raises(ValueError, match="between 0 and 65535"):
        resolve_bind()


@pytest.mark.parametrize("raw", ["0", "1", "65535"])
def test_the_range_boundaries_are_inclusive(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(PORT_ENV, raw)
    assert resolve_bind()[1] == int(raw)


def test_the_server_refuses_to_start_on_an_out_of_range_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # proxy/server.run() resolves the bind before anything else, so a bad SHUNT_PORT must fail
    # with the actionable message rather than after logging is configured or uvicorn is entered.
    import shunt.proxy.server as server

    def _never(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uvicorn must not be reached with an invalid bind")

    monkeypatch.setattr(server.uvicorn, "run", _never)
    monkeypatch.setenv(PORT_ENV, "99999")
    with pytest.raises(ValueError, match="between 0 and 65535"):
        server.run()
