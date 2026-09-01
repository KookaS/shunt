"""Keep the provider credential out of everything the live scaffold serialises."""

# mini-swe-agent persists its model config verbatim into the trajectory dump: ``DefaultAgent.save``
# merges ``LitellmModel.serialize()``, which is ``self.config.model_dump(mode="json")``, into the
# JSON written at ``agent.output_path``. Every key of ``model_kwargs`` is therefore written to disk
# UNREDACTED — so an ``api_key`` passed there lands in plaintext in each message-list dump a live
# run produces, once per cell, for the life of the file.
#
# The fix is positional, not cosmetic: the config carries the NAME of the environment variable and
# the value is read on the way into ``litellm.completion``, where nothing serialises it. Redacting
# the dump afterwards would be strictly weaker — it leaves a window between write and scrub, and it
# fails open the moment a new dump path is added. Same reasoning as ``shunt.log_config``'s ceiling
# on the HTTP libraries: a credential in the request path must not be reachable by any code that
# writes to disk.

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any, Final

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

MODEL_CLASS_PATH: Final[str] = "benchmark.runner.scaffold_model.EnvKeyLitellmModel"

# The request kwarg the credential used to travel in. Stripped from the persisted config and
# re-supplied per call; every OTHER key is left alone and then checked by the wall below.
CREDENTIAL_KWARG: Final[str] = "api_key"

# Self-identifying provider-key shapes, matching the scan in
# ``tests/test_step_snapshots.py`` (the forensic half of this defence). Assembled from prefix
# alternatives so no literal key-shaped string appears here.
_SECRET_SHAPE: Final[re.Pattern[str]] = re.compile(
    r"sk-ant-api03-[A-Za-z0-9_-]{20,}"
    r"|\bsk-[A-Za-z0-9]{20,}"
    r"|\brq-[A-Za-z0-9]{20,}"
    r"|\bxai-[A-Za-z0-9]{20,}"
    r"|\bAIza[0-9A-Za-z_-]{20,}"
    r"|\bAKIA[0-9A-Z]{16}"
    r"|\bgh[pousr]_[0-9A-Za-z]{20,}"
)

# Env vars whose VALUE is a secret. Checked by exact value, so a provider key with no
# self-identifying prefix (which no regex can recognise) is caught too.
_SECRET_ENV_NAME: Final[re.Pattern[str]] = re.compile(
    r"(API_KEY|APIKEY|_TOKEN|TOKEN_|SECRET|PASSWORD|CREDENTIAL)"
)
# Below this, an env value is too short to be a provider credential and matching it by
# substring would fire on ordinary config ("1", "true", "local").
_MIN_SECRET_LEN: Final[int] = 16


class CredentialInScaffoldConfigError(RuntimeError):
    """A model config that would write a provider credential into the trajectory dump."""


class MissingScaffoldCredentialError(RuntimeError):
    """The env var the scaffold config names holds no key at request time."""


class EnvKeyLitellmModelConfig(LitellmModelConfig):
    """``LitellmModelConfig`` plus the NAME of the env var holding the credential."""

    api_key_env_var: str | None = None


class EnvKeyLitellmModel(LitellmModel):
    """A ``LitellmModel`` that reads its credential from the environment per request."""

    # Serialisation is inherited unchanged and is safe by construction: the config holds only the
    # variable's name, so there is nothing to redact.
    #
    # It is also the benchmark's only seam around a provider call, so it is where per-call latency
    # is measured (see ``call_latencies_s``).

    def __init__(self, *, config_class: Any = EnvKeyLitellmModelConfig, **kwargs: Any) -> None:
        super().__init__(config_class=config_class, **kwargs)
        # CLIENT-side wall-clock seconds of each SUCCESSFUL provider call, in call order.
        #
        # This is the innermost seam that exists: the scaffold's `query` wraps `_query` in a
        # retry loop, so timing `_query` measures one round trip and excludes the retry
        # backoff sleeps that would otherwise be billed to the model as latency.
        #
        # Only a call that RETURNED is recorded. A call that raised (rate limit, timeout,
        # connection reset) has a duration, but it is the duration of a failure, not of a
        # generation; appending it would quietly shift the mean of a column named for
        # per-call latency. A failure is counted by `retry_count`, a different column.
        #
        # NOT a time-to-first-token: `LitellmModel._query` calls `litellm.completion`
        # WITHOUT `stream=True`, so the first and last token of a response arrive in one
        # event and TTFT is not observable at this seam at all. `runner.infer` therefore
        # leaves `ttft_s` blank rather than writing time-to-full-response under its name.
        self.call_latencies_s: list[float] = []

    def _timed(self, started: float) -> None:
        """Record one completed provider call's client-side duration."""
        self.call_latencies_s.append(time.perf_counter() - started)

    def _credential_kwargs(self) -> dict[str, str]:
        name = getattr(self.config, "api_key_env_var", None)
        if not name:
            # No name declared: litellm resolves the provider's own canonical env var itself
            # (the `deepseek/`-style routes), so there is nothing to inject.
            return {}
        key = os.environ.get(name)
        if not key:
            raise MissingScaffoldCredentialError(
                f"scaffold model {self.config.model_name!r} needs ${name}, which is unset"
            )
        return {CREDENTIAL_KWARG: key}

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        # Caller kwargs last so an explicit per-call override still wins.
        started = time.perf_counter()
        response = super()._query(messages, **{**self._credential_kwargs(), **kwargs})
        # Deliberately after the call returns, and deliberately not in a `finally`: a raised
        # call must record nothing (see `call_latencies_s`).
        self._timed(started)
        return response


def assert_credential_free(payload: Mapping[str, Any], *, what: str = "model config") -> None:
    """Raise if *payload* would serialise a provider credential. The wall.

    Two independent checks, because either alone fails open: an exact-value scan catches a key
    this process holds whatever its shape, and a shape scan catches a key it does not hold.
    """
    blob = json.dumps(payload, default=str)
    for name, value in os.environ.items():
        if len(value) >= _MIN_SECRET_LEN and _SECRET_ENV_NAME.search(name) and value in blob:
            raise CredentialInScaffoldConfigError(
                f"{what} carries the value of ${name}; it would be written to the trajectory dump"
            )
    if _SECRET_SHAPE.search(blob):
        raise CredentialInScaffoldConfigError(
            f"{what} carries a credential-shaped value; it would be written to the trajectory dump"
        )


def credential_free_model_block(
    model_string: str,
    model_kwargs: Mapping[str, Any],
    *,
    api_key_env_var: str | None,
) -> dict[str, Any]:
    """The scaffold's ``model`` config block, with the credential replaced by its env-var name."""
    if CREDENTIAL_KWARG in model_kwargs and not api_key_env_var:
        # Stripping the key without naming where to read it back would silently produce an
        # unauthenticated run — a quiet failure at the first paid request. Refuse now instead.
        raise CredentialInScaffoldConfigError(
            f"{model_string!r} supplies a credential but names no env var to re-supply it at "
            "request time; the scaffold would authenticate as nobody"
        )
    block: dict[str, Any] = {
        "model_name": model_string,
        "model_kwargs": {k: v for k, v in model_kwargs.items() if k != CREDENTIAL_KWARG},
        "cost_tracking": "ignore_errors",
        "model_class": MODEL_CLASS_PATH,
        "api_key_env_var": api_key_env_var,
    }
    assert_credential_free(block)
    return block
