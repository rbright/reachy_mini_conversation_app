"""Tests for the `/rpc` origin check and the settings PIN."""

import os
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock
from collections.abc import Iterator

import pytest
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from reachy_mini_conversation_app.config import SETTINGS_PIN_HASH_ENV, config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.settings_auth import (
    MAX_PIN_FAILURES,
    PRIVILEGED_METHODS,
    PIN_LOCKOUT_SECONDS,
    SettingsPinGuard,
    SettingsRpcServer,
    hash_settings_pin,
)


PIN = "correct-pin"


@pytest.fixture(autouse=True)
def restore_environment_and_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start without a PIN and restore `os.environ` and `config`; the PIN setter writes both."""
    monkeypatch.delenv(SETTINGS_PIN_HASH_ENV, raising=False)
    environ = dict(os.environ)
    settings = dict(vars(config))
    yield
    for name in os.environ.keys() - environ.keys():
        del os.environ[name]
    os.environ.update(environ)
    vars(config).clear()
    vars(config).update(settings)


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        """Start at an arbitrary time."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


def _call(client: TestClient, method: str, params: dict[str, Any] | None = None, **headers: str) -> dict[str, Any]:
    """Send one request over `/rpc` with `headers` and return the response."""
    with client.websocket_connect("/rpc", headers=headers) as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = ws.receive_json()
        return response


def _reason(response: dict[str, Any]) -> str:
    """Return the error reason of a response."""
    return str(response["error"]["data"]["reason"])


@pytest.fixture
def clock() -> Clock:
    """Return a settable clock for the PIN lockout."""
    return Clock()


@pytest.fixture
def seen() -> list[dict[str, Any]]:
    """Params that reached the privileged handler."""
    return []


@pytest.fixture
def client(clock: Clock, seen: list[dict[str, Any]]) -> TestClient:
    """Return a client for an app with one privileged method (`backend.config`) and one open method."""

    def persist(updates: dict[str, str]) -> None:
        os.environ.update(updates)

    app = FastAPI()
    rpc = SettingsRpcServer(SettingsPinGuard(persist, clock=clock))

    @rpc.method("backend.config")
    def _privileged(params: dict[str, Any]) -> dict[str, object]:
        seen.append(params)
        return {"ok": True}

    @rpc.method("conversation.status")
    def _open(_params: dict[str, Any]) -> dict[str, object]:
        return {"ok": True}

    rpc.mount(app)
    return TestClient(app)


@pytest.mark.parametrize("origin", ["http://evil.example", "http://testserver:8000", "https://testserver", "null"])
def test_upgrade_from_another_origin_is_refused(client: TestClient, origin: str) -> None:
    """A browser page on another origin cannot open `/rpc`."""
    with pytest.raises(WebSocketDisconnect) as refused:
        _call(client, "conversation.status", origin=origin)

    assert refused.value.code == 1008


@pytest.mark.parametrize("headers", [{}, {"origin": "http://testserver"}, {"origin": "http://TestServer"}])
def test_same_origin_and_non_browser_clients_connect(client: TestClient, headers: dict[str, str]) -> None:
    """A same-origin page, in any letter case, and a client without `Origin` (the daemon relay) connect."""
    assert _call(client, "conversation.status", **headers)["result"] == {"ok": True}


def test_privileged_method_is_refused_until_a_pin_is_set(client: TestClient, seen: list[dict[str, Any]]) -> None:
    """With no PIN set, a privileged method is refused and its handler does not run."""
    assert _reason(_call(client, "backend.config", {"settings_pin": PIN})) == "settings_pin_not_set"
    assert seen == []


@pytest.mark.parametrize("pin", [None, "", 123456])
def test_privileged_method_without_a_pin_is_refused(
    client: TestClient, seen: list[dict[str, Any]], pin: object
) -> None:
    """A privileged call without a PIN string is refused."""
    os.environ[SETTINGS_PIN_HASH_ENV] = hash_settings_pin(PIN)
    params = {} if pin is None else {"settings_pin": pin}

    assert _reason(_call(client, "backend.config", params)) == "settings_pin_required"
    assert seen == []


def test_wrong_pin_and_malformed_hash_are_refused(client: TestClient, seen: list[dict[str, Any]]) -> None:
    """A wrong PIN is refused; a malformed stored hash accepts no PIN."""
    os.environ[SETTINGS_PIN_HASH_ENV] = hash_settings_pin(PIN)
    assert _reason(_call(client, "backend.config", {"settings_pin": "wrong-pin"})) == "settings_pin_invalid"

    os.environ[SETTINGS_PIN_HASH_ENV] = "not-a-hash"
    assert _reason(_call(client, "backend.config", {"settings_pin": PIN})) == "settings_pin_invalid"
    assert seen == []


def test_correct_pin_runs_the_handler_without_the_pin(client: TestClient, seen: list[dict[str, Any]]) -> None:
    """The correct PIN runs the handler, and the handler does not get the PIN."""
    os.environ[SETTINGS_PIN_HASH_ENV] = hash_settings_pin(PIN)

    response = _call(client, "backend.config", {"settings_pin": PIN, "backend": "openai"})

    assert response["result"] == {"ok": True}
    assert seen == [{"backend": "openai"}]


def test_repeated_wrong_pins_lock_the_pin_for_a_while(client: TestClient, clock: Clock) -> None:
    """After too many wrong PINs even the correct PIN is refused until the lockout ends."""
    os.environ[SETTINGS_PIN_HASH_ENV] = hash_settings_pin(PIN)
    for _ in range(MAX_PIN_FAILURES):
        assert _reason(_call(client, "backend.config", {"settings_pin": "wrong-pin"})) == "settings_pin_invalid"

    assert _reason(_call(client, "backend.config", {"settings_pin": PIN})) == "settings_pin_locked"

    clock.now += PIN_LOCKOUT_SECONDS
    assert _call(client, "backend.config", {"settings_pin": PIN})["result"] == {"ok": True}


def test_first_pin_is_set_once(client: TestClient, seen: list[dict[str, Any]]) -> None:
    """The first PIN can be set; after that, nobody can replace it over RPC."""
    assert _reason(_call(client, "settings.set_pin", {"pin": "12345"})) == "settings_pin_too_short"
    assert _call(client, "settings.set_pin", {"pin": PIN})["result"] == {"ok": True}

    assert _reason(_call(client, "settings.set_pin", {"pin": "attacker-pin"})) == "settings_pin_already_set"
    assert _reason(_call(client, "backend.config", {"settings_pin": "attacker-pin"})) == "settings_pin_invalid"
    assert _call(client, "backend.config", {"settings_pin": PIN})["result"] == {"ok": True}
    assert len(seen) == 1


def _settings_app(instance_path: Path | None) -> FastAPI:
    """Return the conversation app settings server."""
    app = FastAPI()
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=SimpleNamespace(audio=None, backend=None)),
        settings_app=app,
        instance_path=str(instance_path) if instance_path is not None else None,
    )
    stream._init_settings_ui_if_needed()
    return app


def test_settings_app_protects_every_privileged_method(tmp_path: Path) -> None:
    """The settings server registers every privileged method behind the PIN; status methods stay open."""
    client = TestClient(_settings_app(tmp_path))
    os.environ[SETTINGS_PIN_HASH_ENV] = hash_settings_pin(PIN)

    refused = {method: _reason(_call(client, method)) for method in sorted(PRIVILEGED_METHODS)}

    assert refused == dict.fromkeys(sorted(PRIVILEGED_METHODS), "settings_pin_required")
    assert "result" in _call(client, "obsidian.status")
    assert "result" in _call(client, "conversation.status")


def test_settings_app_stores_only_a_pin_hash_and_accepts_the_pin(tmp_path: Path) -> None:
    """Setting the PIN stores a hash in `.env`, and the UI flow then reads vault access with the PIN."""
    client = TestClient(_settings_app(tmp_path))

    assert _call(client, "settings.set_pin", {"pin": PIN})["result"] == {"ok": True}

    stored = dotenv_values(tmp_path / ".env")[SETTINGS_PIN_HASH_ENV]
    assert stored is not None and stored.startswith("scrypt:") and PIN not in stored
    response = _call(client, "profile_vault_access.get", {"profile": "default", "settings_pin": PIN})
    assert "result" in response


def test_settings_app_refuses_a_pin_it_cannot_save() -> None:
    """Without an instance `.env` to keep the hash, setting a PIN fails and leaves no PIN in memory."""
    client = TestClient(_settings_app(None))

    assert _reason(_call(client, "settings.set_pin", {"pin": PIN})) == "settings_pin_not_saved"
    assert SETTINGS_PIN_HASH_ENV not in os.environ
    assert _reason(_call(client, "backend.config", {"settings_pin": PIN})) == "settings_pin_not_set"
