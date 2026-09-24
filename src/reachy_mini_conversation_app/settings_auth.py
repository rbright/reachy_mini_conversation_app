"""Access control for the settings JSON-RPC endpoint (`/rpc`).

Two checks protect the endpoint:

- The WebSocket upgrade is refused when the request has an `Origin` header that is not the app's own
  origin (same host and port). A browser always sends `Origin`, so another site cannot drive `/rpc`.
  Clients that are not browsers (the daemon relay) send no `Origin` and stay accepted.
- A privileged settings method (secrets, Obsidian account and vault access, backend and tool configuration)
  runs only when the call carries the settings PIN in `settings_pin`. The app keeps only a scrypt hash of
  the PIN, in the instance `.env`. When no PIN is set, privileged methods are refused and the first
  `settings.set_pin` call sets it. To reset a PIN, remove its `.env` line and restart the app.
"""

from __future__ import annotations
import os
import hmac
import time
import asyncio
import hashlib
import inspect
import logging
import secrets
from typing import Any
from urllib.parse import urlsplit
from collections.abc import Callable

from fastapi import FastAPI, WebSocket

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import SETTINGS_PIN_HASH_ENV


logger = logging.getLogger(__name__)

SETTINGS_PIN_PARAM = "settings_pin"
MIN_SETTINGS_PIN_LENGTH = 6
MAX_PIN_FAILURES = 5
PIN_LOCKOUT_SECONDS = 60.0

PRIVILEGED_METHODS = frozenset(
    {
        "backend.config",
        "obsidian.login",
        "obsidian.list_vaults",
        "obsidian.configure",
        "obsidian.logout",
        "profile_vault_access.get",
        "profile_vault_access.save",
        "profile_tools.save",
        "profile_tools.reset",
        "tool_spaces.add",
        "tool_spaces.remove",
    }
)

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


def hash_settings_pin(pin: str) -> str:
    """Return a salted scrypt hash of `pin` as `scrypt:<salt hex>:<digest hex>`."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt:{salt.hex()}:{digest.hex()}"


def _pin_matches(pin: str, stored: str) -> bool:
    """Return whether `pin` matches a stored hash. A malformed hash matches no PIN."""
    try:
        scheme, salt_hex, digest_hex = stored.split(":")
        salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except ValueError:
        return False
    if scheme != "scrypt" or not salt or not expected:
        return False
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=len(expected))
    return hmac.compare_digest(digest, expected)


def is_allowed_origin(origin: str | None, host: str | None) -> bool:
    """Return whether a WebSocket upgrade with this `Origin` and `Host` may connect."""
    if origin is None:
        return True
    if not host:
        return False
    return urlsplit(origin).netloc.lower() == host.lower()


class SettingsPinGuard:
    """Check the settings PIN, with a lockout after repeated wrong PINs."""

    def __init__(
        self,
        persist_env: Callable[[dict[str, str]], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep the PIN hash with `persist_env` (the instance `.env` writer)."""
        self._persist_env = persist_env
        self._clock = clock
        self._failures = 0
        self._locked_until = 0.0
        # One check at a time, so concurrent guesses cannot pass the failure count.
        self._lock = asyncio.Lock()

    @staticmethod
    def _stored_hash() -> str:
        return (os.environ.get(SETTINGS_PIN_HASH_ENV) or "").strip()

    async def verify(self, pin: object) -> None:
        """Raise a `JsonRpcError` unless `pin` is the settings PIN."""
        stored = self._stored_hash()
        if not stored:
            raise JsonRpcError("set a settings PIN first", reason="settings_pin_not_set")
        if not isinstance(pin, str) or not pin:
            raise JsonRpcError("settings PIN required", reason="settings_pin_required")
        async with self._lock:
            if self._clock() < self._locked_until:
                raise JsonRpcError("too many wrong settings PINs; try again later", reason="settings_pin_locked")
            if await asyncio.to_thread(_pin_matches, pin, stored):
                self._failures = 0
                return
            self._failures += 1
            if self._failures >= MAX_PIN_FAILURES:
                self._failures = 0
                self._locked_until = self._clock() + PIN_LOCKOUT_SECONDS
                logger.warning(
                    "Settings PIN locked for %.0f s after %d wrong PINs", PIN_LOCKOUT_SECONDS, MAX_PIN_FAILURES
                )
            raise JsonRpcError("wrong settings PIN", reason="settings_pin_invalid")

    async def set_pin(self, params: dict[str, Any]) -> dict[str, object]:
        """Set the first settings PIN. A PIN that is already set cannot be replaced over RPC."""
        pin = params.get("pin")
        if not isinstance(pin, str) or len(pin) < MIN_SETTINGS_PIN_LENGTH:
            raise JsonRpcError(
                f"settings PIN needs {MIN_SETTINGS_PIN_LENGTH} or more characters",
                reason="settings_pin_too_short",
                code=-32602,
            )
        async with self._lock:
            if self._stored_hash():
                raise JsonRpcError("a settings PIN is already set", reason="settings_pin_already_set")
            self._persist_env({SETTINGS_PIN_HASH_ENV: await asyncio.to_thread(hash_settings_pin, pin)})
        logger.info("Settings PIN set")
        return {"ok": True}

    def protect(self, handler: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        """Return `handler` behind a PIN check; the handler never sees the PIN."""

        async def _guarded(params: dict[str, Any]) -> Any:
            await self.verify(params.get(SETTINGS_PIN_PARAM))
            result = handler({name: value for name, value in params.items() if name != SETTINGS_PIN_PARAM})
            if inspect.isawaitable(result):
                result = await result
            return result

        return _guarded


class SettingsRpcServer(JsonRpcServer):  # type: ignore[misc]  # the SDK is not typed
    """The SDK JSON-RPC server with the origin check and the settings PIN on privileged methods."""

    def __init__(self, guard: SettingsPinGuard) -> None:
        """Register `settings.set_pin` and protect every privileged method registered later."""
        super().__init__()
        self._guard = guard
        self.register("settings.set_pin", guard.set_pin)

    def register(self, name: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a handler; a privileged method gets the PIN check."""
        super().register(name, self._guard.protect(handler) if name in PRIVILEGED_METHODS else handler)

    def mount(self, app: FastAPI, path: str = "/rpc") -> None:
        """Add the `/rpc` WebSocket route, refusing upgrades from another origin."""

        @app.websocket(path)
        async def _rpc_ws(websocket: WebSocket) -> None:
            origin = websocket.headers.get("origin")
            if not is_allowed_origin(origin, websocket.headers.get("host")):
                logger.warning("Refused /rpc connection from origin %r", origin)
                await websocket.close(code=1008)
                return
            await self._serve(websocket)
