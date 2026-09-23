"""Obsidian Headless Sync: keeps the configured vault synced with `ob sync --continuous`."""

import os
import json
import shutil
import signal
import asyncio
import logging
import threading
import contextlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

from reachy_mini_conversation_app.config import (
    OBSIDIAN_SYNC_MODES,
    OBSIDIAN_SYNC_CONFLICT_STRATEGIES,
    config,
)


logger = logging.getLogger(__name__)

RESTART_DELAYS_SECONDS = (5.0, 30.0, 120.0, 300.0)
STOP_GRACE_SECONDS = 10.0
COMMAND_TIMEOUT_SECONDS = 120.0
_NOT_SIGNED_IN_MARKER = "No account logged in"


class ObsidianSyncError(Exception):
    """An `ob` command failed. The message contains no secret values."""


@dataclass(frozen=True)
class _ObResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def message(self) -> str:
        """Return the last non-empty output line, preferring stderr."""
        for text in (self.stderr, self.stdout):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                return lines[-1]
        return f"exit code {self.returncode}"


def configured_vault_path() -> Path | None:
    """Return the configured local vault path; the default is `<instance>/obsidian/<vault>`."""
    if config.OBSIDIAN_SYNC_PATH:
        return Path(config.OBSIDIAN_SYNC_PATH).expanduser()
    vault = config.OBSIDIAN_SYNC_VAULT
    # A vault name that is not a plain folder name must not choose a path outside the instance directory.
    if not vault or config.INSTANCE_PATH is None or vault in {".", ".."} or Path(vault).name != vault:
        return None
    return config.INSTANCE_PATH / "obsidian" / vault


def current_vault_path() -> Path | None:
    """Return the local vault directory when Obsidian Sync is enabled and the directory exists."""
    if not config.OBSIDIAN_SYNC_ENABLED:
        return None
    path = configured_vault_path()
    return path if path is not None and path.is_dir() else None


def _redact(text: str, secrets: tuple[str | None, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _require_executable() -> str:
    executable = shutil.which(config.OBSIDIAN_HEADLESS_BIN)
    if executable is None:
        raise ObsidianSyncError(f"Obsidian Headless ({config.OBSIDIAN_HEADLESS_BIN}) is not installed.")
    return executable


async def _run_ob(
    executable: str,
    args: list[str],
    *,
    stdin_text: str | None = None,
    secrets: tuple[str | None, ...] = (),
) -> _ObResult:
    """Run one `ob` command to completion and return its redacted output."""
    # Without stdin text, stdin is empty so that an unexpected `ob` prompt reads EOF instead of blocking.
    process = await asyncio.create_subprocess_exec(
        executable,
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin_text.encode() if stdin_text is not None else None),
            COMMAND_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ObsidianSyncError(f"ob {args[0]} timed out.") from None
    return _ObResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=_redact(stdout.decode(errors="replace"), secrets),
        stderr=_redact(stderr.decode(errors="replace"), secrets),
    )


class ObsidianSyncSupervisor:
    """Run and restart `ob sync --continuous` for the configured vault in a background thread."""

    def __init__(
        self,
        *,
        restart_delays_seconds: tuple[float, ...] = RESTART_DELAYS_SECONDS,
        stop_grace_seconds: float = STOP_GRACE_SECONDS,
    ) -> None:
        """Create a stopped supervisor."""
        self._restart_delays_seconds = restart_delays_seconds
        self._stop_grace_seconds = stop_grace_seconds
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._state = "stopped"
        self._signed_in: bool | None = None
        self._last_sync: str | None = None
        self._last_error: str | None = None

    def status(self) -> dict[str, object]:
        """Return the sync status and non-secret settings for the settings UI."""
        path = configured_vault_path()
        return {
            "enabled": config.OBSIDIAN_SYNC_ENABLED,
            "available": shutil.which(config.OBSIDIAN_HEADLESS_BIN) is not None,
            "signed_in": self._signed_in,
            "vault": config.OBSIDIAN_SYNC_VAULT,
            "path": str(path) if path is not None else None,
            "state": self._state,
            "last_sync": self._last_sync,
            "last_error": self._last_error,
            "has_e2ee_password": bool(config.OBSIDIAN_SYNC_E2EE_PASSWORD),
            "headless_bin": config.OBSIDIAN_HEADLESS_BIN,
            "custom_path": config.OBSIDIAN_SYNC_PATH,
            "device_name": config.OBSIDIAN_SYNC_DEVICE_NAME,
            "mode": config.OBSIDIAN_SYNC_MODE,
            "modes": list(OBSIDIAN_SYNC_MODES),
            "conflict_strategy": config.OBSIDIAN_SYNC_CONFLICT_STRATEGY,
            "conflict_strategies": list(OBSIDIAN_SYNC_CONFLICT_STRATEGIES),
        }

    def start(self) -> None:
        """Start syncing in a background thread when Obsidian Sync is enabled."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._last_error = None
            if not config.OBSIDIAN_SYNC_ENABLED:
                self._state = "stopped"
                return
            self._state = "starting"
            loop = asyncio.new_event_loop()
            task = loop.create_task(self._supervise(), name="obsidian-sync")
            thread = threading.Thread(target=self._run_loop, args=(loop, task), name="obsidian-sync", daemon=True)
            self._loop, self._task, self._thread = loop, task, thread
            thread.start()

    def stop(self) -> None:
        """Stop `ob sync` gracefully and wait for the background thread to finish."""
        with self._lock:
            loop, task, thread = self._loop, self._task, self._thread
            self._loop = self._task = self._thread = None
        if loop is None or task is None or thread is None:
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            logger.debug("Obsidian Sync loop already finished")
        thread.join()

    def restart(self) -> None:
        """Stop and start again so that changed settings take effect."""
        self.stop()
        self.start()

    async def login(self, email: str, password: str, mfa_code: str | None) -> str:
        """Sign in with `ob login`; the password is passed on stdin and is not kept."""
        # When stdin is not a TTY, `ob` reads a prompt answer from stdin until EOF. That keeps the password out of
        # argv, but only one prompt can read stdin, so the MFA code (a one-time value) goes in argv.
        args = ["login", "--email", email]
        if mfa_code:
            args += ["--mfa", mfa_code]
        result = await _run_ob(_require_executable(), args, stdin_text=password, secrets=(password, mfa_code))
        if result.returncode != 0:
            self._signed_in = False
            raise ObsidianSyncError(f"Sign-in failed: {result.message}")
        if "Logged in as" not in result.stdout:
            self._signed_in = False
            raise ObsidianSyncError("Sign-in did not finish. Enter the MFA code from your authenticator.")
        self._signed_in = True
        return result.message

    async def logout(self) -> None:
        """Sign out with `ob logout`."""
        result = await _run_ob(_require_executable(), ["logout"])
        if result.returncode != 0:
            raise ObsidianSyncError(f"Sign-out failed: {result.message}")
        self._signed_in = False

    async def list_vaults(self) -> list[dict[str, str]]:
        """Return the remote vaults (`id`, `name`, `region`) of the signed-in account."""
        result = await _run_ob(_require_executable(), ["sync-list-remote", "--json"])
        if result.returncode != 0:
            if _NOT_SIGNED_IN_MARKER in result.stderr + result.stdout:
                self._signed_in = False
                raise ObsidianSyncError("Not signed in to Obsidian.")
            raise ObsidianSyncError(f"Listing vaults failed: {result.message}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ObsidianSyncError(f"Unexpected ob sync-list-remote output: {error}") from None
        self._signed_in = True
        entries = [*payload.get("vaults", []), *payload.get("shared", [])] if isinstance(payload, dict) else []
        return [
            {"id": str(entry["id"]), "name": str(entry["name"]), "region": str(entry.get("region") or "")}
            for entry in entries
            if isinstance(entry, dict) and "id" in entry and "name" in entry
        ]

    def _run_loop(self, loop: asyncio.AbstractEventLoop, task: asyncio.Task[None]) -> None:
        asyncio.set_event_loop(loop)
        try:
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)
        finally:
            loop.close()

    async def _supervise(self) -> None:
        executable = shutil.which(config.OBSIDIAN_HEADLESS_BIN)
        if executable is None:
            self._state = "stopped"
            self._last_error = f"Obsidian Headless ({config.OBSIDIAN_HEADLESS_BIN}) is not installed."
            logger.warning("%s Obsidian Sync is off; the conversation is not affected.", self._last_error)
            return
        vault = config.OBSIDIAN_SYNC_VAULT
        path = configured_vault_path()
        if not vault or path is None:
            self._state = "error"
            self._last_error = "Choose a remote vault and a local path."
            logger.warning("Obsidian Sync is enabled but not configured: %s", self._last_error)
            return

        password = config.OBSIDIAN_SYNC_E2EE_PASSWORD
        failures = 0
        try:
            while True:
                self._state = "starting"
                reached_sync = False
                try:
                    await self._prepare_vault(executable, vault, path, password)
                    returncode, last_line, reached_sync = await self._run_sync(executable, path, password)
                    self._last_error = f"ob sync exited with code {returncode}: {last_line or 'no output'}"
                except ObsidianSyncError as error:
                    self._last_error = str(error)
                failures = 1 if reached_sync else failures + 1
                delay = self._restart_delays_seconds[min(failures, len(self._restart_delays_seconds)) - 1]
                self._state = "error"
                logger.warning("Obsidian Sync stopped: %s. Restarting in %.0f s.", self._last_error, delay)
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self._state = "stopped"
            raise

    async def _prepare_vault(self, executable: str, vault: str, path: Path, password: str | None) -> None:
        """Link the local path to the remote vault if needed, then apply mode and conflict strategy."""
        secrets = (password,)
        status = await _run_ob(executable, ["sync-status", "--path", str(path), "--json"], secrets=secrets)
        if status.returncode == 0:
            try:
                linked = json.loads(status.stdout)
            except json.JSONDecodeError as error:
                raise ObsidianSyncError(f"Unexpected ob sync-status output: {error}") from None
            linked_to = {linked.get("vaultId"), linked.get("vaultName")} if isinstance(linked, dict) else set()
            if vault not in linked_to:
                raise ObsidianSyncError(f"{path} is linked to another vault. Choose another local path.")
        else:
            logger.info("Linking %s to Obsidian vault %s", path, vault)
            # `ob sync-setup` prompts for the E2EE password on stdin (see login); it asks only for E2EE vaults.
            setup = await _run_ob(
                executable,
                [
                    "sync-setup",
                    "--vault",
                    vault,
                    "--path",
                    str(path),
                    "--device-name",
                    config.OBSIDIAN_SYNC_DEVICE_NAME,
                ],
                stdin_text=password or "",
                secrets=secrets,
            )
            if setup.returncode != 0:
                if _NOT_SIGNED_IN_MARKER in setup.stderr + setup.stdout:
                    self._signed_in = False
                raise ObsidianSyncError(f"ob sync-setup failed: {setup.message}")
            self._signed_in = True

        sync_config = await _run_ob(
            executable,
            [
                "sync-config",
                "--path",
                str(path),
                "--mode",
                config.OBSIDIAN_SYNC_MODE,
                "--conflict-strategy",
                config.OBSIDIAN_SYNC_CONFLICT_STRATEGY,
                "--device-name",
                config.OBSIDIAN_SYNC_DEVICE_NAME,
            ],
            secrets=secrets,
        )
        if sync_config.returncode != 0:
            raise ObsidianSyncError(f"ob sync-config failed: {sync_config.message}")

    async def _run_sync(self, executable: str, path: Path, password: str | None) -> tuple[int, str, bool]:
        """Run `ob sync --continuous` until it exits; return its exit code, last line, and whether it synced."""
        process = await asyncio.create_subprocess_exec(
            executable,
            "sync",
            "--continuous",
            "--path",
            str(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        last_line = ""
        reached_sync = False
        try:
            assert process.stdout is not None
            async for raw_line in process.stdout:
                line = _redact(raw_line.decode(errors="replace").rstrip(), (password,))
                if not line:
                    continue
                last_line = line
                logger.info("ob sync: %s", line)
                if line.startswith("Starting sync"):
                    self._signed_in = True
                elif "Fully synced" in line:
                    reached_sync = True
                    self._state = "syncing"
                    self._last_error = None
                    self._last_sync = datetime.now(timezone.utc).isoformat(timespec="seconds")
                elif _NOT_SIGNED_IN_MARKER in line:
                    self._signed_in = False
            returncode = await process.wait()
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        return returncode, last_line, reached_sync

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Stop `ob sync` with SIGINT, then SIGTERM, then kill, waiting the grace period between steps."""
        try:
            # Windows cannot send SIGINT to a child process; terminate() is the only graceful option there.
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            for escalate in (process.terminate, process.kill):
                try:
                    await asyncio.wait_for(process.wait(), self._stop_grace_seconds)
                    return
                except asyncio.TimeoutError:
                    logger.warning("ob sync did not stop within %.0f s; escalating", self._stop_grace_seconds)
                    escalate()
            await process.wait()
        except ProcessLookupError:
            return


supervisor = ObsidianSyncSupervisor()
