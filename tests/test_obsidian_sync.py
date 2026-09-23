"""Tests for the Obsidian Headless Sync supervisor, driven by a fake `ob` executable."""

import os
import sys
import json
import time
import logging
from typing import Any
from pathlib import Path
from collections.abc import Callable

import pytest

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.obsidian_sync import (
    ObsidianSyncError,
    ObsidianSyncSupervisor,
    current_vault_path,
    configured_vault_path,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="the fake ob executable is a POSIX script")

E2EE_PASSWORD = "e2ee #secret $pw"
ACCOUNT_PASSWORD = "account-secret"

# Behavior comes from files next to the script so each test controls one `ob`. Every call is logged as JSON.
FAKE_OB = (
    r'''
import os, sys, json, time, signal
from pathlib import Path

home = Path(__file__).parent
args = sys.argv[1:]
command = args[0]
options = dict(zip(args[1::2], args[2::2]))
stdin_text = sys.stdin.read() if command in ("login", "sync-setup") else None

def record(event, **extra):
    with open(home / "calls.jsonl", "a") as log:
        log.write(json.dumps({"event": event, "args": args, "stdin": stdin_text, "time": time.monotonic(), **extra}) + "\n")

record("call")
if command == "sync-status":
    linked = Path(options["--path"]) / ".linked"
    if not linked.exists():
        print("No sync configuration found", file=sys.stderr)
        sys.exit(3)
    print(json.dumps({"vaultId": "id-1", "vaultName": linked.read_text()}))
elif command == "sync-setup":
    if stdin_text != (home / "e2ee").read_text():
        print("Failed to validate password.", file=sys.stderr)
        sys.exit(2)
    Path(options["--path"]).mkdir(parents=True, exist_ok=True)
    (Path(options["--path"]) / ".linked").write_text(options["--vault"])
    print("Vault configured successfully!")
elif command == "sync-config":
    print("Configuration updated:")
elif command == "sync":
    behavior = (home / "sync_behavior").read_text()
    print("Starting sync:", flush=True)
    print("  password echo " + (home / "e2ee").read_text(), flush=True)
    if behavior == "fail":
        print("Sync failed: boom", file=sys.stderr, flush=True)
        sys.exit(1)
    def on_signal(signum, frame):
        record(signal.Signals(signum).name)
        if behavior != "ignore-sigint" or signum != signal.SIGINT:
            sys.exit(0)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print("Fully synced", flush=True)
    while True:
        time.sleep(0.05)
elif command == "login":
    if stdin_text != "'''
    + ACCOUNT_PASSWORD
    + r"""":
        print("Login failed: rejected " + stdin_text, file=sys.stderr)
        sys.exit(2)
    print("Logged in as Robot (" + options["--email"] + ")")
elif command == "sync-list-remote":
    if (home / "signed_out").exists():
        print('No account logged in. Run "ob login" first.', file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"vaults": [{"id": "id-1", "name": "Luna", "region": "eu"}],
                      "shared": [{"id": "id-2", "name": "Shared", "region": "us"}]}, indent=2))
elif command == "logout":
    print("Logged out.")
"""
)


class FakeOb:
    """A fake `ob` executable in a temporary folder, with its call log."""

    def __init__(self, home: Path) -> None:
        """Write the script and its default behavior into `home`."""
        self.home = home
        self.executable = home / "ob"
        self.executable.write_text(f"#!{sys.executable}\n{FAKE_OB}", encoding="utf-8")
        self.executable.chmod(0o755)
        (home / "e2ee").write_text(E2EE_PASSWORD)
        self.set_sync_behavior("run")

    def set_sync_behavior(self, behavior: str) -> None:
        """Choose how `ob sync` behaves: `run`, `fail`, or `ignore-sigint`."""
        (self.home / "sync_behavior").write_text(behavior)

    def calls(self) -> list[dict[str, Any]]:
        """Return every logged call and signal, in order."""
        log = self.home / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def commands(self) -> list[str]:
        """Return the `ob` subcommands called, in order."""
        return [str(call["args"][0]) for call in self.calls() if call["event"] == "call"]


@pytest.fixture
def fake_ob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOb:
    """Configure enabled Obsidian Sync that runs a fake `ob`."""
    home = tmp_path / "ob-home"
    home.mkdir()
    ob = FakeOb(home)
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "OBSIDIAN_HEADLESS_BIN", str(ob.executable))
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_VAULT", "Luna")
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_PATH", str(tmp_path / "vault"))
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_DEVICE_NAME", "reachy-test")
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_MODE", "pull-only")
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_CONFLICT_STRATEGY", "conflict")
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_E2EE_PASSWORD", E2EE_PASSWORD)
    return ob


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


def test_supervisor_links_vault_syncs_and_stops_with_sigint(
    fake_ob: FakeOb, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unlinked path is set up with the password on stdin, configured, synced, and stopped with SIGINT."""
    caplog.set_level(logging.INFO, logger="reachy_mini_conversation_app.obsidian_sync")
    supervisor = ObsidianSyncSupervisor()
    supervisor.start()
    try:
        _wait_until(lambda: supervisor.status()["state"] == "syncing")
        status = supervisor.status()
    finally:
        supervisor.stop()

    assert fake_ob.commands() == ["sync-status", "sync-setup", "sync-config", "sync"]
    setup, sync_config = (call for call in fake_ob.calls() if call["args"][0] in ("sync-setup", "sync-config"))
    assert setup["stdin"] == E2EE_PASSWORD
    assert setup["args"] == [
        "sync-setup",
        "--vault",
        "Luna",
        "--path",
        str(tmp_path / "vault"),
        "--device-name",
        "reachy-test",
    ]
    assert sync_config["args"][3:] == [
        "--mode",
        "pull-only",
        "--conflict-strategy",
        "conflict",
        "--device-name",
        "reachy-test",
    ]
    assert [call["event"] for call in fake_ob.calls()][-1] == "SIGINT"
    assert status["signed_in"] is True
    assert status["last_sync"] is not None
    assert supervisor.status()["state"] == "stopped"
    assert "password echo [redacted]" in caplog.text
    assert E2EE_PASSWORD not in caplog.text
    assert E2EE_PASSWORD not in json.dumps(status)


def test_supervisor_restarts_failed_sync_with_backoff(fake_ob: FakeOb) -> None:
    """A failing `ob sync` restarts after each configured delay and reports its last error."""
    fake_ob.set_sync_behavior("fail")
    delays = (0.2, 0.5, 1.0)
    supervisor = ObsidianSyncSupervisor(restart_delays_seconds=delays)
    supervisor.start()
    try:
        _wait_until(lambda: fake_ob.commands().count("sync") >= 4)
        status = supervisor.status()
    finally:
        supervisor.stop()

    sync_times = [
        float(call["time"]) for call in fake_ob.calls() if call["event"] == "call" and call["args"][0] == "sync"
    ]
    gaps = [later - earlier for earlier, later in zip(sync_times, sync_times[1:])]
    assert all(gap >= delay for gap, delay in zip(gaps, delays))
    assert fake_ob.commands().count("sync-setup") == 1
    assert status["state"] == "error"
    assert "Sync failed: boom" in str(status["last_error"])
    assert supervisor.status()["state"] == "stopped"


def test_supervisor_sends_sigterm_when_sigint_is_ignored(fake_ob: FakeOb) -> None:
    """Stop escalates to SIGTERM after the grace period."""
    fake_ob.set_sync_behavior("ignore-sigint")
    supervisor = ObsidianSyncSupervisor(stop_grace_seconds=0.3)
    supervisor.start()
    _wait_until(lambda: supervisor.status()["state"] == "syncing")
    started = time.monotonic()
    supervisor.stop()

    assert time.monotonic() - started >= 0.3
    assert [call["event"] for call in fake_ob.calls()][-2:] == ["SIGINT", "SIGTERM"]


def test_supervisor_refuses_a_path_linked_to_another_vault(fake_ob: FakeOb, tmp_path: Path) -> None:
    """A local path already linked to another vault is reported, not set up again."""
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / ".linked").write_text("Other")
    supervisor = ObsidianSyncSupervisor(restart_delays_seconds=(30.0,))
    supervisor.start()
    try:
        _wait_until(lambda: supervisor.status()["state"] == "error")
        status = supervisor.status()
    finally:
        supervisor.stop()

    assert "linked to another vault" in str(status["last_error"])
    assert fake_ob.commands() == ["sync-status"]


def test_supervisor_without_ob_reports_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing `ob` leaves sync stopped with a clear status."""
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "OBSIDIAN_HEADLESS_BIN", str(tmp_path / "missing-ob"))
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_VAULT", "Luna")
    supervisor = ObsidianSyncSupervisor()
    supervisor.start()
    _wait_until(lambda: supervisor.status()["last_error"] is not None)
    supervisor.stop()

    status = supervisor.status()
    assert status["available"] is False
    assert status["state"] == "stopped"
    assert "not installed" in str(status["last_error"])


@pytest.mark.asyncio
async def test_login_passes_password_on_stdin_only(fake_ob: FakeOb) -> None:
    """The account password goes to `ob login` on stdin and never in argv."""
    supervisor = ObsidianSyncSupervisor()

    message = await supervisor.login("robot@example.com", ACCOUNT_PASSWORD, "123456")

    login = fake_ob.calls()[0]
    assert login["args"] == ["login", "--email", "robot@example.com", "--mfa", "123456"]
    assert login["stdin"] == ACCOUNT_PASSWORD
    assert message == "Logged in as Robot (robot@example.com)"
    assert supervisor.status()["signed_in"] is True


@pytest.mark.asyncio
async def test_login_error_redacts_the_password(fake_ob: FakeOb) -> None:
    """A failed sign-in reports the `ob` error without the password."""
    supervisor = ObsidianSyncSupervisor()

    with pytest.raises(ObsidianSyncError) as raised:
        await supervisor.login("robot@example.com", "wrong-password", None)

    assert "rejected [redacted]" in str(raised.value)
    assert "wrong-password" not in str(raised.value)
    assert supervisor.status()["signed_in"] is False


@pytest.mark.asyncio
async def test_list_vaults_returns_owned_and_shared_vaults(fake_ob: FakeOb) -> None:
    """Remote vaults come from `ob sync-list-remote --json`."""
    supervisor = ObsidianSyncSupervisor()

    vaults = await supervisor.list_vaults()

    assert vaults == [
        {"id": "id-1", "name": "Luna", "region": "eu"},
        {"id": "id-2", "name": "Shared", "region": "us"},
    ]
    assert supervisor.status()["signed_in"] is True


@pytest.mark.asyncio
async def test_list_vaults_when_signed_out(fake_ob: FakeOb) -> None:
    """A signed-out account is reported as such."""
    (fake_ob.home / "signed_out").touch()
    supervisor = ObsidianSyncSupervisor()

    with pytest.raises(ObsidianSyncError, match="Not signed in"):
        await supervisor.list_vaults()
    assert supervisor.status()["signed_in"] is False


def test_vault_path_defaults_to_the_instance_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default local path is `<instance>/obsidian/<vault>`; vault tools see it only when enabled and present."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_PATH", None)
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_VAULT", "Luna")
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_ENABLED", True)

    assert configured_vault_path() == tmp_path / "obsidian" / "Luna"
    assert current_vault_path() is None
    (tmp_path / "obsidian" / "Luna").mkdir(parents=True)
    assert current_vault_path() == tmp_path / "obsidian" / "Luna"
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_ENABLED", False)
    assert current_vault_path() is None


@pytest.mark.parametrize("vault", ["..", "../escape", "/etc"])
def test_vault_names_that_are_not_folder_names_have_no_default_path(
    vault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vault name cannot move the default path outside the instance folder."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_PATH", None)
    monkeypatch.setattr(config, "OBSIDIAN_SYNC_VAULT", vault)

    assert configured_vault_path() is None
