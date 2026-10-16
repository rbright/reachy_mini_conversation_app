"""Tests for the instance directory resolution and its one-time migration."""

import os
import shutil
import asyncio
from pathlib import Path
from collections.abc import Iterator

import pytest
from dotenv import load_dotenv, dotenv_values

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.config import (
    OPENAI_BACKEND,
    INSTANCE_PATH_ENV,
    SLEEP_PHRASES_ENV,
    BACKEND_PROVIDER_ENV,
    SETTINGS_PIN_HASH_ENV,
    WAKE_WORD_ENABLED_ENV,
    OBSIDIAN_SYNC_VAULT_ENV,
    WAKE_WORD_THRESHOLD_ENV,
    config,
    set_instance_path,
    get_backend_choice,
    resolve_instance_path,
    resolve_wake_word_settings,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpacesManifest,
    normalize_space_alias,
    read_installed_tool_spaces,
    write_installed_tool_spaces,
)
from reachy_mini_conversation_app.obsidian_sync import configured_vault_path
from reachy_mini_conversation_app.settings_auth import SettingsPinGuard, hash_settings_pin
from reachy_mini_conversation_app.profile_vault_access import (
    ProfileVaultAccess,
    read_profile_vault_access,
    write_profile_vault_access,
)


@pytest.fixture(autouse=True)
def _config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default instance directory at a temporary config home."""
    config_home = tmp_path / "config"
    monkeypatch.setattr(config_mod, "user_config_dir", lambda name: str(config_home / name))
    monkeypatch.delenv(INSTANCE_PATH_ENV, raising=False)
    return config_home


@pytest.fixture
def restore_runtime_config() -> Iterator[None]:
    """Restore the runtime config that a simulated start rewrites."""
    settings = dict(vars(config))
    yield
    vars(config).clear()
    vars(config).update(settings)


def _legacy(tmp_path: Path, memory: bool = False) -> Path:
    """Build a package directory holding data the old layout wrote there."""
    legacy = tmp_path / "site-packages"
    (legacy / "user_personalities" / "guide").mkdir(parents=True)
    (legacy / "user_personalities" / "guide" / "profile.md").write_text("+++\n+++\nBe a guide.")
    if memory:
        (legacy / "memory.v1.json").write_text('{"version": 1, "facts": []}')
    return legacy


def test_defaults_beside_the_daemon_config(tmp_path: Path, _config_home: Path) -> None:
    """Without an override the data lives next to the daemon's own config."""
    resolved = resolve_instance_path(tmp_path / "site-packages")

    assert resolved == _config_home / "reachy_mini" / "conversation_app"
    assert resolved.is_dir()


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_INSTANCE_PATH overrides the default and is created."""
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(tmp_path / "mine"))

    assert resolve_instance_path(tmp_path / "site-packages") == tmp_path / "mine"
    assert (tmp_path / "mine").is_dir()


def test_migrates_legacy_data_once(tmp_path: Path) -> None:
    """Data written in the package directory moves over on the first run."""
    legacy = _legacy(tmp_path, memory=True)

    resolved = resolve_instance_path(legacy)

    assert (resolved / "memory.v1.json").read_text() == '{"version": 1, "facts": []}'
    assert (resolved / "user_personalities" / "guide" / "profile.md").exists()

    assert not list(resolved.glob(".*.migrating"))

    # A second run must not overwrite live data with the stale legacy copy.
    (resolved / "memory.v1.json").write_text('{"version": 1, "facts": ["kept"]}')
    resolve_instance_path(legacy)
    assert "kept" in (resolved / "memory.v1.json").read_text()


def test_an_interrupted_migration_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy that dies halfway must not pass for a finished migration."""
    legacy = _legacy(tmp_path)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    real_copytree = shutil.copytree

    def _dies_halfway(src: Path, dst: Path, **kwargs: object) -> None:
        Path(dst).mkdir(parents=True)
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", _dies_halfway)
    resolve_instance_path(legacy)
    assert not (target / "user_personalities").exists()

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    resolve_instance_path(legacy)
    assert (target / "user_personalities" / "guide" / "profile.md").exists()


def test_a_migration_that_fails_between_items_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory landing must not make the next start skip the personalities."""
    legacy = _legacy(tmp_path, memory=True)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    real_copytree = shutil.copytree

    def _fails(src: Path, dst: Path, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", _fails)
    resolve_instance_path(legacy)
    assert (target / "memory.v1.json").exists()
    assert not (target / ".migrated").exists()

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    resolve_instance_path(legacy)
    assert (target / "user_personalities" / "guide" / "profile.md").exists()
    assert (target / ".migrated").exists()


def test_data_deleted_after_migration_stays_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once migrated, the legacy copy never comes back."""
    legacy = _legacy(tmp_path)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    resolve_instance_path(legacy)
    shutil.rmtree(target / "user_personalities")
    resolve_instance_path(legacy)

    assert not (target / "user_personalities").exists()


def test_migration_keeps_vault_symlinks_as_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A link out of a legacy vault stays a link, so its target's bytes never land in the vault."""
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the vault")
    legacy = tmp_path / "site-packages"
    note = legacy / "obsidian" / "Emma" / "note.md"
    note.parent.mkdir(parents=True)
    note.symlink_to(secret)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    resolve_instance_path(legacy)

    migrated = target / "obsidian" / "Emma" / "note.md"
    assert migrated.is_symlink()
    assert migrated.resolve() == secret.resolve()


def test_unusable_path_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad override warns and keeps the previous directory, never crashes."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(blocker / "nope"))

    legacy = tmp_path / "site-packages"
    assert resolve_instance_path(legacy) == legacy


@pytest.mark.usefixtures("restore_runtime_config")
def test_fork_instance_data_survives_the_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fork's PIN, wake word, provider, Obsidian vault, vault access, and tool Spaces reach the new directory."""
    legacy = tmp_path / "site-packages"
    note = legacy / "obsidian" / "Emma" / "Daily" / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text("Walked the dog.")
    pin_hash = hash_settings_pin("123456")
    (legacy / ".env").write_text(
        f"{SETTINGS_PIN_HASH_ENV}={pin_hash}\n"
        f"{WAKE_WORD_ENABLED_ENV}=true\n"
        f"{WAKE_WORD_THRESHOLD_ENV}=0.65\n"
        f"{SLEEP_PHRASES_ENV}=good night emma\n"
        f"{BACKEND_PROVIDER_ENV}=openai\n"
        f"{OBSIDIAN_SYNC_VAULT_ENV}=Emma\n"
    )
    access = ProfileVaultAccess(agent="emma", read=("Daily",))
    write_profile_vault_access("emma", access, legacy)
    slug = "emma-owner/weather"
    space = InstalledToolSpace(
        slug=slug,
        alias=normalize_space_alias(slug),
        mcp_url="https://emma-owner-weather.hf.space/gradio_api/mcp/",
        private=True,
    )
    write_installed_tool_spaces(legacy, InstalledToolSpacesManifest(spaces=[space]))
    installed = read_installed_tool_spaces(legacy)

    resolved = resolve_instance_path(legacy)
    # The first start then loads the instance `.env` as `main.run` does; monkeypatch undoes the values.
    for name in dotenv_values(resolved / ".env"):
        monkeypatch.setenv(name, "")
    load_dotenv(resolved / ".env", override=True)
    refresh_runtime_config_from_env()
    set_instance_path(resolved)

    assert resolved != legacy
    asyncio.run(SettingsPinGuard(lambda _updates: None).verify("123456"))
    assert os.environ[SETTINGS_PIN_HASH_ENV] == pin_hash
    wake = resolve_wake_word_settings(("sleep",))
    assert (wake.enabled, wake.threshold, wake.sleep_phrases) == (True, 0.65, ("good night emma",))
    assert get_backend_choice() == OPENAI_BACKEND
    assert configured_vault_path() == resolved / "obsidian" / "Emma"
    assert (resolved / "obsidian" / "Emma" / "Daily" / "note.md").read_text() == "Walked the dog."
    assert read_profile_vault_access(resolved) == {"emma": access}
    assert read_installed_tool_spaces(resolved) == installed
    assert slug in [installed_space.slug for installed_space in installed.spaces]
