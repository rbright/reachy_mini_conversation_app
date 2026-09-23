"""Persist instance-local Obsidian vault access for personality profiles."""

import os
import json
import logging
import threading
from pathlib import Path

from pydantic import Field, BaseModel, ConfigDict, ValidationError, field_validator

from reachy_mini_conversation_app.vault import AGENT_ID, normalize_folder
from reachy_mini_conversation_app.profile_store import canonical_profile_name
from reachy_mini_conversation_app.profile_toolsets import get_profile_toolsets_path


logger = logging.getLogger(__name__)

PROFILE_VAULT_ACCESS_FILENAME = "profile_vault_access.json"
PROFILE_VAULT_ACCESS_VERSION = 1
_STORE_LOCK = threading.RLock()


class _AccessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionNoteTarget(_AccessModel):
    """Where the session end hook writes one note: folder, schema type, and property templates."""

    folder: str = Field(min_length=1)
    type: str = Field(min_length=1)
    properties: dict[str, str] = Field(default_factory=dict)

    @field_validator("folder")
    @classmethod
    def _normalize_folder(cls, folder: str) -> str:
        return normalize_folder(folder)


class WeeklyMemoryTarget(SessionNoteTarget):
    """Weekly memory target; `date_weekday` picks the ISO weekday of the summarized week used for `{date}`."""

    date_weekday: int = Field(default=1, ge=1, le=7)


class ProfileVaultAccess(_AccessModel):
    """Vault access for one profile. The vault schema must also grant every folder."""

    agent: str
    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()
    session_context: tuple[str, ...] = ()
    session_log: SessionNoteTarget | None = None
    weekly_memory: WeeklyMemoryTarget | None = None

    @field_validator("agent")
    @classmethod
    def _valid_agent(cls, agent: str) -> str:
        if not AGENT_ID.fullmatch(agent):
            raise ValueError("agent id must be lowercase letters, digits, '.', '_', or '-'")
        return agent

    @field_validator("read", "write")
    @classmethod
    def _normalize_folders(cls, folders: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(folder for folder in map(normalize_folder, folders) if folder))

    @field_validator("session_context")
    @classmethod
    def _strip_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(path.strip().strip("/") for path in paths if path.strip()))


def get_profile_vault_access_path(instance_path: str | Path | None) -> Path:
    """Return the vault access store path, next to `profile_toolsets.json`."""
    return get_profile_toolsets_path(instance_path).with_name(PROFILE_VAULT_ACCESS_FILENAME)


def read_profile_vault_access(instance_path: str | Path | None) -> dict[str, ProfileVaultAccess]:
    """Read vault access for every configured profile."""
    with _STORE_LOCK:
        settings_path = get_profile_vault_access_path(instance_path)
        if not settings_path.exists():
            return {}
        try:
            payload: object = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read profile vault access from {settings_path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != PROFILE_VAULT_ACCESS_VERSION:
            raise RuntimeError(
                f"Unsupported profile vault access payload in {settings_path}: "
                f"expected an object with version {PROFILE_VAULT_ACCESS_VERSION}."
            )
        raw_profiles = payload.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise RuntimeError(
                f"Invalid profile vault access payload in {settings_path}: 'profiles' must be an object."
            )
        try:
            return {
                canonical_profile_name(profile): ProfileVaultAccess.model_validate(entry)
                for profile, entry in raw_profiles.items()
            }
        except ValidationError as exc:
            raise RuntimeError(f"Invalid profile vault access entry in {settings_path}: {exc}") from exc


def write_profile_vault_access(
    profile: str | None,
    access: ProfileVaultAccess | None,
    instance_path: str | Path | None,
) -> Path:
    """Set or clear (`access=None`) the vault access of one profile."""
    with _STORE_LOCK:
        profiles = read_profile_vault_access(instance_path)
        profile_name = canonical_profile_name(profile)
        if access is None:
            profiles.pop(profile_name, None)
        else:
            profiles[profile_name] = access
        settings_path = get_profile_vault_access_path(instance_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROFILE_VAULT_ACCESS_VERSION,
            "profiles": {
                name: entry.model_dump(mode="json", exclude_none=True) for name, entry in sorted(profiles.items())
            },
        }
        temporary_path = settings_path.with_name(f".{settings_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
            temporary_path.replace(settings_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temporary profile vault access file %s: %s", temporary_path, exc)
        return settings_path
