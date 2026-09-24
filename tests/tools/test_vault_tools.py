from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.vault_session as vault_session_mod
from reachy_mini_conversation_app.vault import parse_note
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.vault_read import VaultRead
from reachy_mini_conversation_app.tools.vault_query import VaultQuery
from reachy_mini_conversation_app.tools.vault_write import VaultWrite
from reachy_mini_conversation_app.profile_vault_access import ProfileVaultAccess, write_profile_vault_access


@pytest.fixture
def deps(fixture_vault: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolDependencies:
    """Tool dependencies for an active `Tutor` profile with access to the fixture vault."""
    monkeypatch.setattr(vault_session_mod, "current_vault_path", lambda: fixture_vault)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "Tutor")
    access = ProfileVaultAccess(
        agent="tutor",
        read=("Tutor/Conversation Playbook", "Tutor/Research"),
        write=("Tutor/Research",),
    )
    write_profile_vault_access("Tutor", access, tmp_path)
    tool_deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=tmp_path)
    tool_deps.vault_session.begin(tmp_path, now=datetime(2026, 9, 23, 14, 5))
    return tool_deps


@pytest.mark.asyncio
async def test_vault_tools_read_query_and_write(deps: ToolDependencies, fixture_vault: Path) -> None:
    """The tools read, list, and write allowed notes with the session run id."""
    read = await VaultRead()(deps, path="Tutor/Conversation Playbook/Current.md")
    assert read["body"] == "Ask about the dinosaur book.\n"

    listed = await VaultQuery()(deps, type="research", status="draft")
    assert [note["path"] for note in listed["notes"]] == ["Tutor/Research/draft-topic.md"]

    written = await VaultWrite()(deps, path="Tutor/Research/owls.md", type="research", body="Owls.\n")
    assert written == {"path": "Tutor/Research/owls.md", "created": True}
    properties, _body = parse_note((fixture_vault / "Tutor/Research/owls.md").read_text(encoding="utf-8"))
    assert properties is not None
    assert properties["run"] == deps.vault_session.run_id("tutor")
    assert properties["status"] == "draft"


@pytest.mark.asyncio
async def test_vault_tools_return_errors(deps: ToolDependencies, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusals and missing configuration come back as error dicts."""
    refused = await VaultWrite()(deps, path="Tutor/Research/approved-topic.md", type="research", body="x")
    assert "approved" in refused["error"]
    assert "cannot read" in (await VaultRead()(deps, path="Tutor/Sessions/x.md"))["error"]
    assert "error" in await VaultQuery()(deps, created_from="last week")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "Other")
    assert "not configured for personality `Other`" in (await VaultRead()(deps, path="Tutor/Research/x.md"))["error"]

    monkeypatch.setattr(vault_session_mod, "current_vault_path", lambda: None)
    assert "Obsidian Sync is off" in (await VaultQuery()(deps))["error"]
