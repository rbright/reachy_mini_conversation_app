from pathlib import Path

import pytest
from pydantic import ValidationError

from reachy_mini_conversation_app.profile_vault_access import (
    ProfileVaultAccess,
    read_profile_vault_access,
    write_profile_vault_access,
)


def test_vault_root_grant_is_kept_and_folders_are_normalized(tmp_path: Path) -> None:
    """A root grant ("." or "/") survives a store round trip; blanks and duplicates go away."""
    access = ProfileVaultAccess(agent="emma", read=(".", "Emma/Sessions/", "Emma/Sessions", " "), write=("/",))
    write_profile_vault_access("Emma", access, tmp_path)

    stored = read_profile_vault_access(tmp_path)["Emma"]

    assert stored.read == ("", "Emma/Sessions")
    assert stored.write == ("",)


def test_property_templates_accept_only_placeholders_the_hook_fills() -> None:
    """A typo or a weekly-only placeholder in a session log template fails on save, not at session end."""
    note = {"folder": "Emma/Sessions", "type": "emma-session"}
    for session_properties in ({"date": "{dat}"}, {"week_of": "{week_start}"}):
        with pytest.raises(ValidationError, match="unknown placeholders"):
            ProfileVaultAccess.model_validate(
                {"agent": "emma", "session_log": {**note, "properties": session_properties}}
            )

    access = ProfileVaultAccess.model_validate(
        {
            "agent": "emma",
            "session_log": {**note, "properties": {"date": "{date}", "title": "{slug} at {time}"}},
            "weekly_memory": {**note, "properties": {"week_of": "{week_start}", "until": "{week_end}"}},
        }
    )
    assert access.weekly_memory is not None and access.weekly_memory.properties["week_of"] == "{week_start}"
