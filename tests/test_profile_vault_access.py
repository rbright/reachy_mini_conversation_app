from pathlib import Path

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
