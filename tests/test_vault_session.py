from pathlib import Path
from datetime import date, datetime

import pytest

import reachy_mini_conversation_app.vault_session as vault_session_mod
from reachy_mini_conversation_app.vault import parse_note
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.vault_session import SESSION_CONTEXT_MAX_CHARS, VaultSession
from reachy_mini_conversation_app.profile_vault_access import (
    ProfileVaultAccess,
    read_profile_vault_access,
    write_profile_vault_access,
)


ACCESS = {
    "agent": "emma",
    "read": ["Emma/Conversation Playbook", "Emma/Sessions", "Emma/Weekly Memories"],
    "write": ["Emma/Sessions", "Emma/Weekly Memories"],
    "session_context": ["Emma/Conversation Playbook/Current.md"],
    "session_log": {"folder": "Emma/Sessions", "type": "emma-session", "properties": {"date": "{date}"}},
    "weekly_memory": {
        "folder": "Emma/Weekly Memories",
        "type": "emma-memory",
        "date_weekday": 5,
        "properties": {"date": "{date}", "week_of": "{week_start}"},
    },
}


@pytest.fixture
def emma_vault(fixture_vault: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the fixture vault the synced vault of an active `Emma` profile with stored access."""
    monkeypatch.setattr(vault_session_mod, "current_vault_path", lambda: fixture_vault)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "Emma")
    write_profile_vault_access("Emma", ProfileVaultAccess.model_validate(ACCESS), tmp_path)
    return tmp_path


def test_store_round_trips_and_clears(tmp_path: Path) -> None:
    """The store keeps one entry per profile next to profile_toolsets.json."""
    path = write_profile_vault_access("Emma", ProfileVaultAccess.model_validate(ACCESS), tmp_path)

    assert path == tmp_path / "profile_vault_access.json"
    assert read_profile_vault_access(tmp_path)["Emma"].session_log is not None
    write_profile_vault_access("Emma", None, tmp_path)
    assert read_profile_vault_access(tmp_path) == {}


def test_session_start_loads_capped_context_once(emma_vault: Path, fixture_vault: Path) -> None:
    """Start loads the agent rules and context notes, caps them, and keeps them across reconnects."""
    (fixture_vault / "Emma/Conversation Playbook/Current.md").write_text(
        "---\ntype: emma-playbook\ncreated: 2026-09-19\n---\nAsk about the dinosaur book.\n" + "x" * 20000,
        encoding="utf-8",
    )
    session = VaultSession()

    session.begin(emma_vault, now=datetime(2026, 9, 23, 14, 5))

    assert "Emma writes one session log per conversation." in session.context
    assert "Warm and concrete." in session.context
    assert "Mira plans the week." not in session.context
    assert "Never overwrite a dated note." not in session.context
    assert "Ask about the dinosaur book." in session.context
    assert "`emma-session`" in session.context
    assert len(session.context) <= SESSION_CONTEXT_MAX_CHARS + 40
    assert session.context.endswith("[Vault context truncated.]")

    first_id = session.session_id
    (fixture_vault / "AGENTS.md").write_text("## Emma\n\nChanged.\n", encoding="utf-8")
    session.begin(emma_vault)
    assert session.session_id == first_id
    assert "Changed." not in session.context


def test_session_end_writes_one_log_and_skips_empty_sessions(emma_vault: Path, fixture_vault: Path) -> None:
    """A session with a user turn writes one contract log; a session without one writes nothing."""
    empty = VaultSession()
    empty.begin(emma_vault, now=datetime(2026, 9, 23, 9, 0))
    empty.record("assistant", "Hi there!")
    empty.end(emma_vault, now=datetime(2026, 9, 23, 9, 5))
    assert not (fixture_vault / "Emma/Sessions").exists()

    session = VaultSession()
    session.begin(emma_vault, now=datetime(2026, 9, 23, 14, 5))
    run = session.run_id("emma")
    session.record("user", "I saw a  T-rex\ntoday")
    session.record("assistant", "Wow, a T-rex!")
    session.end(emma_vault, now=datetime(2026, 9, 23, 14, 20))

    properties, body = parse_note((fixture_vault / "Emma/Sessions/2026-09-23-1405.md").read_text(encoding="utf-8"))
    assert properties is not None
    assert properties["type"] == "emma-session"
    assert properties["date"] == date(2026, 9, 23)
    assert properties["author"] == "agent/emma"
    assert properties["run"] == run
    assert "**User:** I saw a T-rex today" in body
    assert "**Emma:** Wow, a T-rex!" in body
    assert session.started_at is None and session.turns == []
    assert not (fixture_vault / "Emma/Weekly Memories").exists()


def test_first_session_end_of_a_week_writes_last_weeks_memory(emma_vault: Path, fixture_vault: Path) -> None:
    """The first session end in a new ISO week summarizes last week's logs once."""
    for start, said in (
        (datetime(2026, 9, 15, 10, 0), "We read about owls"),
        (datetime(2026, 9, 18, 16, 30), "Owls fly"),
    ):
        session = VaultSession()
        session.begin(emma_vault, now=start)
        session.record("user", said)
        session.end(emma_vault, now=start)
    assert not (fixture_vault / "Emma/Weekly Memories").exists()

    session = VaultSession()
    session.begin(emma_vault, now=datetime(2026, 9, 22, 9, 0))
    session.record("user", "Good morning")
    session.end(emma_vault, now=datetime(2026, 9, 22, 9, 10))

    memory = fixture_vault / "Emma/Weekly Memories/2026-09-18.md"
    properties, body = parse_note(memory.read_text(encoding="utf-8"))
    assert properties is not None
    assert properties["type"] == "emma-memory"
    assert properties["week_of"] == date(2026, 9, 14)
    assert "[[Emma/Sessions/2026-09-15-1000]] (1 user turns)" in body
    assert "  - Owls fly" in body
    assert "Good morning" not in body

    before = memory.read_text(encoding="utf-8")
    later = VaultSession()
    later.begin(emma_vault, now=datetime(2026, 9, 23, 9, 0))
    later.record("user", "Hello again")
    later.end(emma_vault, now=datetime(2026, 9, 23, 9, 10))
    assert memory.read_text(encoding="utf-8") == before
    assert len(list((fixture_vault / "Emma/Weekly Memories").iterdir())) == 1


def test_session_hooks_do_nothing_without_a_synced_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With sync off the session still runs and nothing is written."""
    monkeypatch.setattr(vault_session_mod, "current_vault_path", lambda: None)
    session = VaultSession()
    session.begin(tmp_path)
    session.record("user", "Hello")
    session.end(tmp_path)
    assert session.context == ""
    assert session.started_at is None
