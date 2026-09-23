import re
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


def test_session_context_lists_writable_types_in_the_vault_root(emma_vault: Path, fixture_vault: Path) -> None:
    """A type whose folder is the vault root is writable when the profile grants the root."""
    schema = fixture_vault / "System/Schema.md"
    schema.write_text(
        schema.read_text(encoding="utf-8").replace("types:\n", "types:\n  inbox: {class: artifact, folders: [.]}\n"),
        encoding="utf-8",
    )
    write_profile_vault_access("Emma", ProfileVaultAccess.model_validate({**ACCESS, "write": ["."]}), emma_vault)
    session = VaultSession()

    session.begin(emma_vault)

    assert "Note types you may write: `inbox`" in session.context


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
    suffix = session.session_id.rsplit("-", 1)[-1]
    session.record("user", "I saw a  T-rex\ntoday")
    session.record("assistant", "Wow, a T-rex!")
    session.end(emma_vault, now=datetime(2026, 9, 23, 14, 20))

    log = fixture_vault / f"Emma/Sessions/2026-09-23-1405-{suffix}.md"
    properties, body = parse_note(log.read_text(encoding="utf-8"))
    assert properties is not None
    assert properties["type"] == "emma-session"
    assert properties["date"] == date(2026, 9, 23)
    assert properties["author"] == "agent/emma"
    assert properties["run"] == run
    assert "**User:** I saw a T-rex today" in body
    assert "**Emma:** Wow, a T-rex!" in body
    assert session.started_at is None and session.turns == []
    assert not (fixture_vault / "Emma/Weekly Memories").exists()


def test_two_sessions_in_one_minute_write_two_logs(emma_vault: Path, fixture_vault: Path) -> None:
    """Each session gets its own log, even when two sessions start in the same minute."""
    for said in ("First chat", "Second chat"):
        session = VaultSession()
        session.begin(emma_vault, now=datetime(2026, 9, 23, 14, 5, 10))
        session.record("user", said)
        session.end(emma_vault, now=datetime(2026, 9, 23, 14, 5, 50))

    bodies = sorted(log.read_text(encoding="utf-8") for log in (fixture_vault / "Emma/Sessions").iterdir())
    assert len(bodies) == 2
    assert "**User:** First chat" in "".join(bodies)
    assert "**User:** Second chat" in "".join(bodies)


def test_profile_change_ends_the_old_profiles_session(
    emma_vault: Path, fixture_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turns of a session stay with the profile that started it; the next connection starts a new session."""
    no_notes = {key: value for key, value in ACCESS.items() if key not in ("session_log", "weekly_memory")}
    write_profile_vault_access("Other", ProfileVaultAccess.model_validate(no_notes), emma_vault)
    session = VaultSession()
    session.begin(emma_vault, now=datetime(2026, 9, 23, 14, 5))
    first_id = session.session_id
    session.record("user", "Hello Emma")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "Other")
    session.begin(emma_vault, now=datetime(2026, 9, 23, 14, 10))

    logs = list((fixture_vault / "Emma/Sessions").iterdir())
    assert len(logs) == 1
    assert "**User:** Hello Emma" in logs[0].read_text(encoding="utf-8")
    assert session.session_id != first_id
    assert session.profile == "Other"
    assert session.turns == []


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
    assert re.search(r"\[\[Emma/Sessions/2026-09-15-1000-[0-9a-f]{6}\]\] \(1 user turns\)", body)
    assert "  - Owls fly" in body
    assert "Good morning" not in body

    before = memory.read_text(encoding="utf-8")
    later = VaultSession()
    later.begin(emma_vault, now=datetime(2026, 9, 23, 9, 0))
    later.record("user", "Hello again")
    later.end(emma_vault, now=datetime(2026, 9, 23, 9, 10))
    assert memory.read_text(encoding="utf-8") == before
    assert len(list((fixture_vault / "Emma/Weekly Memories").iterdir())) == 1


def test_session_end_after_an_idle_week_writes_the_older_weeks_memory(emma_vault: Path, fixture_vault: Path) -> None:
    """A week without sessions does not hide an earlier week that has logs but no memory yet."""
    for start, said in ((datetime(2026, 9, 15, 10, 0), "Owls fly"), (datetime(2026, 9, 29, 9, 0), "Back again")):
        session = VaultSession()
        session.begin(emma_vault, now=start)
        session.record("user", said)
        session.end(emma_vault, now=start)

    memories = list((fixture_vault / "Emma/Weekly Memories").iterdir())
    assert [memory.name for memory in memories] == ["2026-09-18.md"]
    assert "  - Owls fly" in memories[0].read_text(encoding="utf-8")


def test_session_hooks_do_nothing_without_a_synced_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With sync off the session still runs and nothing is written."""
    monkeypatch.setattr(vault_session_mod, "current_vault_path", lambda: None)
    session = VaultSession()
    session.begin(tmp_path)
    session.record("user", "Hello")
    session.end(tmp_path)
    assert session.context == ""
    assert session.started_at is None
