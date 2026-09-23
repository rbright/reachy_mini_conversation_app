import os
from pathlib import Path
from datetime import date

import pytest

from reachy_mini_conversation_app.vault import (
    VaultError,
    SchemaError,
    RefusedError,
    read_note,
    parse_note,
    write_note,
    query_notes,
    parse_schema,
)


READ = ("Emma/Conversation Playbook", "Emma/Sessions", "Emma/Weekly Memories", "Emma/Research")
WRITE = ("Emma/Sessions", "Emma/Weekly Memories", "Emma/Research")
RUN = "reachy:emma:20260923T140500-abc123"
TODAY = date(2026, 9, 23)


def _write(
    vault: Path,
    path: str,
    properties: dict[str, object],
    body: str = "Body.\n",
    *,
    agent: str = "emma",
    folders: tuple[str, ...] = WRITE,
) -> bool:
    return write_note(
        vault, path, agent=agent, folders=folders, run=RUN, properties=properties, body=body, today=TODAY
    )


def test_write_creates_contract_note_atomically(fixture_vault: Path) -> None:
    """A new log gets tool-set keys, keeps key order, and leaves no temporary file."""
    path = "Emma/Sessions/2026-09-23-1405.md"

    created = _write(fixture_vault, path, {"type": "emma-session", "date": "2026-09-23"}, "Hello.\n")

    assert created is True
    text = (fixture_vault / path).read_text(encoding="utf-8")
    properties, body = parse_note(text)
    assert properties == {
        "type": "emma-session",
        "date": date(2026, 9, 23),
        "created": TODAY,
        "author": "agent/emma",
        "run": RUN,
        "updated": TODAY,
    }
    assert body == "Hello.\n"
    assert os.listdir(fixture_vault / "Emma" / "Sessions") == ["2026-09-23-1405.md"]


def test_write_refuses_folders_outside_either_allowlist(fixture_vault: Path) -> None:
    """The store and the vault schema must both allow a folder."""
    with pytest.raises(RefusedError, match="cannot write"):
        _write(fixture_vault, "Private/note.md", {"type": "emma-session", "date": "2026-09-23"})
    with pytest.raises(RefusedError, match="cannot write"):
        _write(
            fixture_vault,
            "Emma/Conversation Playbook/Conversation Playbook - 2026-09-23.md",
            {"type": "emma-playbook"},
            folders=(*WRITE, "Emma/Conversation Playbook"),
        )
    with pytest.raises(RefusedError, match="not in the schema `agents`"):
        _write(fixture_vault, "Emma/Sessions/x.md", {"type": "emma-session", "date": "2026-09-23"}, agent="mira")


def test_write_refuses_system_folder_even_when_allowed(fixture_vault: Path) -> None:
    """No agent writes System/, whatever the store says."""
    with pytest.raises(RefusedError, match="System"):
        _write(fixture_vault, "System/Notes.md", {"type": "emma-session"}, folders=("*",))


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "Emma/Sessions/../../System/Schema.md",
        "/Emma/Sessions/x.md",
        "Emma/.hidden/x.md",
        "Emma/Sessions/x.txt",
    ],
)
def test_write_refuses_traversal_and_non_markdown(fixture_vault: Path, path: str) -> None:
    """Traversal, absolute, hidden, and non-Markdown targets are refused."""
    with pytest.raises(RefusedError):
        _write(fixture_vault, path, {"type": "emma-session", "date": "2026-09-23"})


def test_write_refuses_symlink_that_leaves_the_vault(fixture_vault: Path, tmp_path: Path) -> None:
    """A symlinked folder cannot move a note out of the vault."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture_vault / "Emma" / "Sessions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RefusedError, match="symbolic link"):
        _write(fixture_vault, "Emma/Sessions/2026-09-23-1405.md", {"type": "emma-session", "date": "2026-09-23"})
    assert list(outside.iterdir()) == []


def test_write_refuses_locked_artifacts_and_existing_logs(fixture_vault: Path) -> None:
    """Approved artifacts and existing dated logs stay unchanged; draft artifacts can be updated."""
    approved = fixture_vault / "Emma/Research/approved-topic.md"
    before = approved.read_text(encoding="utf-8")
    with pytest.raises(RefusedError, match="approved"):
        _write(fixture_vault, "Emma/Research/approved-topic.md", {}, "Changed.\n")
    assert approved.read_text(encoding="utf-8") == before

    _write(fixture_vault, "Emma/Sessions/2026-09-23-1405.md", {"type": "emma-session", "date": "2026-09-23"})
    with pytest.raises(RefusedError, match="dated note or a log"):
        _write(fixture_vault, "Emma/Sessions/2026-09-23-1405.md", {"type": "emma-session", "date": "2026-09-23"})

    assert _write(fixture_vault, "Emma/Research/draft-topic.md", {"confidence": "high"}, "Better.\n") is False
    properties, body = parse_note((fixture_vault / "Emma/Research/draft-topic.md").read_text(encoding="utf-8"))
    assert properties is not None and properties["confidence"] == "high" and properties["run"] == RUN
    assert body == "Better.\n"


@pytest.mark.parametrize(
    ("path", "properties", "message"),
    [
        ("Emma/Sessions/a.md", {"type": "unknown"}, "must name a type"),
        ("Emma/Sessions/a.md", {"type": "research"}, "belong in"),
        ("Emma/Sessions/a.md", {"type": "emma-session"}, "missing required key `date`"),
        ("Emma/Sessions/a.md", {"type": "emma-session", "date": "soon"}, "must be of type date"),
        ("Emma/Sessions/a.md", {"type": "emma-session", "date": "2026-09-23", "mood": "ok"}, "not in the schema keys"),
        ("Emma/Research/a.md", {"type": "research", "confidence": "total"}, "must be one of"),
        ("Emma/Research/a.md", {"type": "research", "status": "approved"}, "draft"),
        ("Emma/Sessions/a.md", {"type": "emma-session", "run": "x"}, "set by the tool"),
    ],
)
def test_write_validates_against_schema(
    fixture_vault: Path, path: str, properties: dict[str, object], message: str
) -> None:
    """Type, folder, required keys, key types, allowed values, and tool keys follow System/Schema.md."""
    with pytest.raises(RefusedError, match=message):
        _write(fixture_vault, path, properties)
    assert not (fixture_vault / path).exists()


def test_write_refuses_credential_content(fixture_vault: Path) -> None:
    """Credential-like text is refused and the message does not repeat it."""
    with pytest.raises(RefusedError, match="credential pattern \\(api-key\\)") as error:
        _write(
            fixture_vault,
            "Emma/Sessions/a.md",
            {"type": "emma-session", "date": "2026-09-23"},
            "key sk-abcdefghijklmnopqrstuvwxyz\n",
        )
    assert "sk-abc" not in str(error.value)


def test_read_and_query_respect_both_allowlists(fixture_vault: Path) -> None:
    """Reads need store and schema access; queries filter by properties and created range."""
    note = read_note(fixture_vault, "Emma/Conversation Playbook/Current.md", agent="emma", folders=READ)
    assert note.properties == {"type": "emma-playbook", "created": "2026-09-19"}
    assert note.body == "Ask about the dinosaur book.\n"
    with pytest.raises(RefusedError):
        read_note(fixture_vault, "System/Schema.md", agent="emma", folders=("*",))
    with pytest.raises(RefusedError):
        read_note(fixture_vault, "Emma/Research/draft-topic.md", agent="emma", folders=("Emma/Sessions",))

    notes, truncated = query_notes(
        fixture_vault, agent="emma", folders=READ, where={"type": "research"}, created_from=date(2026, 9, 2)
    )
    assert [summary.path for summary in notes] == ["Emma/Research/draft-topic.md"]
    assert truncated is False


def test_schema_needs_exactly_one_valid_block() -> None:
    """A schema note without one valid block is a clear error."""
    with pytest.raises(SchemaError, match="exactly one"):
        parse_schema("# Schema\n")
    with pytest.raises(SchemaError, match="System/"):
        parse_schema(
            "```yaml vault-schema\nversion: 1\nvault: V\nkeys: {}\ntypes: {}\n"
            "agents: {emma: {write: [System/Reports]}}\n```\n"
        )


def test_read_missing_note_is_an_error(fixture_vault: Path) -> None:
    """A missing note is a VaultError, not a crash."""
    with pytest.raises(VaultError, match="not found"):
        read_note(fixture_vault, "Emma/Sessions/none.md", agent="emma", folders=READ)
