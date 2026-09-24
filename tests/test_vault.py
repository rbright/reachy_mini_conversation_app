import os
import tracemalloc
from pathlib import Path
from datetime import date

import pytest

from reachy_mini_conversation_app.vault import (
    READ_BODY_MAX_CHARS,
    FRONTMATTER_MAX_CHARS,
    VaultError,
    SchemaError,
    RefusedError,
    read_note,
    open_vault,
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
        open_vault(vault), path, agent=agent, folders=folders, run=RUN, properties=properties, body=body, today=TODAY
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


TASKS = (*WRITE, "Emma/Tasks", "Emma/Sources")


def test_records_are_written_only_by_their_owner(fixture_vault: Path) -> None:
    """A new record gets the writing agent as owner; a foreign owner or another agent's record is refused."""
    assert _write(fixture_vault, "Emma/Tasks/Mine.md", {"type": "task", "state": "open"}, folders=TASKS) is True
    properties, _body = parse_note((fixture_vault / "Emma/Tasks/Mine.md").read_text(encoding="utf-8"))
    assert properties is not None and properties["owner"] == "agent/emma" and properties["updated"] == TODAY
    assert _write(fixture_vault, "Emma/Tasks/Mine.md", {"state": "done"}, "Done.\n", folders=TASKS) is False

    with pytest.raises(RefusedError, match="must have `owner: agent/emma`"):
        _write(
            fixture_vault, "Emma/Tasks/New.md", {"type": "task", "state": "open", "owner": "agent/mira"}, folders=TASKS
        )
    assert not (fixture_vault / "Emma/Tasks/New.md").exists()

    theirs = fixture_vault / "Emma/Tasks/Theirs.md"
    theirs.write_text("---\ntype: task\ncreated: 2026-09-01\nstate: open\nowner: agent/mira\n---\nMira's.\n")
    before = theirs.read_text(encoding="utf-8")
    with pytest.raises(RefusedError, match="only the record owner"):
        _write(fixture_vault, "Emma/Tasks/Theirs.md", {"state": "done"}, folders=TASKS)
    assert theirs.read_text(encoding="utf-8") == before


def test_reference_notes_are_never_written(fixture_vault: Path) -> None:
    """Agents neither create nor update reference notes, even in a folder they may write."""
    with pytest.raises(RefusedError, match="never write reference"):
        _write(fixture_vault, "Emma/Sources/New.md", {"type": "source"}, folders=TASKS)
    assert not (fixture_vault / "Emma/Sources/New.md").exists()

    book = fixture_vault / "Emma/Sources/Book.md"
    book.parent.mkdir(parents=True)
    book.write_text("---\ntype: source\ncreated: 2026-09-01\n---\nA book.\n", encoding="utf-8")
    with pytest.raises(RefusedError, match="never write reference"):
        _write(fixture_vault, "Emma/Sources/Book.md", {}, "Changed.\n", folders=TASKS)
    assert book.read_text(encoding="utf-8").endswith("A book.\n")


@pytest.mark.parametrize(
    ("body", "pattern", "secret"),
    [
        ("key sk-abcdefghijklmnopqrstuvwxyz\n", "api-key", "sk-abc"),
        # Built at runtime, so that secret scanners do not flag the fake values in this file.
        ("Use hf_" + "AbCd" * 9 + " here\n", "huggingface-token", "hf_AbC"),
        ("HF_TOKEN=Zx81kQ2mP0aa\n", "secret-assignment", "Zx81kQ2m"),
        ("AWS_SECRET_ACCESS_KEY = '" + "wJalr/Xut" * 4 + "'\n", "secret-assignment", "wJalr"),
        ("github token: Zx81kQ2mP0aa\n", "secret-assignment", "Zx81kQ2m"),
    ],
)
def test_write_refuses_credential_content(fixture_vault: Path, body: str, pattern: str, secret: str) -> None:
    """Credential-like text is refused and the message names the pattern, never the text."""
    with pytest.raises(RefusedError, match=f"credential pattern \\({pattern}\\)") as error:
        _write(fixture_vault, "Emma/Sessions/a.md", {"type": "emma-session", "date": "2026-09-23"}, body)
    assert secret not in str(error.value)
    assert not (fixture_vault / "Emma/Sessions/a.md").exists()


def test_read_and_query_respect_both_allowlists(fixture_vault: Path) -> None:
    """Reads need store and schema access; queries filter by properties and created range."""
    note = read_note(open_vault(fixture_vault), "Emma/Conversation Playbook/Current.md", agent="emma", folders=READ)
    assert note.properties == {"type": "emma-playbook", "created": "2026-09-19"}
    assert note.body == "Ask about the dinosaur book.\n"
    with pytest.raises(RefusedError):
        read_note(open_vault(fixture_vault), "System/Schema.md", agent="emma", folders=("*",))
    with pytest.raises(RefusedError):
        read_note(open_vault(fixture_vault), "Emma/Research/draft-topic.md", agent="emma", folders=("Emma/Sessions",))

    notes, truncated = query_notes(
        open_vault(fixture_vault),
        agent="emma",
        folders=READ,
        where={"type": "research"},
        created_from=date(2026, 9, 2),
    )
    assert [summary.path for summary in notes] == ["Emma/Research/draft-topic.md"]
    assert truncated is False


def test_query_reports_truncation_at_the_limit(fixture_vault: Path) -> None:
    """More matches than `limit` return the first `limit` matches and a truncation flag."""
    notes, truncated = query_notes(
        open_vault(fixture_vault), agent="emma", folders=READ, where={"type": "research"}, limit=1
    )

    assert [summary.path for summary in notes] == ["Emma/Research/approved-topic.md"]
    assert truncated is True


def test_query_skips_and_logs_notes_that_cannot_be_read(fixture_vault: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A note that vault_read would refuse is not listed, and the skip is logged with its path."""
    research = fixture_vault / "Emma/Research"
    (research / "bad-yaml.md").write_text("---\ntype: [research\n---\nText.\n", encoding="utf-8")
    (research / "bad-bytes.md").write_bytes(b"---\ntype: research\ncreated: 2026-09-05\n---\n\xff\xfe\n")

    notes, _truncated = query_notes(open_vault(fixture_vault), agent="emma", folders=READ, where={"type": "research"})

    assert [summary.path for summary in notes] == ["Emma/Research/approved-topic.md", "Emma/Research/draft-topic.md"]
    assert "Skipping vault note Emma/Research/bad-yaml.md" in caplog.text
    assert "Skipping vault note Emma/Research/bad-bytes.md" in caplog.text


def test_reads_and_queries_load_only_a_bounded_prefix(fixture_vault: Path) -> None:
    """A large synced note does not load into memory: reads return a capped body, queries only frontmatter."""
    large = fixture_vault / "Emma/Research/large.md"
    large.write_text("---\ntype: research\ncreated: 2026-09-10\n---\n" + "x" * (8 * 1024 * 1024), encoding="utf-8")
    vault = open_vault(fixture_vault)

    tracemalloc.start()
    try:
        note = read_note(vault, "Emma/Research/large.md", agent="emma", folders=READ)
        notes, _truncated = query_notes(vault, agent="emma", folders=READ, where={"type": "research"})
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert note.truncated is True
    assert len(note.body) == READ_BODY_MAX_CHARS
    assert "Emma/Research/large.md" in [summary.path for summary in notes]
    assert peak < 1024 * 1024


def test_read_refuses_frontmatter_over_the_cap(fixture_vault: Path) -> None:
    """Frontmatter that does not close within the cap is an error, not a whole-file read."""
    (fixture_vault / "Emma/Research/long-yaml.md").write_text(
        "---\ntype: research\n" + "# filler\n" * (FRONTMATTER_MAX_CHARS // 9 + 1) + "---\nBody.\n", encoding="utf-8"
    )

    with pytest.raises(VaultError, match="frontmatter is longer"):
        read_note(open_vault(fixture_vault), "Emma/Research/long-yaml.md", agent="emma", folders=READ)


def test_schema_needs_exactly_one_valid_block() -> None:
    """A schema note without one valid block is a clear error."""
    with pytest.raises(SchemaError, match="exactly one"):
        parse_schema("# Schema\n")
    with pytest.raises(SchemaError, match="System/"):
        parse_schema(
            "```yaml vault-schema\nversion: 1\nvault: V\nkeys: {}\ntypes: {}\n"
            "agents: {emma: {write: [System/Reports]}}\n```\n"
        )


def test_schema_that_is_not_utf8_is_a_schema_error(fixture_vault: Path) -> None:
    """Invalid UTF-8 in the schema note is a vault error, so session start can continue without vault context."""
    (fixture_vault / "System/Schema.md").write_bytes(b"# Schema\n\xff\xfe\n")

    with pytest.raises(SchemaError, match="UTF-8"):
        open_vault(fixture_vault)


def test_read_missing_note_is_an_error(fixture_vault: Path) -> None:
    """A missing note is a VaultError, not a crash."""
    with pytest.raises(VaultError, match="not found"):
        read_note(open_vault(fixture_vault), "Emma/Sessions/none.md", agent="emma", folders=READ)
