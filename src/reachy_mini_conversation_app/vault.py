"""Read, query, and write Obsidian notes under the vault contract v1 (`System/Schema.md`)."""

import os
import re
import json
import tempfile
from typing import Literal
from pathlib import Path
from datetime import date, datetime
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import yaml
from pydantic import Field, BaseModel, JsonValue, ConfigDict, ValidationError, field_validator, model_validator


SCHEMA_PATH = "System/Schema.md"
SYSTEM_FOLDER = "System"
STATUSES = ("draft", "approved", "superseded")
LOCKED_STATUSES = ("approved", "superseded")
# The write path sets these keys. A caller cannot supply them.
TOOL_KEYS = frozenset({"author", "run", "created", "updated"})
READ_BODY_MAX_CHARS = 8000

KeyType = Literal["text", "list", "number", "checkbox", "date", "datetime", "tags"]
NoteClass = Literal["artifact", "record", "log", "reference"]

_ID = r"[a-z0-9][a-z0-9._-]*"
_ACTOR = re.compile(rf"(?:agent|human)/{_ID}")
_HUMAN = re.compile(rf"human/{_ID}")
AGENT_ID = re.compile(_ID)
_RUN = re.compile(
    r"(?:hermes|reachy):[A-Za-z0-9._-]+:[A-Za-z0-9._-]+"
    r"|script:[a-z0-9][a-z0-9._-]*:(?:\d{8}T\d{6}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z"
    r"|[A-Za-z0-9][A-Za-z0-9._-]*",
)
_DATE_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATETIME_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?")
_SCHEMA_BLOCK = re.compile(r"^```yaml vault-schema[ \t]*\r?\n(.*?)^```[ \t]*\r?$", re.MULTILINE | re.DOTALL)
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)(?:\r?\n)?^---[ \t]*(?:\r?\n|\Z)", re.MULTILINE | re.DOTALL)
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_NAME_PLACEHOLDERS = frozenset({"date", "slug", "title", "week", "month", "year", "quarter", "time"})
# Obvious credentials only. A match refuses the write; the message names the pattern, never the text.
_CREDENTIALS = (
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("api-key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    (
        "secret-assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*[\"']?[^\s\"'\[]{8,}",
        ),
    ),
)


class VaultError(Exception):
    """A vault operation cannot run. The message names paths and keys, never note content."""


class SchemaError(VaultError):
    """`System/Schema.md` is missing or invalid."""


class RefusedError(VaultError):
    """The contract or the access allowlist forbids the operation."""


def normalize_folder(value: str) -> str:
    """Return a vault-relative folder without surrounding slashes; the vault root is ""."""
    folder = value.strip().strip("/")
    return "" if folder == "." else folder


def covers(folders: Sequence[str], path: str) -> bool:
    """Return whether a vault-relative file path sits in one of `folders` ("*" is the whole vault)."""
    return any(
        folder == "*" or (not folder and "/" not in path) or (folder and path.startswith(f"{folder}/"))
        for folder in folders
    )


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TypeRule(_SchemaModel):
    """One note type of the vault schema."""

    note_class: NoteClass = Field(alias="class")
    folders: tuple[str, ...] = Field(min_length=1)
    name: str | None = None
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    values: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("folders")
    @classmethod
    def _normalize_folders(cls, folders: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_folder(folder) for folder in folders)

    @field_validator("name")
    @classmethod
    def _known_placeholders(cls, name: str | None) -> str | None:
        unknown = set(PLACEHOLDER.findall(name or "")) - _NAME_PLACEHOLDERS
        if unknown:
            raise ValueError(f"unknown name placeholders: {sorted(unknown)}")
        return name


class AgentAccess(_SchemaModel):
    """Folders one agent may read and write, as the vault grants them."""

    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()

    @field_validator("read", "write")
    @classmethod
    def _normalize(cls, folders: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_folder(folder) for folder in folders)


class Schema(_SchemaModel):
    """The `yaml vault-schema` block of `System/Schema.md`."""

    version: Literal[1]
    vault: str
    keys: dict[str, KeyType]
    types: dict[str, TypeRule]
    agents: dict[str, AgentAccess] = Field(default_factory=dict)
    ignore: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _empty_sections(cls, data: object) -> object:
        # YAML reads an empty section (`agents:`) as null.
        if isinstance(data, dict):
            return {
                key: ({} if value is None and key in {"agents", "keys", "types"} else value)
                for key, value in data.items()
            }
        return data

    @field_validator("agents")
    @classmethod
    def _no_system_writes(cls, agents: dict[str, AgentAccess]) -> dict[str, AgentAccess]:
        for agent, access in agents.items():
            if any(folder == SYSTEM_FOLDER or folder.startswith(f"{SYSTEM_FOLDER}/") for folder in access.write):
                raise ValueError(f"agent {agent!r} cannot have write access to {SYSTEM_FOLDER}/")
        return agents


@dataclass(frozen=True)
class NoteView:
    """One note as the read path returns it."""

    path: str
    properties: dict[str, JsonValue]
    body: str
    truncated: bool


@dataclass(frozen=True)
class NoteSummary:
    """One query match: path and properties only."""

    path: str
    properties: dict[str, JsonValue]


def parse_schema(text: str) -> Schema:
    """Parse the one `yaml vault-schema` block of a schema note."""
    blocks = _SCHEMA_BLOCK.findall(text)
    if len(blocks) != 1:
        raise SchemaError(f"{SCHEMA_PATH} must contain exactly one `yaml vault-schema` block; found {len(blocks)}")
    try:
        document = yaml.safe_load(blocks[0])
    except yaml.YAMLError as error:
        raise SchemaError(f"{SCHEMA_PATH} vault-schema block is not valid YAML: {type(error).__name__}") from None
    try:
        return Schema.model_validate(document)
    except ValidationError as error:
        # Input values can hold note text. Report locations and messages only.
        details = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc']) or 'value'}: {detail['msg']}"
            for detail in error.errors(include_input=False, include_url=False)
        )
        raise SchemaError(f"{SCHEMA_PATH} is invalid: {details}") from None


def load_schema(vault: Path) -> Schema:
    """Read and parse `System/Schema.md` of a vault."""
    path = vault / SCHEMA_PATH
    if not path.is_file():
        raise SchemaError(f"{SCHEMA_PATH} not found; this vault has no contract schema yet")
    return parse_schema(path.read_text(encoding="utf-8"))


def parse_note(text: str) -> tuple[dict[str, object] | None, str]:
    """Return the frontmatter (None when the note has none) and the body after it."""
    match = _FRONTMATTER.match(text)
    if match is None:
        return None, text
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise VaultError(f"frontmatter is not valid YAML: {type(error).__name__}") from None
    if frontmatter is None:
        frontmatter = {}
    if not isinstance(frontmatter, dict):
        raise VaultError("frontmatter is not a mapping")
    return {str(key): value for key, value in frontmatter.items()}, text[match.end() :]


def render_note(properties: Mapping[str, object], body: str) -> str:
    """Render frontmatter in key order, then the body."""
    frontmatter = (
        yaml.safe_dump(dict(properties), sort_keys=False, allow_unicode=True, default_flow_style=False, width=4096)
        if properties
        else ""
    )
    return f"---\n{frontmatter}---\n{body}"


def _json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _missing(value: object) -> bool:
    return value is None or value in ("", [])


def _type_ok(kind: KeyType, value: object) -> bool:
    match kind:
        case "text":
            return isinstance(value, str)
        case "list":
            return isinstance(value, list) and all(not isinstance(item, (list, dict)) for item in value)
        case "tags":
            return isinstance(value, list) and all(
                isinstance(item, str) and not item.startswith("#") for item in value
            )
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "checkbox":
            return isinstance(value, bool)
        case "date":
            return (isinstance(value, date) and not isinstance(value, datetime)) or (
                isinstance(value, str) and _DATE_TEXT.fullmatch(value) is not None
            )
        case "datetime":
            return isinstance(value, datetime) or (
                isinstance(value, str) and _DATETIME_TEXT.fullmatch(value) is not None
            )


def schema_errors(schema: Schema, path: str, properties: Mapping[str, object]) -> list[str]:
    """Return contract section 1 and 2 errors for one note's frontmatter."""
    errors: list[str] = []
    note_type = properties.get("type")
    rule = schema.types.get(note_type) if isinstance(note_type, str) else None
    if _missing(note_type):
        errors.append("missing required key `type`")
    elif rule is None:
        errors.append("`type` is not a type in the schema")
    elif not covers(rule.folders, path):
        errors.append(f"type `{note_type}` notes belong in: {', '.join(f or '.' for f in rule.folders)}")

    required = ["created"]
    if rule is not None:
        required += {"artifact": ["status"], "record": ["state"], "log": [], "reference": []}[rule.note_class]
        required += rule.required
        if rule.note_class in {"artifact", "log"}:
            required += ["author", "run"]
    errors += [f"missing required key `{key}`" for key in dict.fromkeys(required) if _missing(properties.get(key))]

    key_types: dict[str, KeyType] = {**schema.keys, "created": "date"}
    for key, value in properties.items():
        # Human-added keys outside the schema are lint warnings; the write path refuses only supplied ones.
        if key not in key_types or _missing(value):
            continue
        if not _type_ok(key_types[key], value):
            errors.append(f"`{key}` must be of type {key_types[key]}")
        allowed = rule.values.get(key) if rule is not None else None
        if allowed is not None and any(
            item not in allowed for item in (value if isinstance(value, list) else [value])
        ):
            errors.append(f"`{key}` must be one of: {', '.join(allowed)}")

    for key, pattern in (("author", _ACTOR), ("owner", _ACTOR), ("run", _RUN), ("waiting_on", _HUMAN)):
        value = properties.get(key)
        if not _missing(value) and not (isinstance(value, str) and pattern.fullmatch(value)):
            errors.append(f"`{key}` has a bad format")
    status = properties.get("status")
    if not _missing(status) and status not in STATUSES:
        errors.append(f"`status` must be one of: {', '.join(STATUSES)}")
    return errors


def fill_placeholders(template: str, values: Mapping[str, str]) -> str:
    """Replace `{name}` placeholders; an unknown placeholder is an error."""
    unknown = sorted(set(PLACEHOLDER.findall(template)) - values.keys())
    if unknown:
        raise VaultError(f"unknown placeholders in `{template}`: {', '.join(unknown)}")
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def note_path(vault: Path, path: str) -> Path:
    """Return the absolute path of a vault-relative Markdown note. Refuse traversal, hidden parts, and symlinks."""
    parts = path.split("/")
    if (
        not path
        or "\\" in path
        or "\0" in path
        or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
    ):
        raise RefusedError("path must be vault-relative, without '..', hidden parts, or backslashes")
    if not path.endswith(".md"):
        raise RefusedError("path must name a Markdown (.md) note")
    root = vault.resolve()
    target = root.joinpath(*parts)
    # Any symlink in the path could move the note out of the vault or into a folder the allowlist does not name.
    if target.resolve() != target:
        raise RefusedError("path contains a symbolic link")
    return target


def _schema_agent(schema: Schema, agent: str) -> AgentAccess:
    if not AGENT_ID.fullmatch(agent):
        raise RefusedError("agent id has a bad format")
    access = schema.agents.get(agent)
    if access is None:
        raise RefusedError(f"agent `{agent}` is not in the schema `agents` section")
    return access


def read_note(vault: Path, path: str, *, agent: str, folders: Sequence[str]) -> NoteView:
    """Read one note that both the vault schema and `folders` let `agent` read."""
    access = _schema_agent(load_schema(vault), agent)
    target = note_path(vault, path)
    if not (covers(folders, path) and covers(access.read, path)):
        raise RefusedError(f"agent `{agent}` cannot read `{path}`")
    if not target.is_file():
        raise VaultError(f"note not found: {path}")
    frontmatter, body = parse_note(target.read_text(encoding="utf-8"))
    return NoteView(
        path=path,
        properties={key: _json(value) for key, value in (frontmatter or {}).items()},
        body=body[:READ_BODY_MAX_CHARS],
        truncated=len(body) > READ_BODY_MAX_CHARS,
    )


def _property_matches(actual: JsonValue, expected: JsonValue) -> bool:
    return actual == expected or (isinstance(actual, list) and expected in actual)


def query_notes(
    vault: Path,
    *,
    agent: str,
    folders: Sequence[str],
    folder: str | None = None,
    where: Mapping[str, JsonValue] | None = None,
    created_from: date | None = None,
    created_to: date | None = None,
    limit: int = 50,
) -> tuple[list[NoteSummary], bool]:
    """Return readable notes whose properties equal `where` (list properties: contain), and a truncation flag."""
    access = _schema_agent(load_schema(vault), agent)
    root = vault.resolve()
    scope = None if folder is None else normalize_folder(folder)
    matches: list[NoteSummary] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            file_path = Path(directory) / name
            path = file_path.relative_to(root).as_posix()
            if (
                name.startswith(".")
                or not name.endswith(".md")
                or file_path.is_symlink()
                or not (covers(folders, path) and covers(access.read, path))
                or (scope is not None and not covers((scope,), path))
            ):
                continue
            try:
                frontmatter, _body = parse_note(file_path.read_text(encoding="utf-8", errors="replace"))
            except VaultError:
                continue
            properties = {key: _json(value) for key, value in (frontmatter or {}).items()}
            created = properties.get("created")
            if (created_from or created_to) and not (isinstance(created, str) and _DATE_TEXT.fullmatch(created)):
                continue
            if isinstance(created, str) and (
                (created_from and created < created_from.isoformat())
                or (created_to and created > created_to.isoformat())
            ):
                continue
            if all(_property_matches(properties.get(key), expected) for key, expected in (where or {}).items()):
                matches.append(NoteSummary(path=path, properties=properties))
    return matches[:limit], len(matches) > limit


def _credential(text: str) -> str | None:
    return next((name for name, pattern in _CREDENTIALS if pattern.search(text)), None)


def write_note(
    vault: Path,
    path: str,
    *,
    agent: str,
    folders: Sequence[str],
    run: str,
    properties: Mapping[str, JsonValue],
    body: str | None,
    today: date,
) -> bool:
    """Create or update one note under the contract and return whether it was created."""
    if not _RUN.fullmatch(run):
        raise RefusedError("run id has a bad format")
    schema = load_schema(vault)
    access = _schema_agent(schema, agent)
    target = note_path(vault, path)
    if covers((SYSTEM_FOLDER,), path):
        raise RefusedError(f"agents never write `{SYSTEM_FOLDER}/`")
    if not (covers(folders, path) and covers(access.write, path)):
        raise RefusedError(f"agent `{agent}` cannot write `{path}`")
    supplied = sorted(TOOL_KEYS & properties.keys())
    if supplied:
        raise RefusedError(f"keys set by the tool cannot be supplied: {', '.join(supplied)}")
    unknown = sorted(properties.keys() - schema.keys.keys())
    if unknown:
        raise RefusedError(f"keys are not in the schema keys: {', '.join(unknown)}")
    found = _credential(body or "") or _credential(json.dumps(properties, default=str))
    if found is not None:
        raise RefusedError(f"content matches a credential pattern ({found}); remove it before writing")

    actor = f"agent/{agent}"
    exists = target.exists()
    if exists and not target.is_file():
        raise RefusedError("path exists and is not a file")
    merged: dict[str, object] = {}
    old_body = ""
    if exists:
        try:
            frontmatter, old_body = parse_note(target.read_text(encoding="utf-8"))
        except VaultError as error:
            raise RefusedError(f"existing note cannot be updated: {error}") from None
        merged = dict(frontmatter or {})
        old_type = merged.get("type")
        old_rule = schema.types.get(old_type) if isinstance(old_type, str) else None
        if old_rule is None:
            raise RefusedError("existing note has no known `type`; a person must classify it first")
        if old_rule.note_class == "reference":
            raise RefusedError("agents never write reference notes")
        if old_rule.note_class == "log" or "{date}" in (old_rule.name or ""):
            raise RefusedError("agents never overwrite a dated note or a log; create a new note")
        if old_rule.note_class == "artifact" and merged.get("status") in LOCKED_STATUSES:
            raise RefusedError(f"artifact status is `{merged.get('status')}`; agents write drafts only")
        if old_rule.note_class == "record" and merged.get("owner") != actor:
            raise RefusedError(f"only the record owner writes it; `{actor}` is not the owner")
        if "type" in properties and properties["type"] != old_type:
            raise RefusedError("an existing note keeps its `type`")

    for key, value in properties.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict):
            raise RefusedError("nested property values are not allowed")
        elif schema.keys.get(key) == "date" and isinstance(value, str) and _DATE_TEXT.fullmatch(value):
            try:
                merged[key] = date.fromisoformat(value)
            except ValueError:
                raise RefusedError(f"`{key}` is not a valid date") from None
        else:
            merged[key] = value

    note_type = merged.get("type")
    rule = schema.types.get(note_type) if isinstance(note_type, str) else None
    if rule is None:
        raise RefusedError("`type` must name a type in the schema")
    if rule.note_class == "reference":
        raise RefusedError("agents never write reference notes")
    if rule.note_class == "artifact":
        merged.setdefault("status", "draft")
        if merged["status"] != "draft":
            raise RefusedError("agents write artifacts with `status: draft` only")
    if rule.note_class == "record":
        merged.setdefault("owner", actor)
        if merged["owner"] != actor:
            raise RefusedError(f"a record written by `{actor}` must have `owner: {actor}`")
    merged.setdefault("created", today)
    merged["author"] = actor
    merged["run"] = run
    if rule.note_class == "record" or "updated" in schema.keys:
        merged["updated"] = today

    errors = schema_errors(schema, path, merged)
    if errors:
        raise RefusedError("note breaks the vault schema: " + "; ".join(errors))
    atomic_write(target, render_note(merged, old_body if body is None else body))
    return not exists


def atomic_write(target: Path, text: str) -> None:
    """Write `text` to a hidden temporary file beside `target`, then rename it over `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
    # A hidden name in the same folder keeps the rename atomic and out of Obsidian's file list.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged.chmod(mode)
        staged.replace(target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
