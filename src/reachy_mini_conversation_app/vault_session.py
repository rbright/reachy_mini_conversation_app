"""Obsidian vault context at session start and session notes at session end."""

import re
import uuid
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import field, dataclass

from reachy_mini_conversation_app.vault import (
    VaultError,
    covers,
    note_path,
    read_note,
    write_note,
    load_schema,
    query_notes,
    fill_placeholders,
)
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.obsidian_sync import current_vault_path
from reachy_mini_conversation_app.profile_store import canonical_profile_name
from reachy_mini_conversation_app.profile_vault_access import ProfileVaultAccess, read_profile_vault_access


logger = logging.getLogger(__name__)

SESSION_CONTEXT_MAX_CHARS = 8000
TRANSCRIPT_MAX_CHARS = 20000
AGENTS_FILENAME = "AGENTS.md"
# The agent section can sit anywhere in AGENTS.md, so this read cap is larger than the context cap.
AGENTS_MAX_CHARS = 32000
_WEEKLY_EXCERPT_TURNS = 3
_WEEKLY_EXCERPT_CHARS = 200
_WEEKLY_MAX_LOGS = 200
# How far back a session end looks for finished weeks without a memory, for a robot that was off for a while.
_WEEKLY_LOOKBACK_WEEKS = 8
_USER_LABEL = "User"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ActiveVault:
    """The synced vault and the active profile's access to it."""

    root: Path
    profile: str
    access: ProfileVaultAccess


def active_vault(instance_path: str | Path | None, profile: str | None = None) -> ActiveVault:
    """Return the synced vault and a profile's vault access (default: the active profile), or raise `VaultError`."""
    root = current_vault_path()
    if root is None:
        raise VaultError("Obsidian Sync is off or the local vault folder does not exist")
    profile = profile or canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
    try:
        access = read_profile_vault_access(instance_path).get(profile)
    except RuntimeError as exc:
        raise VaultError(str(exc)) from exc
    if access is None:
        raise VaultError(f"vault access is not configured for personality `{profile}`")
    return ActiveVault(root=root, profile=profile, access=access)


def _calendar_values(day: date) -> dict[str, str]:
    iso = day.isocalendar()
    return {
        "date": day.isoformat(),
        "week": f"{iso.year}-W{iso.week:02d}",
        "month": f"{day:%Y-%m}",
        "year": f"{day:%Y}",
        "quarter": str((day.month - 1) // 3 + 1),
    }


def _agent_rules(root: Path, agent: str) -> str:
    """Return the `AGENTS.md` section whose heading names the agent, through the next heading of its level."""
    agents_file = root / AGENTS_FILENAME
    if not agents_file.is_file() or agents_file.is_symlink():
        return ""
    with agents_file.open(encoding="utf-8") as handle:
        text = handle.read(AGENTS_MAX_CHARS)
    headings = list(_HEADING.finditer(text))
    for index, heading in enumerate(headings):
        if heading.group(2).strip().lower() != agent:
            continue
        level = len(heading.group(1))
        end = next((later.start() for later in headings[index + 1 :] if len(later.group(1)) <= level), len(text))
        return text[heading.end() : end].strip()
    return ""


def build_session_context(active: ActiveVault) -> str:
    """Return the capped vault rules and `session_context` notes for the session instructions."""
    access = active.access
    sections: list[str] = []
    try:
        schema = load_schema(active.root)
    except VaultError as exc:
        logger.warning("Vault schema unavailable for session context: %s", exc)
    else:
        writable = [
            f"`{name}` ({rule.note_class}; required: {', '.join(rule.required) or 'none'})"
            for name, rule in schema.types.items()
            if rule.note_class != "reference" and any(covers(access.write, f"{folder}/_") for folder in rule.folders)
        ]
        summary = [
            f"Vault `{schema.vault}`. You write as `agent/{access.agent}` with the vault tools.",
            f"Read folders: {', '.join(access.read) or 'none'}.",
            f"Write folders: {', '.join(access.write) or 'none'}.",
            f"Note types you may write: {', '.join(writable) or 'none'}.",
        ]
        sections.append("### Vault rules\n\n" + "\n".join(summary))
    rules = _agent_rules(active.root, access.agent)
    if rules:
        sections.append(f"### Rules for {access.agent} (from the vault AGENTS.md)\n\n{rules}")
    for path in access.session_context:
        try:
            note = read_note(active.root, path, agent=access.agent, folders=access.read)
        except VaultError as exc:
            logger.warning("Skipping session context note %s: %s", path, exc)
            continue
        sections.append(f"### {path}\n\n{note.body.strip()}")
    if not sections:
        return ""
    context = "## Obsidian vault context\n\n" + "\n\n".join(sections)
    if len(context) > SESSION_CONTEXT_MAX_CHARS:
        context = context[:SESSION_CONTEXT_MAX_CHARS].rstrip() + "\n\n[Vault context truncated.]"
    return context


@dataclass
class VaultSession:
    """One wake session of one profile: its run id, vault context, and transcript. Reconnects keep the session."""

    session_id: str = ""
    profile: str = ""
    started_at: datetime | None = None
    context: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)

    def begin(self, instance_path: str | Path | None, now: datetime | None = None) -> None:
        """Start the session and load its vault context, unless it already runs for the active profile."""
        profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        if self.started_at is not None:
            if self.profile == profile:
                return
            # A profile change ends the old profile's session, so its turns stay in its own vault notes.
            self.end(instance_path, now)
        self.started_at = now or datetime.now().astimezone()
        self.session_id = f"{self.started_at:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.profile = profile
        self.turns = []
        self.context = ""
        try:
            self.context = build_session_context(active_vault(instance_path, profile))
        except VaultError as exc:
            logger.info("No vault context for this session: %s", exc)
        except OSError as exc:
            logger.warning("Failed to read vault context: %s", exc)

    def run_id(self, agent: str) -> str:
        """Return the contract run id `reachy:<agent>:<session-id>`."""
        return f"reachy:{agent}:{self.session_id}"

    def record(self, role: str, text: str) -> None:
        """Add one final transcript turn on one line."""
        text = " ".join(text.split())
        if self.started_at is not None and text:
            self.turns.append((role, text))

    def end(self, instance_path: str | Path | None, now: datetime | None = None) -> None:
        """Write the session log and the weekly memory of each finished week that has none; then close the session."""
        if self.started_at is None:
            return
        try:
            if any(role == "user" for role, _text in self.turns):
                active = active_vault(instance_path, self.profile)
                ended_at = now or datetime.now().astimezone()
                self._write_session_log(active)
                self._write_weekly_memories(active, ended_at)
        except (VaultError, OSError) as exc:
            logger.warning("Vault session note not written: %s", exc)
        finally:
            self.started_at = None
            self.session_id = ""
            self.profile = ""
            self.context = ""
            self.turns = []

    def _write_session_log(self, active: ActiveVault) -> None:
        target = active.access.session_log
        if target is None or self.started_at is None:
            logger.info("No session log target for personality %s", active.profile)
            return
        rule = load_schema(active.root).types.get(target.type)
        if rule is None:
            raise VaultError(f"session log type `{target.type}` is not in the vault schema")
        started = self.started_at
        # The session id suffix keeps two sessions in one minute apart; a log is never overwritten.
        suffix = self.session_id.rsplit("-", 1)[-1]
        values = {
            **_calendar_values(started.date()),
            "time": f"{started:%H%M}",
            "slug": f"{started:%H%M}-{suffix}",
            "title": f"{started:%Y-%m-%d %H%M} {suffix}",
        }
        path = f"{target.folder}/{fill_placeholders(rule.name or '{date}-{slug}', values)}.md"
        lines = [f"# {active.profile} session {started:%Y-%m-%d %H:%M}", ""]
        size = 0
        for role, text in self.turns:
            line = f"**{_USER_LABEL if role == 'user' else active.profile}:** {text}"
            size += len(line)
            if size > TRANSCRIPT_MAX_CHARS:
                lines.append("_Transcript truncated._")
                break
            lines.append(line)
        write_note(
            active.root,
            path,
            agent=active.access.agent,
            folders=active.access.write,
            run=self.run_id(active.access.agent),
            properties={
                "type": target.type,
                **{key: fill_placeholders(v, values) for key, v in target.properties.items()},
            },
            body="\n".join(lines) + "\n",
            today=started.date(),
        )
        logger.info("Wrote vault session log %s", path)

    def _write_weekly_memories(self, active: ActiveVault, ended_at: datetime) -> None:
        target = active.access.weekly_memory
        source = active.access.session_log
        if target is None or source is None:
            return
        today = ended_at.date()
        this_week = today - timedelta(days=today.weekday())
        rule = load_schema(active.root).types.get(target.type)
        if rule is None:
            raise VaultError(f"weekly memory type `{target.type}` is not in the vault schema")
        # Newest first, back to the latest week with a memory: the session end that wrote it summarized all before.
        missing: list[tuple[date, str, dict[str, str]]] = []
        for weeks_back in range(1, _WEEKLY_LOOKBACK_WEEKS + 1):
            week_start = this_week - timedelta(weeks=weeks_back)
            values = {
                **_calendar_values(week_start + timedelta(days=target.date_weekday - 1)),
                "week_start": week_start.isoformat(),
                "week_end": (week_start + timedelta(days=6)).isoformat(),
                "time": f"{ended_at:%H%M}",
            }
            values |= {"slug": values["week"].lower(), "title": values["week"]}
            path = f"{target.folder}/{fill_placeholders(rule.name or '{week}', values)}.md"
            if note_path(active.root, path).exists():
                break
            missing.append((week_start, path, values))
        if not missing:
            return
        logs, truncated = query_notes(
            active.root,
            agent=active.access.agent,
            folders=active.access.read,
            folder=source.folder,
            where={"type": source.type},
            created_from=missing[-1][0],
            created_to=this_week - timedelta(days=1),
            limit=_WEEKLY_MAX_LOGS * len(missing),
        )
        if truncated:
            logger.warning("More than %d session logs; weekly memories list the first ones only", len(logs))
        logs_by_week: dict[date, list[str]] = {}
        for log in logs:
            # The created range filter passes only `YYYY-MM-DD` text, but that text can still be an invalid date.
            try:
                created = date.fromisoformat(str(log.properties["created"]))
            except ValueError:
                logger.warning("Skipping session log %s: `created` is not a valid date", log.path)
                continue
            logs_by_week.setdefault(created - timedelta(days=created.weekday()), []).append(log.path)
        prefix = f"**{_USER_LABEL}:** "
        for week_start, path, values in reversed(missing):
            log_paths = logs_by_week.get(week_start)
            if not log_paths:
                logger.info("No session logs for week %s; weekly memory skipped", values["week"])
                continue
            lines = [f"# {active.profile} weekly memory {values['week']} (week of {week_start.isoformat()})", ""]
            for log_path in log_paths:
                note = read_note(active.root, log_path, agent=active.access.agent, folders=active.access.read)
                said = [line.removeprefix(prefix) for line in note.body.splitlines() if line.startswith(prefix)]
                lines.append(f"- [[{log_path.removesuffix('.md')}]] ({len(said)} user turns)")
                lines += [f"  - {text[:_WEEKLY_EXCERPT_CHARS]}" for text in said[:_WEEKLY_EXCERPT_TURNS]]
            write_note(
                active.root,
                path,
                agent=active.access.agent,
                folders=active.access.write,
                run=self.run_id(active.access.agent),
                properties={
                    "type": target.type,
                    **{key: fill_placeholders(v, values) for key, v in target.properties.items()},
                },
                body="\n".join(lines) + "\n",
                today=today,
            )
            logger.info("Wrote vault weekly memory %s", path)
