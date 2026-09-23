import asyncio
import logging
from typing import Any
from datetime import datetime

from reachy_mini_conversation_app.vault import VaultError, write_note
from reachy_mini_conversation_app.vault_session import active_vault
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class VaultWrite(Tool):
    """Write one note to the synced Obsidian vault under the vault contract."""

    name = "vault_write"
    description = (
        "Create a note in the Obsidian vault, or update a draft artifact or a record that this agent owns, in a "
        "folder this personality may write. An update replaces the whole body and merges the given properties; "
        "to revise your own draft, write to its path again instead of creating a new note. The note type and "
        "properties must follow the vault schema. The tool sets created, author, and run. It refuses dated notes "
        "and logs that exist, approved notes, reference notes, and anything outside the allowed folders."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative note path ending in .md."},
            "type": {"type": "string", "description": "Note type from the vault schema."},
            "properties": {
                "type": "object",
                "description": "Other frontmatter properties from the vault schema, for example a date key.",
            },
            "body": {"type": "string", "description": "Markdown body of the note."},
        },
        "required": ["path", "type", "body"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Create or update one allowed note."""
        path = kwargs.get("path")
        note_type = kwargs.get("type")
        body = kwargs.get("body")
        properties = kwargs.get("properties") or {}
        if not isinstance(path, str) or not path.strip():
            return {"error": "path must be a non-empty string"}
        if not isinstance(note_type, str) or not note_type.strip():
            return {"error": "type must be a non-empty string"}
        if not isinstance(body, str):
            return {"error": "body must be a string"}
        if not isinstance(properties, dict) or "type" in properties:
            return {"error": "properties must be an object without `type`"}
        session = deps.vault_session
        if not session.session_id:
            return {"error": "no conversation session is active"}
        try:
            active = await asyncio.to_thread(active_vault, deps.instance_path)
            created = await asyncio.to_thread(
                write_note,
                active.vault,
                path.strip(),
                agent=active.access.agent,
                folders=active.access.write,
                run=session.run_id(active.access.agent),
                properties={"type": note_type.strip(), **properties},
                body=body,
                today=datetime.now().astimezone().date(),
            )
        except (VaultError, OSError) as exc:
            logger.warning("vault_write refused %s: %s", path, exc)
            return {"error": str(exc)}
        logger.info("Tool call: vault_write path=%s created=%s", path, created)
        return {"path": path.strip(), "created": created}
