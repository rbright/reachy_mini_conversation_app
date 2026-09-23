import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.vault import VaultError, read_note
from reachy_mini_conversation_app.vault_session import active_vault
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class VaultRead(Tool):
    """Read one note from the synced Obsidian vault."""

    name = "vault_read"
    description = (
        "Read one note from the Obsidian vault. Returns its properties and body (long bodies are cut). "
        "You can only read the folders this personality is allowed to read."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Vault-relative note path ending in .md, for example a path returned by vault_query.",
            },
        },
        "required": ["path"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Read one allowed note."""
        path = kwargs.get("path")
        if not isinstance(path, str) or not path.strip():
            return {"error": "path must be a non-empty string"}
        try:
            active = await asyncio.to_thread(active_vault, deps.instance_path)
            note = await asyncio.to_thread(
                read_note, active.root, path.strip(), agent=active.access.agent, folders=active.access.read
            )
        except (VaultError, OSError) as exc:
            logger.warning("vault_read failed for %s: %s", path, exc)
            return {"error": str(exc)}
        return {"path": note.path, "properties": note.properties, "body": note.body, "truncated": note.truncated}
