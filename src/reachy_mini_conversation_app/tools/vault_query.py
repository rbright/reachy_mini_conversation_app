import asyncio
import logging
from typing import Any
from datetime import date

from pydantic import JsonValue

from reachy_mini_conversation_app.vault import VaultError, query_notes
from reachy_mini_conversation_app.vault_session import active_vault
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_MAX_LIMIT = 200


class VaultQuery(Tool):
    """List notes in the synced Obsidian vault by their properties."""

    name = "vault_query"
    description = (
        "List notes in the Obsidian vault that this personality may read. Filter by folder, note type, status, "
        "and created date range. Returns paths and properties only; use vault_read to read a note."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Vault-relative folder to search, including subfolders."},
            "type": {"type": "string", "description": "Only notes with this `type` property."},
            "status": {"type": "string", "description": "Only notes with this `status` property."},
            "created_from": {"type": "string", "description": "Earliest `created` date, YYYY-MM-DD."},
            "created_to": {"type": "string", "description": "Latest `created` date, YYYY-MM-DD."},
            "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT, "description": "Maximum results."},
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """List allowed notes that match the filters."""
        where: dict[str, JsonValue] = {}
        for key in ("type", "status"):
            value = kwargs.get(key)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    return {"error": f"{key} must be a non-empty string"}
                where[key] = value.strip()
        folder = kwargs.get("folder")
        if folder is not None and not isinstance(folder, str):
            return {"error": "folder must be a string"}
        limit = kwargs.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_LIMIT:
            return {"error": f"limit must be an integer from 1 to {_MAX_LIMIT}"}
        dates: dict[str, date | None] = {}
        for key in ("created_from", "created_to"):
            value = kwargs.get(key)
            try:
                dates[key] = None if value is None else date.fromisoformat(str(value))
            except ValueError:
                return {"error": f"{key} must be a date in YYYY-MM-DD form"}
        try:
            active = await asyncio.to_thread(active_vault, deps.instance_path)
            notes, truncated = await asyncio.to_thread(
                query_notes,
                active.root,
                agent=active.access.agent,
                folders=active.access.read,
                folder=folder,
                where=where,
                created_from=dates["created_from"],
                created_to=dates["created_to"],
                limit=limit,
            )
        except (VaultError, OSError) as exc:
            logger.warning("vault_query failed: %s", exc)
            return {"error": str(exc)}
        return {
            "notes": [{"path": note.path, "properties": note.properties} for note in notes],
            "truncated": truncated,
        }
