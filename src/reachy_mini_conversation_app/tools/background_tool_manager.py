"""Background tool orchestrator for non-blocking tool execution.

Allows tools to run long operations asynchronously while the robot
continues conversing. Tools can be tracked, cancelled, and their
completion is announced vocally via a silent notification queue.
"""

import time
import asyncio
import logging
from typing import Any, Callable, Coroutine

from pydantic import Field, BaseModel, PrivateAttr

from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from reachy_mini_conversation_app.tools.tool_constants import SYSTEM_TOOL_NAMES, ToolState


logger = logging.getLogger(__name__)


class ToolProgress(BaseModel):
    """Progress of a background tool."""

    progress: float = Field(..., ge=0.0, le=1.0)
    message: str | None = None


class ToolCallRoutine(BaseModel):
    """A deferred tool call: the tool name, its JSON arguments and the tool dependencies."""

    model_config = {"arbitrary_types_allowed": True}

    tool_name: str
    args_json_str: str
    deps: ToolDependencies

    async def __call__(self, tool_manager: "BackgroundToolManager") -> dict[str, Any]:
        """Dispatch the tool call."""
        if self.tool_name in SYSTEM_TOOL_NAMES:
            # Only system tools get the manager, so other tools cannot inspect or cancel tasks.
            return await dispatch_tool_call_with_manager(
                tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps, tool_manager=tool_manager
            )
        return await dispatch_tool_call(tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps)


class ToolNotification(BaseModel):
    """Notification payload for completed tools."""

    id: str
    tool_name: str
    is_idle_tool_call: bool
    status: ToolState
    result: dict[str, Any] | None = None
    error: str | None = None


class BackgroundTool(ToolNotification):
    """Represents a background tool."""

    progress: ToolProgress | None = None
    started_at: float = Field(default_factory=time.monotonic)
    completed_at: float | None = None
    _task: asyncio.Task[None] | None = PrivateAttr(default=None)

    @property
    def tool_id(self) -> str:
        """Return a unique composite identifier for this tool instance."""
        return f"{self.tool_name}-{self.id}-{self.started_at}"

    def get_notification(self) -> ToolNotification:
        """Get the notification for the tool."""
        return ToolNotification(
            id=self.id,
            tool_name=self.tool_name,
            is_idle_tool_call=self.is_idle_tool_call,
            status=self.status,
            result=self.result,
            error=self.error,
        )


class BackgroundToolManager(BaseModel):
    """Run tool calls as background tasks, track their state and deliver completion notifications."""

    _tools: dict[str, BackgroundTool] = PrivateAttr(default_factory=dict)
    _notification_queue: asyncio.Queue[ToolNotification] = PrivateAttr(default_factory=asyncio.Queue)
    _lifecycle_tasks: list[asyncio.Task[None]] = PrivateAttr(default_factory=list)
    _max_tool_duration_seconds: float = PrivateAttr(default=86400)  # 1 day
    _max_tool_memory_seconds: float = PrivateAttr(default=3600)  # 1 hour

    async def start_tool(
        self,
        call_id: str,
        tool_call_routine: ToolCallRoutine,
        is_idle_tool_call: bool,
        with_progress: bool = False,
    ) -> BackgroundTool:
        """Register a tool call and run it as a background task."""
        tool_name = tool_call_routine.tool_name
        background_tool = BackgroundTool(
            id=call_id,
            tool_name=tool_name,
            is_idle_tool_call=is_idle_tool_call,
            progress=ToolProgress(progress=0.0) if with_progress else None,
            status=ToolState.RUNNING,
        )
        self._tools[background_tool.tool_id] = background_tool

        background_tool._task = asyncio.create_task(
            self._run_tool(background_tool, tool_call_routine),
            name=f"bg-{tool_name}-{call_id}",
        )

        logger.info("Started background tool: %s (id=%s)", tool_name, call_id)

        return background_tool

    async def _run_tool(
        self,
        background_tool: BackgroundTool,
        tool_call_routine: ToolCallRoutine,
    ) -> None:
        """Execute the tool and handle completion."""
        result = await tool_call_routine(self)
        background_tool.completed_at = time.monotonic()
        error = result.get("error")

        if error is not None:
            if error == "Tool cancelled":
                background_tool.status = ToolState.CANCELLED
                logger.debug("Background tool cancelled: %s (id=%s)", background_tool.tool_name, background_tool.id)
            else:
                background_tool.status = ToolState.FAILED
                logger.debug(
                    "Background tool failed: %s (id=%s): %s",
                    background_tool.tool_name,
                    background_tool.id,
                    background_tool.error,
                )
            background_tool.error = error

        else:
            background_tool.result = result
            background_tool.status = ToolState.COMPLETED
            logger.debug("Background tool completed: %s (id=%s)", background_tool.tool_name, background_tool.id)

        await self._notification_queue.put(background_tool.get_notification())
        logger.debug("Queued notification for tool: %s (id=%s)", background_tool.tool_name, background_tool.id)

    async def update_progress(
        self,
        tool_id: str,
        progress: float,
        message: str | None = None,
    ) -> bool:
        """Set a progress-tracking tool's progress, clamped to [0, 1]; return False if it is unknown or untracked."""
        tool = self._tools.get(tool_id)
        if tool is None or tool.progress is None:
            return False

        tool.progress = ToolProgress(progress=max(0.0, min(1.0, progress)), message=message)
        logger.debug("Tool %s progress: %.1f%% - %s", tool_id, progress * 100, message or "")
        return True

    async def cancel_tool(self, tool_id: str, log: bool = True) -> bool:
        """Cancel a running tool; return False only when the tool is unknown or has no task."""
        tool = self._tools.get(tool_id)
        if tool is None:
            if log:
                logger.warning("Cannot cancel tool %s: not found", tool_id)
            return False

        if tool.status != ToolState.RUNNING:
            if log:
                logger.warning("Cannot cancel tool %s: status is %s", tool_id, tool.status.value)
            return True

        if tool._task:
            tool._task.cancel()
            if log:
                logger.info("Cancelled tool: %s (id=%s)", tool.tool_name, tool_id)
            return True

        return False

    def start_up(self, tool_callbacks: list[Callable[[ToolNotification], Coroutine[Any, Any, None]]]) -> None:
        """Start the task that passes notifications to the callbacks and the periodic cleanup and timeout task."""

        async def _listener() -> None:
            while True:
                background_tool = await self._notification_queue.get()
                for callback in tool_callbacks:
                    await callback(background_tool)

        async def _cleanup() -> None:
            while True:
                await asyncio.sleep(5 * 60)
                await self.cleanup_tools()
                await self.timeout_tools()

        self._lifecycle_tasks = [
            asyncio.create_task(_cleanup(), name="bg-tool-cleanup"),
            asyncio.create_task(_listener(), name="bg-tool-listener-callback"),
        ]

        logger.info(
            "BackgroundToolManager started. "
            "Max tool execution duration: %s seconds (tools running longer will be auto-cancelled). "
            "Max tool memory retention: %s seconds (completed/failed/cancelled tools older than this are purged).",
            self._max_tool_duration_seconds,
            self._max_tool_memory_seconds,
        )

    async def shutdown(self) -> None:
        """Cancel all background tasks (listener, cleanup) and running tools."""
        for task in self._lifecycle_tasks:
            task.cancel()
        for task in self._lifecycle_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._lifecycle_tasks.clear()

        for tool_id in list(self._tools):
            await self.cancel_tool(tool_id, log=False)

        logger.info("BackgroundToolManager shut down")

    async def timeout_tools(self) -> int:
        """Cancel tools running longer than the max duration and return how many."""
        now = time.monotonic()
        to_cancel = [
            tool_id
            for tool_id, tool in self._tools.items()
            if tool.status == ToolState.RUNNING
            and tool.started_at
            and now - tool.started_at > self._max_tool_duration_seconds
        ]

        for tool_id in to_cancel:
            await self.cancel_tool(tool_id)

        if to_cancel:
            logger.debug("Timed out %d tools", len(to_cancel))

        return len(to_cancel)

    async def cleanup_tools(self) -> int:
        """Remove finished tools older than the memory retention window and return how many."""
        now = time.monotonic()
        to_remove = [
            tool_id
            for tool_id, tool in self._tools.items()
            if tool.status in (ToolState.COMPLETED, ToolState.FAILED, ToolState.CANCELLED)
            and tool.completed_at
            and now - tool.completed_at > self._max_tool_memory_seconds
        ]

        for tool_id in to_remove:
            del self._tools[tool_id]

        if to_remove:
            logger.debug("Cleaned up %d old tools", len(to_remove))

        return len(to_remove)

    def get_tool(self, tool_id: str) -> BackgroundTool | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def get_running_tools(self) -> list[BackgroundTool]:
        """Get all currently running tools."""
        return [t for t in self._tools.values() if t.status == ToolState.RUNNING]

    def get_all_tools(self, limit: int | None = None) -> list[BackgroundTool]:
        """Return tracked tools, most recent first, up to ``limit`` when given."""
        sorted_tools = sorted(
            self._tools.values(),
            key=lambda t: t.started_at,
            reverse=True,
        )
        if limit is not None:
            return sorted_tools[:limit]
        return sorted_tools
