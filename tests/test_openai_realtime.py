import json
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as openai_mod
import reachy_mini_conversation_app.huggingface_realtime as realtime_mod
from reachy_mini_conversation_app.config import OPENAI_DEFAULT_VOICE, config
from reachy_mini_conversation_app.openai_realtime import OpenAIRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


class _FakeRealtimeClient:
    def __init__(self, captured_connect: dict[str, Any], captured_update: dict[str, Any]) -> None:
        class Session:
            async def update(inner_self, **kwargs: Any) -> None:
                captured_update.update(kwargs)

        class Noop:
            async def append(inner_self, **_kwargs: Any) -> None:
                pass

            async def create(inner_self, **_kwargs: Any) -> None:
                pass

        class Connection:
            session = Session()
            input_audio_buffer = Noop()
            conversation = SimpleNamespace(item=Noop())
            response = Noop()

            async def __aenter__(inner_self) -> "Connection":
                return inner_self

            async def __aexit__(inner_self, *_args: Any) -> bool:
                return False

            def __aiter__(inner_self) -> "Connection":
                return inner_self

            async def __anext__(inner_self) -> Any:
                raise StopAsyncIteration

            async def close(inner_self) -> None:
                pass

        class Realtime:
            def connect(inner_self, **kwargs: Any) -> Connection:
                captured_connect.update(kwargs)
                return Connection()

        self.realtime = Realtime()


def _handler() -> OpenAIRealtimeHandler:
    return OpenAIRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


@pytest.mark.asyncio
async def test_openai_session_uses_direct_model_audio_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct OpenAI sessions send the selected model, native audio, and app tools."""
    tool_specs = [
        {
            "name": "camera",
            "description": "Look through the robot camera.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        }
    ]
    captured_connect: dict[str, Any] = {}
    captured_update: dict[str, Any] = {}
    monkeypatch.setattr(config, "OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    monkeypatch.setattr(realtime_mod, "get_tool_specs", lambda: tool_specs)
    monkeypatch.setattr(realtime_mod, "get_session_instructions", lambda _instance_path=None: "Be concise.")
    monkeypatch.setattr(realtime_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: default)
    monkeypatch.setattr(realtime_mod, "get_session_greeting_prompt", lambda: "")

    handler = _handler()
    handler.client = _FakeRealtimeClient(captured_connect, captured_update)  # type: ignore[assignment]
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert captured_connect == {"model": "gpt-realtime-2.1"}
    session = captured_update["session"]
    assert session["model"] == "gpt-realtime-2.1"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == OPENAI_DEFAULT_VOICE
    assert session["tools"][0]["name"] == "camera"


@pytest.mark.asyncio
async def test_openai_camera_tool_result_attaches_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI function-call results attach camera images without echoing base64 in tool output."""
    handler = _handler()
    created_items: list[dict[str, Any]] = []

    async def create_item(*, item: dict[str, Any]) -> None:
        created_items.append(item)

    handler.connection = MagicMock()
    handler.connection.conversation.item.create = create_item
    handler.output_queue = asyncio.Queue()
    handler._in_flight_tool_calls = {"camera-call"}
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_safe_response_create", AsyncMock())
    monkeypatch.setattr(realtime_mod.core_tools, "get_tools", lambda: {})

    await handler._handle_tool_result(
        ToolNotification(
            id="camera-call",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"seen": "cat", "b64_im": "aW1hZ2U=", "image_width": 2, "image_height": 2},
        )
    )

    function_output = json.loads(created_items[0]["output"])
    assert function_output == {"seen": "cat", "image_width": 2, "image_height": 2, "image_attached": True}
    assert created_items[1]["content"][0]["image_url"] == "data:image/jpeg;base64,aW1hZ2U="


@pytest.mark.asyncio
async def test_openai_client_requires_and_uses_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct client refuses missing credentials and passes a configured key only to the SDK."""
    handler = _handler()
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
        await handler._build_realtime_client()

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(config, "OPENAI_API_KEY", "replacement-secret")
    monkeypatch.setattr(openai_mod, "AsyncOpenAI", FakeClient)

    await handler._build_realtime_client()

    assert captured == {"api_key": "replacement-secret"}


@pytest.mark.asyncio
async def test_openai_voice_change_reconnects_instead_of_updating_locked_session() -> None:
    """Changing an OpenAI voice closes the session because emitted-session voices are immutable."""
    handler = _handler()
    handler.connection = MagicMock()
    handler.connection.close = AsyncMock()
    handler.connection.session.update = AsyncMock()

    message = await handler.change_voice("cedar")

    assert message == "Voice changed to cedar. Reconnecting."
    assert handler.get_current_voice() == "cedar"
    handler.connection.close.assert_awaited_once_with()
    handler.connection.session.update.assert_not_awaited()
