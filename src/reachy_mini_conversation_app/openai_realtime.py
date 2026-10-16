from openai import AsyncOpenAI
from openai.types.realtime import RealtimeSessionCreateRequestParam
from openai.types.realtime.realtime_audio_formats_param import AudioPCM

from reachy_mini_conversation_app.config import (
    OPENAI_BACKEND,
    config,
    get_openai_realtime_model,
)
from reachy_mini_conversation_app.tools.core_tools import ToolSpec
from reachy_mini_conversation_app.huggingface_realtime import OpenAICompatibleRealtimeHandler


__all__ = ["OpenAIRealtimeHandler"]


class OpenAIRealtimeHandler(OpenAICompatibleRealtimeHandler):
    """Realtime handler for the direct OpenAI Realtime API."""

    BACKEND_PROVIDER = OPENAI_BACKEND
    SAMPLE_RATE = 24000
    VOICE_UPDATE_REQUIRES_RECONNECT = True

    def _audio_format(self) -> AudioPCM:
        """Return OpenAI's native 24 kHz PCM format."""
        return AudioPCM(type="audio/pcm", rate=24000)

    def _get_model_name(self) -> str:
        """Return the validated OpenAI Realtime model."""
        return get_openai_realtime_model()

    def _get_session_config(self, tool_specs: list[ToolSpec]) -> RealtimeSessionCreateRequestParam:
        """Return the session config, letting OpenAI detect the language when it is `auto`."""
        session = super()._get_session_config(tool_specs)
        if config.REALTIME_TRANSCRIPTION_LANGUAGE == "auto":
            # OpenAI expects an ISO-639-1 code; without one it detects the language itself.
            del session["audio"]["input"]["transcription"]["language"]
        return session

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the direct OpenAI Realtime SDK client."""
        api_key = (config.OPENAI_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        self._realtime_connect_query = {}
        return AsyncOpenAI(api_key=api_key)
