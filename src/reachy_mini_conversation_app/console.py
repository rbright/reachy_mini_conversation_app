"""Bidirectional local audio stream with optional web settings UI.

If the selected backend is missing its required API key, a settings page is
served via the Reachy Mini Apps settings server so users can configure it.
"""

import os
import re
import time
import asyncio
import logging
import contextlib
from math import gcd
from typing import Any, List, Optional
from pathlib import Path
from collections.abc import Callable

import numpy as np
from dotenv import dotenv_values
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter

from reachy_mini import ReachyMini
from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app import obsidian_sync
from reachy_mini_conversation_app.config import (
    HF_BACKEND,
    LOCKED_PROFILE,
    OPENAI_BACKEND,
    OPENAI_API_KEY_ENV,
    OBSIDIAN_SYNC_MODES,
    BACKEND_PROVIDER_ENV,
    HF_REALTIME_WS_URL_ENV,
    OBSIDIAN_SYNC_MODE_ENV,
    OBSIDIAN_SYNC_PATH_ENV,
    OPENAI_REALTIME_MODELS,
    OBSIDIAN_SYNC_VAULT_ENV,
    HF_LOCAL_CONNECTION_MODE,
    OBSIDIAN_HEADLESS_BIN_ENV,
    OBSIDIAN_SYNC_ENABLED_ENV,
    OPENAI_REALTIME_MODEL_ENV,
    HF_DEPLOYED_CONNECTION_MODE,
    OBSIDIAN_SYNC_DEVICE_NAME_ENV,
    HF_REALTIME_CONNECTION_MODE_ENV,
    OBSIDIAN_SYNC_E2EE_PASSWORD_ENV,
    OBSIDIAN_SYNC_CONFLICT_STRATEGIES,
    OBSIDIAN_SYNC_CONFLICT_STRATEGY_ENV,
    config,
    get_default_voice,
    get_backend_choice,
    get_hf_session_url,
    has_openai_api_key,
    set_custom_profile,
    get_available_voices,
    get_hf_direct_ws_url,
    build_hf_direct_ws_url,
    has_hf_realtime_target,
    parse_hf_direct_target,
    get_openai_realtime_model,
    get_hf_connection_selection,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_float32
from reachy_mini_conversation_app.wake_word import (
    WAKE_WORD_SAMPLE_RATE,
    WakeWordEvent,
    WakeWordDetector,
    matches_sleep_phrase,
)
from reachy_mini_conversation_app.obsidian_sync import ObsidianSyncError
from reachy_mini_conversation_app.settings_auth import SettingsPinGuard, SettingsRpcServer
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings
from reachy_mini_conversation_app.tools.core_tools import initialize_tools
from reachy_mini_conversation_app.tool_space_routes import register_tool_space_methods
from reachy_mini_conversation_app.personality_routes import (
    build_personality_ops,
    register_personality_methods,
)
from reachy_mini_conversation_app.profile_tool_routes import register_profile_tool_methods
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore

logger = logging.getLogger(__name__)


def _detach_framework_root_routes(app: "FastAPI") -> None:
    """Strip framework routes that would shadow the settings UI."""
    routes = getattr(app, "router", None)
    routes = getattr(routes, "routes", None) if routes else getattr(app, "routes", None)
    if routes is None:
        return
    survivors = []
    for route in routes:
        path = getattr(route, "path", None)
        is_catch_all = isinstance(path, str) and path.startswith("/{") and path.endswith(":path}")
        if path in ("/", "/static") or is_catch_all:
            logger.debug("detaching framework-provided route %r (%s)", path, type(route).__name__)
            continue
        survivors.append(route)
    routes[:] = survivors


LOCAL_PLAYER_BACKEND = (
    getattr(MediaBackend, "LOCAL", None)
    or getattr(MediaBackend, "GSTREAMER", None)
    or getattr(MediaBackend, "DEFAULT", None)
)

HandlerFactory = Callable[[Optional[str]], ConversationHandler]

LEGACY_STARTUP_ENV_NAMES = (
    "REACHY_MINI_CUSTOM_PROFILE",
    "REACHY_MINI_VOICE_OVERRIDE",
)
BACKEND_RETRY_DELAY_SECONDS = 5.0
# Values outside this set are single-quoted in `.env` so that python-dotenv keeps `#`, spaces, and `$` literally.
_PLAIN_ENV_VALUE = re.compile(r"[A-Za-z0-9_./:@+,=-]*")


class _StreamingResampler:
    """Resample consecutive audio chunks without introducing boundary artifacts."""

    def __init__(self, source_sample_rate: int, target_sample_rate: int) -> None:
        if source_sample_rate <= 0 or target_sample_rate <= 0:
            raise ValueError("Audio sample rates must be positive")

        common_divisor = gcd(source_sample_rate, target_sample_rate)
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self._up = target_sample_rate // common_divisor
        self._down = source_sample_rate // common_divisor
        max_rate = max(self._up, self._down)
        self._taps = np.asarray(
            firwin(20 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self._up,
            dtype=np.float32,
        )
        self._filter_state: NDArray[np.float32] | None = None
        self._sample_shape: tuple[int, ...] | None = None
        self._downsample_phase = 0

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample the next contiguous audio chunk."""
        if audio.size == 0:
            return audio

        sample_axis = 1 if audio.ndim == 2 and audio.shape[1] > audio.shape[0] else 0
        samples_first = np.moveaxis(audio, sample_axis, 0)
        sample_shape = samples_first.shape[1:]
        if self._sample_shape is not None and sample_shape != self._sample_shape:
            raise ValueError("Audio channel shape changed while resampling")
        self._sample_shape = sample_shape

        upsampled = np.zeros((samples_first.shape[0] * self._up, *sample_shape), dtype=np.float32)
        upsampled[:: self._up] = samples_first
        if self._filter_state is None:
            self._filter_state = np.zeros((self._taps.size - 1, *sample_shape), dtype=np.float32)

        filtered, self._filter_state = lfilter(
            self._taps,
            [1.0],
            upsampled,
            axis=0,
            zi=self._filter_state,
        )
        first_sample = (-self._downsample_phase) % self._down
        self._downsample_phase = (self._downsample_phase + upsampled.shape[0]) % self._down
        resampled = np.asarray(filtered[first_sample :: self._down], dtype=np.float32)
        return np.moveaxis(resampled, 0, sample_axis)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: ConversationHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
        handler_factory: HandlerFactory | None = None,
        startup_voice: Optional[str] = None,
        wake_word_detector: WakeWordDetector | None = None,
        sleep_phrases: tuple[str, ...] = (),
        on_wake_word: Callable[[], None] | None = None,
        on_sleep_phrase: Callable[[], None] | None = None,
    ):
        """Initialize the stream with a realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        - ``handler_factory``: builds a fresh handler for the currently selected backend.
        """
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._restart_requested = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self._handler_factory = handler_factory
        self._voice_override = startup_voice
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None
        self._mic_muted = False  # mic starts live; the UI toggles it via the settings API
        self._wake_word_detector = wake_word_detector
        self._sleep_phrases = sleep_phrases
        self._on_wake_word = on_wake_word
        self._on_sleep_phrase = on_sleep_phrase
        self._wake_gate_open = wake_word_detector is None
        self._wake_gate_event = asyncio.Event()
        self._conversation_ready = asyncio.Event()
        if self._wake_gate_open:
            self._wake_gate_event.set()
            self._conversation_ready.set()
        self._sleep_transition_complete = asyncio.Event()
        self._sleep_transition_complete.set()
        self._wake_handler_ready = asyncio.Event()
        self._wake_handler_ready.set()
        self._active_backend_name = get_backend_choice()
        self._backend_connection_state = "not_started"
        self._backend_error: str | None = None
        self._backend_retry_delay = BACKEND_RETRY_DELAY_SECONDS
        # JSON-RPC control surface (mounted at /rpc in _init_settings_ui_if_needed).
        # Notifications (conversation.turn/phase/transcript/activity) are pushed
        # here from activity + transcripts. Survives handler rebuilds (mounted once).
        self._rpc: Optional[SettingsRpcServer] = None
        self._last_turn_state: Optional[str] = None
        # Per-role throttle timestamps for conversation.level (orb audio meter).
        self._last_level_emit: dict[str, float] = {}
        self._install_handler(handler)

    def _install_handler(self, handler: ConversationHandler) -> None:
        """Set the active handler and wire LocalStream-owned helpers into it."""
        self.handler = handler
        self.handler._clear_queue = self.clear_audio_queue
        self._attach_observers_to_handler()

    def _attach_observers_to_handler(self) -> None:
        """Wire the handler's activity + transcript observers to JSON-RPC pushes."""
        setter = getattr(self.handler, "set_activity_observer", None)
        if callable(setter):
            setter(self._dispatch_activity)
        transcript_setter = getattr(self.handler, "set_transcript_observer", None)
        if callable(transcript_setter):
            transcript_setter(self._dispatch_transcript)
        command_setter = getattr(self.handler, "set_transcript_command_handler", None)
        if callable(command_setter):
            command_setter(self._handle_transcript_command)

    def _dispatch_transcript(self, role: str, text: str, final: bool) -> None:
        """Push a conversation.transcript notification to JSON-RPC clients."""
        if self._rpc is not None:
            self._rpc.broadcast_threadsafe(
                "conversation.transcript",
                {"role": role, "text": text, "final": final},
            )

    # Audio level meter for the client orb. RMS is scaled into a visible 0..1
    # range and capped to ~15 Hz so it stays light on the DataChannel.
    _LEVEL_INTERVAL_S = 1.0 / 15.0
    _LEVEL_GAIN = 6.0

    def _emit_level(self, role: str, frame: Any) -> None:
        """Emit a throttled conversation.level (RMS) for ``role`` (user/assistant)."""
        if self._rpc is None:
            return
        now = time.monotonic()
        if now - self._last_level_emit.get(role, 0.0) < self._LEVEL_INTERVAL_S:
            return
        self._last_level_emit[role] = now
        try:
            samples = audio_to_float32(frame)
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        except Exception:
            return
        level = max(0.0, min(1.0, rms * self._LEVEL_GAIN))
        self._rpc.broadcast_threadsafe("conversation.level", {"role": role, "rms": round(level, 3)})

    # Map backend activity reasons to the orb's turn states (mirrors the old
    # browser orb's mapActivityToState so the orb reliably reaches listening/
    # thinking/speaking across the reasons the HF handler actually emits).
    # Transcript text is delivered separately via the transcript observer.
    _REASON_TO_TURN = {
        "user_speech_started": "listening",
        "user_transcription_delta": "listening",
        "user_speech_stopped": "thinking",
        "user_transcription_completed": "thinking",
        "response_created": "thinking",
        "tool_call_received": "thinking",
        "tool_result_ready": "thinking",
        "assistant_audio_delta": "speaking",
        "assistant_transcript_done": "ready",
    }

    def _dispatch_activity(self, reason: str) -> None:
        """Fan one activity reason out to JSON-RPC clients."""
        if self._rpc is not None:
            # Raw reason (the browser orb maps it exactly like the old SSE feed)...
            self._rpc.broadcast_threadsafe("conversation.activity", {"reason": reason})
            # ...plus a semantic turn state for clients without that mapping (mobile).
            state = self._REASON_TO_TURN.get(reason)
            if state and state != self._last_turn_state:
                self._last_turn_state = state
                self._rpc.broadcast_threadsafe("conversation.turn", {"state": state})

    def _emit_phase(self, phase: str, reason: Optional[str] = None) -> None:
        """Push a conversation.phase notification to JSON-RPC clients."""
        if self._rpc is not None:
            self._rpc.broadcast_threadsafe("conversation.phase", {"phase": phase, "reason": reason})

    def _handle_transcript_command(self, transcript: str) -> bool:
        """Consume configured sleep phrases before the realtime model responds."""
        if self._wake_word_detector is None or not self._wake_gate_open:
            return False
        if not matches_sleep_phrase(transcript, self._sleep_phrases):
            return False

        logger.info("Sleep phrase detected")
        self._wake_gate_open = False
        self._wake_gate_event.clear()
        self._conversation_ready.clear()
        self._sleep_transition_complete.clear()
        self._wake_handler_ready.clear()
        self._restart_requested.set()
        self.clear_audio_queue()
        asyncio.create_task(self._enter_sleep_mode(), name="wake-word-sleep")
        return True

    async def _enter_sleep_mode(self) -> None:
        """Close the active conversation and move the robot to sleep."""
        try:
            await self._shutdown_active_handler()
            await asyncio.to_thread(self.handler.deps.vault_session.end, self._instance_path)
            if self._on_sleep_phrase is not None:
                try:
                    await asyncio.to_thread(self._on_sleep_phrase)
                except Exception:
                    logger.exception("Failed to put Reachy Mini to sleep after a sleep phrase")
            self._emit_phase("sleeping", "sleep_phrase")
        finally:
            self._sleep_transition_complete.set()

    async def _handle_wake_event(self, event: WakeWordEvent) -> None:
        """Wake the robot and open a fresh realtime conversation."""
        if self._wake_gate_open:
            return

        await self._sleep_transition_complete.wait()
        logger.info("Wake word detected via %s (%.2f)", event.model, event.score)
        if self._on_wake_word is not None:
            try:
                await asyncio.to_thread(self._on_wake_word)
            except Exception:
                logger.exception("Failed to wake Reachy Mini after wake-word detection")
                return

        self._wake_gate_event.set()
        await self._wake_handler_ready.wait()
        self._wake_gate_open = True
        self._conversation_ready.set()
        self._emit_phase("ready", "wake_word")

    async def _disable_wake_word_gate(self) -> None:
        """Restore always-on behavior after a detector failure."""
        await self._sleep_transition_complete.wait()
        if self._on_wake_word is not None:
            try:
                await asyncio.to_thread(self._on_wake_word)
            except Exception:
                logger.exception("Failed to restore the active robot state after disabling wake gating")
                return
        self._wake_word_detector = None

        self._wake_gate_event.set()
        await self._wake_handler_ready.wait()
        self._wake_gate_open = True
        self._conversation_ready.set()

    def seconds_since_activity(self) -> float:
        """Seconds since the live handler last saw conversation activity."""
        return time.monotonic() - self.handler.last_activity_time

    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    @staticmethod
    def _has_required_configuration(backend: str) -> bool:
        """Return whether a provider can start without more configuration."""
        if backend == OPENAI_BACKEND:
            return has_openai_api_key()
        return has_hf_realtime_target()

    @staticmethod
    def _requirement_name(backend: str) -> str:
        """Return the missing provider configuration name."""
        if backend == OPENAI_BACKEND:
            return OPENAI_API_KEY_ENV
        return HF_REALTIME_WS_URL_ENV

    def _backend_connected(self) -> bool:
        """Return whether the active handler currently has a realtime connection."""
        try:
            handler_state = vars(self.handler)
        except TypeError:
            handler_state = {}
        return handler_state.get("connection") is not None

    def _can_rebuild_handler(self) -> bool:
        """Return whether LocalStream can construct handlers for backend changes."""
        return self._handler_factory is not None

    def _build_handler_for_current_backend(self) -> ConversationHandler:
        """Create and install a fresh handler for the current runtime backend config."""
        if self._handler_factory is None:
            return self.handler
        selected_backend = get_backend_choice()
        if selected_backend != self._active_backend_name:
            self._voice_override = None
        handler = self._handler_factory(self._voice_override)
        self._install_handler(handler)
        self._active_backend_name = selected_backend
        return handler

    async def _shutdown_active_handler(self) -> None:
        """Best-effort shutdown for the currently active handler."""
        try:
            await self.handler.shutdown()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Active handler shutdown ignored during restart: %s", e)

    async def request_backend_restart(self, reason: str, *, rebuild: bool = True) -> None:
        """Stop the current handler and ask the stream loop to rebuild the backend, unless `rebuild` is false."""
        loop = self._asyncio_loop
        if loop is not None and loop.is_running() and asyncio.get_running_loop() is not loop:
            future = asyncio.run_coroutine_threadsafe(self.request_backend_restart(reason, rebuild=rebuild), loop)
            await asyncio.wrap_future(future)
            return

        logger.info("Backend restart requested: %s", reason)
        self._set_backend_connection_state("connecting")
        if rebuild:
            if self._wake_word_detector is not None and not self._wake_gate_open:
                self._wake_handler_ready.clear()
            self._restart_requested.set()
        await self._shutdown_active_handler()

    async def _sleep_or_restart_requested(self, delay: float) -> None:
        """Sleep for a retry interval, waking early if a restart is requested."""
        if self._restart_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._restart_requested.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    def _format_backend_error(self, error: BaseException | str) -> str:
        """Return a compact user-facing backend error without credential values."""
        message = error if isinstance(error, str) else str(error).strip()
        api_key = (config.OPENAI_API_KEY or "").strip()
        if api_key:
            message = message.replace(api_key, "[redacted]")
        if isinstance(error, str):
            return message
        if message:
            return f"{type(error).__name__}: {message}"
        return type(error).__name__

    def _set_backend_connection_state(self, state: str, error: BaseException | str | None = None) -> None:
        """Update backend connection status exposed through the settings UI."""
        self._backend_connection_state = state
        if error is not None:
            self._backend_error = self._format_backend_error(error)
        elif state != "disconnected":
            self._backend_error = None

    def _backend_connection_status(self) -> dict[str, object]:
        """Return the backend connection state exposed in the settings API."""
        connected = self._backend_connected()
        state = "connected" if connected else self._backend_connection_state
        return {
            "backend_connected": connected,
            "backend_connection_state": state,
            "backend_error": None if connected else self._backend_error,
        }

    def _persist_env_values(self, updates: dict[str, str]) -> None:
        """Persist non-empty environment values in memory and in the instance `.env`."""
        normalized_updates = {name: (value or "").strip() for name, value in updates.items()}
        normalized_updates = {name: value for name, value in normalized_updates.items() if value}
        if not normalized_updates:
            return

        for env_name, value in normalized_updates.items():
            try:
                os.environ[env_name] = value
            except Exception:
                pass
        refresh_runtime_config_from_env()

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            for env_name, value in normalized_updates.items():
                if not _PLAIN_ENV_VALUE.fullmatch(value):
                    value = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
                replaced = False
                for i, ln in enumerate(lines):
                    if ln.strip().startswith(f"{env_name}="):
                        lines[i] = f"{env_name}={value}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"{env_name}={value}")
            final_text = "\n".join(lines) + "\n"
            env_path.touch(mode=0o600, exist_ok=True)
            if os.name == "posix":
                env_path.chmod(0o600)
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted %s to %s", ", ".join(sorted(normalized_updates)), env_path)

            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                pass
            refresh_runtime_config_from_env()
        except Exception as e:
            logger.warning("Failed to persist %s: %s", ", ".join(sorted(normalized_updates)), e)

    def _persist_env_values_or_raise(self, updates: dict[str, str]) -> None:
        """Persist values like `_persist_env_values`; raise `OSError` unless the instance `.env` now holds them."""
        self._persist_env_values(updates)
        env_path = Path(self._instance_path) / ".env" if self._instance_path else None
        saved = dotenv_values(env_path) if env_path is not None and env_path.is_file() else {}
        missing = sorted(name for name, value in updates.items() if saved.get(name) != value)
        if missing:
            raise OSError(f"{', '.join(missing)} not saved to the instance .env")

    def _remove_persisted_env_values(self, env_names: tuple[str, ...]) -> None:
        """Remove keys from the instance `.env` without mutating the current runtime."""
        normalized_names = tuple(sorted({name.strip() for name in env_names if name and name.strip()}))
        if not normalized_names or not self._instance_path:
            return

        env_path = Path(self._instance_path) / ".env"
        if not env_path.exists():
            return

        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
            filtered_lines = [
                line
                for line in lines
                if not any(line.strip().startswith(f"{env_name}=") for env_name in normalized_names)
            ]
            if filtered_lines == lines:
                return

            final_text = "\n".join(filtered_lines)
            if final_text:
                final_text += "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Removed %s from %s", ", ".join(normalized_names), env_path)
        except Exception as e:
            logger.warning("Failed to remove %s: %s", ", ".join(normalized_names), e)

    def _persist_personality(self, profile: Optional[str], voice_override: Optional[str] = None) -> None:
        """Persist startup profile and voice in instance-local UI settings."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        normalized_voice_override = (voice_override or "").strip() or None
        set_custom_profile(selection)

        if not self._instance_path:
            return
        try:
            write_startup_settings(
                self._instance_path,
                profile=selection,
                voice=normalized_voice_override,
            )
            self._remove_persisted_env_values(LEGACY_STARTUP_ENV_NAMES)
            logger.info("Persisted startup personality settings to %s", Path(self._instance_path))
        except Exception as e:
            logger.warning("Failed to persist startup personality settings: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read the saved startup personality from instance-local UI settings."""
        return read_startup_settings(self._instance_path).profile

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Apply a personality by updating config and restarting the active backend."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            get_session_instructions()
            get_session_voice(default=get_default_voice(get_backend_choice()))
            initialize_tools(force=True)
        except Exception:
            set_custom_profile(previous_profile)
            raise

        await self.request_backend_restart("personality_changed")
        return "Applied personality and restarting backend."

    async def get_available_voices(self) -> list[str]:
        """Return voices available for the selected provider."""
        return get_available_voices(get_backend_choice())

    def get_current_voice(self) -> str:
        """Return the current provider-supported voice override or profile voice."""
        backend = get_backend_choice()
        default_voice = get_default_voice(backend)
        voice = self._voice_override
        if not voice:
            try:
                voice = get_session_voice(default=default_voice)
            except Exception as exc:
                logger.warning("Failed to resolve the current profile voice: %s", exc)
                return default_voice

        voices_by_lowercase = {candidate.lower(): candidate for candidate in get_available_voices(backend)}
        return voices_by_lowercase.get(voice.lower(), default_voice)

    async def change_voice(self, voice: str) -> str:
        """Change the voice through the active handler without rebuilding the backend."""
        try:
            status = await self.handler.change_voice(voice)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error changing voice to %r: %s", voice, e)
            return f"Failed to change voice: {e}"

        try:
            current_voice = self.handler.get_current_voice()
            if isinstance(current_voice, str) and current_voice.strip():
                self._voice_override = current_voice
        except Exception as e:
            logger.debug("Could not sync LocalStream voice override after voice change: %s", e)
        if self._voice_override:
            self._persist_voice_override(self._voice_override)
        return status

    def _persist_voice_override(self, voice: str) -> None:
        """Persist the chosen voice as the startup voice, keeping the startup profile."""
        if not self._instance_path:
            return
        try:
            existing = read_startup_settings(self._instance_path)
            write_startup_settings(self._instance_path, profile=existing.profile, voice=voice)
        except Exception as e:
            logger.warning("Failed to persist startup voice: %s", e)

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return
        settings_app = self._settings_app

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"
        logger.info("Serving settings UI from %s", static_dir)

        # Framework pre-registers GET / and /static; strip them so our routes aren't shadowed.
        _detach_framework_root_routes(settings_app)

        if hasattr(settings_app, "mount"):
            try:
                settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                logger.exception("Failed to mount settings UI static assets")
                raise

        def _status_payload() -> dict[str, object]:
            backend_provider = get_backend_choice()
            active_backend = self._active_backend_name
            has_openai_key = has_openai_api_key()
            hf_session_url = get_hf_session_url()
            hf_ws_url = get_hf_direct_ws_url()
            hf_direct_host, hf_direct_port = parse_hf_direct_target(hf_ws_url)
            hf_connection_selection = get_hf_connection_selection()
            has_hf_connection = hf_connection_selection.has_target
            readiness_backend = backend_provider if self._can_rebuild_handler() else active_backend
            can_proceed = self._has_required_configuration(readiness_backend)
            backend_connection = self._backend_connection_status()
            return {
                "backend": backend_provider,
                "backend_provider": backend_provider,
                "active_backend": active_backend,
                "has_key": can_proceed,
                "has_openai_key": has_openai_key,
                "openai_model": get_openai_realtime_model(),
                "openai_models": list(OPENAI_REALTIME_MODELS),
                "has_hf_session_url": bool(hf_session_url),
                "has_hf_ws_url": bool(hf_ws_url),
                "has_hf_connection": has_hf_connection,
                "hf_connection_mode": hf_connection_selection.mode,
                "hf_direct_host": hf_direct_host,
                "hf_direct_port": hf_direct_port,
                "can_proceed": can_proceed,
                "can_proceed_with_openai": has_openai_key,
                "can_proceed_with_hf": has_hf_connection,
                "requires_restart": backend_provider != active_backend and not self._can_rebuild_handler(),
                **backend_connection,
            }

        # GET / -> index.html
        @settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # ── JSON-RPC control surface (/rpc) ──────────────────────────────
        # The single wire format both the local browser UI and remote WebRTC
        # clients use (the daemon relays it over the DataChannel). Notifications
        # (conversation.turn/phase/transcript/activity) are pushed from activity.
        # Privileged settings methods need the settings PIN (see settings_auth).
        rpc = SettingsRpcServer(SettingsPinGuard(self._persist_env_values_or_raise))

        @rpc.method("conversation.status")
        def _rpc_status(_params: dict[str, object]) -> dict[str, object]:
            return _status_payload()

        @rpc.method("conversation.say")
        async def _rpc_say(params: dict[str, object]) -> dict[str, object]:
            text = str(params.get("text", "")).strip()
            if not text:
                raise JsonRpcError("say requires 'text'", reason="invalid_params", code=-32602)
            if not self.handler._is_connected():
                raise JsonRpcError("no active session", reason="not_running")
            self.clear_audio_queue()  # barge in if mid-utterance
            await self.handler.say(text)
            return {"ok": True}

        @rpc.method("conversation.interrupt")
        def _rpc_interrupt(_params: dict[str, object]) -> dict[str, object]:
            if not self.handler._is_connected():
                raise JsonRpcError("no active session", reason="not_running")
            self.clear_audio_queue()
            self._last_turn_state = "listening"
            rpc.broadcast_threadsafe("conversation.turn", {"state": "listening", "reason": "interrupted"})
            return {"ok": True}

        @rpc.method("conversation.mic")
        def _rpc_mic(params: dict[str, object]) -> dict[str, object]:
            if "muted" in params:
                self._mic_muted = bool(params["muted"])
                logger.info("Microphone %s via /rpc", "muted" if self._mic_muted else "unmuted")
            return {"muted": self._mic_muted}

        @rpc.method("backend.config")
        async def _rpc_backend_config(params: dict[str, object]) -> dict[str, object]:
            backend = str(params.get("backend") or get_backend_choice()).strip().lower()
            if backend not in {HF_BACKEND, OPENAI_BACKEND}:
                raise JsonRpcError("invalid backend", reason="invalid_backend", code=-32602)

            updates = {BACKEND_PROVIDER_ENV: backend}
            remove_stale_hf_session_url = False
            if backend == OPENAI_BACKEND:
                api_key = str(params.get("api_key") or "").strip()
                if not api_key and not has_openai_api_key():
                    raise JsonRpcError("OpenAI API key required", reason="empty_key", code=-32602)
                model = str(params.get("openai_model") or get_openai_realtime_model()).strip()
                if model not in OPENAI_REALTIME_MODELS:
                    raise JsonRpcError("invalid OpenAI Realtime model", reason="invalid_openai_model", code=-32602)
                updates[OPENAI_REALTIME_MODEL_ENV] = model
                if api_key:
                    updates[OPENAI_API_KEY_ENV] = api_key
            else:
                hf_selection = get_hf_connection_selection()
                hf_mode = str(params.get("hf_mode") or hf_selection.mode).strip().lower()
                if hf_mode == HF_LOCAL_CONNECTION_MODE:
                    existing_host, existing_port = parse_hf_direct_target(hf_selection.direct_ws_url)
                    host = str(params.get("hf_host") or "").strip() or existing_host or ""
                    if not host:
                        raise JsonRpcError("Hugging Face host required", reason="empty_hf_host", code=-32602)
                    if "://" in host or "/" in host or "?" in host or "#" in host:
                        raise JsonRpcError("invalid Hugging Face host", reason="invalid_hf_host", code=-32602)
                    raw_port = params.get("hf_port")
                    port = int(raw_port) if isinstance(raw_port, (int, float, str)) else (existing_port or 8765)
                    if port < 1 or port > 65535:
                        raise JsonRpcError("invalid Hugging Face port", reason="invalid_hf_port", code=-32602)
                    updates[HF_REALTIME_CONNECTION_MODE_ENV] = HF_LOCAL_CONNECTION_MODE
                    updates[HF_REALTIME_WS_URL_ENV] = build_hf_direct_ws_url(host, port)
                elif hf_mode == HF_DEPLOYED_CONNECTION_MODE:
                    if not bool(get_hf_session_url()):
                        raise JsonRpcError(
                            "missing Hugging Face session url", reason="missing_hf_session_url", code=-32602
                        )
                    updates[HF_REALTIME_CONNECTION_MODE_ENV] = HF_DEPLOYED_CONNECTION_MODE
                    remove_stale_hf_session_url = True
                else:
                    raise JsonRpcError("invalid Hugging Face mode", reason="invalid_hf_mode", code=-32602)

            self._persist_env_values(updates)
            if remove_stale_hf_session_url:
                self._remove_persisted_env_values(("HF_REALTIME_SESSION_URL",))
            if self._can_rebuild_handler():
                await self.request_backend_restart("backend_config_changed")
                message = "Backend saved. Reconnecting backend."
            else:
                message = "Backend saved. Restart Reachy Mini Conversation from the desktop app to apply it."
            return {"ok": True, "message": message, **_status_payload()}

        @rpc.method("obsidian.status")
        def _rpc_obsidian_status(_params: dict[str, object]) -> dict[str, object]:
            return obsidian_sync.supervisor.status()

        @rpc.method("obsidian.login")
        async def _rpc_obsidian_login(params: dict[str, object]) -> dict[str, object]:
            email = str(params.get("email") or "").strip()
            password = str(params.get("password") or "")
            mfa_code = str(params.get("mfa_code") or "").strip() or None
            if not email or not password:
                raise JsonRpcError(
                    "Obsidian email and password required", reason="obsidian_credentials_required", code=-32602
                )
            try:
                message = await obsidian_sync.supervisor.login(email, password, mfa_code)
            except ObsidianSyncError as e:
                raise JsonRpcError(str(e), reason="obsidian_login_failed") from None
            await asyncio.to_thread(obsidian_sync.supervisor.restart)
            return {"ok": True, "message": message, **obsidian_sync.supervisor.status()}

        @rpc.method("obsidian.list_vaults")
        async def _rpc_obsidian_list_vaults(_params: dict[str, object]) -> dict[str, object]:
            try:
                vaults = await obsidian_sync.supervisor.list_vaults()
            except ObsidianSyncError as e:
                raise JsonRpcError(str(e), reason="obsidian_list_failed") from None
            return {"vaults": vaults}

        @rpc.method("obsidian.configure")
        async def _rpc_obsidian_configure(params: dict[str, object]) -> dict[str, object]:
            texts = {
                name: str(params.get(name) or "").strip()
                for name in ("vault", "headless_bin", "device_name", "mode", "conflict_strategy", "path")
            }
            if params.get("enabled") is True and not (texts["vault"] or config.OBSIDIAN_SYNC_VAULT):
                raise JsonRpcError("choose a remote vault", reason="obsidian_vault_required", code=-32602)
            if texts["mode"] == "mirror-remote":
                raise JsonRpcError("mirror-remote reverts local writes", reason="obsidian_mode_refused", code=-32602)
            if texts["mode"] and texts["mode"] not in OBSIDIAN_SYNC_MODES:
                raise JsonRpcError("invalid Obsidian Sync mode", reason="invalid_obsidian_mode", code=-32602)
            if texts["conflict_strategy"] and texts["conflict_strategy"] not in OBSIDIAN_SYNC_CONFLICT_STRATEGIES:
                raise JsonRpcError(
                    "invalid Obsidian conflict strategy", reason="invalid_obsidian_conflict_strategy", code=-32602
                )
            if texts["path"] and not Path(texts["path"]).expanduser().is_absolute():
                raise JsonRpcError("Obsidian local path must be absolute", reason="invalid_obsidian_path", code=-32602)

            # A blank vault or E2EE password keeps the stored value (like the OpenAI key); a blank defaulted
            # setting that the request names goes back to its default.
            updates = {
                OBSIDIAN_SYNC_VAULT_ENV: texts["vault"],
                OBSIDIAN_SYNC_E2EE_PASSWORD_ENV: str(params.get("e2ee_password") or "").strip(),
            }
            if "enabled" in params:
                updates[OBSIDIAN_SYNC_ENABLED_ENV] = "true" if params["enabled"] is True else "false"
            cleared: list[str] = []
            for name, env_name in (
                ("headless_bin", OBSIDIAN_HEADLESS_BIN_ENV),
                ("device_name", OBSIDIAN_SYNC_DEVICE_NAME_ENV),
                ("mode", OBSIDIAN_SYNC_MODE_ENV),
                ("conflict_strategy", OBSIDIAN_SYNC_CONFLICT_STRATEGY_ENV),
                ("path", OBSIDIAN_SYNC_PATH_ENV),
            ):
                if texts[name]:
                    updates[env_name] = texts[name]
                elif name in params:
                    cleared.append(env_name)

            # A session never spans a vault change: it ends under the old settings, and the next one begins after it.
            vault_changed = (
                ("enabled" in params and (params["enabled"] is True) != config.OBSIDIAN_SYNC_ENABLED)
                or (bool(texts["vault"]) and texts["vault"] != config.OBSIDIAN_SYNC_VAULT)
                or (
                    "path" in params
                    and Path(texts["path"]).expanduser() != Path(config.OBSIDIAN_SYNC_PATH or "").expanduser()
                )
            )
            vault_session = self.handler.deps.vault_session
            with vault_session.vault_change() if vault_changed else contextlib.nullcontext():
                if vault_changed:
                    # Without a rebuild, the stream loop reconnects the same handler after its retry delay.
                    await self.request_backend_restart("obsidian_vault_changed", rebuild=False)
                    await asyncio.to_thread(vault_session.end, self._instance_path)

                if cleared:
                    # Removed from the instance `.env` first, so that the reload in `_persist_env_values` cannot restore them.
                    self._remove_persisted_env_values(tuple(cleared))
                    for env_name in cleared:
                        os.environ.pop(env_name, None)
                    refresh_runtime_config_from_env()
                self._persist_env_values(updates)
                await asyncio.to_thread(obsidian_sync.supervisor.restart)
            if vault_changed and self._can_rebuild_handler():
                await self.request_backend_restart("obsidian_vault_changed")
            return {"ok": True, "message": "Obsidian Sync settings saved.", **obsidian_sync.supervisor.status()}

        @rpc.method("obsidian.logout")
        async def _rpc_obsidian_logout(_params: dict[str, object]) -> dict[str, object]:
            try:
                await obsidian_sync.supervisor.logout()
            except ObsidianSyncError as e:
                raise JsonRpcError(str(e), reason="obsidian_logout_failed") from None
            await asyncio.to_thread(obsidian_sync.supervisor.stop)
            return {"ok": True, "message": "Signed out of Obsidian.", **obsidian_sync.supervisor.status()}

        rpc.mount(settings_app)
        self._rpc = rpc

        try:
            personality_ops = build_personality_ops(
                self.handler,
                lambda: self._asyncio_loop,
                persist_personality=self._persist_personality,
                get_persisted_personality=self._read_persisted_personality,
                apply_personality=self.apply_personality,
                get_voices=self.get_available_voices,
                get_current_voice=self.get_current_voice,
                change_voice=self.change_voice,
            )
            # personalities.* / voices.* over JSON-RPC — the local UI and remote
            # clients drive personalities the same way, one control surface.
            register_personality_methods(rpc, personality_ops)
        except Exception:
            logger.exception("Failed to register personality methods; the personality UI will be unavailable")

        try:
            register_tool_space_methods(
                rpc,
                lambda: self._asyncio_loop,
                self.request_backend_restart,
                instance_path=self._instance_path,
            )
        except Exception:
            logger.exception("Failed to register Tool Space methods; remote tool settings will be unavailable")

        try:
            register_profile_tool_methods(
                rpc,
                lambda: self._asyncio_loop,
                self.request_backend_restart,
                instance_path=self._instance_path,
            )
        except Exception:
            logger.exception("Failed to register profile tool methods; personality tool settings will be unavailable")

        self._settings_initialized = True

    async def _run_handler_startup_loop(self) -> None:
        """Start the realtime handler and keep settings UI alive after backend failures."""
        rebuild_required = False
        while not self._stop_event.is_set():
            if self._wake_word_detector is not None and not self._wake_gate_open:
                self._set_backend_connection_state("sleeping")
                await self._wake_gate_event.wait()
            selected_backend = get_backend_choice()
            if selected_backend != self._active_backend_name or self._restart_requested.is_set() or rebuild_required:
                await self._shutdown_active_handler()
                if not self._can_rebuild_handler():
                    self._restart_requested.clear()
                    self._set_backend_connection_state("restart_required")
                    await self._sleep_or_restart_requested(0.5)
                    continue
                self._restart_requested.clear()
                try:
                    self._build_handler_for_current_backend()
                    rebuild_required = False
                    self._wake_handler_ready.set()
                except Exception as e:
                    rebuild_required = True
                    self._set_backend_connection_state("disconnected", e)
                    logger.warning(
                        "%s backend handler failed to initialize: %s. Retrying in %.1f seconds.",
                        selected_backend,
                        self._format_backend_error(e),
                        self._backend_retry_delay,
                    )
                    await self._sleep_or_restart_requested(self._backend_retry_delay)
                    continue

            active_backend = self._active_backend_name
            if not self._has_required_configuration(active_backend):
                requirement_name = self._requirement_name(active_backend)
                self._set_backend_connection_state("waiting_for_config", f"{requirement_name} is not configured.")
                await self._sleep_or_restart_requested(0.5)
                continue

            self._set_backend_connection_state("connecting")
            try:
                await self.handler.start_up()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._set_backend_connection_state("disconnected", e)
                logger.warning(
                    "%s backend failed to start: %s. Settings UI remains available; retrying in %.1f seconds.",
                    active_backend,
                    self._format_backend_error(e),
                    self._backend_retry_delay,
                )
            else:
                if self._stop_event.is_set():
                    return
                self._set_backend_connection_state("disconnected")
                if self._restart_requested.is_set():
                    logger.info("%s backend stopped for requested restart.", active_backend)
                    continue
                logger.info(
                    "%s backend session ended. Settings UI remains available; retrying in %.1f seconds.",
                    active_backend,
                    self._backend_retry_delay,
                )

            await self._sleep_or_restart_requested(self._backend_retry_delay)

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the selected backend is missing its required key, expose a tiny
        settings UI via the Reachy Mini settings server to collect it before
        starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    refresh_runtime_config_from_env()
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        # Always expose settings UI if a settings app is available
        # (do this AFTER loading the instance .env so status endpoint sees the right value)
        self._init_settings_ui_if_needed()

        selected_backend = get_backend_choice()
        if not self._has_required_configuration(selected_backend):
            requirement_name = self._requirement_name(selected_backend)
            self._set_backend_connection_state("waiting_for_config", f"{requirement_name} is not configured.")
            if self._settings_app is None:
                logger.error("%s not found. Set it in the app .env before starting.", requirement_name)
                return
            logger.warning("%s not found. Open the app settings page to configure it.", requirement_name)
            try:
                while not self._stop_event.is_set():
                    selected_backend = get_backend_choice()
                    if self._has_required_configuration(selected_backend):
                        break
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for backend configuration.")
                return
            if self._stop_event.is_set():
                return
            self._set_backend_connection_state("not_started")

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Connect the backend first so it overlaps detector loading, warmup, and audio config below.
            handler_task = asyncio.create_task(self._run_handler_startup_loop(), name="realtime-handler")
            self._tasks = [handler_task]
            if self._wake_word_detector is not None:
                try:
                    await asyncio.to_thread(self._wake_word_detector.load)
                except Exception:
                    logger.exception("Wake-word detector failed to load; continuing without wake gating")
                    await self._disable_wake_word_gate()
            await asyncio.gather(
                asyncio.sleep(1),  # give the pipelines time to start
                asyncio.to_thread(apply_audio_startup_config, self._robot, logger=logger),
            )
            self._tasks += [
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()
                # App stop and shutdown end the wake session; write its note before Obsidian Sync stops.
                await asyncio.to_thread(self.handler.deps.vault_session.end, self._instance_path)

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # close() runs on watcher threads, loop-owned state must change on the loop.
        loop = self._asyncio_loop
        if loop is None or not loop.is_running():
            self._stop_event.set()
            return
        loop.call_soon_threadsafe(self._stop_event.set)
        for task in self._tasks:
            if not task.done():
                loop.call_soon_threadsafe(task.cancel)

    def clear_audio_queue(self) -> None:
        """Flush queued playback audio immediately on user barge-in.

        Calls the SDK's ``clear_player()`` — now a first-class flush on both
        the local GStreamer and WebRTC backends (the WebRTC one also tells the
        daemon to drop audio already queued for the speaker). Falls back to the
        deprecated ``clear_output_buffer()`` only for older SDKs.
        """
        logger.info("User intervention: flushing player queue")
        audio = getattr(self._robot.media, "audio", None)
        if audio is not None:
            if hasattr(audio, "clear_player") and callable(audio.clear_player):
                audio.clear_player()
            elif hasattr(audio, "clear_output_buffer") and callable(audio.clear_output_buffer):
                # Older SDK without clear_player(); best-effort.
                audio.clear_output_buffer()
        # Drain the handler's pending output in place — do NOT replace the
        # queue object, since emit() may be awaiting it (wait_for_item).
        self._drain_output_queue()

    def _drain_output_queue(self) -> None:
        """Empty the handler's output queue in place without replacing it."""
        queue = getattr(self.handler, "output_queue", None)
        if queue is None:
            return
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def record_loop(self) -> None:
        """Detect wake words and forward audio while the wake gate is open."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        backend_resampler: _StreamingResampler | None = None
        wake_word_resampler: _StreamingResampler | None = None
        logger.debug("Audio recording started at %s Hz", input_sample_rate)

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None and not self._mic_muted:
                detector = self._wake_word_detector
                if detector is not None and not self._wake_gate_open:
                    wake_word_audio = audio_frame
                    if input_sample_rate != WAKE_WORD_SAMPLE_RATE:
                        if wake_word_resampler is None:
                            wake_word_resampler = _StreamingResampler(input_sample_rate, WAKE_WORD_SAMPLE_RATE)
                        wake_word_audio = wake_word_resampler.process(audio_to_float32(audio_frame))
                    try:
                        event = detector.process(wake_word_audio)
                    except Exception:
                        logger.exception("Wake-word inference failed; continuing without wake gating")
                        await self._disable_wake_word_gate()
                    else:
                        if event is not None:
                            await self._handle_wake_event(event)
                else:
                    wake_word_resampler = None

                if self._wake_gate_open:
                    handler = self.handler
                    handler_sample_rate = handler.SAMPLE_RATE
                    backend_audio = audio_frame
                    if input_sample_rate != handler_sample_rate:
                        if backend_resampler is None or backend_resampler.target_sample_rate != handler_sample_rate:
                            backend_resampler = _StreamingResampler(input_sample_rate, handler_sample_rate)
                        backend_audio = backend_resampler.process(audio_to_float32(audio_frame))
                    else:
                        backend_resampler = None
                    await handler.receive((handler_sample_rate, backend_audio))

                self._emit_level("user", audio_frame)
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        output_sample_rate = self._robot.media.get_output_audio_samplerate()
        resampler: _StreamingResampler | None = None
        while not self._stop_event.is_set():
            if self._wake_word_detector is not None and not self._wake_gate_open:
                await self._conversation_ready.wait()
                continue
            handler = self.handler
            try:
                handler_output = await asyncio.wait_for(handler.emit(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if (self._wake_word_detector is not None and not self._wake_gate_open) or handler is not self.handler:
                continue

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                sample_rate, audio_data = handler_output

                # Skip empty audio frames
                if audio_data.size == 0:
                    continue

                # Reshape if needed
                if audio_data.ndim == 2:
                    # channels-last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)
                if sample_rate != output_sample_rate:
                    if resampler is None or resampler.source_sample_rate != sample_rate:
                        resampler = _StreamingResampler(sample_rate, output_sample_rate)
                    audio_frame = resampler.process(audio_frame)
                else:
                    resampler = None

                self._robot.media.push_audio_sample(audio_frame)
                self._emit_level("assistant", audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
