"""Entrypoint for the Reachy Mini conversation app."""

from __future__ import annotations
import os
import sys
import time
import signal
import asyncio
import logging
import argparse
import threading
from typing import TYPE_CHECKING, Any, Optional
from pathlib import Path
from collections.abc import Callable, Awaitable

from fastapi import FastAPI, Request, Response

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    log_connection_troubleshooting,
)


if TYPE_CHECKING:
    from reachy_mini_conversation_app.console import LocalStream


_import_warmup_thread: threading.Thread | None = None


def start_import_warmup() -> threading.Thread:
    """Preload heavy modules in the background while the robot connects.

    The openai realtime tree (~1.5s on the robot) is needed by the realtime
    handler, but not before the robot / media pipelines are up. Importing it
    on a side thread overlaps that cost with robot initialization instead of
    paying it serially. Idempotent; errors are deferred to the real import
    site, which reports them with full context.
    """
    global _import_warmup_thread
    if _import_warmup_thread is None:

        def _preload() -> None:
            try:
                # AsyncOpenAI.realtime is a cached_property whose module tree
                # (~1.6s on the CM4) otherwise loads lazily INSIDE the ws
                # connect phase; same for the websockets client the SDK pulls
                # in __aenter__. Importing plain openai does NOT cover these.
                import openai.resources.realtime  # noqa: F401
                import websockets.asyncio.client  # noqa: F401

                import reachy_mini_conversation_app.huggingface_realtime  # noqa: F401
            except Exception:
                logging.getLogger(__name__).debug("Import warmup failed", exc_info=True)

        _import_warmup_thread = threading.Thread(target=_preload, name="import-warmup", daemon=True)
        _import_warmup_thread.start()
    return _import_warmup_thread


def _start_inactivity_timeout_thread(
    timeout_minutes: float,
    stream_manager: LocalStream,
    logger: logging.Logger,
    app_stop_event: threading.Event | None,
    go_to_sleep: Callable[[], dict[str, Any]] | None = None,
) -> threading.Thread:
    """Start a daemon that puts the app to sleep after inactivity."""
    timeout_seconds = timeout_minutes * 60.0

    def poll_inactivity_timeout() -> None:
        logger.info("App inactivity timeout enabled: %.1f minutes.", timeout_minutes)
        while app_stop_event is None or not app_stop_event.is_set():
            elapsed = stream_manager.seconds_since_activity()
            if elapsed >= timeout_seconds:
                logger.info("No activity for %.1f minutes; going to sleep.", elapsed / 60.0)
                try:
                    if go_to_sleep is not None:
                        go_to_sleep()
                    else:
                        stream_manager.close()
                except Exception as e:
                    logger.error("Error while going to sleep after inactivity timeout: %s", e)
                    try:
                        stream_manager.close()
                    except Exception as close_error:
                        logger.error("Error while closing stream manager after inactivity timeout: %s", close_error)
                return
            time.sleep(1.0)

    thread = threading.Thread(target=poll_inactivity_timeout, daemon=True)
    thread.start()
    return thread


def _handle_sigterm_as_sigint() -> None:
    """Route SIGTERM to the SIGINT handler, so that a service stop takes the orderly shutdown path."""
    signal.signal(signal.SIGTERM, lambda _signum, _frame: signal.raise_signal(signal.SIGINT))


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    if args.command == "tool-spaces":
        from reachy_mini_conversation_app.tool_spaces import handle_tool_spaces_command

        logger = setup_logger(args.debug)
        try:
            raise SystemExit(handle_tool_spaces_command(args))
        except Exception as exc:
            logger.error("tool-spaces command failed: %s", exc)
            raise SystemExit(1) from exc
    instance_path = None
    if args.ui:
        data_home = os.getenv("XDG_DATA_HOME")
        data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
        standalone_instance_path = data_root / "reachy_mini_conversation_app"
        standalone_instance_path.mkdir(parents=True, exist_ok=True)
        instance_path = str(standalone_instance_path)

    run(args, instance_path=instance_path)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    start_import_warmup()
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app import obsidian_sync
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.config import (
        OPENAI_BACKEND,
        HF_LOCAL_CONNECTION_MODE,
        config,
        set_instance_path,
        get_backend_choice,
        resolve_wake_word_settings,
        get_hf_connection_selection,
        resolve_app_timeout_minutes,
        refresh_runtime_config_from_env,
    )
    from reachy_mini_conversation_app.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    set_instance_path(instance_path)
    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    backend_provider = get_backend_choice()
    if backend_provider == OPENAI_BACKEND:
        logger.info(
            "Configured backend provider: %s, model: %s",
            "OpenAI Realtime",
            config.OPENAI_REALTIME_MODEL,
        )
    else:
        logger.info(
            "Configured backend provider: %s, connection mode: %s",
            "Hugging Face",
            get_hf_connection_selection().mode,
        )

    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.wake_word import (
        DEFAULT_SLEEP_PHRASES,
        PACKAGED_WAKE_WORD_MODEL,
        WakeWordDetector,
    )
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.conversation_handler import ConversationHandler

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    movement_manager = MovementManager(current_robot=robot)
    wake_word_settings = resolve_wake_word_settings(DEFAULT_SLEEP_PHRASES)
    wake_word_detector: WakeWordDetector | None = None
    if wake_word_settings.enabled:
        wake_word_model_path = (
            Path(wake_word_settings.model_path).expanduser()
            if wake_word_settings.model_path is not None
            else PACKAGED_WAKE_WORD_MODEL
        )
        wake_word_detector = WakeWordDetector(
            model_path=wake_word_model_path,
            threshold=wake_word_settings.threshold,
        )

    def wake_from_wake_word() -> None:
        app_lifecycle.wake_robot_for_conversation(robot, movement_manager, logger)

    def sleep_from_phrase() -> None:
        sleep_error = app_lifecycle.move_robot_to_sleep(robot, movement_manager, logger)
        if sleep_error is not None:
            raise RuntimeError(sleep_error)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        instance_path=instance_path,
        camera_enabled=not args.no_camera,
    )

    def build_handler(startup_voice: Optional[str] = None) -> ConversationHandler:
        """Build a realtime handler for the current provider configuration."""
        if get_backend_choice() == OPENAI_BACKEND:
            from reachy_mini_conversation_app.openai_realtime import OpenAIRealtimeHandler

            logger.info("Using direct OpenAI Realtime handler")
            return OpenAIRealtimeHandler(
                deps,
                instance_path=instance_path,
                startup_voice=startup_voice,
            )

        from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

        hf_connection_selection = get_hf_connection_selection()
        transport_label = (
            "Hugging Face direct websocket"
            if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
            else "Hugging Face session proxy"
        )
        logger.info("Using Hugging Face realtime handler (%s)", transport_label)
        return HuggingFaceRealtimeHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    handler = build_handler(startup_settings.voice)

    stream_manager: LocalStream | None = None
    own_ui_server = None

    effective_settings_app = settings_app
    if args.ui and settings_app is None:
        effective_settings_app = FastAPI()

        @effective_settings_app.middleware("http")
        async def _no_cache(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            """Serve everything no-store so browsers don't keep stale UI modules."""
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=effective_settings_app,
        instance_path=instance_path,
        handler_factory=build_handler,
        startup_voice=startup_settings.voice,
        wake_word_detector=wake_word_detector,
        sleep_phrases=wake_word_settings.sleep_phrases,
        on_wake_word=wake_from_wake_word,
        on_sleep_phrase=sleep_from_phrase,
    )

    # The page is served immediately, so the API must be live before the slow startup work below.
    if effective_settings_app is not None:
        stream_manager._init_settings_ui_if_needed()

    go_to_sleep_lock = threading.Lock()
    go_to_sleep_requested = threading.Event()

    def go_to_sleep_and_stop_app() -> dict[str, Any]:
        """Put Reachy to sleep, then stop the current app."""
        if not go_to_sleep_lock.acquire(blocking=False):
            return {"status": "already_requested"}

        try:
            if go_to_sleep_requested.is_set():
                return {"status": "already_requested"}
            go_to_sleep_requested.set()

            logger.info("Going to sleep before stopping conversation app.")
            sleep_error = app_lifecycle.move_robot_to_sleep(robot, movement_manager, logger)

            stop_current_app_requested = False
            if app_stop_event is None or not app_stop_event.is_set():
                stop_current_app_requested = app_lifecycle.request_stop_current_app(robot, logger)
            local_stop_requested = True
            if app_stop_event is not None:
                app_stop_event.set()
            else:
                try:
                    stream_manager.close()
                except Exception as e:
                    local_stop_requested = False
                    logger.error("Error while closing stream manager after go_to_sleep: %s", e)

            result: dict[str, Any] = {
                "status": "sleeping" if sleep_error is None else "stop_requested",
                "stop_current_app_requested": stop_current_app_requested,
                "local_stop_requested": local_stop_requested,
            }
            if sleep_error is not None:
                result["error"] = f"go_to_sleep movement failed: {sleep_error}"
            return result
        finally:
            go_to_sleep_lock.release()

    deps.go_to_sleep = go_to_sleep_and_stop_app

    def run_go_to_sleep_tool() -> dict[str, Any]:
        return app_lifecycle.run_go_to_sleep_tool(deps, logger)

    if args.ui and settings_app is None and effective_settings_app is not None:
        import uvicorn

        own_ui_server = uvicorn.Server(
            uvicorn.Config(effective_settings_app, host="0.0.0.0", port=7860, log_level="warning")
        )
        threading.Thread(target=own_ui_server.run, daemon=True, name="ui-server").start()
        logger.info("Web UI available at http://localhost:7860")

    try:
        app_lifecycle.initialize_tools_with_default_fallback(instance_path, logger)
    except Exception as e:
        logger.error("Failed to initialize tools: %s", e)
        sys.exit(1)

    # Each async service gets its own thread or event loop.
    if wake_word_detector is None:
        app_lifecycle.wake_up_if_sleeping(robot, logger)
        movement_manager.start()
        robot.enable_wobbling()
    else:
        try:
            robot.disable_wobbling()
        except Exception as e:
            logger.debug("Error disabling wobbling while waiting for the wake word: %s", e)
        logger.info("Wake-word gate enabled; say the configured wake phrase to start a conversation.")

    timeout_minutes = resolve_app_timeout_minutes()
    if timeout_minutes is not None:
        _start_inactivity_timeout_thread(timeout_minutes, stream_manager, logger, app_stop_event, run_go_to_sleep_tool)

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown.

        Deliberately does NOT put the robot to sleep: an external stop
        (mobile app, dashboard, app switch) means "stop this app", not
        "power the robot down" — the daemon returns it to the neutral
        pose afterwards, awake and ready for the next app. Sleeping is
        reserved for the explicit paths (the voice go_to_sleep tool and
        the inactivity timeout).
        """
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    # Started here, right before the try block, so that every exit path stops the `ob sync` child process.
    # Only the main thread can set signal handlers; the daemon stops apps with SIGINT, systemd with SIGTERM.
    if threading.current_thread() is threading.main_thread():
        _handle_sigterm_as_sigint()
    obsidian_sync.supervisor.start()
    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        if own_ui_server is not None:
            own_ui_server.should_exit = True

        obsidian_sync.supervisor.stop()

        # Stop the motion writes without changing the robot's posture. If
        # the shutdown came from the voice go_to_sleep tool the robot is
        # already in the sleep pose; on a plain stop it stays awake and
        # the daemon returns it to neutral once the process exits.
        movement_manager.stop(reset_to_neutral=False)
        try:
            robot.disable_wobbling()
        except Exception as e:
            logger.debug(f"Error disabling wobbling during shutdown: {e}")

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start preloading heavy imports before the robot/media init in wrapped_run."""
        start_import_warmup()
        super().__init__(*args, **kwargs)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
