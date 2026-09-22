"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import reachy_mini_conversation_app.main as main_mod


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def test_standalone_ui_uses_stable_writable_instance_path(tmp_path, monkeypatch) -> None:
    """Two standalone UI launches share the same persistent settings directory."""
    args = SimpleNamespace(command=None, ui=True)
    paths = []
    previous_contents = []

    def run(_args, *, instance_path=None) -> None:
        path = main_mod.Path(instance_path)
        paths.append(path)
        settings_path = path / ".env"
        previous_contents.append(settings_path.read_text(encoding="utf-8") if settings_path.exists() else None)
        settings_path.write_text("BACKEND_PROVIDER=openai\n", encoding="utf-8")

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(main_mod, "parse_args", lambda: (args, []))
    monkeypatch.setattr(main_mod, "run", run)

    main_mod.main()
    main_mod.main()

    assert paths == [tmp_path / "reachy_mini_conversation_app"] * 2
    assert previous_contents == [None, "BACKEND_PROVIDER=openai\n"]
