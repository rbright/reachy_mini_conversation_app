"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


def test_provider_voice_catalogs_and_defaults(monkeypatch) -> None:
    """Each provider exposes only its native voices and default."""
    assert config.get_available_voices(config.HF_BACKEND) == config.HF_AVAILABLE_VOICES
    assert config.get_default_voice(config.HF_BACKEND) == "Aiden"
    assert config.get_available_voices(config.OPENAI_BACKEND) == [
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    ]
    assert config.get_default_voice(config.OPENAI_BACKEND) == "marin"

    monkeypatch.setattr(config.config, "BACKEND_PROVIDER", config.OPENAI_BACKEND)
    assert config.get_available_voices() == config.OPENAI_AVAILABLE_VOICES


def test_openai_model_validation_uses_safe_default(monkeypatch) -> None:
    """Unsupported OpenAI model configuration falls back to the current safe default."""
    monkeypatch.setattr(config.config, "OPENAI_REALTIME_MODEL", "not-a-realtime-model")

    assert config.get_openai_realtime_model() == "gpt-realtime-2.1"


def test_backend_choice_defaults_invalid_values_to_huggingface(monkeypatch) -> None:
    """Invalid persisted provider values cannot select an unknown runtime backend."""
    monkeypatch.setattr(config.config, "BACKEND_PROVIDER", "unknown")

    assert config.get_backend_choice() == config.HF_BACKEND
