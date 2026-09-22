"""Tests for local wake-word and sleep-phrase behavior."""

from reachy_mini_conversation_app.wake_word import matches_sleep_phrase


def test_sleep_phrase_normalizes_configured_punctuation() -> None:
    """Configured punctuation must match normalized transcript punctuation."""
    phrases = ("good night!", "sign-off")

    assert matches_sleep_phrase("Okay, good night.", phrases)
    assert matches_sleep_phrase("Please sign off now.", phrases)
