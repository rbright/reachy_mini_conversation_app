import asyncio

import numpy as np
import pytest

from reachy_mini_conversation_app.streaming import (
    AdditionalOutputs,
    audio_to_mono,
    wait_for_item,
    audio_to_int16,
    audio_to_float32,
)


def test_additional_outputs_stores_messages() -> None:
    """Additional outputs should keep the emitted role/content payloads."""
    message = {"role": "assistant", "content": "hello"}

    outputs = AdditionalOutputs(message)

    assert outputs.args == (message,)


@pytest.mark.asyncio
async def test_wait_for_item_returns_none_on_timeout() -> None:
    """Queue waits should time out without raising."""
    queue: asyncio.Queue[int] = asyncio.Queue()

    assert await wait_for_item(queue, timeout=0.01) is None


def test_audio_to_int16_scales_float32() -> None:
    """Float audio should scale into the int16 sample range."""
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(audio_to_int16(audio), np.array([-32767, 0, 32767], dtype=np.int16))


@pytest.mark.parametrize("stereo", [np.array([[1, 10], [2, 20], [3, 30]]), np.array([[1, 2, 3], [10, 20, 30]])])
def test_audio_to_mono_keeps_first_channel_of_either_layout(stereo: np.ndarray) -> None:
    """Channels-last and channels-first stereo both reduce to the first channel; mono passes through."""
    mono = np.array([1, 2, 3], dtype=np.int16)

    np.testing.assert_array_equal(audio_to_mono(stereo.astype(np.int16)), mono)
    assert audio_to_mono(mono) is mono


def test_audio_to_float32_scales_int16() -> None:
    """Int16 audio should scale into the float sample range."""
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    np.testing.assert_allclose(audio_to_float32(audio), np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_audio_converters_reject_unsupported_dtype() -> None:
    """Unsupported audio dtypes should fail explicitly."""
    audio = np.array([1.0], dtype=np.float64)

    with pytest.raises(TypeError, match="Unsupported audio data type"):
        audio_to_int16(audio)
