"""Local wake-word detection and sleep-phrase matching."""

import re
import time
import logging
from typing import Protocol, cast
from pathlib import Path
from importlib import import_module
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import AudioArray, audio_to_int16


logger = logging.getLogger(__name__)

WAKE_WORD_SAMPLE_RATE = 16000
_WAKE_WORD_RESOURCE_DIR = Path(str(files("reachy_mini_conversation_app").joinpath("resources/wake_words")))
PACKAGED_WAKE_WORD_MODEL = _WAKE_WORD_RESOURCE_DIR / "hey_emma.onnx"
_MELSPECTROGRAM_MODEL = _WAKE_WORD_RESOURCE_DIR / "melspectrogram.onnx"
_EMBEDDING_MODEL = _WAKE_WORD_RESOURCE_DIR / "embedding_model.onnx"
WAKE_WORD_WINDOW_SAMPLES = 1280
DEFAULT_SLEEP_PHRASES = (
    "goodbye emma",
    "go to sleep",
    "go sleep",
    "sleep now",
    "time to sleep",
    "good night",
)


class _WakeWordModel(Protocol):
    def predict(self, frame: NDArray[np.int16]) -> dict[str, float]: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class WakeWordEvent:
    """A detected wake phrase."""

    model: str
    score: float


class WakeWordDetector:
    """Score 16 kHz microphone audio with an openWakeWord model."""

    def __init__(self, model_path: Path, threshold: float = 0.5, cooldown_s: float = 2.0) -> None:
        """Configure a detector for one packaged or external ONNX model."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Wake-word threshold must be greater than 0 and at most 1")
        if cooldown_s < 0.0:
            raise ValueError("Wake-word cooldown cannot be negative")

        self.model_path = model_path
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._prediction_key = model_path.stem
        self._model: _WakeWordModel | None = None
        self._window = np.empty(WAKE_WORD_WINDOW_SAMPLES, dtype=np.int16)
        self._window_size = 0
        self._last_detection_at = float("-inf")

    def load(self) -> None:
        """Load the openWakeWord runtime and configured model."""
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Wake-word model does not exist: {self.model_path}")

        model_module = import_module("openwakeword.model")
        self._model = cast(
            "_WakeWordModel",
            model_module.Model(
                inference_framework="onnx",
                wakeword_models=[str(self.model_path)],
                melspec_model_path=str(_MELSPECTROGRAM_MODEL),
                embedding_model_path=str(_EMBEDDING_MODEL),
            ),
        )
        logger.info("Loaded wake-word model %s with threshold %.2f", self.model_path.name, self.threshold)

    def process(self, audio: AudioArray) -> WakeWordEvent | None:
        """Process 16 kHz audio and return the first wake event, if any."""
        if self._model is None:
            raise RuntimeError("Wake-word detector is not loaded")

        samples = audio
        if samples.ndim == 2:
            if samples.shape[1] > samples.shape[0]:
                samples = samples.T
            if samples.shape[1] > 1:
                samples = samples[:, 0]

        mono = audio_to_int16(samples).reshape(-1)
        sample_offset = 0
        while sample_offset < mono.size:
            copy_size = min(WAKE_WORD_WINDOW_SAMPLES - self._window_size, mono.size - sample_offset)
            next_window_size = self._window_size + copy_size
            next_sample_offset = sample_offset + copy_size
            self._window[self._window_size : next_window_size] = mono[sample_offset:next_sample_offset]
            self._window_size = next_window_size
            sample_offset = next_sample_offset

            if self._window_size < WAKE_WORD_WINDOW_SAMPLES:
                continue

            self._window_size = 0
            predictions = self._model.predict(self._window)
            score = float(predictions.get(self._prediction_key, 0.0))
            now = time.monotonic()
            if score < self.threshold or now - self._last_detection_at < self.cooldown_s:
                continue

            self._last_detection_at = now
            self._model.reset()
            return WakeWordEvent(model=self._prediction_key, score=score)

        return None


def matches_sleep_phrase(transcript: str, phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES) -> bool:
    """Return whether a transcript contains a configured sleep phrase."""
    normalized = re.sub(r"[^\w']+", " ", transcript.casefold()).strip()
    padded_transcript = f" {normalized} "
    return any(f" {' '.join(phrase.casefold().split())} " in padded_transcript for phrase in phrases)
