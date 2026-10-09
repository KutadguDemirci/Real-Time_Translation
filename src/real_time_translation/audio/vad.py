"""Energy-based voice activity and configurable trailing-silence endpointing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any


@dataclass(frozen=True)
class VADResult:
    speech: bool
    started: bool
    ended: bool
    trailing_silence_ms: int
    rms: float


class VoiceActivityDetector:
    """Detect speech frames and signal one endpoint after sustained quiet.

    This lightweight RMS detector is dependency-free and intentionally exposes its threshold
    so deployments can tune it to microphone gain and ambient noise. It signals *when* the
    recognizer should be flushed; it does not alter ASR text or perform a flush itself.
    """

    def __init__(self, endpoint_pause_ms: int = 800, frame_ms: int = 20,
                 threshold: float = 0.015, release_ratio: float = 0.6,
                 min_speech_ms: int = 40) -> None:
        if endpoint_pause_ms <= 0 or frame_ms <= 0:
            raise ValueError("endpoint_pause_ms and frame_ms must be positive.")
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("threshold must be a finite non-negative RMS amplitude.")
        if not 0 < release_ratio <= 1 or not math.isfinite(release_ratio):
            raise ValueError("release_ratio must be finite and in the range (0, 1].")
        if min_speech_ms < 0:
            raise ValueError("min_speech_ms cannot be negative.")
        self.endpoint_pause_ms = endpoint_pause_ms
        self.frame_ms = frame_ms
        self.threshold = threshold
        self.release_threshold = threshold * release_ratio
        self.min_speech_ms = min_speech_ms
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def trailing_silence_ms(self) -> int:
        """Duration of the current quiet tail while a phrase is active."""
        return self._silence_ms

    def process(self, samples: Any, sample_width: int = 4) -> VADResult:
        """Consume float32 samples (or signed 16-bit PCM bytes) for one frame."""
        rms = _rms(samples, sample_width)
        threshold = self.release_threshold if self._active else self.threshold
        speech = rms >= threshold
        started = False
        ended = False

        if speech:
            self._speech_ms += self.frame_ms
            self._silence_ms = 0
            if not self._active and self._speech_ms >= self.min_speech_ms:
                self._active = True
                started = True
        elif self._active:
            self._silence_ms += self.frame_ms
            if self._silence_ms >= self.endpoint_pause_ms:
                ended = True
                self.reset()
        else:
            self._speech_ms = 0

        trailing = self.endpoint_pause_ms if ended else self._silence_ms
        return VADResult(self._active or ended, started, ended, trailing, rms)

    def reset(self) -> None:
        """Reset after an endpoint or explicit session cancellation."""
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0


def _rms(samples: Any, sample_width: int) -> float:
    if isinstance(samples, (bytes, bytearray, memoryview)):
        raw = bytes(samples)
        if sample_width == 2:
            usable = len(raw) - len(raw) % 2
            values = struct.unpack(f"<{usable // 2}h", raw[:usable]) if usable else ()
            scale = 32768.0
        elif sample_width == 4:
            usable = len(raw) - len(raw) % 4
            values = struct.unpack(f"<{usable // 4}f", raw[:usable]) if usable else ()
            scale = 1.0
        else:
            raise ValueError("sample_width must be 2 (PCM16 bytes) or 4 (float32 bytes).")
        return math.sqrt(sum((float(value) / scale) ** 2 for value in values) / len(values)) if values else 0.0
    try:
        count = int(samples.size)
        if count == 0:
            return 0.0
        # NumPy arrays expose mean/square operations without copying into Python lists.
        return float((samples.astype("float64", copy=False) ** 2).mean() ** 0.5)
    except AttributeError:
        values = [float(value) for value in samples]
        return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0

