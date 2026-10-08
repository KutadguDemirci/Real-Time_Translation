"""Backend-neutral contract for incremental speech recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RecognitionUpdate:
    """A replaceable hypothesis or a final utterance result."""

    text: str
    language: str
    updated_at: float
    revision: int
    is_final: bool = False


@runtime_checkable
class StreamingRecognizer(Protocol):
    """Incrementally consume mono float PCM and emit revised transcript text.

    Implementations return ``None`` when the hypothesis did not change. ``finish`` finalizes
    the current utterance and resets the decoder for the next one.
    """

    language: str
    sample_rate: int

    def accept_audio(
        self, samples: Any, sample_rate: int, captured_at: float
    ) -> RecognitionUpdate | None:
        """Consume one frame and return a changed partial hypothesis, if any."""

    def finish(self, captured_at: float) -> RecognitionUpdate | None:
        """Flush the active utterance at an externally detected endpoint."""

    def close(self) -> None:
        """Release backend resources."""
