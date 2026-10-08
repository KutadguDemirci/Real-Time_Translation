"""TTS protocol for phrase text to PCM audio."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeechResult:
    """Synthesized mono signed 16-bit little-endian PCM and its metadata."""

    pcm: bytes
    sample_rate: int
    channels: int
    sample_width: int
    synthesis_time_ms: float
    model: str
    target_lang: str
    device: str


class SpeechSynthesizer(Protocol):
    """A local speech engine with an explicit target-language model lifecycle."""

    def load(self, target_lang: str) -> None:
        """Load the configured voice once for the selected target language."""
        ...

    def synthesize(self, text: str) -> SpeechResult:
        """Synthesize text using the voice already loaded for the target language."""
        ...

    def unload(self) -> None:
        """Release the loaded voice and its inference session."""
        ...

