"""TTS requests and protocol for phrase text to PCM audio."""

from dataclasses import dataclass
from typing import Protocol


SUPPORTED_TTS_LANGUAGES: dict[str, str] = {
    "en": "English", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ru": "Russian",
}
BUILT_IN_VOICES: dict[str, str] = {"voice_1": "Agent", "voice_2": "Cartoon"}


@dataclass(frozen=True, slots=True)
class SynthesisOptions:
    """Options for Qwen3-TTS Base voice cloning."""

    target_lang: str = "en"
    voice_id: str = "voice_1"
    temperature: float = 0.9

    def validate(self) -> "SynthesisOptions":
        """Validate controls before they reach the model adapter."""
        if self.target_lang not in SUPPORTED_TTS_LANGUAGES:
            raise ValueError(f"Unsupported TTS language: {self.target_lang!r}.")
        if self.voice_id not in (*BUILT_IN_VOICES, "custom"):
            raise ValueError("Choose one of the two built-in voices or an active custom voice.")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("Speech variation must be a number between 0.1 and 1.5.")
        if not 0.1 <= float(self.temperature) <= 1.5:
            raise ValueError("Speech variation must be a number between 0.1 and 1.5.")
        return self


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

    def synthesize(self, text: str, options: SynthesisOptions | None = None) -> SpeechResult:
        """Synthesize text using the voice already loaded for the target language."""
        ...

    def unload(self) -> None:
        """Release the loaded voice and its inference session."""
        ...

