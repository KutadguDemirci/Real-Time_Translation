"""Immutable, typed messages crossing asynchronous component boundaries."""

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import TypeAlias

from .languages import SPEECH_LANGUAGES, SUPPORTED_LANGUAGES
from .segments import Phrase, TranscriptSegment


class SessionStatus(StrEnum):
    """High-level lifecycle state for the active translation session."""

    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A mono or multichannel audio block represented as normalized float samples."""

    samples: tuple[float, ...]
    sample_rate: int
    channels: int
    captured_at: float
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        _timestamp(self.captured_at)
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not all(math.isfinite(sample) for sample in self.samples):
            raise ValueError("audio samples must be finite")


@dataclass(frozen=True, slots=True)
class AudioFrameEvent:
    frame: AudioFrame


@dataclass(frozen=True, slots=True)
class TranscriptUpdated:
    segment: TranscriptSegment


@dataclass(frozen=True, slots=True)
class PhraseCommitted:
    phrase: Phrase


@dataclass(frozen=True, slots=True)
class TranslationCompleted:
    phrase_id: str
    text: str
    source_language: str
    target_language: str
    completed_at: float
    inference_time_ms: float
    model: str

    def __post_init__(self) -> None:
        _validate_phrase_id(self.phrase_id)
        if self.source_language not in SUPPORTED_LANGUAGES or self.target_language not in SUPPORTED_LANGUAGES:
            raise ValueError("translation languages must be supported")
        if self.source_language == self.target_language:
            raise ValueError("source and target languages must differ")
        _timestamp(self.completed_at)
        _duration(self.inference_time_ms)
        if not self.text.strip():
            raise ValueError("translated text must not be empty")


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    phrase_id: str
    pcm: bytes
    sample_rate: int
    channels: int
    sample_width: int
    target_language: str
    completed_at: float
    synthesis_time_ms: float
    model: str

    def __post_init__(self) -> None:
        _validate_phrase_id(self.phrase_id)
        if self.target_language not in SPEECH_LANGUAGES:
            raise ValueError(f"Unsupported target language: {self.target_language!r}")
        _timestamp(self.completed_at)
        _duration(self.synthesis_time_ms)
        if self.sample_rate <= 0 or self.channels <= 0 or self.sample_width <= 0:
            raise ValueError("audio format values must be positive")
        if len(self.pcm) % (self.channels * self.sample_width):
            raise ValueError("PCM byte length must contain complete audio frames")


@dataclass(frozen=True, slots=True)
class LatencyRecorded:
    """One measured processing or endpoint delay associated with a committed phrase."""

    phrase_id: str
    stage: str
    duration_ms: float
    recorded_at: float

    def __post_init__(self) -> None:
        _validate_phrase_id(self.phrase_id)
        if not self.stage.strip():
            raise ValueError("latency stage must not be empty")
        _duration(self.duration_ms)
        _timestamp(self.recorded_at)


@dataclass(frozen=True, slots=True)
class StatusChanged:
    status: SessionStatus
    message: str
    changed_at: float

    def __post_init__(self) -> None:
        _timestamp(self.changed_at)


@dataclass(frozen=True, slots=True)
class RecoverableError:
    component: str
    message: str
    occurred_at: float
    phrase_id: str | None = None
    details: str | None = None

    def __post_init__(self) -> None:
        _timestamp(self.occurred_at)
        if not self.component.strip() or not self.message.strip():
            raise ValueError("component and message must not be empty")
        if self.phrase_id is not None:
            _validate_phrase_id(self.phrase_id)


@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    requested_at: float
    reason: str = "user"

    def __post_init__(self) -> None:
        _timestamp(self.requested_at)


DomainEvent: TypeAlias = (
    AudioFrameEvent
    | TranscriptUpdated
    | PhraseCommitted
    | TranslationCompleted
    | SynthesizedAudio
    | LatencyRecorded
    | StatusChanged
    | RecoverableError
    | ShutdownRequested
)


def _timestamp(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("timestamp must be finite, non-negative monotonic seconds")


def _duration(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("duration must be finite and non-negative")


def _validate_phrase_id(value: str) -> None:
    if not value or not value.strip():
        raise ValueError("phrase_id must not be empty")
