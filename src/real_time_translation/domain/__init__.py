"""Typed domain objects shared by UI, pipeline, and model adapters."""

from .events import (
    AudioFrame,
    AudioFrameEvent,
    DomainEvent,
    LatencyRecorded,
    PhraseCommitted,
    RecoverableError,
    SessionStatus,
    ShutdownRequested,
    StatusChanged,
    SynthesizedAudio,
    TranscriptUpdated,
    TranslationCompleted,
)
from .languages import SUPPORTED_LANGUAGES, target_for
from .segments import Phrase, TranscriptSegment

__all__ = [
    "AudioFrame",
    "AudioFrameEvent",
    "DomainEvent",
    "LatencyRecorded",
    "Phrase",
    "PhraseCommitted",
    "RecoverableError",
    "SUPPORTED_LANGUAGES",
    "SessionStatus",
    "ShutdownRequested",
    "StatusChanged",
    "SynthesizedAudio",
    "TranscriptSegment",
    "TranscriptUpdated",
    "TranslationCompleted",
    "target_for",
]

