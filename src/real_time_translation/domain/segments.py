"""Immutable transcript and phrase values shared across the translation pipeline."""

from dataclasses import dataclass
import math

from .languages import target_for


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A recognizer hypothesis; later revisions may replace earlier text."""

    text: str
    language: str
    started_at: float
    updated_at: float
    revision: int = 0
    is_final: bool = False

    def __post_init__(self) -> None:
        target_for(self.language)
        _validate_times(self.started_at, self.updated_at)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")


@dataclass(frozen=True, slots=True)
class Phrase:
    """Stable source phrase committed exactly once for downstream processing."""

    phrase_id: str
    text: str
    source_language: str
    target_language: str
    started_at: float
    committed_at: float

    def __post_init__(self) -> None:
        expected_target = target_for(self.source_language)
        if self.target_language != expected_target:
            raise ValueError(
                f"target language for {self.source_language!r} must be {expected_target!r}"
            )
        if not self.phrase_id or not self.phrase_id.strip():
            raise ValueError("phrase_id must not be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("phrase text must not be empty")
        _validate_times(self.started_at, self.committed_at)


def _validate_times(started_at: float, ended_at: float) -> None:
    if not (math.isfinite(started_at) and math.isfinite(ended_at)):
        raise ValueError("timestamps must be finite monotonic seconds")
    if started_at < 0 or ended_at < started_at:
        raise ValueError("timestamps must be non-negative and ordered")
