"""Turn replaceable recognizer hypotheses into exactly-once phrase commits."""

from __future__ import annotations

from real_time_translation.asr.base import RecognitionUpdate
from real_time_translation.domain.languages import SUPPORTED_LANGUAGES, target_for
from real_time_translation.domain.segments import Phrase, TranscriptSegment


class PhraseCommitter:
    """Keep the latest ASR revision and commit it once at an audio endpoint.

    ASR updates replace the whole active hypothesis, so no token-diff heuristic is needed:
    only the most recent text is committed. An endpoint clears that hypothesis, allowing a
    later identical utterance to become a distinct phrase.
    """

    def __init__(self, source_language: str, target_language: str | None = None) -> None:
        fallback_target = target_for(source_language) if target_language is None else target_language
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_language!r}")
        if fallback_target not in SUPPORTED_LANGUAGES or fallback_target == source_language:
            raise ValueError(f"Unsupported target language for source {source_language!r}.")
        self.source_language = source_language
        self.target_language = target_language or fallback_target
        self._started_at: float | None = None
        self._latest: RecognitionUpdate | None = None
        self._last_revision = -1
        self._next_id = 1

    @property
    def active(self) -> bool:
        return self._started_at is not None

    def observe(self, update: RecognitionUpdate, started_at: float | None = None) -> TranscriptSegment:
        """Record a hypothesis revision and return its immutable UI representation."""
        if update.language != self.source_language:
            raise ValueError(
                f"Recognizer language {update.language!r} does not match "
                f"session language {self.source_language!r}."
            )
        if update.revision <= self._last_revision:
            # Stale revisions can arrive from an adapter callback after a newer update.
            if self._latest is not None and self._started_at is not None:
                return self._segment(self._latest, self._started_at)
            return self._segment(update, update.updated_at)
        if self._started_at is None:
            self._started_at = update.updated_at if started_at is None else started_at
        self._last_revision = update.revision
        self._latest = update
        return self._segment(update, self._started_at)

    def commit(self, committed_at: float) -> Phrase | None:
        """Commit the current final or latest partial text and reset the active span."""
        latest, started_at = self._latest, self._started_at
        self.reset()
        if latest is None or started_at is None or not latest.text.strip():
            return None
        phrase = Phrase(
            phrase_id=f"phrase-{self._next_id}",
            text=latest.text.strip(),
            source_language=self.source_language,
            target_language=self.target_language,
            started_at=started_at,
            committed_at=committed_at,
        )
        self._next_id += 1
        return phrase

    def reset(self) -> None:
        """Discard the active hypothesis, for cancellation or a completed endpoint."""
        self._started_at = None
        self._latest = None
        self._last_revision = -1

    @staticmethod
    def _segment(update: RecognitionUpdate, started_at: float) -> TranscriptSegment:
        return TranscriptSegment(
            text=update.text,
            language=update.language,
            started_at=started_at,
            updated_at=update.updated_at,
            revision=update.revision,
            is_final=update.is_final,
        )
