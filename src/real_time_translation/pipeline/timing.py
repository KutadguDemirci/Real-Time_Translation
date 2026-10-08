"""In-memory monotonic timing records for one phrase's local processing stages."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class PhraseTiming:
    phrase_id: str
    phrase_started_at: float
    phrase_committed_at: float
    translation_started_at: float | None = None
    translation_completed_at: float | None = None
    synthesis_started_at: float | None = None
    synthesis_completed_at: float | None = None
    playback_enqueued_at: float | None = None

    @property
    def phrase_to_translation_ms(self) -> float | None:
        return _elapsed_ms(self.phrase_committed_at, self.translation_started_at)

    @property
    def translation_ms(self) -> float | None:
        return _elapsed_ms(self.translation_started_at, self.translation_completed_at)

    @property
    def synthesis_ms(self) -> float | None:
        return _elapsed_ms(self.synthesis_started_at, self.synthesis_completed_at)


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


class TimingCollector:
    """Collect bounded timing snapshots without retaining audio or transcript content."""

    def __init__(self, max_records: int = 500) -> None:
        if max_records < 1:
            raise ValueError("max_records must be at least 1.")
        self.max_records = max_records
        self._records: dict[str, PhraseTiming] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def begin_phrase(self, phrase_id: str, started_at: float, committed_at: float) -> None:
        self._put(PhraseTiming(phrase_id, started_at, committed_at))

    def mark(self, phrase_id: str, stage: str, timestamp: float | None = None) -> PhraseTiming:
        """Set a stage boundary; stage is one of translation_started/completed,
        synthesis_started/completed, or playback_enqueued.
        """
        allowed = {
            "translation_started": "translation_started_at",
            "translation_completed": "translation_completed_at",
            "synthesis_started": "synthesis_started_at",
            "synthesis_completed": "synthesis_completed_at",
            "playback_enqueued": "playback_enqueued_at",
        }
        try:
            field_name = allowed[stage]
        except KeyError as error:
            raise ValueError(f"Unknown timing stage: {stage!r}.") from error
        at = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            current = self._records.get(phrase_id)
            if current is None:
                raise KeyError(f"No timing record for phrase {phrase_id!r}.")
            updated = PhraseTiming(
                **{name: getattr(current, name) for name in PhraseTiming.__dataclass_fields__}
                | {field_name: at}
            )
            self._records[phrase_id] = updated
            return updated

    def get(self, phrase_id: str) -> PhraseTiming | None:
        with self._lock:
            return self._records.get(phrase_id)

    def snapshot(self) -> tuple[PhraseTiming, ...]:
        with self._lock:
            return tuple(self._records[key] for key in self._order)

    def _put(self, record: PhraseTiming) -> None:
        with self._lock:
            if record.phrase_id not in self._records:
                self._order.append(record.phrase_id)
            self._records[record.phrase_id] = record
            while len(self._order) > self.max_records:
                expired = self._order.pop(0)
                self._records.pop(expired, None)
