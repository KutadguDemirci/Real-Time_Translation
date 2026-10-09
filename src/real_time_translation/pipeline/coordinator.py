"""Session lifecycle and ordered phrase processing for local live translation."""

from __future__ import annotations

from collections.abc import Callable
from collections import deque
import logging
import queue
import threading
import time
from typing import Any

from real_time_translation.audio.capture import AudioCapture, AudioFrame
from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.audio.vad import VoiceActivityDetector
from real_time_translation.asr.base import StreamingRecognizer
from real_time_translation.domain.events import (
    DomainEvent, LatencyRecorded, PhraseCommitted, RecoverableError, SessionStatus, StatusChanged,
    SynthesizedAudio, TranscriptUpdated, TranslationCompleted,
)
from real_time_translation.domain.languages import target_for
from real_time_translation.domain.segments import Phrase
from real_time_translation.pipeline.phrase_committer import PhraseCommitter
from real_time_translation.pipeline.timing import TimingCollector
from real_time_translation.translation.base import Translator
from real_time_translation.tts.base import SpeechSynthesizer

logger = logging.getLogger(__name__)


class SessionCoordinator:
    """Connect microphone capture, VAD, streaming ASR, translation, TTS and playback.

    ``start`` returns after devices/models are ready. Audio recognition runs on a dedicated
    worker, while a second worker serializes committed phrases. UI code receives immutable
    domain events through ``on_event`` and never runs inference itself.
    """

    def __init__(
        self,
        capture: AudioCapture,
        vad: VoiceActivityDetector,
        recognizer: StreamingRecognizer,
        translator: Translator,
        synthesizer: SpeechSynthesizer,
        playback: AudioPlayback,
        source_language: str,
        *,
        on_event: Callable[[DomainEvent], None] | None = None,
        phrase_queue_size: int = 8,
        frame_timeout: float = 0.1,
        timing: TimingCollector | None = None,
    ) -> None:
        target_for(source_language)
        if recognizer.language != source_language:
            raise ValueError("Recognizer language must match the selected source language.")
        if phrase_queue_size < 1 or frame_timeout <= 0:
            raise ValueError("phrase_queue_size and frame_timeout must be positive.")
        self.capture = capture
        self.vad = vad
        self.recognizer = recognizer
        self._listening_detail = (
            "Listening · transcript appears after each phrase"
            if getattr(recognizer, "mode", "streaming") == "phrase"
            else "Listening"
        )
        self.translator = translator
        self.synthesizer = synthesizer
        self.playback = playback
        self.source_language = source_language
        self.target_language = target_for(source_language)
        self.on_event = on_event or (lambda event: None)
        self.frame_timeout = frame_timeout
        self.timing = timing or TimingCollector()
        self.committer = PhraseCommitter(source_language)
        self._phrases: queue.Queue[Phrase | None] = queue.Queue(phrase_queue_size)
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._state = SessionStatus.IDLE
        self._started_once = False
        self._audio_thread: threading.Thread | None = None
        self._phrase_thread: threading.Thread | None = None
        self._active_latency: dict[str, float] | None = None
        preroll_frames = max(1, round(100 / max(1, self.vad.frame_ms)))
        self._speech_preroll: deque[AudioFrame] = deque(maxlen=preroll_frames)
        self._active_phrase_started_at: float | None = None
        self._last_speech_at: float | None = None
        self._phrase_speech_end: dict[str, float | None] = {}
        self._phrase_queued_at: dict[str, float] = {}
        self._playback_enqueued: dict[str, float] = {}
        self._playback_started: dict[str, float] = {}
        # The playback worker can report when sounddevice has accepted a phrase for output.
        # This lets the UI show actual pipeline-to-playback latency, not only synthesis time.
        if hasattr(self.playback, "on_playback_started"):
            self.playback.on_playback_started = self._on_playback_started
            self.playback.on_playback_finished = self._on_playback_finished

    @property
    def status(self) -> SessionStatus:
        with self._state_lock:
            return self._state

    def start(self) -> None:
        """Load the selected direction, start playback/capture, and launch workers."""
        with self._state_lock:
            if self._state in (SessionStatus.STARTING, SessionStatus.LISTENING,
                               SessionStatus.PROCESSING):
                return
            if self._started_once:
                raise RuntimeError("A stopped translation session cannot be restarted; create a new coordinator.")
            self._started_once = True
            self._state = SessionStatus.STARTING
        self._emit(StatusChanged(SessionStatus.STARTING, "Preparing local translation", time.monotonic()))
        self._stop.clear()
        self.committer.reset()
        try:
            self.translator.load(self.source_language, self.target_language)
            self.synthesizer.load(self.target_language)
            self.capture.start()
            self._phrase_thread = threading.Thread(
                target=self._phrase_loop, name="translation-phrases", daemon=True
            )
            self._audio_thread = threading.Thread(
                target=self._audio_loop, name="translation-audio", daemon=True
            )
            self._phrase_thread.start()
            self._audio_thread.start()
            self._set_status(SessionStatus.LISTENING, self._listening_detail)
        except Exception as error:
            self._stop.set()
            self._cleanup_devices()
            self._unload_models()
            self._set_status(SessionStatus.ERROR, "Could not start translation")
            self._error("session", str(error))
            raise

    def stop(self, *, flush_playback: bool = True, timeout: float = 5.0) -> None:
        """Stop capture, flush the active recognizer span, drain phrases, then release I/O."""
        if self.status in (SessionStatus.IDLE, SessionStatus.STOPPED):
            return
        self._set_status(SessionStatus.STOPPING, "Stopping translation")
        self._stop.set()
        self.capture.stop()
        self._join(self._audio_thread, timeout)
        current = time.monotonic()
        flush_started = time.perf_counter()
        try:
            update = self.recognizer.finish(current)
            if update is not None:
                self._publish_update(update, self._active_phrase_started_at)
        except Exception as error:
            self._error("asr", f"Could not finalize the active phrase: {error}")
        finally:
            flush_ms = (time.perf_counter() - flush_started) * 1000.0
        try:
            flush_started = time.perf_counter()
            phrase = self.committer.commit(current)
            commit_ms = (time.perf_counter() - flush_started) * 1000.0
            if phrase is not None:
                metrics = self._take_active_latency()
                metrics["Endpoint silence"] = float(self.vad.trailing_silence_ms)
                metrics["ASR flush"] = flush_ms
                metrics["Phrase commit"] = commit_ms
                self._publish_phrase(phrase, metrics, self._last_speech_at)
            else:
                self._active_latency = None
            self._active_phrase_started_at = None
        except Exception as error:
            self._error("asr", f"Could not commit the active phrase: {error}")
        self._queue_sentinel(timeout)
        self._join(self._phrase_thread, timeout)
        if flush_playback:
            self.playback.flush()
        self._cleanup_devices()
        self._unload_models()
        self._set_status(SessionStatus.STOPPED, "Translation stopped")

    close = stop

    def process_frame(self, frame: AudioFrame) -> None:
        """Process one captured frame; exposed for deterministic integrations and adapters."""
        if self._stop.is_set():
            return
        vad_started = time.perf_counter()
        result = self.vad.process(frame.samples)
        vad_ms = (time.perf_counter() - vad_started) * 1000.0
        if result.started:
            self._active_latency = {
                "Audio backlog": 0.0,
                "VAD compute": 0.0,
                "Streaming ASR": 0.0,
            }
        if result.trailing_silence_ms == 0:
            self._last_speech_at = frame.captured_at
        if self._active_latency is not None:
            self._active_latency["Audio backlog"] = max(
                self._active_latency["Audio backlog"],
                max(0.0, (time.monotonic() - frame.captured_at) * 1000.0),
            )
            self._active_latency["VAD compute"] += vad_ms
        if result.started:
            # Preserve a short onset window so the VAD's minimum-speech delay cannot clip
            # the first phoneme. Idle audio is not sent to the recognizer or retained longer.
            asr_frames = [*self._speech_preroll, frame]
            self._speech_preroll.clear()
            self._active_phrase_started_at = asr_frames[0].captured_at
        elif result.speech:
            asr_frames = [frame]
        else:
            self._speech_preroll.append(frame)
            asr_frames = []

        if asr_frames:
            asr_started = time.perf_counter()
            try:
                for asr_frame in asr_frames:
                    update = self.recognizer.accept_audio(
                        asr_frame.samples, asr_frame.sample_rate, asr_frame.captured_at
                    )
                    if update is not None:
                        self._publish_update(update, self._active_phrase_started_at)
            except Exception as error:
                self._error("asr", str(error))
            finally:
                if self._active_latency is not None:
                    self._active_latency["Streaming ASR"] += (time.perf_counter() - asr_started) * 1000.0
        if result.ended:
            self._speech_preroll.clear()
            endpoint = frame.captured_at
            flush_started = time.perf_counter()
            try:
                update = self.recognizer.finish(endpoint)
                if update is not None:
                    self._publish_update(update, self._active_phrase_started_at)
            except Exception as error:
                self._error("asr", f"Could not finalize phrase: {error}")
            flush_ms = (time.perf_counter() - flush_started) * 1000.0
            commit_started = time.perf_counter()
            phrase = self.committer.commit(endpoint)
            commit_ms = (time.perf_counter() - commit_started) * 1000.0
            if phrase is not None:
                metrics = self._take_active_latency()
                metrics["Endpoint silence"] = float(result.trailing_silence_ms)
                metrics["ASR flush"] = flush_ms
                metrics["Phrase commit"] = commit_ms
                self._publish_phrase(phrase, metrics, self._last_speech_at)
                self._set_status(SessionStatus.PROCESSING, "Translating committed phrase")
            else:
                self._active_latency = None
                self._set_status(SessionStatus.LISTENING, self._listening_detail)
            self._active_phrase_started_at = None

    def _audio_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.capture.read(timeout=self.frame_timeout)
            except queue.Empty:
                continue
            except Exception as error:
                self._error("audio", f"Microphone capture stopped: {error}")
                break
            self.process_frame(frame)

    def _phrase_loop(self) -> None:
        while True:
            phrase = self._phrases.get()
            if phrase is None:
                return
            self._process_phrase(phrase)

    def _process_phrase(self, phrase: Phrase) -> None:
        speech_end_at = self._phrase_speech_end.get(phrase.phrase_id)
        phrase_started = time.monotonic()
        queued_at = self._phrase_queued_at.pop(phrase.phrase_id, phrase.committed_at)
        self._record_latency(
            phrase.phrase_id,
            "Phrase queue",
            max(0.0, (phrase_started - queued_at) * 1000.0),
        )
        self.timing.begin_phrase(phrase.phrase_id, phrase.started_at, phrase.committed_at)
        try:
            self.timing.mark(phrase.phrase_id, "translation_started")
            translation_started = time.perf_counter()
            translated = self.translator.translate(phrase.text)
            translation_wall_ms = (time.perf_counter() - translation_started) * 1000.0
            translated_at = time.monotonic()
            self.timing.mark(phrase.phrase_id, "translation_completed", translated_at)
            event = TranslationCompleted(
                phrase_id=phrase.phrase_id,
                text=translated.text.strip(),
                source_language=phrase.source_language,
                target_language=phrase.target_language,
                completed_at=translated_at,
                inference_time_ms=translated.inference_time_ms,
                model=translated.model,
            )
            self._emit(event)
            self._record_latency(phrase.phrase_id, "MT inference", translated.inference_time_ms)
            self._record_latency(phrase.phrase_id, "MT total", translation_wall_ms)
            self.timing.mark(phrase.phrase_id, "synthesis_started")
            synthesis_started = time.perf_counter()
            speech = self.synthesizer.synthesize(event.text)
            synthesis_wall_ms = (time.perf_counter() - synthesis_started) * 1000.0
            speech_at = time.monotonic()
            self.timing.mark(phrase.phrase_id, "synthesis_completed", speech_at)
            self._emit(SynthesizedAudio(
                phrase_id=phrase.phrase_id, pcm=speech.pcm, sample_rate=speech.sample_rate,
                channels=speech.channels, sample_width=speech.sample_width,
                target_language=speech.target_lang, completed_at=speech_at,
                synthesis_time_ms=speech.synthesis_time_ms, model=speech.model,
            ))
            self._record_latency(phrase.phrase_id, "TTS inference", speech.synthesis_time_ms)
            self._record_latency(phrase.phrase_id, "TTS total", synthesis_wall_ms)
            enqueue_started = time.perf_counter()
            self._playback_enqueued[phrase.phrase_id] = time.monotonic()
            try:
                if hasattr(self.playback, "on_playback_started"):
                    self.playback.play(speech, timeout=2.0, phrase_id=phrase.phrase_id)
                else:
                    self.playback.play(speech, timeout=2.0)
            except Exception:
                self._playback_enqueued.pop(phrase.phrase_id, None)
                self._phrase_speech_end.pop(phrase.phrase_id, None)
                raise
            enqueue_ms = (time.perf_counter() - enqueue_started) * 1000.0
            self._record_latency(phrase.phrase_id, "Playback enqueue", enqueue_ms)
            # A no-op fallback keeps custom playback adapters useful if they do not expose
            # the AudioPlayback start callback.
            if not hasattr(self.playback, "on_playback_started"):
                playback_started = time.monotonic()
                self._record_latency(
                    phrase.phrase_id,
                    "Playback queue",
                    max(0.0, (playback_started - self._playback_enqueued.pop(phrase.phrase_id, playback_started)) * 1000.0),
                )
                if speech_end_at is not None:
                    self._record_latency(phrase.phrase_id, "Speech → playback", max(0.0, (playback_started - speech_end_at) * 1000.0))
                self._phrase_speech_end.pop(phrase.phrase_id, None)
            self.timing.mark(phrase.phrase_id, "playback_enqueued")
            self._set_status(SessionStatus.LISTENING, self._listening_detail)
        except Exception as error:
            self._error("translation_pipeline", str(error), phrase.phrase_id)
            self._set_status(SessionStatus.LISTENING, "Listening after a phrase error")

    def _publish_update(self, update: Any, started_at: float | None = None) -> None:
        try:
            segment = self.committer.observe(update, started_at=started_at)
        except (TypeError, ValueError) as error:
            self._error("asr", f"Invalid recognizer update: {error}")
            return
        self._emit(TranscriptUpdated(segment))

    def _publish_phrase(
        self, phrase: Phrase, metrics: dict[str, float] | None = None,
        speech_end_at: float | None = None,
    ) -> None:
        self._phrase_speech_end[phrase.phrase_id] = speech_end_at
        self._emit(PhraseCommitted(phrase))
        for stage, duration_ms in (metrics or {}).items():
            self._record_latency(phrase.phrase_id, stage, duration_ms)
        self._phrase_queued_at[phrase.phrase_id] = time.monotonic()
        try:
            self._phrases.put(phrase, timeout=0.25)
        except queue.Full:
            self._phrase_queued_at.pop(phrase.phrase_id, None)
            self._phrase_speech_end.pop(phrase.phrase_id, None)
            self._error(
                "pipeline", "Phrase queue is full; this phrase was not translated. "
                "Pause briefly to let translation catch up.", phrase.phrase_id,
            )

    def _take_active_latency(self) -> dict[str, float]:
        metrics, self._active_latency = self._active_latency or {}, None
        return metrics

    def _record_latency(self, phrase_id: str, stage: str, duration_ms: float) -> None:
        self._emit(LatencyRecorded(phrase_id, stage, max(0.0, duration_ms), time.monotonic()))

    def _on_playback_started(self, phrase_id: str | None, started_at: float) -> None:
        if phrase_id is None:
            return
        enqueued_at = self._playback_enqueued.pop(phrase_id, started_at)
        self._record_latency(phrase_id, "Playback queue", max(0.0, (started_at - enqueued_at) * 1000.0))
        speech_end_at = self._phrase_speech_end.pop(phrase_id, None)
        if speech_end_at is not None:
            self._record_latency(phrase_id, "Speech → playback", max(0.0, (started_at - speech_end_at) * 1000.0))
        self._playback_started[phrase_id] = started_at

    def _on_playback_finished(self, phrase_id: str | None, finished_at: float) -> None:
        if phrase_id is None:
            return
        started_at = self._playback_started.pop(phrase_id, None)
        if started_at is not None:
            self._record_latency(
                phrase_id,
                "Playback duration",
                max(0.0, (finished_at - started_at) * 1000.0),
            )

    def _queue_sentinel(self, timeout: float) -> None:
        if self._phrase_thread is None or not self._phrase_thread.is_alive():
            return
        try:
            self._phrases.put(None, timeout=timeout)
        except queue.Full:
            self._error("pipeline", "Timed out while draining the phrase queue.")

    def _cleanup_devices(self) -> None:
        for component in (self.capture, self.playback):
            try:
                component.close()
            except Exception:
                logger.exception("Error while closing %s", type(component).__name__)

    def _unload_models(self) -> None:
        for component in (self.recognizer, self.translator, self.synthesizer):
            try:
                component.close() if component is self.recognizer else component.unload()
            except Exception:
                logger.exception("Error while releasing %s", type(component).__name__)

    @staticmethod
    def _join(thread: threading.Thread | None, timeout: float) -> None:
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))

    def _set_status(self, status: SessionStatus, message: str) -> None:
        with self._state_lock:
            self._state = status
        self._emit(StatusChanged(status, message, time.monotonic()))

    def _error(self, component: str, message: str, phrase_id: str | None = None) -> None:
        logger.warning("%s: %s", component, message)
        self._emit(RecoverableError(component, message, time.monotonic(), phrase_id))

    def _emit(self, event: DomainEvent) -> None:
        try:
            self.on_event(event)
        except Exception:
            # A UI observer must not bring down microphone or inference workers.
            logger.exception("Translation event listener failed")
