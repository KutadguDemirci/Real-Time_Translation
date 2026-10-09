"""Ordered, bounded PCM playback with mute, stop and stale-queue flushing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackItem:
    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2
    phrase_id: str | None = None


class AudioPlayback:
    """Play phrase PCM sequentially on a worker thread using ``sounddevice``."""

    def __init__(self, device: int | str | None = None, queue_size: int = 8) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1.")
        self.device = device
        self._queue: queue.Queue[PlaybackItem | None] = queue.Queue(queue_size)
        self._worker: threading.Thread | None = None
        self._lock = threading.RLock()
        self._muted = False
        self._closed = False
        self.last_error: str | None = None
        self.on_playback_started: Callable[[str | None, float], None] | None = None
        self.on_playback_finished: Callable[[str | None, float], None] | None = None

    @staticmethod
    def enumerate_devices() -> list[dict[str, Any]]:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("Install audio support with `pip install -e .[audio]`.") from error
        try:
            return [dict(id=i, name=str(d["name"]), channels=int(d["max_output_channels"]),
                         default_sample_rate=int(d["default_samplerate"]))
                    for i, d in enumerate(sd.query_devices()) if int(d["max_output_channels"]) > 0]
        except Exception as error:
            raise RuntimeError(f"Could not enumerate speaker devices: {error}") from error

    def enqueue(self, pcm: bytes | Any, sample_rate: int, channels: int = 1,
                sample_width: int = 2, timeout: float | None = None,
                phrase_id: str | None = None) -> None:
        """Queue one phrase; block up to ``timeout`` when the bounded queue is full."""
        if sample_rate <= 0 or channels < 1 or sample_width != 2:
            raise ValueError("Playback requires a positive sample rate and signed 16-bit PCM.")
        payload = pcm if isinstance(pcm, bytes) else getattr(pcm, "pcm", None)
        if not isinstance(payload, bytes):
            raise TypeError("pcm must be bytes or a SpeechResult containing PCM bytes.")
        if len(payload) % (channels * sample_width):
            raise ValueError("PCM byte length is not aligned to the channel/sample format.")
        if not payload:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Playback is closed.")
            self._ensure_worker()
        self._queue.put(
            PlaybackItem(payload, sample_rate, channels, sample_width, phrase_id),
            timeout=timeout,
        )

    def play(self, result: Any, timeout: float | None = None,
             phrase_id: str | None = None) -> None:
        """Enqueue a ``SpeechResult`` without requiring callers to unpack its metadata."""
        self.enqueue(result.pcm, result.sample_rate, result.channels,
                     result.sample_width, timeout=timeout, phrase_id=phrase_id)

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = bool(muted)

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    def flush(self) -> None:
        """Discard queued audio. Audio already being rendered may finish its current phrase."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self, flush: bool = True) -> None:
        if flush:
            self.flush()
        with self._lock:
            worker, self._worker = self._worker, None
            self._closed = True
        if worker is not None:
            self._queue.put(None)
            worker.join(timeout=5)
            if worker.is_alive():
                raise RuntimeError("Audio playback worker did not stop within five seconds.")

    close = stop

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="audio-playback", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self.muted:
                continue
            try:
                import numpy as np
                import sounddevice as sd

                samples = np.frombuffer(item.pcm, dtype="<i2").astype(np.float32) / 32768.0
                if item.channels > 1:
                    samples = samples.reshape(-1, item.channels)
                output_info = sd.query_devices(self.device, "output")
                output_rate = int(output_info["default_samplerate"])
                output_channels = min(2, int(output_info["max_output_channels"]))
                if output_channels < 1:
                    raise RuntimeError("The selected speaker has no output channels.")
                if item.sample_rate != output_rate:
                    old_length = samples.shape[0]
                    new_length = max(1, round(old_length * output_rate / item.sample_rate))
                    positions = np.linspace(0, old_length - 1, new_length)
                    if samples.ndim == 1:
                        samples = np.interp(positions, np.arange(old_length), samples)
                    else:
                        samples = np.column_stack([
                            np.interp(positions, np.arange(old_length), samples[:, channel])
                            for channel in range(samples.shape[1])
                        ])
                if samples.ndim == 1 and output_channels > 1:
                    samples = np.repeat(samples[:, None], output_channels, axis=1)
                elif samples.ndim == 2 and samples.shape[1] > output_channels:
                    samples = samples[:, :output_channels]
                elif samples.ndim == 2 and samples.shape[1] < output_channels:
                    samples = np.repeat(samples[:, :1], output_channels, axis=1)
                # Keep sounddevice's supported high-level playback path. The two
                # AudioPlayback instances share sounddevice's module-global stream,
                # so callers must serialize playback instead of opening competing
                # output streams.
                sd.play(samples, samplerate=output_rate, device=self.device, blocking=False)
                callback = self.on_playback_started
                if callback is not None:
                    try:
                        callback(item.phrase_id, time.monotonic())
                    except Exception:
                        logger.exception("Playback start callback failed")
                sd.wait()
                callback = self.on_playback_finished
                if callback is not None:
                    try:
                        callback(item.phrase_id, time.monotonic())
                    except Exception:
                        logger.exception("Playback completion callback failed")
                self.last_error = None
            except Exception as error:
                self.last_error = (
                    f"Could not play audio: {error}. Check speaker selection, output permissions, "
                    "and whether the device is available."
                )


