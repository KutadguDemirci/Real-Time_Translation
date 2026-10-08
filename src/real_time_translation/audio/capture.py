"""Bounded, mono microphone capture with conversion to recognizer-rate float PCM."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    channels: int
    default_sample_rate: int


@dataclass(frozen=True)
class AudioFrame:
    """A mono float32 frame in the configured recognizer sample rate."""

    samples: Any
    sample_rate: int
    captured_at: float


class AudioCapture:
    """Capture frames on the PortAudio callback and expose them through a bounded queue.

    ``sounddevice`` and NumPy are optional until an audio device is actually opened.
    Overflow drops the oldest frame so a slow recognizer catches up to live speech.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        device: int | str | None = None,
        queue_size: int = 50,
    ) -> None:
        if sample_rate <= 0 or frame_ms <= 0 or 1000 % frame_ms:
            raise ValueError("sample_rate must be positive and frame_ms must divide 1000.")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1.")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.device = device
        self.frames: queue.Queue[AudioFrame] = queue.Queue(maxsize=queue_size)
        self._stream: Any = None
        self._closed = threading.Event()
        self._status = "stopped"
        self._status_lock = threading.Lock()
        self.dropped_frames = 0

    @staticmethod
    def enumerate_devices() -> list[AudioDevice]:
        """List input-capable devices, or explain how to enable audio support."""
        sd, _ = _audio_modules()
        try:
            return [
                AudioDevice(i, str(item["name"]), int(item["max_input_channels"]),
                            int(item["default_samplerate"]))
                for i, item in enumerate(sd.query_devices())
                if int(item["max_input_channels"]) > 0
            ]
        except Exception as error:
            raise RuntimeError(f"Could not enumerate microphone devices: {error}") from error

    @property
    def status(self) -> str:
        with self._status_lock:
            return self._status

    def start(self) -> None:
        if self._stream is not None:
            return
        sd, np = _audio_modules()
        self._closed.clear()
        blocksize = self.sample_rate * self.frame_ms // 1000
        try:
            info = sd.query_devices(self.device, "input")
            native_rate = int(info["default_samplerate"])
            channels = min(2, int(info["max_input_channels"]))
            if channels < 1:
                raise RuntimeError("The selected device has no input channels.")
            sd.check_input_settings(device=self.device, channels=channels,
                                    dtype="float32", samplerate=native_rate)

            def callback(indata: Any, frames: int, timing: Any, status: Any) -> None:
                if self._closed.is_set():
                    raise sd.CallbackStop
                if status:
                    with self._status_lock:
                        self._status = f"audio callback warning: {status}"
                # The callback only averages channels, resamples and queues fixed frames.
                mono = np.asarray(indata, dtype=np.float32).mean(axis=1)
                if native_rate != self.sample_rate and mono.size:
                    size = max(1, round(mono.size * self.sample_rate / native_rate))
                    mono = np.interp(np.linspace(0, mono.size - 1, size),
                                     np.arange(mono.size), mono).astype(np.float32)
                for offset in range(0, mono.size - blocksize + 1, blocksize):
                    frame = AudioFrame(mono[offset:offset + blocksize].copy(), self.sample_rate,
                                       time.monotonic())
                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        self.dropped_frames += 1
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            self.dropped_frames += 1

            self._stream = sd.InputStream(device=self.device, samplerate=native_rate,
                                          channels=channels, dtype="float32", callback=callback)
            self._stream.start()
        except Exception as error:
            self._stream = None
            with self._status_lock:
                self._status = "error"
            raise RuntimeError(
                f"Could not start microphone capture: {error}. Check microphone selection, "
                "system privacy permissions, and whether another application has exclusive access."
            ) from error
        with self._status_lock:
            self._status = "running"

    def read(self, timeout: float | None = None) -> AudioFrame:
        """Wait for the next frame; raise queue.Empty when the timeout expires."""
        return self.frames.get(timeout=timeout)

    def stop(self) -> None:
        self._closed.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        with self._status_lock:
            self._status = "stopped"

    close = stop

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _audio_modules() -> tuple[Any, Any]:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Audio support is not installed. Install the audio extra with "
            "`pip install -e .[audio]`."
        ) from error
    return sd, np
