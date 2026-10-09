"""Write microphone frames from the app's callback capture into a continuous WAV."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import queue
import threading
import wave
from typing import Any


def record_capture_to_wav(
    capture: Any,
    path: str | Path,
    stop_event: threading.Event,
    cancel_event: threading.Event,
    *,
    max_seconds: float,
    on_progress: Callable[[float], None] | None = None,
) -> float:
    """Drain callback-captured frames promptly and fail if the stream dropped any audio."""
    import numpy as np

    capture.start()
    sample_rate = capture.sample_rate
    samples_written = 0
    buffered: list[Any] = []
    buffered_samples = 0
    write_chunk_samples = max(1, sample_rate // 5)
    last_progress = 0.0
    input_status = ""
    dropped_frames = 0
    try:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)

            def flush() -> None:
                nonlocal buffered_samples
                if buffered:
                    samples = np.concatenate(buffered)
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
                    output.writeframesraw(pcm.tobytes())
                    buffered.clear()
                    buffered_samples = 0

            while not stop_event.is_set() and not cancel_event.is_set():
                duration = samples_written / sample_rate
                if duration >= max_seconds:
                    break
                try:
                    frame = capture.read(timeout=0.25)
                except queue.Empty:
                    continue
                values = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
                remaining = max(0, int(max_seconds * sample_rate) - samples_written)
                if values.size > remaining:
                    values = values[:remaining]
                if not values.size:
                    break
                buffered.append(values)
                buffered_samples += values.size
                samples_written += values.size
                if buffered_samples >= write_chunk_samples:
                    flush()
                duration = samples_written / sample_rate
                if on_progress is not None and duration - last_progress >= 1.0:
                    on_progress(duration)
                    last_progress = duration
            flush()
            output.writeframes(b"")
        input_status = str(capture.status)
        dropped_frames = int(capture.dropped_frames)
    finally:
        capture.stop()

    if cancel_event.is_set():
        return samples_written / sample_rate
    if dropped_frames:
        raise RuntimeError(
            f"Microphone capture dropped {dropped_frames} audio frames. "
            "The recording was discarded; try again with a different input device."
        )
    if input_status.startswith("audio callback warning"):
        raise RuntimeError(
            f"Microphone input reported an overrun ({input_status}). "
            "The recording was discarded; try again with a different input device."
        )
    return samples_written / sample_rate
