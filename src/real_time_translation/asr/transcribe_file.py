"""Read long local WAV recordings and send them through the active recognizer."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Any


def transcribe_wav(path: str | Path, recognizer: Any) -> str:
    """Transcribe a local PCM WAV in bounded frames using the app's ASR adapter."""
    import numpy as np

    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            if audio.getcomptype() != "NONE" or channels not in (1, 2):
                raise ValueError("Transcription needs an uncompressed mono or stereo PCM WAV.")
            if sample_width not in (2, 3, 4) or sample_rate < 1:
                raise ValueError("Transcription needs 16-bit, 24-bit, or 32-bit PCM WAV audio.")
            raw = audio.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("Could not read the voice recording as PCM WAV audio.") from error

    waveform = _decode_pcm(raw, sample_width, channels)
    if sample_rate != recognizer.sample_rate:
        waveform = _resample(waveform, sample_rate, recognizer.sample_rate)
    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)
    frame_size = recognizer.sample_rate  # one second at the ASR's expected rate
    captured_at = time.monotonic()
    try:
        for offset in range(0, waveform.size, frame_size):
            recognizer.accept_audio(
                waveform[offset:offset + frame_size], recognizer.sample_rate, captured_at
            )
        result = recognizer.finish(time.monotonic())
        text = " ".join(str(result.text).split()) if result is not None else ""
        if not text:
            raise RuntimeError("The selected ASR model did not detect any speech in this recording.")
        return text
    finally:
        recognizer.close()


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> Any:
    import numpy as np

    if sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (bytes_[:, 0].astype(np.int32)
                  | (bytes_[:, 1].astype(np.int32) << 8)
                  | (bytes_[:, 2].astype(np.int32) << 16))
        values = np.where(values & 0x800000, values - 0x1000000, values)
        values = values.astype(np.float32) / 8388608.0
    else:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    return values.reshape(-1, channels).mean(axis=1)


def _resample(waveform: Any, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    if not waveform.size:
        return waveform.astype(np.float32)
    output_size = max(1, round(waveform.size * target_rate / source_rate))
    positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    return np.interp(positions, np.arange(waveform.size), waveform).astype(np.float32)
