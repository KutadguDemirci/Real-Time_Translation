"""Strict local reference-audio validation and temporary copies for voice conditioning."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import wave
from collections.abc import Iterator


MAX_REFERENCE_BYTES = 40 * 1024 * 1024
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ReferenceAudioInfo:
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int


def validate_reference_audio(path: str | Path) -> ReferenceAudioInfo:
    """Validate actual WAV content, not just a filename or UI file filter."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Choose a readable local WAV recording.")
    if source.suffix.lower() != ".wav":
        raise ValueError("Voice references must be PCM WAV files.")
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ValueError("The selected recording could not be read.") from error
    if size <= 0 or size > MAX_REFERENCE_BYTES:
        raise ValueError("The WAV file must be between 1 byte and 40 MB.")

    try:
        with wave.open(str(source), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            if compression != "NONE":
                raise ValueError("Use an uncompressed PCM WAV recording.")
            if not 1 <= channels <= 2:
                raise ValueError("The recording must have one or two channels.")
            if not 16_000 <= sample_rate <= 48_000:
                raise ValueError("The recording sample rate must be between 16 and 48 kHz.")
            if sample_width not in (2, 3, 4):
                raise ValueError("Use 16-bit, 24-bit, or 32-bit PCM audio.")
            duration = frame_count / sample_rate if sample_rate else 0.0
            if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
                raise ValueError("Use a clear recording between 2 seconds and 5 minutes long.")
            frames = audio.readframes(frame_count)
    except ValueError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("The selected file is not a readable PCM WAV recording.") from error

    if len(frames) != frame_count * channels * sample_width:
        raise ValueError("The WAV file is incomplete or corrupted.")
    if not _has_audible_content(frames, sample_width):
        raise ValueError("The recording is silent or too quiet. Choose a clear spoken sample.")
    return ReferenceAudioInfo(duration, sample_rate, channels, sample_width)


@contextmanager
def private_reference_copy(path: str | Path) -> Iterator[Path]:
    """Copy validated audio into an app-owned private temp directory, then remove it."""
    source = Path(path)
    validate_reference_audio(source)
    directory = Path(tempfile.mkdtemp(prefix="local-translator-voice-"))
    try:
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        destination = directory / "reference.wav"
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            try:
                os.chmod(destination, 0o600)
            except OSError:
                pass
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        validate_reference_audio(destination)
        yield destination
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _has_audible_content(frames: bytes, sample_width: int) -> bool:
    """Reject silence and near-silence while accepting ordinary quiet speech."""
    import numpy as np

    if sample_width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.int32)
        return bool(values.size and np.max(np.abs(values)) >= 164)
    if sample_width == 3:
        bytes_ = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (bytes_[:, 0].astype(np.int32)
                  | (bytes_[:, 1].astype(np.int32) << 8)
                  | (bytes_[:, 2].astype(np.int32) << 16))
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return bool(values.size and np.max(np.abs(values)) >= 4195)
    values = np.frombuffer(frames, dtype="<i4").astype(np.int64)
    return bool(values.size and np.max(np.abs(values)) >= 1_073_741)
