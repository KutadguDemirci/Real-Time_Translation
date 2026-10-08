"""Sherpa-ONNX streaming transducer adapter.

The adapter intentionally accepts an explicit model file set. The project manifest owns model
identity, hashes, and licensing; this module only checks the files needed by the runtime. Sherpa
and NumPy are imported lazily so non-ASR parts of the app remain importable without the extra.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

from real_time_translation.asr.base import RecognitionUpdate
from real_time_translation.domain.languages import SUPPORTED_LANGUAGES


@dataclass(frozen=True, slots=True)
class TransducerModel:
    """Paths for one Sherpa-ONNX streaming transducer model."""

    tokens: Path
    encoder: Path
    decoder: Path
    joiner: Path

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (self.tokens, self.encoder, self.decoder, self.joiner)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Streaming ASR model files are missing: " + ", ".join(missing)
                + ". Install the selected local Sherpa-ONNX model and configure its exact files."
            )

    @classmethod
    def discover(cls, directory: str | Path) -> "TransducerModel":
        """Resolve common Sherpa transducer filenames, rejecting ambiguous matches."""
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"ASR model directory does not exist: {root}. See models/README.md."
            )

        def choose(role: str, patterns: tuple[str, ...]) -> Path:
            matches = sorted(
                {path for pattern in patterns for path in root.rglob(pattern) if path.is_file()}
            )
            if not matches:
                raise FileNotFoundError(
                    f"Could not find the {role} file under {root}. Configure the exact "
                    "Sherpa-ONNX model file paths."
                )
            if len(matches) > 1:
                rendered = ", ".join(str(path.relative_to(root)) for path in matches)
                raise ValueError(f"Multiple {role} files found under {root}: {rendered}")
            return matches[0]

        return cls(
            tokens=choose("tokens", ("tokens.txt",)),
            encoder=choose("encoder", ("encoder*.onnx",)),
            decoder=choose("decoder", ("decoder*.onnx",)),
            joiner=choose("joiner", ("joiner*.onnx",)),
        )


class SherpaStreamingRecognizer:
    """Incremental mono 16 kHz float32 recognition using Sherpa-ONNX transducers."""

    sample_rate = 16_000

    def __init__(
        self,
        model: TransducerModel,
        language: str,
        *,
        num_threads: int | None = None,
        provider: str = "cpu",
        sherpa_module: Any | None = None,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported ASR language {language!r}; expected en or fr.")
        if provider not in ("cpu", "cuda", "coreml"):
            raise ValueError("provider must be one of: cpu, cuda, coreml")
        if num_threads is not None and num_threads < 1:
            raise ValueError("num_threads must be positive when specified")
        model.validate()
        if sherpa_module is None:
            try:
                import sherpa_onnx as sherpa_module
            except ImportError as error:
                raise RuntimeError(
                    "Sherpa-ONNX is not installed. Install ASR support with "
                    "`pip install -e .[inference]`."
                ) from error

        self.language = language
        self._np = _load_numpy()
        threads = num_threads or max(1, (os.cpu_count() or 1) // 2)
        try:
            self._recognizer = sherpa_module.OnlineRecognizer.from_transducer(
                tokens=str(model.tokens),
                encoder=str(model.encoder),
                decoder=str(model.decoder),
                joiner=str(model.joiner),
                num_threads=threads,
                sample_rate=self.sample_rate,
                enable_endpoint_detection=False,
                provider=provider,
                decoding_method="greedy_search",
            )
            self._stream = self._recognizer.create_stream()
        except Exception as error:
            raise RuntimeError(
                f"Could not initialize the {language} Sherpa-ONNX streaming recognizer "
                f"using provider {provider!r}: {error}. Check model compatibility and runtime."
            ) from error
        self._revision = 0
        self._last_text = ""
        self._closed = False

    def accept_audio(
        self, samples: Any, sample_rate: int, captured_at: float
    ) -> RecognitionUpdate | None:
        """Decode a frame and return only changed, normalized partial text."""
        self._ensure_open()
        _validate_timestamp(captured_at)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"ASR expects {self.sample_rate} Hz audio; received {sample_rate} Hz. "
                "Resample before calling the recognizer."
            )
        waveform = self._to_waveform(samples)
        if waveform.size == 0:
            return None
        try:
            self._stream.accept_waveform(sample_rate, waveform)
            self._decode_ready()
            text = _clean_text(self._recognizer.get_result(self._stream))
        except Exception as error:
            raise RuntimeError(
                f"Sherpa-ONNX failed while decoding {self.language} audio: {error}"
            ) from error
        return self._update_if_changed(text, captured_at, is_final=False)

    def finish(self, captured_at: float) -> RecognitionUpdate | None:
        """Signal end of input, drain decoder, emit a final result, and reset the stream."""
        self._ensure_open()
        _validate_timestamp(captured_at)
        try:
            self._stream.input_finished()
            self._decode_ready()
            text = _clean_text(self._recognizer.get_result(self._stream))
        except Exception as error:
            self._reset_stream()
            raise RuntimeError(f"Sherpa-ONNX failed to finalize {self.language} audio: {error}") from error

        result = self._update_if_changed(text, captured_at, is_final=True) if text else None
        self._reset_stream()
        return result

    def close(self) -> None:
        """Drop stream and recognizer references; sherpa owns native object cleanup."""
        self._closed = True
        self._stream = None
        self._recognizer = None
        self._last_text = ""

    def _decode_ready(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _reset_stream(self) -> None:
        try:
            if self._stream is not None and self._recognizer is not None:
                self._recognizer.reset(self._stream)
        finally:
            self._last_text = ""

    def _update_if_changed(
        self, text: str, timestamp: float, *, is_final: bool
    ) -> RecognitionUpdate | None:
        if not text or (text == self._last_text and not is_final):
            return None
        self._revision += 1
        self._last_text = text
        return RecognitionUpdate(
            text=text,
            language=self.language,
            updated_at=timestamp,
            revision=self._revision,
            is_final=is_final,
        )

    def _to_waveform(self, samples: Any) -> Any:
        try:
            waveform = self._np.asarray(samples, dtype=self._np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Audio samples must be a one-dimensional float waveform.") from error
        if waveform.ndim != 1:
            raise ValueError("ASR accepts mono audio only; provide a one-dimensional waveform.")
        if not self._np.isfinite(waveform).all():
            raise ValueError("Audio samples must contain only finite values.")
        if waveform.size and (float(waveform.min()) < -1.0 or float(waveform.max()) > 1.0):
            raise ValueError("Float audio samples must be normalized to the range [-1, 1].")
        return self._np.ascontiguousarray(waveform)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The ASR recognizer is closed.")


def _clean_text(value: Any) -> str:
    """Normalize whitespace without changing recognizer punctuation or casing."""
    if hasattr(value, "text"):
        value = value.text
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _validate_timestamp(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("captured_at must be finite, non-negative monotonic seconds")


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for streaming ASR. Install ASR support with "
            "`pip install -e .[inference]`."
        ) from error
    return np
