"""Qwen3-ASR adapter for phrase-preserving local speech recognition.

On Linux with vLLM installed, Qwen's native incremental decoder produces revisable partials.
On other supported app platforms, the Transformers backend transcribes the complete VAD phrase
at its endpoint. Both paths expose the same whole-hypothesis recognizer contract.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import platform
from pathlib import Path
from typing import Any

from real_time_translation.asr.base import RecognitionUpdate
from real_time_translation.domain.languages import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "tr": "Turkish", "de": "German",
    "es": "Spanish", "it": "Italian", "zh": "Chinese",
}


class QwenStreamingRecognizer:
    """Recognize one VAD-delimited phrase at a time with the local Qwen3-ASR model.

    Qwen's official streaming API is backed by vLLM. When it is unavailable, this adapter
    preserves the complete phrase and runs one Transformers transcription at ``finish``; it
    does not repeatedly decode growing audio prefixes or fabricate interim partials.
    """

    sample_rate = 16_000
    # Feed a short cadence while letting Qwen aggregate enough audio for its own
    # streaming chunks. Qwen's reference streaming example uses 2-second chunks.
    feed_chunk_seconds = 1.0
    model_chunk_seconds = 2.0
    unfixed_chunk_count = 2
    unfixed_token_count = 5

    def __init__(
        self,
        model_dir: str | Path,
        language: str,
        *,
        context: str = "",
        torch_device: str = "cpu",
        qwen_module: Any | None = None,
        numpy_module: Any | None = None,
        enable_streaming: bool | None = None,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported ASR language {language!r}; expected one of {', '.join(SUPPORTED_LANGUAGES)}.")
        if len(context) > 2000:
            raise ValueError("Qwen ASR context must be 2000 characters or fewer.")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Qwen3-ASR model directory does not exist: {self.model_dir}")
        self.language = language
        self._language_name = _LANGUAGE_NAMES[language]
        self._context = context.strip()
        self._np = numpy_module or _load_numpy()
        if qwen_module is None:
            try:
                import qwen_asr as qwen_module
            except ImportError as error:
                raise RuntimeError(
                    "Qwen3-ASR support is not installed. Install it with "
                    "`pip install -e .[qwen-asr]`."
                ) from error

        if enable_streaming is None:
            enable_streaming = _vllm_available()
        if enable_streaming and platform.system() != "Linux":
            raise RuntimeError(
                "Qwen3-ASR live streaming uses vLLM, which is not natively supported on "
                f"{platform.system()}. Qwen3-ASR can still run in phrase mode with the "
                "Transformers backend."
            )

        self._mode = "streaming" if enable_streaming else "phrase"
        try:
            if self._mode == "streaming":
                self._model = qwen_module.Qwen3ASRModel.LLM(
                    model=str(self.model_dir),
                    max_new_tokens=32,
                )
                self._state = self._new_stream_state()
            else:
                import torch

                dtype = torch.float16 if torch_device == "cuda" else torch.float32
                self._model = qwen_module.Qwen3ASRModel.from_pretrained(
                    str(self.model_dir),
                    dtype=dtype,
                    device_map=torch_device,
                    max_inference_batch_size=1,
                    max_new_tokens=256,
                )
                if torch_device == "mps":
                    # Qwen's outer config accepts attn_implementation, but its
                    # nested Thinker/audio/text configs retain SDPA. PyTorch
                    # 2.6 MPS SDPA crashes on GQA (16 query heads, 8 KV heads).
                    # Eager attention explicitly expands KV heads and runs on MPS.
                    for module in self._model.model.modules():
                        config = getattr(module, "config", None)
                        if config is not None and hasattr(config, "_attn_implementation"):
                            config._attn_implementation = "eager"
                    logger.info("Configured Qwen nested attention blocks for MPS-compatible eager GQA")
                self._state = None
        except Exception as error:
            if self._mode == "streaming":
                detail = (
                    "Could not initialize Qwen3-ASR's vLLM streaming backend. "
                    "Check the Qwen/vLLM installation, model files, and available device."
                )
            else:
                detail = (
                    "Could not initialize Qwen3-ASR's Transformers backend. Check the "
                    "Qwen model files, PyTorch device support, and available memory."
                )
            raise RuntimeError(f"{detail} Original error: {error}") from error

        self._revision = 0
        self._last_text = ""
        self._phrase_parts: list[Any] = []
        self._pending_parts: list[Any] = []
        self._pending_samples = 0
        self._closed = False
        model_device = getattr(self._model, "device", torch_device)
        logger.info(
            "Qwen3-ASR initialized in %s mode for %s on %s",
            self._mode,
            language,
            model_device,
        )

    @property
    def mode(self) -> str:
        """Return ``streaming`` or ``phrase`` (final transcript only) for status reporting."""
        return self._mode

    def accept_audio(
        self, samples: Any, sample_rate: int, captured_at: float
    ) -> RecognitionUpdate | None:
        self._ensure_open()
        _validate_timestamp(captured_at)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Qwen3-ASR expects {self.sample_rate} Hz audio; received {sample_rate} Hz."
            )
        waveform = self._to_waveform(samples)
        if waveform.size == 0:
            return None

        if self._mode == "phrase":
            self._phrase_parts.append(waveform)
            return None

        self._pending_parts.append(waveform)
        self._pending_samples += waveform.size
        chunk_samples = int(self.sample_rate * self.feed_chunk_seconds)
        update = None
        while self._pending_samples >= chunk_samples:
            chunk = self._take_pending(chunk_samples)
            try:
                self._model.streaming_transcribe(chunk, self._state)
                text = _clean_text(self._state.text)
            except Exception as error:
                raise RuntimeError(f"Qwen3-ASR failed while decoding {self.language} audio: {error}") from error
            changed = self._update_if_changed(text, captured_at, is_final=False)
            if changed is not None:
                update = changed
        return update

    def finish(self, captured_at: float) -> RecognitionUpdate | None:
        self._ensure_open()
        _validate_timestamp(captured_at)
        try:
            if self._mode == "phrase":
                if not self._phrase_parts:
                    return None
                waveform = self._np.concatenate(self._phrase_parts)
                result = self._model.transcribe(
                    audio=(waveform, self.sample_rate),
                    language=self._language_name,
                    context=self._context,
                )[0]
                text = _clean_text(result.text)
            else:
                if self._pending_samples:
                    self._model.streaming_transcribe(
                        self._take_pending(self._pending_samples), self._state
                    )
                self._model.finish_streaming_transcribe(self._state)
                text = _clean_text(self._state.text)
        except Exception as error:
            raise RuntimeError(
                f"Qwen3-ASR failed to finalize the {self.language} phrase: {error}"
            ) from error
        finally:
            self._phrase_parts.clear()
            self._pending_parts.clear()
            self._pending_samples = 0
            if self._mode == "streaming" and not self._closed:
                self._state = self._new_stream_state()

        self._last_text = ""
        return self._update_if_changed(text, captured_at, is_final=True) if text else None

    def close(self) -> None:
        self._closed = True
        self._state = None
        self._model = None
        self._phrase_parts.clear()
        self._pending_parts.clear()
        self._pending_samples = 0
        self._last_text = ""

    def _new_stream_state(self) -> Any:
        return self._model.init_streaming_state(
            context=self._context,
            language=self._language_name,
            chunk_size_sec=self.model_chunk_seconds,
            unfixed_chunk_num=self.unfixed_chunk_count,
            unfixed_token_num=self.unfixed_token_count,
        )

    def _take_pending(self, sample_count: int) -> Any:
        """Remove exactly ``sample_count`` samples from the short streaming buffer."""
        chunks: list[Any] = []
        remaining = sample_count
        while remaining:
            first = self._pending_parts[0]
            take = min(remaining, first.size)
            chunks.append(first[:take])
            remaining -= take
            if take == first.size:
                self._pending_parts.pop(0)
            else:
                self._pending_parts[0] = first[take:]
        self._pending_samples -= sample_count
        return self._np.ascontiguousarray(self._np.concatenate(chunks), dtype=self._np.float32)

    def _to_waveform(self, samples: Any) -> Any:
        try:
            waveform = self._np.asarray(samples, dtype=self._np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Audio samples must be a one-dimensional float waveform.") from error
        if waveform.ndim != 1:
            raise ValueError("Qwen3-ASR accepts mono audio only.")
        if not self._np.isfinite(waveform).all():
            raise ValueError("Audio samples must contain only finite values.")
        if waveform.size and (float(waveform.min()) < -1.0 or float(waveform.max()) > 1.0):
            raise ValueError("Float audio samples must be normalized to the range [-1, 1].")
        return self._np.ascontiguousarray(waveform)

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

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Qwen3-ASR recognizer is closed.")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _validate_timestamp(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("captured_at must be finite, non-negative monotonic seconds")


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for Qwen3-ASR. Install ASR support with "
            "`pip install -e .[qwen-asr]`."
        ) from error
    return np


def _vllm_available() -> bool:
    """Only auto-select Qwen's official streaming implementation on supported Linux hosts."""
    if platform.system() != "Linux":
        return False
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False
