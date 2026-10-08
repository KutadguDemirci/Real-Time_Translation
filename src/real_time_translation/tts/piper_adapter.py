"""Local Piper synthesis with one resident voice for the selected target language."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import threading
import time
from typing import Mapping

from ..device.probe import detect_device_profile, onnx_providers_for
from ..domain.languages import SUPPORTED_LANGUAGES
from .base import SpeechResult


class PiperSynthesizer:
    """Load and retain one local Piper voice until the target language changes.

    ``voice_paths`` maps ``en`` and ``fr`` to local Piper ``.onnx`` files. Piper's
    adjacent ``.onnx.json`` config is required. The default device profile is shared
    with translation: CUDA, Apple MPS/CoreML, then CPU.
    """

    def __init__(
        self,
        voice_paths: Mapping[str, str | Path],
    ) -> None:
        self.voice_paths = {language: Path(path) for language, path in voice_paths.items()}
        self.profile = detect_device_profile()
        self._voice = None
        self._session = None
        self._target_lang: str | None = None
        self._model_path: Path | None = None
        self._active_provider = "CPUExecutionProvider"
        self._lock = threading.RLock()

    @property
    def loaded_target_lang(self) -> str | None:
        """The language currently resident in memory, if any."""
        return self._target_lang

    def load(self, target_lang: str) -> None:
        """Load the target-language voice; repeated selection does not reload it."""
        language = target_lang.lower().strip()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported TTS target language: {target_lang!r}")

        model_path = self.voice_paths.get(language)
        if model_path is None:
            raise ValueError(f"No local Piper voice is configured for language {language!r}.")
        config_path = Path(f"{model_path}.json")
        if not model_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Piper voice for {language} is incomplete. Expected both "
                f"{model_path} and {config_path}. Add the local voice files and retry."
            )

        with self._lock:
            if self._voice is not None and self._target_lang == language:
                return
            self.unload()

            try:
                import onnxruntime
                from piper import PiperConfig, PiperVoice
            except ImportError as error:
                raise RuntimeError(
                    "Piper TTS is unavailable. Install the project's optional TTS dependencies."
                ) from error

            try:
                with config_path.open("r", encoding="utf-8") as config_file:
                    voice_config = PiperConfig.from_dict(json.load(config_file))
            except Exception as error:
                raise RuntimeError(
                    f"Could not read the Piper voice configuration at {config_path}: {error}"
                ) from error

            providers = onnx_providers_for(self.profile)
            try:
                session = onnxruntime.InferenceSession(
                    str(model_path),
                    sess_options=onnxruntime.SessionOptions(),
                    providers=providers,
                )
                active_provider = self.profile.onnx_provider
            except Exception as accelerated_error:
                # Provider discovery does not guarantee that a specific model can create
                # a session. Retry on CPU, matching the application's fallback policy.
                if providers == ["CPUExecutionProvider"]:
                    raise RuntimeError(
                        f"Could not load the local Piper voice at {model_path}: "
                        f"{accelerated_error}"
                    ) from accelerated_error
                try:
                    session = onnxruntime.InferenceSession(
                        str(model_path),
                        sess_options=onnxruntime.SessionOptions(),
                        providers=["CPUExecutionProvider"],
                    )
                    active_provider = "CPUExecutionProvider"
                except Exception as cpu_error:
                    raise RuntimeError(
                        f"Could not load the Piper voice on {self.profile.onnx_provider} "
                        f"or CPU. Accelerated error: {accelerated_error}; CPU error: {cpu_error}"
                    ) from cpu_error

            self._voice = PiperVoice(config=voice_config, session=session)
            self._session = session
            self._target_lang = language
            self._model_path = model_path
            self._active_provider = active_provider

    def synthesize(self, text: str) -> SpeechResult:
        """Synthesize a phrase using the already-loaded local voice."""
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        with self._lock:
            if self._voice is None or self._target_lang is None or self._model_path is None:
                raise RuntimeError("No Piper voice is loaded. Call load(target_lang) first.")

            start = time.perf_counter()
            pcm_chunks: list[bytes] = []
            sample_rate: int | None = None
            channels: int | None = None
            sample_width: int | None = None
            for chunk in self._voice.synthesize(text.strip()):
                current_format = (chunk.sample_rate, chunk.sample_channels, chunk.sample_width)
                if sample_rate is None:
                    sample_rate, channels, sample_width = current_format
                elif current_format != (sample_rate, channels, sample_width):
                    raise RuntimeError("Piper returned inconsistent audio formats for one phrase.")
                pcm_chunks.append(chunk.audio_int16_bytes)

            if sample_rate is None or channels is None or sample_width is None or not pcm_chunks:
                raise RuntimeError("Piper returned no audio for the phrase.")

            return SpeechResult(
                pcm=b"".join(pcm_chunks),
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
                synthesis_time_ms=(time.perf_counter() - start) * 1000.0,
                model=str(self._model_path),
                target_lang=self._target_lang,
                device=self._active_provider,
            )

    def unload(self) -> None:
        """Release the resident voice and its ONNX Runtime session."""
        with self._lock:
            self._voice = None
            self._session = None
            self._target_lang = None
            self._model_path = None
            self._active_provider = "CPUExecutionProvider"
            gc.collect()
