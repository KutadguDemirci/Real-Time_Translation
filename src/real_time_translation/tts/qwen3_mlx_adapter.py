"""Local Qwen3-TTS 0.6B Base inference through MLX Audio on Apple Silicon."""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping
import wave

from .audio_reference import validate_reference_audio
from .base import BUILT_IN_VOICES, SUPPORTED_TTS_LANGUAGES, SpeechResult, SynthesisOptions

logger = logging.getLogger(__name__)


class Qwen3MLXSynthesizer:
    """Keep one quantized MLX model loaded and reuse reference-audio conditioning."""

    model_name = "Qwen3-TTS 0.6B Base (MLX 8-bit)"
    sample_rate = 24_000

    def __init__(
        self,
        model_dir: str | Path,
        voice_references: Mapping[str, tuple[str | Path, str]],
        *,
        model_loader: Callable[[str | Path], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.voice_references = {
            key: (Path(path).expanduser().resolve(), transcript.strip())
            for key, (path, transcript) in voice_references.items()
        }
        if set(self.voice_references) != set(BUILT_IN_VOICES):
            raise ValueError("Exactly two local preset reference voices must be configured.")
        if any(not transcript for _, transcript in self.voice_references.values()):
            raise ValueError("Each preset voice must include the exact reference transcript.")
        self._model_loader = model_loader or _load_local_model
        self._model: Any | None = None
        self._target_lang: str | None = None
        self._custom_reference: Path | None = None
        self._custom_directory: Path | None = None
        self._custom_transcript: str | None = None
        self._default_options = SynthesisOptions()
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def supported_languages(self) -> Mapping[str, str]:
        return SUPPORTED_TTS_LANGUAGES

    def load(self, target_lang: str) -> None:
        language = target_lang.strip().lower()
        if language not in SUPPORTED_TTS_LANGUAGES:
            raise ValueError(f"Unsupported TTS language: {target_lang!r}.")
        if __import__("sys").platform != "darwin" or __import__("platform").machine() != "arm64":
            raise RuntimeError("Qwen3-TTS MLX 8-bit requires macOS on Apple Silicon.")
        with self._lock:
            if self._model is None:
                if not self.model_dir.is_dir() or not any(self.model_dir.iterdir()):
                    raise FileNotFoundError(
                        "Qwen3-TTS MLX 8-bit files are missing. Run python models/fetch_qwen_tts_mlx.py."
                    )
                try:
                    self._model = self._model_loader(self.model_dir)
                except Exception as error:
                    raise RuntimeError("Could not load the local Qwen3-TTS MLX 8-bit model.") from error
                logger.info("Loaded %s on Apple MLX/Metal from %s", self.model_name, self.model_dir)
            self._target_lang = language

    def set_custom_reference(self, path: str | Path, transcript: str) -> None:
        """Keep a validated, private copy for repeated Base-model voice-clone requests."""
        clean_transcript = transcript.strip()
        if not clean_transcript:
            raise ValueError("Enter the exact words spoken in the reference recording.")
        validate_reference_audio(path)
        with self._lock:
            self.clear_custom_reference()
            directory = Path(tempfile.mkdtemp(prefix="local-translator-voice-"))
            try:
                directory.chmod(0o700)
            except OSError:
                pass
            destination = directory / "reference.wav"
            shutil.copyfile(path, destination)
            try:
                destination.chmod(0o600)
            except OSError:
                pass
            self._custom_directory = directory
            self._custom_reference = destination
            self._custom_transcript = clean_transcript

    def clear_custom_reference(self) -> None:
        with self._lock:
            if self._custom_directory is not None:
                shutil.rmtree(self._custom_directory, ignore_errors=True)
            self._custom_reference = None
            self._custom_directory = None
            self._custom_transcript = None

    def set_default_options(self, options: SynthesisOptions) -> None:
        """Set voice and generation options used by real-time translation synthesis."""
        self._default_options = options.validate()

    def synthesize(self, text: str, options: SynthesisOptions | None = None) -> SpeechResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a phrase to synthesize.")
        selected = options or replace(self._default_options, target_lang=self._target_lang or "en")
        selected.validate()
        with self._lock:
            self.load(selected.target_lang)
            assert self._model is not None
            ref_path, ref_text = self._reference(selected.voice_id)
            started = time.perf_counter()
            try:
                audio, sample_rate = _generate(self._model, text.strip(), selected.target_lang,
                                               ref_path, ref_text, float(selected.temperature))
                pcm = _to_pcm16(audio)
            except Exception as error:
                raise RuntimeError("Qwen3-TTS could not synthesize this phrase. Check its language and reference transcript.") from error
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not pcm:
                raise RuntimeError("Qwen3-TTS returned no usable audio.")
            return SpeechResult(pcm, sample_rate, 1, 2, elapsed_ms, self.model_name,
                                selected.target_lang, "mlx-metal")

    def unload(self) -> None:
        with self._lock:
            self.clear_custom_reference()
            self._model = None
            self._target_lang = None
            gc.collect()

    def _reference(self, voice_id: str) -> tuple[Path, str]:
        if voice_id == "custom":
            if self._custom_reference is None or self._custom_transcript is None:
                raise ValueError("Add a valid reference recording and its exact transcript before cloning.")
            return self._custom_reference, self._custom_transcript
        reference, transcript = self.voice_references[voice_id]
        if not reference.is_file():
            raise FileNotFoundError(f"The {BUILT_IN_VOICES[voice_id]} reference recording is missing.")
        return reference, transcript


def _load_local_model(model_dir: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
        from mlx_audio.tts.utils import load_model
    except ImportError as error:
        raise RuntimeError('Install MLX TTS support with `pip install -e ".[tts]"`.') from error

    # Transformers 4.57's tokenizer warning is substantive for this model: its
    # Mistral regex needs the compatibility repair. Scope the override to model
    # loading so unrelated tokenizers in ASR/translation keep their defaults.
    from unittest.mock import patch

    original_from_pretrained = AutoTokenizer.from_pretrained

    def _load_tokenizer(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("fix_mistral_regex", True)
        return original_from_pretrained(*args, **kwargs)

    with patch.object(AutoTokenizer, "from_pretrained", side_effect=_load_tokenizer):
        return load_model(str(model_dir))


def _generate(model: Any, text: str, language_code: str, ref_audio: Path,
              ref_text: str, temperature: float = 0.9) -> tuple[Any, int]:
    # mlx-audio's Qwen3-TTS API calls this parameter ``lang_code``. Passing
    # ``language`` is silently swallowed by **kwargs and leaves language on auto.
    language = SUPPORTED_TTS_LANGUAGES[language_code].lower()
    result_chunks = list(model.generate(
        text=text,
        lang_code=language,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        temperature=temperature,
    ))
    if not result_chunks:
        raise RuntimeError("No audio chunks were generated.")
    import numpy as np

    arrays = [np.asarray(chunk.audio).reshape(-1) for chunk in result_chunks]
    return np.concatenate(arrays), int(getattr(result_chunks[0], "sample_rate", 24_000))


def _to_pcm16(audio: Any) -> bytes:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        return b""
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
