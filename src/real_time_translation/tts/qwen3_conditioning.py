"""Reusable speaker-only conditioning for MLX Audio's Qwen3-TTS Base model.

MLX Audio 0.5.8 selects ICL when both ref_audio and ref_text are present. With
ref_text=None it uses extract_speaker_embedding instead. Its public generate API
does not accept precomputed embeddings, so temporarily replace that extraction
hook while running the normal token/codec generator. The caller must serialize
all access to the model, including the lifetime of the returned generator.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import closing
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

PROFILE_VERSION = "qwen3-speaker-only-mean-6s-v1"


class SpeakerEmbeddingCache:
    """Cache fixed-size embeddings in memory and as local, non-pickle NumPy files."""

    def __init__(self, model_dir: Path, directory: Path) -> None:
        self.directory = directory
        # Evaluate model identity once, rather than hashing model weights per phrase.
        files = sorted(
            (str(path.relative_to(model_dir)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix in {".json", ".safetensors"}
        )
        try:
            runtime_version = importlib.metadata.version("mlx-audio")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = "injected-model"
        identity = json.dumps((PROFILE_VERSION, runtime_version, str(model_dir), files))
        # Keep filenames short enough for Windows' default path-length limit.
        self._model_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        self._memory: dict[tuple[Path, int, int], Any] = {}

    def get(self, model: Any, reference: Path) -> Any:
        from mlx import core as mx
        import numpy as np

        stat = reference.stat()
        memory_key = (reference, stat.st_size, stat.st_mtime_ns)
        if memory_key in self._memory:
            return self._memory[memory_key]
        path = self.directory / f"{_reference_digest(reference)}-{self._model_key}.npy"
        values = None
        try:
            if path.stat().st_size > 128 * 1024:
                raise ValueError("Invalid speaker embedding cache size.")
            values = np.load(path, allow_pickle=False)
            _validate_embedding(values)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, EOFError):
            logger.warning("Rebuilding an unreadable speaker embedding cache entry.")
            values = None
        if values is None:
            values = _encode_reference(model, reference)
            self._save(path, values)
        embedding = mx.array(values)
        mx.eval(embedding)
        # Replace stale versions of this recording in the memory cache.
        self._memory = {key: value for key, value in self._memory.items() if key[0] != reference}
        self._memory[memory_key] = embedding
        return embedding

    def forget(self, reference: Path) -> None:
        self._memory = {key: value for key, value in self._memory.items() if key[0] != reference}

    def _save(self, path: Path, values: Any) -> None:
        import numpy as np

        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as output:
                temporary = Path(output.name)
                os.chmod(temporary, 0o600)
                np.save(output, values, allow_pickle=False)
            os.replace(temporary, path)
        except OSError as error:
            # A read-only cache directory must not prevent in-memory reuse or speech.
            logger.warning("Could not persist speaker embedding; using the in-memory cache: %s",
                           error.strerror or type(error).__name__)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove a temporary speaker cache entry.")


def _validate_embedding(values: Any) -> None:
    import numpy as np

    if (not isinstance(values, np.ndarray) or values.ndim != 2
            or values.shape[0] != 1 or not 1 <= values.shape[1] <= 16_384
            or values.dtype != np.float32 or not np.isfinite(values).all()):
        raise ValueError("Invalid speaker embedding; expected one finite float32 vector.")


def _reference_digest(reference: Path) -> str:
    digest = hashlib.sha256()
    with reference.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def remove_reference_embeddings(reference: Path, directory: Path) -> None:
    """Remove derived profiles for this recording when its saved voice is deleted."""
    if not directory.is_dir():
        return
    for path in directory.glob(f"{_reference_digest(reference)}-*.npy"):
        path.unlink(missing_ok=True)


def _encode_reference(model: Any, reference: Path) -> Any:
    from mlx_audio.utils import load_audio
    import numpy as np

    if not callable(getattr(model, "extract_speaker_embedding", None)):
        raise RuntimeError("The MLX model does not support speaker embedding extraction.")
    rate = int(model.sample_rate)
    audio = load_audio(str(reference), sample_rate=rate)
    # Bound profiling memory for recordings up to five minutes. More recording
    # affects this one-time computation, never the generation prompt length.
    chunk_size = 6 * rate
    embeddings = []
    for start in range(0, audio.shape[0], chunk_size):
        chunk = audio[start:start + chunk_size]
        if chunk.shape[0] < rate and embeddings:
            break
        value = np.asarray(model.extract_speaker_embedding(chunk, sr=rate), dtype=np.float32)
        _validate_embedding(value)
        embeddings.append(value)
    if not embeddings:
        raise ValueError("No audio available for speaker profiling.")
    values = np.mean(np.stack(embeddings), axis=0, dtype=np.float32)
    _validate_embedding(values)
    return values


def generate_with_speaker_embedding(
    model: Any, embedding: Any, **options: Any,
) -> Generator[Any, None, None]:
    """Use one cached vector, with no reference text or speech tokens in the prompt."""
    from mlx import core as mx

    if getattr(model, "speaker_encoder", None) is None:
        raise RuntimeError("Speaker-only cloning requires a Qwen3-TTS Base speaker encoder.")
    original = model.extract_speaker_embedding
    used_embedding = False

    def cached_embedding(_audio: Any, sr: int = 24_000) -> Any:
        nonlocal used_embedding
        used_embedding = True
        return embedding

    # A constant-size marker activates MLX's speaker-only input path. No actual
    # reference waveform is passed to inference or encoded by its speech tokenizer.
    model.extract_speaker_embedding = cached_embedding
    try:
        with closing(model.generate(ref_audio=mx.zeros(1), ref_text=None, **options)) as chunks:
            for chunk in chunks:
                if not used_embedding:
                    raise RuntimeError(
                        "This MLX Audio version did not consume the cached speaker embedding. "
                        "Install the pinned runtime with models/install_qwen_tts_mlx_runtime.py."
                    )
                yield chunk
    finally:
        model.extract_speaker_embedding = original
