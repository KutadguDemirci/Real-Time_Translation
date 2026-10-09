"""Command-line entry point for the local translation desktop application."""

from __future__ import annotations

import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from real_time_translation.asr.sherpa_adapter import SherpaStreamingRecognizer, TransducerModel
from real_time_translation.asr.transcribe_file import transcribe_wav
from real_time_translation.asr.qwen_adapter import QwenStreamingRecognizer
from real_time_translation.audio.capture import AudioCapture
from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.audio.vad import VoiceActivityDetector
from real_time_translation.config import Settings, default_settings_path, load_settings
from real_time_translation.device.probe import DeviceProfile, detect_device_profile
from real_time_translation.domain.events import DomainEvent
from real_time_translation.models.manifest import ModelManifest
from real_time_translation.models.resolver import (
    ModelResolver,
    ModelValidationError,
)
from real_time_translation.pipeline.coordinator import SessionCoordinator
from real_time_translation.translation.opus_mt import OpusTranslator
from real_time_translation.tts.qwen3_mlx_adapter import Qwen3MLXSynthesizer
from real_time_translation.tts.base import SynthesisOptions
from real_time_translation.tts.voice_profiles import VoiceProfileStore
from real_time_translation.ui.desktop import run_desktop

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StartupState:
    """Local setup information presented to the desktop UI."""

    settings: Settings
    model_status: str
    setup_error: str
    device: DeviceProfile
    manifest: ModelManifest | None = None


def create_coordinator(
    source_language: str,
    on_event: Callable[[DomainEvent], None],
    *,
    target_language: str | None = None,
    resolver: ModelResolver,
    manifest: ModelManifest,
    device: DeviceProfile,
    settings: Settings,
    synthesizer: Qwen3MLXSynthesizer | None = None,
) -> SessionCoordinator:
    """Build a session entirely from the assets listed in the verified manifest."""
    def resolve(model_id: str):
        return resolver.resolve(model_id, manifest)

    recognizer = create_recognizer(
        source_language, resolver=resolver, manifest=manifest, device=device, settings=settings
    )

    translation_dirs: dict[tuple[str, str], Path] = {}
    for source, target in OpusTranslator.MODEL_NAMES:
        try:
            translation = resolve(f"translation-{source}-{target}")
        except KeyError:
            continue
        translation_dirs[(source, target)] = translation.files[0].parent
    translator = OpusTranslator(
        device=device.torch_device,
        model_dirs=translation_dirs,
    )

    synthesizer = synthesizer or create_synthesizer(
        resolver=resolver, manifest=manifest, device=device, settings=settings
    )
    capture = AudioCapture(sample_rate=16_000, device=settings.input_device_id)
    playback = AudioPlayback(device=settings.output_device_id)
    return SessionCoordinator(
        capture=capture,
        vad=VoiceActivityDetector(endpoint_pause_ms=settings.endpoint_pause_ms),
        recognizer=recognizer,
        translator=translator,
        synthesizer=synthesizer,
        playback=playback,
        source_language=source_language,
        target_language=target_language or settings.target_language,
        on_event=on_event,
    )


def create_recognizer(
    source_language: str,
    *,
    resolver: ModelResolver,
    manifest: ModelManifest,
    device: DeviceProfile,
    settings: Settings,
) -> Any:
    """Create the currently configured local recognizer for sessions or Studio transcription."""
    def resolve(model_id: str):
        return resolver.resolve(model_id, manifest)

    if settings.asr_backend == "qwen":
        try:
            asr = resolve("asr-qwen3")
        except KeyError as error:
            raise ValueError(
                "Qwen3-ASR is not registered in the local model manifest. Run "
                "`python models/fetch_qwen_asr.py` and select Qwen again."
            ) from error
        if source_language not in asr.asset.languages:
            raise ValueError(
                f"Qwen ASR asset {asr.asset.id} does not list support for {source_language}."
            )
        recognizer = QwenStreamingRecognizer(
            asr.files[0].parent,
            source_language,
            context=settings.asr_context,
            torch_device=device.torch_device,
        )
    else:
        if source_language not in {"en", "fr"}:
            raise ValueError(
                f"Sherpa-ONNX is configured only for English and French. Select Qwen3-ASR "
                f"to recognize {source_language}."
            )
        asr = resolve(f"asr-{source_language}")
        # English model files carry versioned encoder/decoder/joiner names.
        tokens = next((path for path in asr.files if path.name == "tokens.txt"), None)
        encoder = next(
            (path for path in asr.files if path.name.startswith("encoder") and path.suffix == ".onnx"),
            None,
        )
        decoder = next(
            (path for path in asr.files if path.name.startswith("decoder") and path.suffix == ".onnx"),
            None,
        )
        joiner = next(
            (path for path in asr.files if path.name.startswith("joiner") and path.suffix == ".onnx"),
            None,
        )
        if tokens is None or encoder is None or decoder is None or joiner is None:
            raise ValueError(
                f"ASR model {asr.asset.id} must list tokens.txt and encoder, decoder, and joiner ONNX files."
            )
        transducer = TransducerModel(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
        )
        provider = {
            "CUDAExecutionProvider": "cuda",
            "CoreMLExecutionProvider": "coreml",
        }.get(device.onnx_provider, "cpu")
        recognizer = SherpaStreamingRecognizer(
            transducer,
            source_language,
            num_threads=settings.cpu_threads,
            provider=provider,
        )

    return recognizer


def create_voice_transcriber(
    *,
    resolver: ModelResolver,
    manifest: ModelManifest,
    device: DeviceProfile,
    settings: Settings,
) -> Callable[[Path, str, str, str], str]:
    """Use the app's selected local ASR engine to transcribe Voice Studio recordings."""
    def transcribe(path: Path, language: str, backend: str, context: str) -> str:
        asr_settings = replace(settings, asr_backend=backend, asr_context=context)
        recognizer = create_recognizer(
            language, resolver=resolver, manifest=manifest, device=device, settings=asr_settings
        )
        return transcribe_wav(path, recognizer)

    return transcribe


def create_synthesizer(
    *,
    resolver: ModelResolver,
    manifest: ModelManifest,
    device: DeviceProfile,
    settings: Settings,
) -> Qwen3MLXSynthesizer:
    """Construct one reusable adapter from the manifest-verified local MLX checkpoint."""
    tts = _manifest_files(resolver, manifest, "tts-qwen3-mlx-8bit")
    cartoon_reference = Path(__file__).resolve().parent / "assets" / "cartoon_reference.wav"
    agent_reference = Path(__file__).resolve().parent / "assets" / "agent_reference.wav"
    voice_references = {
        "voice_1": (agent_reference, _CARTOON_REFERENCE_TEXT),
        "voice_2": (cartoon_reference, _CARTOON_REFERENCE_TEXT),
    }
    if not cartoon_reference.is_file():
        raise FileNotFoundError("The bundled Cartoon voice reference is missing from the app installation.")
    if not agent_reference.is_file():
        raise FileNotFoundError("The bundled Agent voice reference is missing from the app installation.")
    synthesizer = Qwen3MLXSynthesizer(
        model_dir=tts[1][0].parent,
        voice_references=voice_references,
    )
    options = SynthesisOptions(
        target_lang="en",
        voice_id=settings.tts_voice_id,
        temperature=settings.tts_temperature,
    )
    if settings.tts_profile_id:
        try:
            profile = VoiceProfileStore().get(settings.tts_profile_id)
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Could not load saved voice profile %s: %s", settings.tts_profile_id, error)
            profile = None
        if profile is not None:
            synthesizer.set_custom_reference(profile.audio_path, profile.transcript)
            options = replace(options, voice_id="custom")
        else:
            logger.warning("Saved voice profile %s was not found; using the preset voice.",
                           settings.tts_profile_id)
    synthesizer.set_default_options(options)
    return synthesizer


def _manifest_files(
    resolver: ModelResolver, manifest: ModelManifest, model_id: str
) -> tuple[object, tuple[Path, ...]]:
    """Resolve paths after startup's manifest validation without rehashing multi-GB weights."""
    asset = manifest.get(model_id)
    paths = tuple((resolver.model_dir / relative).resolve() for relative in asset.files)
    return asset, paths


_CARTOON_REFERENCE_TEXT = "Hello there. I have a clever little plan, and I think we're going to have a rather splendid day."


def inspect_startup(
    *,
    settings: Settings | None = None,
    resolver: ModelResolver | None = None,
    device: DeviceProfile | None = None,
) -> StartupState:
    """Collect settings, model validation, and device information without loading models."""
    user_settings = settings or load_settings()
    model_resolver = resolver or ModelResolver()
    device_profile = device or detect_device_profile()

    try:
        manifest = model_resolver.load()
    except (ModelValidationError, FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        logger.info("Local models are not ready: %s", error)
        return StartupState(
            settings=user_settings,
            model_status="Models need setup",
            setup_error=str(error),
            device=device_profile,
        )

    return StartupState(
        settings=user_settings,
        model_status=f"Models verified · {len(manifest.models)} assets",
        setup_error="",
        device=device_profile,
        manifest=manifest,
    )


def main() -> int:
    """Open the desktop UI and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    resolver = ModelResolver()
    state = inspect_startup(resolver=resolver)
    profile = state.device
    device_label = (
        f"{profile.operating_system} {profile.architecture} · "
        f"PyTorch {profile.torch_device.upper()} · ONNX {profile.onnx_provider}"
    )
    logger.info(
        "Starting Local Translator (Python %s, settings %s)",
        platform.python_version(),
        default_settings_path(),
    )
    try:
        coordinator_factory = None
        synthesizer = None
        if state.manifest is not None:
            try:
                synthesizer = create_synthesizer(
                    resolver=resolver, manifest=state.manifest, device=profile, settings=state.settings
                )
                coordinator_factory = lambda source, target, emit, settings: create_coordinator(
                    source, emit, target_language=target, resolver=resolver, manifest=state.manifest,
                    device=profile, settings=settings, synthesizer=synthesizer,
                )
            except (KeyError, FileNotFoundError, ValueError, OSError, RuntimeError) as error:
                state = StartupState(
                    settings=state.settings,
                    model_status="TTS setup needed",
                    setup_error=str(error),
                    device=profile,
                    manifest=state.manifest,
                )
        transcriber = (
            create_voice_transcriber(
                resolver=resolver, manifest=state.manifest, device=profile, settings=state.settings
            )
            if state.manifest is not None else None
        )
        run_desktop(
            model_status=state.model_status,
            setup_error=state.setup_error,
            device_label=device_label,
            settings=state.settings,
            coordinator_factory=coordinator_factory,
            synthesizer=synthesizer,
            voice_transcriber=transcriber,
        )
    except (ImportError, OSError, RuntimeError) as error:
        logger.exception("Could not start the desktop application")
        print(f"Could not start Local Translator: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
