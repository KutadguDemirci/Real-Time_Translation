"""Command-line entry point for the local translation desktop application."""

from __future__ import annotations

import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from real_time_translation.asr.sherpa_adapter import SherpaStreamingRecognizer, TransducerModel
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
from real_time_translation.tts.piper_adapter import PiperSynthesizer
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
    resolver: ModelResolver,
    manifest: ModelManifest,
    device: DeviceProfile,
    settings: Settings,
) -> SessionCoordinator:
    """Build a session entirely from the assets listed in the verified manifest."""
    def resolve(model_id: str):
        return resolver.resolve(model_id, manifest)

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

    translation_dirs: dict[tuple[str, str], Path] = {}
    for source, target in (("en", "fr"), ("fr", "en")):
        translation = resolve(f"translation-{source}-{target}")
        translation_dirs[(source, target)] = translation.files[0].parent
    translator = OpusTranslator(
        device=device.torch_device,
        model_dirs=translation_dirs,
    )

    voice_paths: dict[str, Path] = {}
    for language in ("en", "fr"):
        voice = resolve(f"tts-{language}")
        model_file = next((path for path in voice.files if path.suffix.lower() == ".onnx"), None)
        if model_file is None:
            raise ValueError(f"TTS model {voice.asset.id} does not list a Piper ONNX voice file.")
        voice_paths[language] = model_file

    capture = AudioCapture(sample_rate=16_000, device=settings.input_device_id)
    playback = AudioPlayback(device=settings.output_device_id)
    return SessionCoordinator(
        capture=capture,
        vad=VoiceActivityDetector(endpoint_pause_ms=settings.endpoint_pause_ms),
        recognizer=recognizer,
        translator=translator,
        synthesizer=PiperSynthesizer(voice_paths),
        playback=playback,
        source_language=source_language,
        on_event=on_event,
    )


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
        if state.manifest is not None:
            coordinator_factory = lambda source, emit, settings: create_coordinator(
                source,
                emit,
                resolver=resolver,
                manifest=state.manifest,
                device=profile,
                settings=settings,
            )
        run_desktop(
            model_status=state.model_status,
            setup_error=state.setup_error,
            device_label=device_label,
            settings=state.settings,
            coordinator_factory=coordinator_factory,
        )
    except (ImportError, OSError, RuntimeError) as error:
        logger.exception("Could not start the desktop application")
        print(f"Could not start Local Translator: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
