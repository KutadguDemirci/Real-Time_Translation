"""Install MLX Audio without replacing Transformers required by Qwen3-ASR."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


MLX_AUDIO_VERSION = "0.5.8"


def main() -> int:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Qwen3-TTS MLX requires macOS on Apple Silicon.")
    try:
        current = importlib.metadata.version("mlx-audio")
    except importlib.metadata.PackageNotFoundError:
        current = ""
    if current != MLX_AUDIO_VERSION:
        print(
            f"Installing mlx-audio {MLX_AUDIO_VERSION} without dependency changes. "
            "The normal dependency resolver would replace Transformers 4.57.6, "
            "which Qwen3-ASR requires.",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", f"mlx-audio=={MLX_AUDIO_VERSION}"],
            check=True,
        )
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    print(f"MLX Audio {MLX_AUDIO_VERSION} ready; default MLX device: {mx.default_device()}.")
    print(f"Local model loader: {load_model.__module__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
