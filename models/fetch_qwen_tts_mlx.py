"""Download the MLX-community Qwen3-TTS 0.6B Base 8-bit checkpoint locally."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    os.environ.get("REAL_TIME_TRANSLATION_MODEL_DIR", ROOT / "models" / "local")
).expanduser().resolve()
REPO_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
DESTINATION = MODEL_DIR / "tts" / "qwen3-tts-0.6b-base-8bit"
LICENSE_URL = f"https://huggingface.co/{REPO_ID}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit('Install MLX dependencies first: pip install -e ".[tts]"') from error

    api = HfApi()
    revision = api.model_info(REPO_ID).sha
    print(f"Downloading {REPO_ID} at revision {revision}", flush=True)
    snapshot_download(repo_id=REPO_ID, revision=revision, local_dir=DESTINATION)

    files: list[Path] = []
    for path in sorted(DESTINATION.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        # Hugging Face local_dir may use links into its cache. Materialize them inside
        # the model directory so the manifest never accepts files outside its root.
        if path.is_symlink():
            resolved = path.resolve()
            temporary = path.with_name(path.name + ".local-copy")
            temporary.write_bytes(resolved.read_bytes())
            temporary.replace(path)
        files.append(path)
    if not files:
        raise RuntimeError("The Qwen3-TTS repository download contained no model files.")

    relative_files = [path.relative_to(MODEL_DIR).as_posix() for path in files]
    checksums = [sha256(path) for path in files]
    manifest_path = MODEL_DIR / "manifest.yaml"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or {}
    else:
        manifest = {"schema_version": 1, "models": []}
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Invalid model manifest: {manifest_path}")
    models = [
        item for item in models
        if isinstance(item, dict)
        and item.get("id") not in {"tts-qwen3-mlx-8bit", "tts-chatterbox-v3"}
    ]
    models.append({
        "id": "tts-qwen3-mlx-8bit",
        "task": "tts",
        "languages": ["en", "fr", "de", "it", "pt", "es", "zh", "ja", "ko", "ru"],
        "revision": revision,
        "runtime": "Qwen3-TTS 0.6B Base 8-bit via MLX Audio",
        "weight_license": "Apache-2.0",
        "license_url": LICENSE_URL,
        "attribution": "Qwen3-TTS by Alibaba Cloud; MLX 8-bit conversion by mlx-community.",
        "files": relative_files,
        "sha256": checksums,
    })
    manifest["schema_version"] = 1
    manifest["models"] = models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    print(f"Registered {len(files)} files in {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
