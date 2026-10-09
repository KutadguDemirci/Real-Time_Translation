"""Download the optional Qwen3-ASR 0.6B checkpoint and register its local hashes.

This is an explicit model-acquisition command. The app itself never downloads models. Run
``python models/fetch_models.py`` first, then this script; it preserves other manifest entries.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import yaml

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as error:
    raise SystemExit(
        "Model download requires Hugging Face Hub. Install Qwen support with "
        "`pip install -e .[qwen-asr]`."
    ) from error

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("REAL_TIME_TRANSLATION_MODEL_DIR", ROOT / "models" / "local")).expanduser().resolve()
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ASSET_ID = "asr-qwen3"
DESTINATION = MODEL_DIR / "asr" / "qwen3-asr-0.6b"
MANIFEST_PATH = MODEL_DIR / "manifest.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            f"Model manifest not found: {MANIFEST_PATH}. Run `python models/fetch_models.py` first, "
            "or set REAL_TIME_TRANSLATION_MODEL_DIR to the configured model directory."
        )

    api = HfApi()
    revision = api.model_info(MODEL_ID).sha
    print(f"Downloading {MODEL_ID} at {revision}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=DESTINATION,
    )

    files = sorted(
        path for path in DESTINATION.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(DESTINATION).parts
    )
    if not files:
        raise RuntimeError(f"No model files were downloaded to {DESTINATION}")
    relative_files = [path.relative_to(MODEL_DIR).as_posix() for path in files]
    asset = {
        "id": ASSET_ID,
        "task": "streaming-asr",
        "languages": ["en", "fr"],
        "revision": revision,
        "runtime": "Qwen3-ASR 0.6B; vLLM streaming or Transformers phrase mode",
        "files": relative_files,
        "sha256": [sha256(path) for path in files],
        "weight_license": "Apache-2.0",
        "license_url": f"https://huggingface.co/{MODEL_ID}",
        "attribution": "Qwen3-ASR model by the Qwen team; see the model card and license.",
        "optional": True,
    }

    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
        raise ValueError(f"Invalid model manifest: {MANIFEST_PATH}")
    manifest["models"] = [entry for entry in manifest["models"] if entry.get("id") != ASSET_ID]
    manifest["models"].append(asset)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    print(f"Registered {len(files)} files and wrote hashes to {MANIFEST_PATH}", flush=True)
    print("The Qwen checkpoint is optional and is validated when selected in the app.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
