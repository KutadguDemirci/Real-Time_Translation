"""Fetch pinned OPUS-MT multilingual pivot models and register their local hashes.

Run after ``python models/fetch_models.py``. The two Marian models translate the added
languages to English and English to any requested output language, using greedy decoding.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as error:
    raise SystemExit(
        "Translation model acquisition needs Hugging Face Hub. Install with "
        "`pip install -e .[translation]`."
    ) from error

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("REAL_TIME_TRANSLATION_MODEL_DIR", ROOT / "models" / "local")).expanduser().resolve()
MANIFEST_PATH = MODEL_DIR / "manifest.yaml"
MODELS = {
    "translation-mul-en": ("Helsinki-NLP/opus-mt-mul-en", "mul", "en"),
    "translation-en-mul": ("Helsinki-NLP/opus-mt-en-mul", "en", "mul"),
}
ALLOW_PATTERNS = (
    "config.json", "generation_config.json", "metadata.json", "pytorch_model.bin",
    "source.spm", "target.spm", "tokenizer_config.json", "vocab.json",
    "special_tokens_map.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            f"Model manifest not found: {MANIFEST_PATH}. Run `python models/fetch_models.py` first."
        )
    api = HfApi()
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
        raise ValueError(f"Invalid model manifest: {MANIFEST_PATH}")
    assets = manifest["models"]
    assets = [entry for entry in assets if entry.get("id") not in {"translation-en-fr", "translation-fr-en"}]
    for asset_id, (repo, source, target) in MODELS.items():
        info = api.model_info(repo)
        license_id = getattr(info.card_data, "license", None)
        if not license_id:
            raise RuntimeError(f"No declared license found in {repo}'s model card; refusing to register it.")
        destination = MODEL_DIR / "translation" / f"{source}-{target}"
        print(f"Downloading {repo} at {info.sha}", flush=True)
        snapshot_download(
            repo_id=repo, revision=info.sha, local_dir=destination,
            allow_patterns=list(ALLOW_PATTERNS),
        )
        files = sorted(
            path for path in destination.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(destination).parts
        )
        required = {"config.json", "pytorch_model.bin", "source.spm", "target.spm", "vocab.json"}
        missing = required - {path.name for path in files}
        if missing:
            raise RuntimeError(f"{repo} is missing required Marian files: {', '.join(sorted(missing))}")
        assets = [entry for entry in assets if entry.get("id") != asset_id]
        assets.append({
            "id": asset_id,
            "task": "translation",
            "source": source,
            "target": target,
            "revision": info.sha,
            "runtime": "Helsinki-NLP OPUS-MT multilingual MarianMT (PyTorch)",
            "files": [path.relative_to(MODEL_DIR).as_posix() for path in files],
            "sha256": [sha256(path) for path in files],
            "weight_license": license_id,
            "license_url": f"https://huggingface.co/{repo}",
            "attribution": f"Helsinki-NLP {repo.rsplit('/', 1)[-1]} OPUS-MT model; see model card.",
        })
    manifest["models"] = assets
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    print(f"Registered multilingual English pivot models in {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
