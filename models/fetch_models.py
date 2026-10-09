"""Download pinned local inference assets and write a checksum-verified manifest.

Run with ``python models/fetch_models.py``. Assets are kept under ``models/local``
and are ignored by Git; this script does not install packages or execute model code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "local"


def hf(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


ASSETS = [
    {
        "id": "asr-en",
        "task": "streaming-asr",
        "languages": ["en"],
        "revision": "bfbb11950302dad9955f8ede01f0e768595e4fba",
        "runtime": "sherpa-onnx streaming Zipformer transducer (int8)",
        "weight_license": "Apache-2.0",
        "license_url": "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21",
        "attribution": "Sherpa-ONNX streaming Zipformer English model, csukuangfj; see model card.",
        "repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21",
        "files": [
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ],
        "destination": "asr/en",
    },
    {
        "id": "asr-fr",
        "task": "streaming-asr",
        "languages": ["fr"],
        "revision": "08b84b7b7cf519be9817e9c16919d96a7a8bad91",
        "runtime": "sherpa-onnx streaming Zipformer transducer",
        "weight_license": "CC-BY-SA",
        "license_url": "https://huggingface.co/Banafo/Kroko-ASR",
        "attribution": "Kroko ASR French community model, provided by Banafo; attribution and share-alike terms apply.",
        "repo": "csukuangfj/sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06",
        "files": ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        "destination": "asr/fr",
    },
    {
        "id": "translation-en-fr",
        "task": "translation",
        "source": "en",
        "target": "fr",
        "revision": "dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18",
        "runtime": "Hugging Face Transformers MarianMT (PyTorch)",
        "weight_license": "Apache-2.0",
        "license_url": "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr",
        "attribution": "Helsinki-NLP OPUS-MT English-to-French; trained on OPUS data.",
        "repo": "Helsinki-NLP/opus-mt-en-fr",
        "files": [
            "config.json", "generation_config.json", "pytorch_model.bin",
            "source.spm", "target.spm", "tokenizer_config.json", "vocab.json",
        ],
        "destination": "translation/en-fr",
    },
    {
        "id": "translation-fr-en",
        "task": "translation",
        "source": "fr",
        "target": "en",
        "revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
        "runtime": "Hugging Face Transformers MarianMT (PyTorch)",
        "weight_license": "Apache-2.0",
        "license_url": "https://huggingface.co/Helsinki-NLP/opus-mt-fr-en",
        "attribution": "Helsinki-NLP OPUS-MT French-to-English; trained on OPUS data.",
        "repo": "Helsinki-NLP/opus-mt-fr-en",
        "files": [
            "config.json", "generation_config.json", "pytorch_model.bin",
            "source.spm", "target.spm", "tokenizer_config.json", "vocab.json",
        ],
        "destination": "translation/fr-en",
    },
    {
        "id": "tts-en",
        "task": "tts",
        "languages": ["en"],
        "revision": "375a0fe641dea077c2a47b4e9a056d6da521eed3",
        "runtime": "Piper ONNX voice",
        "weight_license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Piper en_US-libritts_r-medium voice; LibriTTS-R dataset, CC BY 4.0.",
        "repo": "rhasspy/piper-voices",
        "files": [
            "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
            "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json",
            "en/en_US/libritts_r/medium/MODEL_CARD",
        ],
        "destination": "tts/en",
        "voice_model": "en_US-libritts_r-medium.onnx",
    },
    {
        "id": "tts-fr",
        "task": "tts",
        "languages": ["fr"],
        "revision": "375a0fe641dea077c2a47b4e9a056d6da521eed3",
        "runtime": "Piper ONNX voice",
        "weight_license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Piper fr_FR-siwis-medium voice; SIWIS speech corpus, CC BY 4.0.",
        "repo": "rhasspy/piper-voices",
        "files": [
            "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
            "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
            "fr/fr_FR/siwis/medium/MODEL_CARD",
        ],
        "destination": "tts/fr",
        "voice_model": "fr_FR-siwis-medium.onnx",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"Keeping existing {destination.relative_to(MODEL_DIR)}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "local-realtime-translation/0.1"})
    print(f"Downloading {destination.relative_to(MODEL_DIR)}", flush=True)
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    existing_optional: list[dict[str, object]] = []
    existing_manifest_path = MODEL_DIR / "manifest.yaml"
    if existing_manifest_path.is_file():
        try:
            with existing_manifest_path.open("r", encoding="utf-8") as stream:
                existing_manifest = yaml.safe_load(stream)
            if isinstance(existing_manifest, dict) and isinstance(existing_manifest.get("models"), list):
                existing_optional = [
                    entry for entry in existing_manifest["models"]
                    if isinstance(entry, dict) and entry.get("optional") is True
                ]
        except (OSError, UnicodeError, yaml.YAMLError):
            print(f"Could not read existing optional model records from {existing_manifest_path}", file=sys.stderr)
    manifest_assets: list[dict[str, object]] = []
    for asset in ASSETS:
        files: list[str] = []
        hashes: list[str] = []
        for source_name in asset["files"]:
            if asset["repo"] == "rhasspy/piper-voices":
                relative = source_name.split("/", 1)[1]
                source_revision = "v1.0.0"
            else:
                relative = source_name
                source_revision = asset["revision"]
            target = MODEL_DIR / asset["destination"] / Path(relative).name
            download(hf(asset["repo"], source_revision, source_name), target)
            files.append(target.relative_to(MODEL_DIR).as_posix())
            hashes.append(sha256(target))

        manifest_asset = {
            key: value for key, value in asset.items()
            if key not in {"repo", "files", "destination", "voice_model"}
        }
        manifest_asset["files"] = files
        manifest_asset["sha256"] = hashes
        if "voice_model" in asset:
            manifest_asset["voice_model"] = asset["voice_model"]
        manifest_assets.append(manifest_asset)

    starter_ids = {asset["id"] for asset in manifest_assets}
    manifest_assets.extend(
        asset for asset in existing_optional if asset.get("id") not in starter_ids
    )

    manifest = {"schema_version": 1, "models": manifest_assets}
    manifest_path = MODEL_DIR / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    print(f"Wrote {manifest_path}", flush=True)
    print(f"Assets: {len(manifest_assets)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
