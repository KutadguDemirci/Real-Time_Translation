# Local model directory

Put model assets on this computer; do not commit weights. By default, the app looks for
`manifest.yaml` and assets in the per-user model directory. Set
`REAL_TIME_TRANSLATION_MODEL_DIR` to use another directory. The resolver validates the local
manifest and files before inference; it never downloads models or contacts a hosted API.

## Download the pinned starter set

From the repository root, run `python models/fetch_models.py`. This downloads the pinned English
and French streaming ASR models to `models/local/`, then writes `models/local/manifest.yaml` with
SHA-256 checksums. The model files are ignored by Git.
The downloader is explicit and separate from normal application startup.

Run `python models/fetch_translation_models.py` to add Helsinki-NLP's multilingual OPUS-MT
English-pivot models. It records the exact upstream revision, declared license, downloaded files,
and SHA-256 hashes in the same manifest. These two models cover Turkish, German, Spanish, Italian,
and Mandarin as well as other languages supported by OPUS-MT. These multilingual models are the
only translation route, including English↔French.

To add the optional Qwen3-ASR 0.6B model, install Qwen support and run
`python models/fetch_qwen_asr.py`. This downloads the official checkpoint, records its exact
revision and per-file hashes, and preserves the other manifest entries. The standard starter-set
command does not download Qwen weights. The Qwen model enables the additional spoken languages;
Sherpa-ONNX remains configured for English and French only.

```powershell
pip install -e ".[audio,inference,translation,tts,qwen-asr]"
python models/install_qwen_tts_mlx_runtime.py
python models/fetch_qwen_asr.py
python models/fetch_qwen_tts_mlx.py
```

The MLX Audio install script pins its version and installs without dependencies because Qwen ASR
uses Transformers 4.57.6. This compatibility path has an upstream metadata mismatch: `pip check`
will report MLX Audio's newer Transformers and Hugging Face Hub requirements even though the tested
Qwen3-TTS path loads with the project's pinned versions. Revalidate it when upgrading either stack.
Qwen3-TTS uses the 0.6B Base 8-bit MLX checkpoint on macOS Apple Silicon. It is about 2 GB and
requires a local voice reference plus the exact transcript spoken in that clip. Voice Studio can
transcribe longer recordings with the selected local ASR backend, save multiple named profiles
under the user's application-data directory, and reuse a selected profile for live translation.
The Base checkpoint exposes speech sampling variation; it does not provide direct emotion or
exaggeration controls. The fetch script
pins the downloaded Hugging Face commit in the manifest and records SHA-256 hashes for every file.
On Linux, add `qwen-asr-streaming` to the extras to use Qwen ASR's native vLLM partial
transcripts when the installed vLLM build supports your hardware. The standard Qwen extra uses
phrase-final Transformers transcription, which is the path used on Windows and macOS.

On Windows, point the app at this directory in the same PowerShell session before launching it:

```powershell
$env:REAL_TIME_TRANSLATION_MODEL_DIR = (Resolve-Path .\models\local).Path
```

The French Kroko community ASR model is marked CC BY-SA; preserve its attribution and share-alike
terms if you redistribute it. The translation and Qwen3-TTS models are Apache-2.0. The
bundled generic Agent and Cartoon reference audio was derived from Piper `en_US-libritts_r-medium`; its model
card identifies LibriTTS-R under CC BY 4.0. The app records attribution in its license register.
Review each upstream license and model card for your intended use.

Suggested directory layout (exact files depend on chosen revisions):

```text
models/local/
  asr/en/       # locally selected streaming English recognizer
  asr/fr/       # locally selected streaming French recognizer
  translation/mul-en/  # OPUS-MT multilingual-to-English files
  translation/en-mul/  # OPUS-MT English-to-multilingual files
  tts/qwen3-tts-0.6b-base-8bit/  # local MLX checkpoint files
```

Copy `manifest.example.yaml` to `manifest.yaml` and fill in exact paths, revisions, hashes,
runtime names, license identifiers, and license URLs. Each downloadable asset may have separate
terms. The manifest requires a 64-character SHA-256 for every listed file. Keep a local license
text copy when redistribution terms require it; reference it with `license_file` relative to the
model directory. An optional `attribution` field records the notice that must accompany the asset.
These checks establish that the manifest is complete and files match its hashes; they do not
decide whether a license permits a particular use.

`real_time_translation.models.ModelResolver` provides the programmatic entry point:

```python
from real_time_translation.models import ModelResolver

resolver = ModelResolver()  # or ModelResolver(model_dir="/path/to/models")
manifest = resolver.load()  # fails with all missing files/hash mismatches and setup instructions
asr = resolver.resolve("asr-en", manifest)
print(asr.files)
```

Use `load_manifest(path)` and `validate_manifest(manifest, directory)` when the application
needs to display all diagnostics without raising an exception.

The Sherpa-ONNX streaming adapter uses one `tokens.txt`, one encoder ONNX file, one decoder ONNX
file, and one joiner ONNX file. Use `TransducerModel.discover()` to find common filenames, or pass
explicit paths when constructing `SherpaStreamingRecognizer`. Install it with
`pip install -e .[inference]`.

Qwen3-ASR uses the same mono, normalized 16 kHz audio and VAD phrase boundaries. Select the engine
in the app and optionally enter names or specialist terms as Qwen context. On Linux with the
optional vLLM runtime installed, Qwen emits revised partial hypotheses during speech and
finalizes at the VAD endpoint. On Windows and macOS, it retains the complete phrase and transcribes
it at the endpoint with Transformers, so interim text is unavailable. Install its model runtime
with `pip install -e ".[audio,inference,translation,tts,qwen-asr]"`. The Qwen asset is optional and is validated only when selected;
runtime loading uses the verified local files and makes no network requests.

