# Local model directory

Put model assets on this computer; do not commit weights. By default, the app looks for
`manifest.yaml` and assets in the per-user model directory. Set
`REAL_TIME_TRANSLATION_MODEL_DIR` to use another directory. The resolver validates the local
manifest and files before inference; it never downloads models or contacts a hosted API.

## Download the pinned starter set

From the repository root, run `python models/fetch_models.py`. This downloads the pinned English
and French streaming ASR models, both OPUS-MT translation directions, and medium Piper voices to
`models/local/`, then writes `models/local/manifest.yaml` with SHA-256 checksums. The model files
are ignored by Git. The downloader is explicit and separate from normal application startup.

To add the optional Qwen3-ASR 0.6B model, install Qwen support and run
`python models/fetch_qwen_asr.py`. This downloads the official checkpoint, records its exact
revision and per-file hashes, and preserves the other manifest entries. The standard starter-set
command does not download Qwen weights.

```powershell
pip install -e ".[qwen-asr]"
python models/fetch_qwen_asr.py
```

On Linux, install `pip install -e ".[qwen-asr-streaming]"` to use Qwen's native vLLM partial
transcripts when the installed vLLM build supports your hardware. The standard Qwen extra uses
phrase-final Transformers transcription, which is the path used on Windows and macOS.

On Windows, point the app at this directory in the same PowerShell session before launching it:

```powershell
$env:REAL_TIME_TRANSLATION_MODEL_DIR = (Resolve-Path .\models\local).Path
```

The French Kroko community ASR model is marked CC BY-SA; preserve its attribution and share-alike
terms if you redistribute it. The translation models are Apache-2.0. The selected Piper voice
model cards identify their training datasets as CC BY 4.0. Review each upstream license and model
card for your intended use.

Suggested directory layout (exact files depend on chosen revisions):

```text
models/local/
  asr/en/       # locally selected streaming English recognizer
  asr/fr/       # locally selected streaming French recognizer
  translation/en-fr/  # OPUS-MT English-to-French files
  translation/fr-en/  # OPUS-MT French-to-English files
  tts/en/       # selected English voice and runtime files
  tts/fr/       # selected French voice and runtime files
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
with `pip install -e .[qwen-asr]`. The Qwen asset is optional and is validated only when selected;
runtime loading uses the verified local files and makes no network requests.

