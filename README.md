# Local Real-Time Translation

Cross-platform, local-first spoken translation for French ↔ English on macOS and Windows. The first implementation is deliberately modular: a streaming recognizer produces provisional text, a short phrase boundary commits stable text, a direction-specific translator converts it, and local TTS speaks the result.

This repository is an implementation scaffold. The Python modules describe the responsibilities and contracts, while `docs/implementation_plan.md` is the end-to-end build checklist. No model weights are included. The user places model files on the computer; the app validates the local manifest and never needs a hosted inference service.

## First-version behavior

- The user chooses French or English as the spoken language. The target language is the other one.
- Microphone audio is processed locally. Live ASR partials may be displayed, but are not translated or spoken until a phrase is committed.
- A voice activity detector and endpointing policy commit a phrase after a configurable pause. The initial pause threshold is a tuning parameter, not a claim about achieved latency.
- The desktop UI shows a per-phrase latency breakdown from microphone-frame backlog through ASR, translation, synthesis, and playback. Its end-of-speech wait defaults to 800 ms; longer waits help keep clauses together when speakers pause mid-sentence.
- Translation and TTS run in order. Playback queues phrase audio so phrases do not overlap or reorder.
- CPU inference is the required baseline. Hardware detection selects a compatible runtime profile; acceleration is optional and may fall back to CPU.
- Raw audio is not persisted by default. Logs contain operational timings and errors, not audio or full transcripts unless the user explicitly enables diagnostic capture.
- Once the models are installed locally, live translation has no network dependency.

## Repository map

See `docs/architecture.md` for the component diagram and data flow, `docs/implementation_plan.md` for the ordered build plan, and `docs/model_licenses.md` for the required license audit. Each Python file contains module-level comments describing the implementation expected there.

## Local model assets

Model files are intentionally kept outside source control. Create the model folders shown in `models/README.md` under the per-user model directory (or set `REAL_TIME_TRANSLATION_MODEL_DIR`), copy the chosen ASR, translation, and TTS assets into them, and fill in `manifest.yaml` with exact model revisions, file paths, checksums, and license metadata. The model resolver checks file hashes and reports missing or invalid assets before they are loaded.

The app can use Sherpa-ONNX or optional Qwen3-ASR for local speech recognition. Qwen uses its native streaming API through vLLM where available; elsewhere it transcribes the complete VAD phrase at the endpoint and therefore does not show interim partials. Both paths commit one final whole-phrase transcript before translation. The two Helsinki-NLP OPUS-MT directions provide translation, and local Piper provides TTS. TTS loads one configured voice for the selected target language and retains it until the language changes. Exact voice files and their individual licenses still need to be selected and audited.

Install the runtime groups with `pip install -e .[audio,inference,translation,tts]`. To enable Qwen3-ASR's phrase mode, add `qwen-asr` to that extra list. To install Qwen's vLLM-backed streaming runtime on Linux, use `pip install -e ".[qwen-asr-streaming]"`; vLLM does not support Windows natively, so Windows uses whole-phrase Qwen transcription. PyTorch inference prefers CUDA, then Apple MPS, then CPU. ONNX Runtime inference prefers CUDA, then Apple CoreML on macOS, then CPU fallback. Piper voices are local model assets; install the exact `.onnx` and adjacent `.onnx.json` files and configure their paths before use. Run `python models/fetch_models.py` to download the pinned starter set and write its local manifest, then optionally run `python models/fetch_qwen_asr.py`; see `models/README.md` for model setup.

## Development environment

The eventual application targets Python 3.11 or newer, macOS and Windows x64/arm64 as supported by the selected dependency wheels. `pyproject.toml` records the dependency groups and developer commands. Create an isolated environment and install the project in editable mode after implementation dependencies have been selected. Use `docs/platform_support.md` for platform-specific checks.

## User settings

Non-secret preferences are stored as YAML in the per-user config directory: `%APPDATA%\\LocalRealTimeTranslation` on Windows, `~/Library/Application Support/LocalRealTimeTranslation` on macOS, or `$XDG_CONFIG_HOME/local-realtime-translation` (default `~/.config/local-realtime-translation`) on Linux. The file is `settings.yaml`. Set `REAL_TIME_TRANSLATION_CONFIG_DIR` to override the directory or `REAL_TIME_TRANSLATION_SETTINGS_FILE` to select a specific file. Settings default safely when the file or an individual value is missing or invalid. The desktop saves the language, ASR engine/context, endpoint pause, and selected audio devices when it closes.

## Privacy and failure behavior

The app must clearly show microphone, model, and inference status. Missing models, unsupported devices, unavailable audio devices, and inference errors should be actionable UI states. Model discovery/validation may read local files; the runtime pipeline must not upload audio or transcript data. Network access is reserved for a separately invoked model acquisition process, if one is added later.

