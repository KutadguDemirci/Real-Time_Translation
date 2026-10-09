# Local Real-Time Translation

Cross-platform, local-first spoken translation for English, French, Turkish, German, Spanish, Italian, and Mandarin Chinese on macOS and Windows. A local recognizer produces text, a short phrase boundary commits stable text, a local translator converts it, and local TTS speaks the result.

This repository is an implementation scaffold. The Python modules describe the responsibilities and contracts, while `docs/implementation_plan.md` is the end-to-end build checklist. No model weights are included. The user places model files on the computer; the app validates the local manifest and never needs a hosted inference service.

## First-version behavior

- The user chooses a spoken language and a separate output language. Turkish is available as an input language; Turkish speech output is not offered.
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

The app can use Sherpa-ONNX for English/French or optional Qwen3-ASR for all seven languages. Qwen uses its native streaming API through vLLM where available; elsewhere it transcribes the complete VAD phrase at the endpoint and therefore does not show interim partials. Both paths commit one final whole-phrase transcript before translation. Every language pair uses the same Helsinki-NLP OPUS-MT multilingual models, translating through English when neither language is English. The live output selector offers English, French, German, Spanish, Italian, and Mandarin Chinese; Turkish speech output is omitted. Open **Voice Studio** to choose the generic **Agent** or **Cartoon** reference voice, or record up to five minutes and transcribe it in any Qwen3-ASR language. Name and save multiple voice profiles, then select them for samples and live translation. Qwen Base uses a reference recording and its transcript for cloning; recordings and profiles stay local. Its Base checkpoint supports sampling variation, not direct emotion or exaggeration controls.

Install the runtime with `pip install -e ".[audio,inference,translation,tts,qwen-asr]"`, then run `python models/install_qwen_tts_mlx_runtime.py`. That installs the MLX Audio library without replacing Transformers 4.57.6, which Qwen ASR requires. Qwen3-TTS runs through Apple's MLX/Metal on Apple Silicon; the selected 8-bit checkpoint is about 2 GB and is verified against a local SHA-256 manifest. Run `python models/fetch_models.py`, `python models/fetch_translation_models.py`, `python models/fetch_qwen_asr.py`, and `python models/fetch_qwen_tts_mlx.py`. See `models/README.md` for setup. To install Qwen ASR's vLLM-backed streaming runtime on Linux, use `pip install -e ".[qwen-asr-streaming]"`; Windows uses phrase-final Qwen transcription. Translation and Qwen ASR still use PyTorch; ONNX Runtime prefers CUDA, then Apple CoreML on macOS, then CPU fallback.

## Development environment

The eventual application targets Python 3.11 or newer, macOS and Windows x64/arm64 as supported by the selected dependency wheels. `pyproject.toml` records the dependency groups and developer commands. Create an isolated environment and install the project in editable mode after implementation dependencies have been selected. Use `docs/platform_support.md` for platform-specific checks.

## User settings

Non-secret preferences are stored as YAML in the per-user config directory: `%APPDATA%\\LocalRealTimeTranslation` on Windows, `~/Library/Application Support/LocalRealTimeTranslation` on macOS, or `$XDG_CONFIG_HOME/local-realtime-translation` (default `~/.config/local-realtime-translation`) on Linux. The file is `settings.yaml`. Set `REAL_TIME_TRANSLATION_CONFIG_DIR` to override the directory or `REAL_TIME_TRANSLATION_SETTINGS_FILE` to select a specific file. Settings default safely when the file or an individual value is missing or invalid. The desktop saves the source and output languages, ASR engine/context, endpoint pause, and selected audio devices when it closes.

## Privacy and failure behavior

The app must clearly show microphone, model, and inference status. Missing models, unsupported devices, unavailable audio devices, and inference errors should be actionable UI states. Model discovery/validation may read local files; the runtime pipeline must not upload audio or transcript data. Network access is reserved for a separately invoked model acquisition process, if one is added later.

