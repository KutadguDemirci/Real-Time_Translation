# Architecture decisions

## Product scope

- Local microphone translation, French ↔ English, both directions.
- macOS and Windows support; no GPU prerequisite.
- Manual source-language selection avoids an auto-detection delay and makes the correct ASR model explicit.
- Translate/speak only after a short phrase boundary in v1. Live transcript text may be provisional.
- Models live on the user's computer. Runtime inference is local and audio is not stored by default.

## Runtime/data flow

```text
Desktop UI ──settings/status──> Application composition root
                                      │
Microphone -> frame queue -> VAD + selected streaming ASR
                                      ├── partial text -> UI
                                      └── phrase committer -> ordered phrase queue
                                                               │
                                              direction-specific local MT
                                                               │
                                               local target-language TTS
                                                               │
                                                  ordered PCM playback -> speaker
```

Audio callbacks do minimal work and write fixed-size frames into bounded queues. Inference runs
on worker tasks/threads. Every event carries a phrase ID and monotonic timestamp. Backpressure
must be explicit: on overloaded queues, report status and avoid unbounded memory growth.

## Phrase policy

The streaming recognizer emits revisions. The UI may replace the current partial transcript.
VAD silence plus a final ASR flush commits a source phrase exactly once. The initial configurable
trailing-silence default is 800 ms; tune it in the desktop UI. Translate only committed source text,
then synthesize and play it in order. Do not speak replaceable partial text.

## Device and runtime policy

Use one device profile across inference components. PyTorch prefers CUDA, then Apple MPS, then
CPU. ONNX Runtime prefers an available CUDA provider, then CoreML on macOS, then CPU; every
accelerated ONNX session keeps CPU fallback. Provider/model initialization failure retries on CPU.
CoreML may dispatch supported operators to the Apple GPU or Neural Engine and unsupported work to
CPU. Report the selected provider and any fallback. Never require a dedicated GPU.

## Local models and licenses

Model assets are external to the repository and referenced by a manifest with exact revisions,
paths, hashes, runtime formats, languages, and license metadata. No model is approved based only
on its library's license. Record code, weights, voices, and training-data restrictions separately.
See `model_licenses.md` and `../models/manifest.example.yaml`.

## Component boundaries

`AudioCapture` owns input format/device; `VoiceActivityDetector` owns silence detection;
`StreamingRecognizer` emits partial/final text; `PhraseCommitter` emits exactly-once source
phrases; `Translator` maps one phrase to the target language; `SpeechSynthesizer` loads the voice
for the selected target language and retains it until direction changes, then returns PCM;
`AudioPlayback` owns ordered output; `SessionCoordinator` connects them and owns cancellation.
The desktop UI consumes events and never calls model inference directly.

