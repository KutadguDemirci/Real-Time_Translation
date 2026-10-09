# Implementation scope and release validation

This document records the intended end-to-end path and release criteria. Core application paths
are implemented, while platform behavior and release acceptance still need the checks below; this
document does not claim every item has been verified.

1. **Project and settings:** complete packaging metadata, cross-platform config/model paths, structured logging, and typed settings. Add a startup screen that reports OS, architecture, memory, and config paths.
2. **Local model contract:** implement manifest parsing, SHA-256 validation, license metadata checks, path containment, compatibility checks, and missing-model instructions. Keep model acquisition manual for now.
3. **Hardware/audio setup:** enumerate microphones/speakers; select defaults; capture mono frames on a low-work callback; resample/normalize; implement reconnect and clean shutdown.
4. **ASR:** integrate local English/French Sherpa-ONNX and optional multilingual Qwen3-ASR through a common protocol; publish provisional transcript revisions; flush final text on endpoint; measure partial latency.
5. **Phrase commit:** implement VAD and exactly-once commit logic. Configure pause length. Prevent duplicated text when ASR rewrites prior partial tokens.
6. **Translation:** load multilingual OPUS-MT models and translate non-English pairs through English; select source and output independently; preserve punctuation, names, and numbers; report per-phrase latency.
7. **TTS and playback:** validate the local Qwen3-TTS 0.6B Base 8-bit MLX checkpoint and generic reference clips; load one model per session; pass exact reference transcripts for preset and cloned voices; synthesize target text on Apple MLX; convert PCM formats; play phrases in order; support mute/stop and stale-queue flushing.
8. **Coordinator and desktop UI:** connect bounded queues, cancellation, UI events, source-language selector, live transcript, translation, device status, and readable errors. Keep inference off the GUI thread.
9. **Reliability and performance:** manually verify both directions, long pauses, rapid phrases, background noise, device changes, missing model files, corrupted hashes, and CPU-only startup on macOS and Windows. Record stage timings and memory use.
10. **Release preparation:** document setup, supported OS/Python/dependency versions, model acquisition and licenses, privacy behavior, known language/accent limits, and build/package instructions. Add optional accelerators only after CPU behavior is stable.

## Release acceptance criteria

The implemented system is complete when a clean macOS or Windows machine can install the app,
point it at verified local model assets, select a supported spoken and output language, translate
microphone speech, and hear ordered synthesized output on CPU without network inference. Startup,
missing-model, audio-device, and runtime failures must be understandable to a non-developer.

