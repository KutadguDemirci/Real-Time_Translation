# End-to-end implementation plan

Implement in order; each stage should leave the app launchable and provide actionable errors.

1. **Project and settings:** complete packaging metadata, cross-platform config/model paths, structured logging, and typed settings. Add a startup screen that reports OS, architecture, memory, and config paths.
2. **Local model contract:** implement manifest parsing, SHA-256 validation, license metadata checks, path containment, compatibility checks, and missing-model instructions. Keep model acquisition manual for now.
3. **Hardware/audio setup:** enumerate microphones/speakers; select defaults; capture mono frames on a low-work callback; resample/normalize; implement reconnect and clean shutdown.
4. **ASR:** integrate selected local streaming English and French models through a common protocol; publish provisional transcript revisions; flush final text on endpoint; measure partial latency.
5. **Phrase commit:** implement VAD and exactly-once commit logic. Configure pause length. Prevent duplicated text when ASR rewrites prior partial tokens.
6. **Translation:** load the two local direction-specific translation models; set source/target from the manual selector; preserve punctuation, names, and numbers; report per-phrase latency.
7. **TTS and playback:** validate both voice assets and licenses; load the selected target-language Piper voice once and reuse it until the target changes; use the shared CUDA/CoreML/MPS/CPU device policy; synthesize target text; convert PCM formats; play phrases in order; support mute/stop and stale-queue flushing.
8. **Coordinator and desktop UI:** connect bounded queues, cancellation, UI events, source-language selector, live transcript, translation, device status, and readable errors. Keep inference off the GUI thread.
9. **Reliability and performance:** manually verify both directions, long pauses, rapid phrases, background noise, device changes, missing model files, corrupted hashes, and CPU-only startup on macOS and Windows. Record stage timings and memory use.
10. **Release preparation:** document setup, supported OS/Python/dependency versions, model acquisition and licenses, privacy behavior, known language/accent limits, and build/package instructions. Add optional accelerators only after CPU behavior is stable.

## Completion definition

The implemented system is complete when a clean macOS or Windows machine can install the app,
point it at verified local model assets, select French or English, translate microphone speech in
both directions, and hear ordered synthesized output on CPU without network inference. Startup,
missing-model, audio-device, and runtime failures must be understandable to a non-developer.

