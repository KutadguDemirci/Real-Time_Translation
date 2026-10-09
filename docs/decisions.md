# Architecture decision record

## Accepted

- Local inference on the user's Mac or Windows computer; no Hugging Face hosting in the current scope.
- English, French, Turkish, German, Spanish, Italian, and Mandarin Chinese source languages, selected manually; output language is selected separately.
- Turkish speech output is omitted; supported Turkish source phrases can be translated into a Qwen3-TTS output language.
- Modular pipeline: streaming ASR → short phrase commit → text MT → TTS → ordered playback.
- Simple phrase-at-pause output for v1; optimize simultaneity later.
- CPU baseline with optional acceleration and fallback; no GPU prerequisite.
- Keep downloaded model files local and outside source control; validate exact assets and licenses.
- No raw audio retention by default; collect local per-stage timing only.

## Open validation items

- Verify the exact ASR revisions and license obligations for redistribution.
- Measure Qwen3-TTS 0.6B Base MLX latency and memory use on Apple Silicon.
- Publish tested macOS/Windows versions, packaging method, and minimum memory/CPU expectations.
- Tune endpoint pause, audio frame size, queue sizes, and latency targets from real-device measurements.

