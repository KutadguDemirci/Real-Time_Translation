# Architecture decision record

## Accepted

- Local inference on the user's Mac or Windows computer; no Hugging Face hosting in the current scope.
- French ↔ English, both directions, selected manually.
- Modular pipeline: streaming ASR → short phrase commit → text MT → TTS → ordered playback.
- Simple phrase-at-pause output for v1; optimize simultaneity later.
- CPU baseline with optional acceleration and fallback; no GPU prerequisite.
- Keep downloaded model files local and outside source control; validate exact assets and licenses.
- No raw audio retention by default; collect local per-stage timing only.

## Open until implementation

- Exact ASR model revisions and licenses.
- Translation runtime/export format and conversion verification.
- Exact English and French TTS voices and their individual licenses.
- Supported macOS/Windows versions, packaging method, and minimum memory/CPU expectations.
- Endpoint pause, audio frame size, queue sizes, and measured latency targets.

