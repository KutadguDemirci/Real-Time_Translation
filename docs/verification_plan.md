# Manual verification plan

Use this checklist for release verification. Keep the suite local and avoid storing test audio
unless the operator explicitly opts in.

- Verify launch, model validation, CPU fallback, and UI responsiveness on macOS and Windows.
- Exercise English, French, Turkish, German, Spanish, Italian, and Mandarin input; check available target languages with short phrases, names, numbers, punctuation, and long phrases.
- Confirm provisional ASR revisions update the UI and each final phrase is translated/spoken once.
- Confirm rapid successive phrases stay ordered and playback never overlaps.
- Measure capture-to-partial, phrase-end-to-translated-text, TTS generation, and playback-start timings.
- Verify no model/audio network calls occur during a session; inspect logs for unintended transcript/audio persistence.
- Verify missing/corrupt weights, invalid license metadata, microphone permission denial, device unplug, and output failure produce actionable messages.
- Compare CPU-only resource use and quality; use the results to decide whether optional acceleration is worthwhile.

