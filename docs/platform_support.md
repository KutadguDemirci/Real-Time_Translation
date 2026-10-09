# Platform support and setup behavior

## Target

The current live translation workflow is supported on macOS Apple Silicon. The bundled session
requires a synthesizer, and the only implemented synthesizer uses MLX; Windows and Intel Macs
therefore cannot currently start live translation. Keep broader OS targets as release goals until
a compatible TTS adapter and end-to-end verification are available.

## Startup device configuration

1. Detect OS, CPU architecture/core count, available memory, default input/output devices, and installed inference providers.
2. Validate local assets and confirm the selected models can load.
3. Use Apple MLX/Metal for Qwen3-TTS 0.6B Base 8-bit on Apple Silicon. PyTorch inference for translation and Qwen ASR prefers CUDA, then Apple MPS, then CPU. On Apple MPS, Qwen ASR uses eager grouped-query attention for compatibility with PyTorch 2.6. ONNX Runtime prefers an available CUDA provider, then CoreML on macOS, then CPU.
4. Report the chosen backend. Qwen3-TTS requires Apple Silicon and fails with an actionable setup message on unsupported platforms.
5. Give actionable guidance for unavailable microphones, speakers, permissions, and model files.

Qwen3-ASR's native streaming adapter uses the optional vLLM backend where that runtime is
available. The app uses phrase-final Transformers inference on Windows and macOS, keeping those
platforms usable without vLLM; this mode has no interim transcript during the active phrase.

## macOS

Request microphone permission through the desktop framework and explain how to enable it in
System Settings. Use MLX/Metal for Qwen3-TTS and PyTorch MPS for supported ASR and translation
models. Use ONNX Runtime CoreML when its provider is available. CoreML can use Apple GPU or Neural
Engine; unsupported model operators may run on CPU.

## Windows

Handle microphone privacy permissions and default-device changes. Use CUDA when the installed
runtime exposes it and CPU otherwise. Do not make DirectML or CUDA a requirement for the base
experience.

## Audio constraints

Support device-native sample rates by resampling to the recognizer's required format. Handle
exclusive-device contention, USB/Bluetooth device changes, and sample-rate/channel mismatches.
Use a clear selector when automatic device choice is ambiguous.

