# Platform support and setup behavior

## Target

Support current macOS and Windows versions that are supported by the selected Python, audio, and
inference dependency wheels. Verify x64 and Apple Silicon separately; publish the actually tested
matrix when implementation begins.

## Startup device configuration

1. Detect OS, CPU architecture/core count, available memory, default input/output devices, and installed inference providers.
2. Validate local assets and confirm the selected models can load.
3. Use the shared device profile: PyTorch prefers CUDA, then Apple MPS, then CPU; ONNX Runtime prefers an available CUDA provider, then CoreML on macOS, then CPU.
4. Report the chosen backend and fall back to CPU if accelerated model initialization fails.
5. Give actionable guidance for unavailable microphones, speakers, permissions, and model files.

## macOS

Request microphone permission through the desktop framework and explain how to enable it in
System Settings. Use PyTorch MPS where supported and ONNX Runtime CoreML when its provider is
available. CoreML can use Apple GPU or Neural Engine; unsupported model operators may run on CPU.

## Windows

Handle microphone privacy permissions and default-device changes. Use CUDA when the installed
runtime exposes it and CPU otherwise. Do not make DirectML or CUDA a requirement for the base
experience.

## Audio constraints

Support device-native sample rates by resampling to the recognizer's required format. Handle
exclusive-device contention, USB/Bluetooth device changes, and sample-rate/channel mismatches.
Use a clear selector when automatic device choice is ambiguous.

