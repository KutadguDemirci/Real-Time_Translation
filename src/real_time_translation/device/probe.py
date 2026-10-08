"""Shared inference-device detection and provider selection.

PyTorch models use CUDA, then Apple MPS, then CPU. ONNX models use the
corresponding available execution provider and always retain CPU fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    """Best device exposed by the installed ML runtime."""

    torch_device: str
    onnx_provider: str
    operating_system: str
    architecture: str


def available_onnx_providers() -> tuple[str, ...]:
    """Return ONNX Runtime providers, or the CPU baseline when it is unavailable."""
    try:
        import onnxruntime
    except ImportError:
        return ("CPUExecutionProvider",)
    return tuple(onnxruntime.get_available_providers())


def detect_device_profile() -> DeviceProfile:
    """Prefer CUDA, then Apple MPS/CoreML, and fall back to CPU."""
    torch_device = "cpu"
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        if torch.cuda.is_available():
            torch_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_device = "mps"

    providers = available_onnx_providers()
    if "CUDAExecutionProvider" in providers:
        onnx_provider = "CUDAExecutionProvider"
    elif platform.system() == "Darwin" and "CoreMLExecutionProvider" in providers:
        onnx_provider = "CoreMLExecutionProvider"
    else:
        onnx_provider = "CPUExecutionProvider"

    return DeviceProfile(
        torch_device=torch_device,
        onnx_provider=onnx_provider,
        operating_system=platform.system(),
        architecture=platform.machine(),
    )


def onnx_providers_for(profile: DeviceProfile) -> list[Any]:
    """Return preferred ONNX provider followed by CPU fallback."""
    providers = available_onnx_providers()
    preferred = profile.onnx_provider
    if preferred == "CoreMLExecutionProvider" and preferred in providers:
        return [
            (
                preferred,
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
    if preferred == "CUDAExecutionProvider" and preferred in providers:
        return [preferred, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
