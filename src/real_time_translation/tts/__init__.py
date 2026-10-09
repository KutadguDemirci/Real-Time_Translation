"""Local target-language speech synthesis adapters."""

from .base import SpeechResult, SpeechSynthesizer, SynthesisOptions
from .qwen3_mlx_adapter import Qwen3MLXSynthesizer

__all__ = ["Qwen3MLXSynthesizer", "SpeechResult", "SpeechSynthesizer", "SynthesisOptions"]

