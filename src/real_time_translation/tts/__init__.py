"""Local target-language speech synthesis adapters."""

from .base import SpeechResult, SpeechSynthesizer
from .piper_adapter import PiperSynthesizer

__all__ = ["PiperSynthesizer", "SpeechResult", "SpeechSynthesizer"]

