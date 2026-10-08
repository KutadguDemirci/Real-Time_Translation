"""
Translation package.

Provides the common translator interface and the local OPUS-MT
implementation used by the real-time translation pipeline.
"""

from .base import TranslationResult, Translator
from .opus_mt import OpusTranslator

__all__ = [
    "TranslationResult",
    "Translator",
    "OpusTranslator",
]