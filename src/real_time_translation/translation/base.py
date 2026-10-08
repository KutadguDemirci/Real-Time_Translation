"""Translator protocol: translate one committed phrase with direction and optional context.

Return translated text plus timing/model metadata. Preserve numbers, names, and meaning; do not
silently summarize. Keep context short and explicit to prevent phrases from contaminating one
another. Translation is invoked only for committed text in the first version.
"""

"""
Base interfaces and data structures for translation.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranslationResult:
    """
    Result returned by a translation model.
    """

    text: str
    inference_time_ms: float
    model: str
    source_lang: str
    target_lang: str


class Translator(Protocol):
    """
    Interface that every translation backend should implement.

    The language pair is selected and the corresponding model is loaded
    before translation begins.
    """

    def load(
        self,
        source_lang: str,
        target_lang: str,
    ) -> None:
        """
        Load the model required for the selected language pair.
        """
        ...

    def translate(
        self,
        text: str,
    ) -> TranslationResult:
        """
        Translate text using the currently loaded model.
        """
        ...

    def unload(self) -> None:
        """
        Release the currently loaded model from memory.
        """
        ...