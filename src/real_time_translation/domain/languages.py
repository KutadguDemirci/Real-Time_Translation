"""Language codes available to recognition and translation."""

SUPPORTED_LANGUAGES = ("en", "fr", "tr", "de", "es", "it", "zh")
SPEECH_LANGUAGES = ("en", "fr", "de", "es", "it", "zh")


def target_for(source_language: str) -> str:
    """Return the legacy opposite direction for old English/French callers."""
    if source_language == "en":
        return "fr"
    if source_language == "fr":
        return "en"
    raise ValueError(f"Unsupported source language: {source_language!r}")

