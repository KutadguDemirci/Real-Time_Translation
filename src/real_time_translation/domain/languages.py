"""Supported language codes and direction rules for the initial English/French pair."""

SUPPORTED_LANGUAGES = ("en", "fr")


def target_for(source_language: str) -> str:
    """Return the opposite language or raise a clear validation error."""
    if source_language == "en":
        return "fr"
    if source_language == "fr":
        return "en"
    raise ValueError(f"Unsupported source language: {source_language!r}")

