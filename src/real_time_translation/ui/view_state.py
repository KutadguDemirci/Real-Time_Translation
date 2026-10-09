"""Pure presentation helpers for the desktop translation window."""

from __future__ import annotations

from dataclasses import dataclass

from real_time_translation.domain.events import SessionStatus


@dataclass(frozen=True)
class SessionViewState:
    """Widget-facing lifecycle labels and button state."""

    label: str
    detail: str
    running: bool
    busy: bool


_STATUS: dict[SessionStatus, SessionViewState] = {
    SessionStatus.IDLE: SessionViewState("Ready", "Choose a language and start listening.", False, False),
    SessionStatus.STARTING: SessionViewState("Preparing", "Loading local models and audio devices…", False, True),
    SessionStatus.LISTENING: SessionViewState("Listening", "Speak naturally. Translation starts after a pause.", True, False),
    SessionStatus.PROCESSING: SessionViewState("Translating", "Your phrase is being translated locally.", True, False),
    SessionStatus.STOPPING: SessionViewState("Stopping", "Finishing the current phrase…", True, True),
    SessionStatus.STOPPED: SessionViewState("Stopped", "Your microphone is off.", False, False),
    SessionStatus.ERROR: SessionViewState("Needs attention", "Check the message below and try again.", False, False),
}


def session_view_state(status: SessionStatus) -> SessionViewState:
    """Map a domain lifecycle value to concise, accessible UI copy."""
    return _STATUS[status]


def language_name(code: str) -> str:
    """Return a human-readable name for a supported language code."""
    return {
        "en": "English", "fr": "French", "tr": "Turkish", "de": "German",
        "es": "Spanish", "it": "Italian", "zh": "Chinese (Mandarin)",
    }.get(code, code.upper())
