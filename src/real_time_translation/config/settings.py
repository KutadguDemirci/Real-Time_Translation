"""Validated user preferences and cross-platform settings persistence."""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret preferences for a local translation session."""

    source_language: str = "en"
    input_device_id: int | None = None
    output_device_id: int | None = None
    endpoint_pause_ms: int = 800
    asr_backend: str = "sherpa"
    asr_context: str = ""
    cpu_threads: int | None = None
    model_profile: str = "auto"
    diagnostic_capture: bool = False
    bubble_opacity: float = 0.6


def default_config_dir() -> Path:
    """Return the platform's standard per-user application config directory."""
    override = os.environ.get("REAL_TIME_TRANSLATION_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "LocalRealTimeTranslation"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LocalRealTimeTranslation"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "local-realtime-translation"


def default_settings_path() -> Path:
    """Return the settings YAML path, honoring a direct file override if provided."""
    override = os.environ.get("REAL_TIME_TRANSLATION_SETTINGS_FILE")
    if override:
        return Path(override).expanduser().resolve()
    return default_config_dir() / "settings.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings, using defaults for absent files and invalid individual values."""
    settings_path = Path(path).expanduser() if path is not None else default_settings_path()
    if not settings_path.exists():
        return Settings()
    try:
        import yaml
        with settings_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError) as error:
        logger.warning("Could not read settings at %s; using defaults: %s", settings_path, error)
        return Settings()
    except ImportError:
        logger.error("Reading settings requires PyYAML; using defaults")
        return Settings()
    except Exception as error:
        # PyYAML's YAMLError is intentionally handled without making it a public dependency here.
        logger.warning("Could not parse settings at %s; using defaults: %s", settings_path, error)
        return Settings()
    if not isinstance(raw, Mapping):
        logger.warning("Settings at %s must contain a YAML mapping; using defaults", settings_path)
        return Settings()

    defaults = Settings()
    values: dict[str, Any] = {}
    for field in fields(Settings):
        candidate = raw.get(field.name, getattr(defaults, field.name))
        try:
            values[field.name] = _validate(field.name, candidate)
        except ValueError as error:
            logger.warning("Ignoring invalid setting %s in %s: %s", field.name, settings_path, error)
            values[field.name] = getattr(defaults, field.name)
    return Settings(**values)


def save_settings(settings: Settings, path: str | Path | None = None) -> Path:
    """Validate and atomically save preferences; returns the destination path."""
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings instance")
    destination = Path(path).expanduser() if path is not None else default_settings_path()
    normalized = Settings(
        **{
            field.name: _validate(field.name, getattr(settings, field.name))
            for field in fields(Settings)
        }
    )
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Saving settings requires PyYAML. Install the project dependencies.") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                {field.name: getattr(normalized, field.name) for field in fields(Settings)},
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temporary settings file %s", temporary, exc_info=True)
    return destination.resolve()


def _validate(name: str, value: Any) -> Any:
    if name == "source_language":
        if not isinstance(value, str) or value.strip().lower() not in {"en", "fr"}:
            raise ValueError("expected 'en' or 'fr'")
        return value.strip().lower()
    if name in {"input_device_id", "output_device_id"}:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("expected a non-negative device ID or null")
        return value
    if name == "endpoint_pause_ms":
        if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 5000:
            raise ValueError("expected an integer between 100 and 5000")
        return value
    if name == "asr_backend":
        if not isinstance(value, str) or value.strip().lower() not in {"sherpa", "qwen"}:
            raise ValueError("expected 'sherpa' or 'qwen'")
        return value.strip().lower()
    if name == "asr_context":
        if not isinstance(value, str) or len(value) > 2000:
            raise ValueError("expected text no longer than 2000 characters")
        return value.strip()
    if name == "cpu_threads":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("expected a positive integer or null")
        return value
    if name == "model_profile":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected a non-empty profile name")
        return value.strip()
    if name == "diagnostic_capture":
        if not isinstance(value, bool):
            raise ValueError("expected true or false")
        return value
    if name == "bubble_opacity":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number between 0 and 1")
        opacity = float(value)
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("expected a number between 0 and 1")
        return opacity
    raise ValueError(f"unknown setting {name!r}")
