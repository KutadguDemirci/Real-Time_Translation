"""Parse and validate the local model manifest and its license records.

The manifest is deliberately data-only: this module never downloads assets or contacts a
model host. Validation is split into structural checks (``load_manifest``) and local file
checks (``validate_manifest``), so callers can show useful setup errors before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SUPPORTED_TASKS = frozenset({"streaming-asr", "translation", "tts"})


class ManifestError(ValueError):
    """A manifest is missing required fields or contains invalid values."""


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """One model entry and the local files that make up that asset."""

    id: str
    task: str
    revision: str
    runtime: str
    files: tuple[str, ...]
    sha256: tuple[str, ...]
    weight_license: str
    license_url: str
    languages: tuple[str, ...] = ()
    source: str | None = None
    target: str | None = None
    attribution: str | None = None
    license_path: str | None = None

    @property
    def language_pair(self) -> tuple[str, str] | None:
        if self.source is None or self.target is None:
            return None
        return self.source, self.target


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated manifest metadata, without making claims about license suitability."""

    schema_version: int
    models: tuple[ModelAsset, ...]
    path: Path | None = None

    def get(self, model_id: str) -> ModelAsset:
        """Return the named model or raise a helpful ``KeyError``."""
        for model in self.models:
            if model.id == model_id:
                return model
        choices = ", ".join(model.id for model in self.models) or "none"
        raise KeyError(f"No model {model_id!r} in manifest. Available model IDs: {choices}.")

    def for_task(self, task: str) -> tuple[ModelAsset, ...]:
        return tuple(model for model in self.models if model.task == task)


def load_manifest(path: str | Path) -> ModelManifest:
    """Read a YAML manifest. PyYAML is a small required dependency of the model contract."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Local model manifest not found: {manifest_path}. Copy "
            "models/manifest.example.yaml to models/manifest.yaml and fill in verified assets."
        )
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Reading model manifests requires PyYAML. Install the project dependencies.") from error
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ManifestError(f"Could not read model manifest {manifest_path}: {error}") from error
    manifest = parse_manifest(raw, path=manifest_path)
    return manifest


def parse_manifest(raw: Any, *, path: Path | None = None) -> ModelManifest:
    """Validate manifest structure and required metadata without touching model files."""
    if not isinstance(raw, Mapping):
        raise ManifestError("Model manifest must be a YAML mapping/object.")
    version = raw.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise ManifestError(f"Unsupported or missing schema_version {version!r}; expected 1.")
    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("Model manifest must contain a non-empty 'models' list.")

    models: list[ModelAsset] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        label = f"models[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{label} must be a mapping/object")
            continue
        model, item_errors = _parse_asset(item, label)
        errors.extend(item_errors)
        if model is not None:
            if model.id in seen:
                errors.append(f"{label}.id duplicates model ID {model.id!r}")
            else:
                seen.add(model.id)
                models.append(model)
    if errors:
        raise ManifestError("Invalid model manifest:\n- " + "\n- ".join(errors))
    return ModelManifest(schema_version=1, models=tuple(models), path=path)


def validate_manifest(
    manifest: ModelManifest,
    model_dir: str | Path,
    *,
    check_hashes: bool = True,
) -> list[str]:
    """Return actionable errors for missing, unsafe, or checksum-mismatched local assets."""
    root = Path(model_dir).expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        return [f"Model directory does not exist: {root}. Create it and install the assets listed in the manifest."]
    for model in manifest.models:
        for relative, expected in zip(model.files, model.sha256):
            try:
                file_path = _contained_path(root, relative)
            except ValueError as error:
                errors.append(f"{model.id}: {error}")
                continue
            if not file_path.is_file():
                errors.append(f"{model.id}: required file is missing: {relative}")
                continue
            if check_hashes:
                actual = _sha256(file_path)
                if actual.lower() != expected.lower():
                    errors.append(
                        f"{model.id}: SHA-256 mismatch for {relative} (expected {expected}, got {actual}). "
                        "Replace the file with the exact manifest revision or update the verified checksum."
                    )
        if model.license_path:
            try:
                license_file = _contained_path(root, model.license_path)
            except ValueError as error:
                errors.append(f"{model.id} license record: {error}")
            else:
                if not license_file.is_file():
                    errors.append(f"{model.id}: license text file is missing: {model.license_path}")
    return errors


def _parse_asset(item: Mapping[str, Any], label: str) -> tuple[ModelAsset | None, list[str]]:
    errors: list[str] = []

    def required_string(key: str) -> str:
        value = item.get(key)
        if not isinstance(value, str) or not value.strip() or _is_placeholder(value):
            errors.append(f"{label}.{key} must be a verified, non-empty string")
            return ""
        return value.strip()

    model_id = required_string("id")
    task = required_string("task")
    revision = required_string("revision")
    runtime = required_string("runtime")
    license_id = required_string("weight_license")
    license_url = required_string("license_url")
    attribution = required_string("attribution")
    if task and task not in SUPPORTED_TASKS:
        errors.append(f"{label}.task must be one of {', '.join(sorted(SUPPORTED_TASKS))}")
    if license_url:
        parsed_url = urlparse(license_url)
        if parsed_url.scheme not in ("https", "http") or not parsed_url.netloc:
            errors.append(f"{label}.license_url must be an http(s) URL")

    raw_files = item.get("files")
    raw_hashes = item.get("sha256")
    files = _string_list(raw_files, f"{label}.files", errors)
    hashes = _string_list(raw_hashes, f"{label}.sha256", errors)
    if not files:
        errors.append(f"{label}.files must list at least one relative asset path")
    if len(files) != len(hashes):
        errors.append(f"{label}.sha256 must contain one checksum for each file")
    for index, checksum in enumerate(hashes):
        if not SHA256_RE.fullmatch(checksum):
            errors.append(f"{label}.sha256[{index}] must be a 64-character hexadecimal SHA-256")
    if any(Path(name).is_absolute() or "\\" in name for name in files):
        errors.append(f"{label}.files paths must be relative and use forward slashes")
    if any(part in ("", ".", "..") for name in files for part in name.split("/")):
        errors.append(f"{label}.files paths must not contain empty, '.' or '..' components")

    languages = tuple(_string_list(item.get("languages", []), f"{label}.languages", errors, allow_empty=True))
    source = item.get("source")
    target = item.get("target")
    if task == "translation":
        for key, value in (("source", source), ("target", target)):
            if not isinstance(value, str) or not value.strip() or _is_placeholder(value):
                errors.append(f"{label}.{key} must be a supported language code")
        if isinstance(source, str) and isinstance(target, str) and source.strip() == target.strip():
            errors.append(f"{label}.source and target must differ")
    elif task in ("streaming-asr", "tts") and not languages:
        errors.append(f"{label}.languages must list at least one language")

    license_path = item.get("license_file")
    for key, value in (("license_file", license_path),):
        if value is not None and (not isinstance(value, str) or not value.strip() or _is_placeholder(value)):
            errors.append(f"{label}.{key} must be a non-empty value when provided")
    if errors:
        return None, errors
    return ModelAsset(
        id=model_id, task=task, revision=revision, runtime=runtime, files=tuple(files),
        sha256=tuple(hashes), weight_license=license_id, license_url=license_url,
        languages=languages,
        source=source.strip().lower() if isinstance(source, str) else None,
        target=target.strip().lower() if isinstance(target, str) else None,
        attribution=attribution.strip() if isinstance(attribution, str) else None,
        license_path=license_path.strip() if isinstance(license_path, str) else None,
    ), []


def _string_list(value: Any, label: str, errors: list[str], *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        errors.append(f"{label} must be a {'possibly empty ' if allow_empty else 'non-empty '}list")
        return []
    result: list[str] = []
    for index, member in enumerate(value):
        if not isinstance(member, str) or not member.strip() or _is_placeholder(member):
            errors.append(f"{label}[{index}] must be a verified, non-empty string")
        else:
            result.append(member.strip())
    return result


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return "replace_with" in normalized or normalized in {"tbd", "todo", "placeholder"}


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"asset path escapes the configured model directory: {relative}") from error
    return candidate


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
