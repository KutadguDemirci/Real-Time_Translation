"""Resolve and verify local model assets without downloading or making network requests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .manifest import ModelAsset, ModelManifest, load_manifest, validate_manifest


def default_model_dir() -> Path:
    """Return the platform's per-user model directory, honoring a user override."""
    override = os.environ.get("REAL_TIME_TRANSLATION_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "LocalRealTimeTranslation" / "models"
    if __import__("sys").platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LocalRealTimeTranslation" / "models"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "local-realtime-translation" / "models"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A manifest asset paired with its verified absolute local paths."""

    asset: ModelAsset
    directory: Path
    files: tuple[Path, ...]


class ModelResolver:
    """Load manifests and resolve model assets under one configured local directory."""

    def __init__(self, model_dir: str | Path | None = None, manifest_path: str | Path | None = None):
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else default_model_dir()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path
            else self.model_dir / "manifest.yaml"
        )

    def load(self) -> ModelManifest:
        """Parse the configured manifest and validate all referenced files and hashes."""
        manifest = load_manifest(self.manifest_path)
        # Optional backends (for example Qwen ASR) need not be installed unless selected.
        # resolve() validates the selected optional asset in full before it is loaded.
        problems = validate_manifest(manifest, self.model_dir, include_optional=False)
        if problems:
            raise ModelValidationError(problems, self.model_dir, self.manifest_path)
        return manifest

    def resolve(self, model: ModelAsset | str, manifest: ModelManifest | None = None) -> ResolvedModel:
        """Return a model's paths after verifying the manifest and files."""
        loaded = manifest or self.load()
        asset = loaded.get(model) if isinstance(model, str) else model
        problems = validate_manifest(
            ModelManifest(loaded.schema_version, (asset,), loaded.path), self.model_dir
        )
        if problems:
            raise ModelValidationError(problems, self.model_dir, self.manifest_path)
        paths = tuple((self.model_dir / relative).resolve() for relative in asset.files)
        return ResolvedModel(asset=asset, directory=self.model_dir, files=paths)

    def validate(self, manifest: ModelManifest | None = None) -> list[str]:
        """Return all setup errors instead of stopping at the first invalid model."""
        loaded = manifest or load_manifest(self.manifest_path)
        return validate_manifest(loaded, self.model_dir)


class ModelValidationError(RuntimeError):
    """Actionable aggregate error raised when local model assets are not ready."""

    def __init__(self, problems: list[str], model_dir: Path, manifest_path: Path):
        self.problems = tuple(problems)
        self.model_dir = model_dir
        self.manifest_path = manifest_path
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(
            "Local model validation failed:\n"
            f"{details}\n"
            f"Model directory: {model_dir}\n"
            f"Manifest: {manifest_path}\n"
            "Install the exact model files, verify their checksums and license records, "
            "then update the manifest. See models/README.md."
        )
