"""Private on-device storage for named voice-cloning profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from real_time_translation.config.settings import default_config_dir
from real_time_translation.domain.languages import SUPPORTED_LANGUAGES
from .audio_reference import validate_reference_audio


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    profile_id: str
    name: str
    audio_path: Path
    transcript: str
    language: str


class VoiceProfileStore:
    """Save validated recordings and transcripts under the user's app config directory."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory or (default_config_dir() / "voice_profiles")).expanduser().resolve()
        self.index_path = self.directory / "profiles.json"

    def list_profiles(self) -> list[VoiceProfile]:
        records = self._read_index()
        profiles = [self._profile(record) for record in records]
        return sorted(profiles, key=lambda profile: profile.name.casefold())

    def get(self, profile_id: str) -> VoiceProfile | None:
        for profile in self.list_profiles():
            if profile.profile_id == profile_id:
                return profile
        return None

    def save(self, name: str, audio_path: str | Path, transcript: str, language: str) -> VoiceProfile:
        clean_name = _validate_name(name)
        clean_transcript = " ".join(transcript.split())
        if not clean_transcript:
            raise ValueError("Transcribe the recording before saving this voice profile.")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Voice recording language must be one of {', '.join(SUPPORTED_LANGUAGES)}.")
        validate_reference_audio(audio_path)

        records = self._read_index()
        if any(str(record.get("name", "")).casefold() == clean_name.casefold() for record in records):
            raise ValueError(f'A voice profile named "{clean_name}" already exists. Choose another name.')

        profile_id = uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        profile_dir = self.directory / profile_id
        profile_dir.mkdir(mode=0o700)
        destination = profile_dir / "reference.wav"
        shutil.copyfile(audio_path, destination)
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
        record = {
            "profile_id": profile_id,
            "name": clean_name,
            "audio": f"{profile_id}/reference.wav",
            "transcript": clean_transcript,
            "language": language,
        }
        records.append(record)
        try:
            self._write_index(records)
        except Exception:
            shutil.rmtree(profile_dir, ignore_errors=True)
            raise
        return self._profile(record)

    def delete(self, profile_id: str) -> bool:
        records = self._read_index()
        remaining = [record for record in records if record.get("profile_id") != profile_id]
        if len(remaining) == len(records):
            return False
        self._write_index(remaining)
        shutil.rmtree(self.directory / profile_id, ignore_errors=True)
        return True

    def _read_index(self) -> list[dict[str, object]]:
        if not self.index_path.is_file():
            return []
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read voice profiles at {self.index_path}.") from error
        if not isinstance(raw, list):
            raise RuntimeError(f"Voice profile index must be a list: {self.index_path}")
        return [record for record in raw if isinstance(record, dict)]

    def _write_index(self, records: list[dict[str, object]]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.index_path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, self.index_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _profile(self, record: dict[str, object]) -> VoiceProfile:
        profile_id = str(record.get("profile_id", ""))
        if not re.fullmatch(r"[0-9a-f]{32}", profile_id):
            raise RuntimeError("A voice profile has an invalid ID.")
        relative = Path(str(record.get("audio", "")))
        expected = Path(profile_id) / "reference.wav"
        if relative != expected:
            raise RuntimeError(f"Voice profile {profile_id} points outside its private directory.")
        path = (self.directory / relative).resolve()
        if self.directory not in path.parents or not path.is_file():
            raise RuntimeError(f"Voice profile audio is missing: {profile_id}")
        return VoiceProfile(
            profile_id=profile_id,
            name=_validate_name(str(record.get("name", ""))),
            audio_path=path,
            transcript=str(record.get("transcript", "")),
            language=str(record.get("language", "en")),
        )


def _validate_name(name: str) -> str:
    clean = " ".join(name.split())
    if not clean or len(clean) > 64 or any(ord(character) < 32 for character in clean):
        raise ValueError("Voice names must contain 1–64 printable characters.")
    return clean
