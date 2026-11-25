from __future__ import annotations

import math
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from real_time_translation.tts.voice_profiles import VoiceProfileStore


def write_reference(path: Path, seconds: float = 3.0) -> None:
    rate = 16_000
    samples = (struct.pack("<h", round(6000 * math.sin(2 * math.pi * 220 * i / rate)))
               for i in range(round(seconds * rate)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"".join(samples))


class VoiceProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.wav"
        write_reference(self.reference)
        self.store = VoiceProfileStore(
            self.root / "profiles", embedding_cache_dir=self.root / "embeddings",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_list_select_and_delete_profile(self) -> None:
        profile = self.store.save("My voice", self.reference, "Hello there, this is a test.", "en")
        self.assertEqual(profile.name, "My voice")
        self.assertEqual(profile.transcript, "Hello there, this is a test.")
        self.assertNotEqual(profile.audio_path, self.reference)
        self.assertTrue(profile.audio_path.is_file())
        self.assertEqual(self.store.get(profile.profile_id), profile)
        self.assertEqual(self.store.list_profiles(), [profile])
        self.assertTrue(self.store.delete(profile.profile_id))
        self.assertFalse(profile.audio_path.exists())
        self.assertEqual(self.store.list_profiles(), [])

    def test_deleting_voice_removes_its_cached_embeddings_only(self) -> None:
        profile = self.store.save("My voice", self.reference, "", "en")
        directory = self.root / "embeddings"
        directory.mkdir()
        digest = hashlib.sha256(profile.audio_path.read_bytes()).hexdigest()[:32]
        own_cache = directory / f"{digest}-model.npy"
        other_cache = directory / "other-voice-model.npy"
        own_cache.write_bytes(b"cached embedding")
        other_cache.write_bytes(b"other voice")
        self.store.delete(profile.profile_id)
        self.assertFalse(own_cache.exists())
        self.assertTrue(other_cache.exists())

    def test_requires_name_and_unique_names_but_allows_audio_only_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Voice names"):
            self.store.save("  ", self.reference, "words", "en")
        profile = self.store.save("My voice", self.reference, " ", "en")
        saved = self.store.get(profile.profile_id)
        assert saved is not None
        self.assertEqual(saved.transcript, "")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save("my VOICE", self.reference, "other words", "en")


if __name__ == "__main__":
    unittest.main()
