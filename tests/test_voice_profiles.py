from __future__ import annotations

import math
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
        self.store = VoiceProfileStore(self.root / "profiles")

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

    def test_requires_name_transcript_and_unique_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Voice names"):
            self.store.save("  ", self.reference, "words", "en")
        with self.assertRaisesRegex(ValueError, "Transcribe"):
            self.store.save("My voice", self.reference, " ", "en")
        self.store.save("My voice", self.reference, "words", "en")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save("my VOICE", self.reference, "other words", "en")


if __name__ == "__main__":
    unittest.main()
