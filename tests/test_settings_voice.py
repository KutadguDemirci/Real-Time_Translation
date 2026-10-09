from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from real_time_translation.config.settings import Settings, load_settings, save_settings


class VoiceSettingsTests(unittest.TestCase):
    def test_named_profile_and_speech_variation_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            settings = Settings(tts_profile_id="a" * 32, tts_temperature=0.65)
            save_settings(settings, path)
            loaded = load_settings(path)
            self.assertEqual(loaded.tts_profile_id, settings.tts_profile_id)
            self.assertEqual(loaded.tts_temperature, 0.65)

    def test_rejects_out_of_range_speech_variation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("tts_temperature: 9.0\n", encoding="utf-8")
            self.assertEqual(load_settings(path).tts_temperature, 0.9)


if __name__ == "__main__":
    unittest.main()
