from __future__ import annotations

import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from types import SimpleNamespace

import numpy as np

from real_time_translation.tts.audio_reference import validate_reference_audio
from real_time_translation.tts.base import BUILT_IN_VOICES, SUPPORTED_TTS_LANGUAGES, SynthesisOptions
from real_time_translation.tts.qwen3_mlx_adapter import Qwen3MLXSynthesizer


def write_reference(path: Path, *, seconds: float = 3.0, amplitude: int = 5000) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", round(amplitude * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(round(seconds * rate))
        )
        output.writeframes(frames)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object):
        self.calls.append(kwargs)
        yield SimpleNamespace(audio=np.asarray([0.1, -0.1], dtype=np.float32), sample_rate=24_000)


class Qwen3MLXAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        self.reference = self.root / "preset.wav"
        self.custom = self.root / "custom.wav"
        write_reference(self.reference)
        write_reference(self.custom)
        self.model = FakeModel()
        references = {
            "voice_1": (self.reference, "Hello there."),
            "voice_2": (self.reference, "Hello there."),
        }
        self.adapter = Qwen3MLXSynthesizer(
            self.model_dir, references, model_loader=lambda _path: self.model
        )

    def tearDown(self) -> None:
        self.adapter.unload()
        self.temporary.cleanup()

    def test_uses_reference_transcript_and_target_language(self) -> None:
        result = self.adapter.synthesize("Bonjour.", SynthesisOptions(target_lang="fr"))
        self.assertEqual(result.model, "Qwen3-TTS 0.6B Base (MLX 8-bit)")
        self.assertEqual(result.sample_rate, 24_000)
        self.assertEqual(result.device, "mlx-metal")
        self.assertEqual(result.pcm, np.asarray([3276, -3276], dtype="<i2").tobytes())
        call = self.model.calls[0]
        self.assertEqual(call["text"], "Bonjour.")
        self.assertEqual(call["lang_code"], "french")
        self.assertNotIn("language", call)
        self.assertEqual(call["ref_text"], "Hello there.")
        self.assertEqual(call["temperature"], 0.9)
        self.assertTrue(Path(str(call["ref_audio"])).is_file())

    def test_default_options_are_used_by_live_translation_calls(self) -> None:
        self.adapter.set_custom_reference(self.custom, "My name is Alex.")
        self.adapter.set_default_options(SynthesisOptions(
            target_lang="fr", voice_id="custom", temperature=0.6,
        ))
        self.adapter.load("fr")
        result = self.adapter.synthesize("Bonjour.")
        self.assertEqual(result.target_lang, "fr")
        self.assertEqual(self.model.calls[-1]["ref_text"], "My name is Alex.")
        self.assertEqual(self.model.calls[-1]["temperature"], 0.6)

    def test_custom_voice_requires_and_uses_exact_reference_transcript(self) -> None:
        self.adapter.set_custom_reference(self.custom, "My name is Alex.")
        result = self.adapter.synthesize("Bonjour.", SynthesisOptions(target_lang="fr", voice_id="custom"))
        self.assertEqual(result.target_lang, "fr")
        self.assertEqual(self.model.calls[-1]["ref_text"], "My name is Alex.")
        private_path = Path(str(self.model.calls[-1]["ref_audio"]))
        self.assertTrue(private_path.is_file())
        self.adapter.clear_custom_reference()
        self.assertFalse(private_path.exists())
        with self.assertRaisesRegex(ValueError, "exact transcript"):
            self.adapter.synthesize("Hello", SynthesisOptions(voice_id="custom"))

    def test_rejects_empty_reference_transcript_and_bad_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact words"):
            self.adapter.set_custom_reference(self.custom, "  ")
        with self.assertRaisesRegex(ValueError, "Unsupported TTS language"):
            SynthesisOptions(target_lang="xx").validate()
        with self.assertRaisesRegex(ValueError, "Speech variation"):
            SynthesisOptions(temperature=2.0).validate()
        self.assertEqual(BUILT_IN_VOICES, {"voice_1": "Agent", "voice_2": "Cartoon"})
        self.assertEqual(set(SUPPORTED_TTS_LANGUAGES),
                         {"en", "fr", "de", "it", "pt", "es", "zh", "ja", "ko", "ru"})

    def test_reference_audio_validation_still_applies(self) -> None:
        bad = self.root / "short.wav"
        write_reference(bad, seconds=1)
        with self.assertRaises(ValueError):
            validate_reference_audio(bad)


if __name__ == "__main__":
    unittest.main()
