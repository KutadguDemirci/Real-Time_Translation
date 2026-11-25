from __future__ import annotations

import math
import queue
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.audio.vad import VoiceActivityDetector
from real_time_translation.domain.events import PlaybackStarted, SynthesizedAudio
from real_time_translation.domain.segments import Phrase
from real_time_translation.pipeline.coordinator import SessionCoordinator
from real_time_translation.tts.audio_reference import validate_reference_audio
from real_time_translation.tts.base import (
    BUILT_IN_VOICES, SUPPORTED_TTS_LANGUAGES, SpeechResult, SynthesisOptions,
)
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
        # The injected model is platform-independent. Real MLX still requires Apple Silicon.
        self.enterContext(patch("sys.platform", "darwin"))
        self.enterContext(patch("platform.machine", return_value="arm64"))
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

    def test_live_playback_writes_first_chunk_before_model_finishes(self) -> None:
        first_write = threading.Event()
        playback_finished = threading.Event()
        writes = []
        opened = []

        class FakeOutputStream:
            def __init__(self, **options):
                opened.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                playback_finished.set()

            def write(self, samples):
                writes.append(samples.copy())
                first_write.set()

        def generate(**options):
            self.model.calls.append(options)
            yield SimpleNamespace(audio=np.asarray([0.1, -0.1]), sample_rate=24_000)
            if not first_write.wait(timeout=2):
                raise AssertionError("Playback did not start before generation resumed.")
            yield SimpleNamespace(audio=np.asarray([0.2, -0.2]), sample_rate=24_000)

        self.enterContext(patch.object(self.model, "generate", side_effect=generate))
        self.adapter.set_custom_reference(self.custom, "My name is Alex.")
        self.adapter.set_default_options(SynthesisOptions(
            target_lang="fr", voice_id="custom", temperature=0.6,
        ))
        self.adapter.load("fr")
        fake_sd = SimpleNamespace(
            query_devices=lambda *_args: {"default_samplerate": 48_000, "max_output_channels": 2},
            OutputStream=FakeOutputStream,
        )
        events = []
        playback = AudioPlayback()
        coordinator = SessionCoordinator(
            capture=None, vad=VoiceActivityDetector(), recognizer=SimpleNamespace(language="en"),
            translator=SimpleNamespace(translate=lambda _: SimpleNamespace(
                text="Bonjour.", inference_time_ms=1.0, model="fake translation",
            )),
            synthesizer=self.adapter, playback=playback, source_language="en",
            target_language="fr", on_event=events.append,
        )
        now = time.monotonic()
        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            try:
                coordinator._process_phrase(Phrase("phrase-1", "Hello.", "en", "fr", now, now))
                self.assertTrue(playback_finished.wait(timeout=2))
            finally:
                playback.close()

        self.assertIsNone(playback.last_error)
        self.assertEqual(len(opened), 1)
        self.assertEqual([chunk.shape for chunk in writes], [(4, 2), (4, 2)])
        self.assertTrue(self.model.calls[0]["stream"])
        self.assertEqual(self.model.calls[0]["streaming_interval"], 0.32)
        self.assertEqual(self.model.calls[0]["ref_text"], "My name is Alex.")
        self.assertEqual(self.model.calls[0]["lang_code"], "french")
        self.assertEqual(self.model.calls[0]["temperature"], 0.6)
        event_types = [type(event) for event in events]
        self.assertLess(event_types.index(PlaybackStarted), event_types.index(SynthesizedAudio))

    def test_stream_rejects_empty_output_and_wraps_model_errors(self) -> None:
        def empty(**_options):
            yield from ()

        with patch.object(self.model, "generate", side_effect=empty):
            with self.assertRaisesRegex(RuntimeError, "no usable audio"):
                list(self.adapter.synthesize_stream("Hello."))
        with patch.object(self.model, "generate", side_effect=ValueError("model failure")):
            with self.assertRaisesRegex(RuntimeError, "could not stream") as raised:
                list(self.adapter.synthesize_stream("Hello."))
            self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_stream_keeps_chunk_volume_and_closes_model_generator(self) -> None:
        closed = threading.Event()

        def generate(**_options):
            try:
                yield SimpleNamespace(audio=np.asarray([0.1, -0.1]), sample_rate=24_000)
                yield SimpleNamespace(audio=np.asarray([2.0, -0.2]), sample_rate=24_000)
            finally:
                closed.set()

        with patch.object(self.model, "generate", side_effect=generate):
            chunks = list(self.adapter.synthesize_stream("Hello."))
            self.assertEqual(np.frombuffer(chunks[1].pcm, dtype="<i2").tolist(), [32767, -6553])
            self.assertTrue(closed.is_set())
            closed.clear()
            stream = self.adapter.synthesize_stream("Hello again.")
            next(stream)
            stream.close()
            self.assertTrue(closed.is_set())


class StreamingLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speech = SpeechResult(b"\x01\x00" * 24, 24_000, 1, 2, 1.0, "fake", "fr", "cpu")
        self.coordinator = SessionCoordinator(
            capture=None, vad=VoiceActivityDetector(), recognizer=SimpleNamespace(language="en"),
            translator=SimpleNamespace(), synthesizer=SimpleNamespace(),
            playback=AudioPlayback(), source_language="en", target_language="fr",
        )
        self.coordinator.timing.begin_phrase("phrase", 0.0, 0.0)

    def test_generation_failure_ends_playback_and_closes_generator(self) -> None:
        remaining = []
        closed = threading.Event()

        def chunks():
            try:
                yield self.speech
                raise RuntimeError("generation failed")
            finally:
                closed.set()

        def play_stream(_first, pending, **_options):
            remaining.append(pending)

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.coordinator._stream_phrase("phrase", chunks(), play_stream)
        self.assertIsNone(remaining[0].get_nowait())
        self.assertTrue(closed.is_set())

    def test_empty_stream_has_clear_error_and_does_not_enqueue(self) -> None:
        def chunks():
            yield from ()

        def play_stream(*_args, **_options):
            self.fail("An empty stream must not enqueue playback")

        with self.assertRaisesRegex(RuntimeError, "no usable audio"):
            self.coordinator._stream_phrase("phrase", chunks(), play_stream)

    def test_format_change_ends_stream(self) -> None:
        remaining = []

        def chunks():
            yield self.speech
            yield SpeechResult(self.speech.pcm, 48_000, 1, 2, 1.0, "fake", "fr", "cpu")

        def play_stream(_first, pending, **_options):
            remaining.append(pending)

        with self.assertRaisesRegex(ValueError, "changed audio format"):
            self.coordinator._stream_phrase("phrase", chunks(), play_stream)
        self.assertIsNone(remaining[0].get_nowait())

    def test_stop_interrupts_stream_waiting_for_more_audio(self) -> None:
        first_write = threading.Event()
        exited = threading.Event()

        class OutputStream:
            def __init__(self, **_options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                exited.set()

            def write(self, _samples):
                first_write.set()

        fake_sd = SimpleNamespace(
            query_devices=lambda *_args: {"default_samplerate": 24_000, "max_output_channels": 1},
            OutputStream=OutputStream,
        )
        playback = self.coordinator.playback
        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            try:
                playback.play_stream(self.speech, queue.SimpleQueue(), timeout=1.0, phrase_id="phrase")
                self.assertTrue(first_write.wait(timeout=2))
            finally:
                playback.close()
        self.assertTrue(exited.is_set())
        self.assertIsNone(playback.last_error)


if __name__ == "__main__":
    unittest.main()
