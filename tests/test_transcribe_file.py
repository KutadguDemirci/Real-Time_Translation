from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
import wave

import numpy as np

from real_time_translation.asr.transcribe_file import transcribe_wav


class FakeRecognizer:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.closed = False

    def accept_audio(self, samples, sample_rate, captured_at) -> None:
        assert sample_rate == self.sample_rate
        self.chunks.append(np.asarray(samples).copy())

    def finish(self, captured_at):
        return type("Update", (), {"text": "Hello there."})()

    def close(self) -> None:
        self.closed = True


class FileTranscriptionTests(unittest.TestCase):
    def test_resamples_long_wav_and_feeds_recognizer_in_bounded_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            rate = 48_000
            frames = b"".join(struct.pack("<h", 3000) for _ in range(rate * 2))
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframes(frames)

            recognizer = FakeRecognizer()
            self.assertEqual(transcribe_wav(path, recognizer), "Hello there.")
            self.assertTrue(recognizer.closed)
            self.assertEqual([chunk.size for chunk in recognizer.chunks], [16_000, 16_000])
            self.assertTrue(all(np.max(np.abs(chunk)) <= 1.0 for chunk in recognizer.chunks))


if __name__ == "__main__":
    unittest.main()
