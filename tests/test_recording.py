from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import threading
import unittest
import wave

import numpy as np

from real_time_translation.audio.capture import AudioFrame
from real_time_translation.audio.recording import record_capture_to_wav


class FakeCapture:
    sample_rate = 24_000
    dropped_frames = 0
    status = "running"

    def __init__(self, frames, stop_event: threading.Event) -> None:
        self.frames = queue.Queue()
        for frame in frames:
            self.frames.put(frame)
        self.stop_event = stop_event
        self.stopped = False

    def start(self) -> None:
        pass

    def read(self, timeout=None):
        frame = self.frames.get(timeout=timeout)
        if self.frames.empty():
            self.stop_event.set()
        return frame

    def stop(self) -> None:
        self.stopped = True
        self.status = "stopped"


class RecordingTests(unittest.TestCase):
    def test_callback_frames_are_written_without_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            stop = threading.Event()
            cancel = threading.Event()
            first = np.full(480, 0.25, dtype=np.float32)
            second = np.full(480, -0.25, dtype=np.float32)
            capture = FakeCapture([
                AudioFrame(first, 24_000, 1.0),
                AudioFrame(second, 24_000, 1.02),
            ], stop)
            duration = record_capture_to_wav(
                capture, path, stop, cancel, max_seconds=1.0,
            )
            self.assertEqual(duration, 0.04)
            self.assertTrue(capture.stopped)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getframerate(), 24_000)
                self.assertEqual(audio.getnframes(), 960)
                samples = np.frombuffer(audio.readframes(960), dtype="<i2")
            self.assertEqual(samples[200], 8191)
            self.assertEqual(samples[700], -8191)

    def test_dropped_capture_frames_discard_the_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            stop = threading.Event()
            capture = FakeCapture(
                [AudioFrame(np.ones(480, dtype=np.float32) * 0.1, 24_000, 1.0)], stop
            )
            capture.dropped_frames = 3
            with self.assertRaisesRegex(RuntimeError, "dropped 3 audio frames"):
                record_capture_to_wav(
                    capture, path, stop, threading.Event(), max_seconds=1.0,
                )
            self.assertTrue(capture.stopped)


if __name__ == "__main__":
    unittest.main()
