"""Voice recording, local transcription, named profiles, and Qwen3-TTS audition."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import wave

from real_time_translation.audio.capture import AudioCapture
from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.audio.recording import record_capture_to_wav
from real_time_translation.config.settings import Settings
from real_time_translation.tts.audio_reference import validate_reference_audio
from real_time_translation.tts.base import BUILT_IN_VOICES, SUPPORTED_TTS_LANGUAGES, SpeechResult, SynthesisOptions
from real_time_translation.tts.voice_profiles import VoiceProfile, VoiceProfileStore

logger = logging.getLogger(__name__)
MAX_RECORDING_SECONDS = 300
REFERENCE_LANGUAGES = {
    "English": "en", "French": "fr", "Turkish": "tr", "German": "de",
    "Spanish": "es", "Italian": "it", "Chinese (Mandarin)": "zh",
}
AUDITION_TEXT = {
    "en": "I’m glad we could find a moment to talk.",
    "fr": "Je suis heureux que nous ayons trouvé un moment pour discuter.",
    "de": "Ich freue mich, dass wir einen Moment zum Reden gefunden haben.",
    "it": "Sono felice che siamo riusciti a trovare un momento per parlare.",
    "pt": "Fico feliz por termos encontrado um momento para conversar.",
    "es": "Me alegra que hayamos encontrado un momento para hablar.",
    "zh": "很高兴我们能找到时间聊一聊。",
    "ja": "お話しする時間ができてうれしいです。",
    "ko": "이야기할 시간을 낼 수 있어서 기쁩니다.",
    "ru": "Я рад, что мы нашли время поговорить.",
}
Transcriber = Callable[[Path, str, str, str], str]
OptionsCallback = Callable[[SynthesisOptions, str | None], None]


class VoiceStudio:
    """Record naturally, transcribe with the selected local ASR, and save reusable voices."""

    def __init__(
        self,
        root: tk.Tk,
        synthesizer: object,
        settings: Settings,
        *,
        output_device: int | None = None,
        input_device: int | None = None,
        source_language: str = "en",
        asr_backend: str = "sherpa",
        asr_context: str = "",
        transcriber: Transcriber | None = None,
        on_options: OptionsCallback | None = None,
    ) -> None:
        self.root = root
        self.synthesizer = synthesizer
        self.settings = settings
        self.input_device = input_device
        self.source_language = source_language if source_language in REFERENCE_LANGUAGES.values() else "en"
        self.asr_backend = asr_backend
        self.asr_context = asr_context
        self.transcriber = transcriber
        self.on_options = on_options or (lambda _options, _profile_id: None)
        self.store = VoiceProfileStore()
        self.profiles = self.store.list_profiles()
        self.closed = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._record_worker: threading.Thread | None = None
        self._record_stop = threading.Event()
        self._record_cancel = threading.Event()
        self._reference_path: Path | None = None
        self._recording_directory: Path | None = None
        self.selected_profile_id: str | None = None
        self.current_builtin_voice_id = settings.tts_voice_id
        self._profile_labels: dict[str, str | None] = {}
        self._samples: list[tuple[SpeechResult, str]] = []
        self.playback = AudioPlayback(device=output_device)

        fallback_target = "fr" if self.source_language == "en" else "en"
        default_target = settings.target_language if settings.target_language in SUPPORTED_TTS_LANGUAGES else fallback_target
        self.target_language = tk.StringVar(value=SUPPORTED_TTS_LANGUAGES[default_target])
        self.text = tk.StringVar(value=AUDITION_TEXT[default_target])
        self.reference_language = tk.StringVar(value=next(
            (name for name, code in REFERENCE_LANGUAGES.items() if code == self.source_language), "English"
        ))
        self.voice = tk.StringVar()
        self.voice_name = tk.StringVar(value="")
        self.reference_name = tk.StringVar(value="No recording selected")
        self.temperature = tk.DoubleVar(value=settings.tts_temperature)
        self.consent = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready. Record up to five minutes; transcription stays on this device.")

        self.window = tk.Toplevel(root)
        self.window.title("Voice Studio")
        self.window.geometry("820x900")
        self.window.minsize(680, 720)
        self.window.configure(bg="#e5eceb")
        self.window.transient(root)
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(100, self._drain_events)
        self._select_initial_voice()

    def _build(self) -> None:
        style = ttk.Style(self.window)
        style.configure("Studio.TFrame", background="#e5eceb")
        style.configure("StudioTitle.TLabel", background="#e5eceb", foreground="#152c2a",
                        font=("Avenir Next", 18, "bold"))
        style.configure("StudioBody.TLabel", background="#e5eceb", foreground="#344a49",
                        font=("Avenir Next", 10))
        style.configure("StudioMuted.TLabel", background="#e5eceb", foreground="#4f6461",
                        font=("Avenir Next", 9))

        outer = ttk.Frame(self.window, style="Studio.TFrame", padding=(22, 18))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Voice Studio", style="StudioTitle.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Record a natural sample, transcribe it locally, then save a named voice.",
                  style="StudioBody.TLabel").pack(anchor="w", pady=(2, 12))

        profile_frame = ttk.LabelFrame(outer, text="Voice profile", padding=(10, 8))
        profile_frame.pack(fill="x", pady=(0, 10))
        voice_row = ttk.Frame(profile_frame)
        voice_row.pack(fill="x")
        ttk.Label(voice_row, text="Use voice", style="StudioBody.TLabel").pack(side="left")
        self.voice_box = ttk.Combobox(voice_row, textvariable=self.voice, state="readonly", width=36)
        self.voice_box.pack(side="left", padx=8, fill="x", expand=True)
        self.voice_box.bind("<<ComboboxSelected>>", self._voice_selected)
        self.delete_profile_button = ttk.Button(voice_row, text="Delete profile", command=self.delete_profile)
        self.delete_profile_button.pack(side="left", padx=(4, 0))

        record_frame = ttk.LabelFrame(outer, text="Create a voice from your recording", padding=(10, 8))
        record_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(record_frame,
                  text="Speak naturally for up to 5 minutes. The full recording is used for cloning; longer reference audio can increase synthesis time. Clear speech and low background noise help most.",
                  style="StudioBody.TLabel", wraplength=730).pack(anchor="w", pady=(0, 7))
        controls = ttk.Frame(record_frame)
        controls.pack(fill="x", pady=(0, 6))
        self.record_button = ttk.Button(controls, text="Start recording", command=self.toggle_recording)
        self.record_button.pack(side="left")
        ttk.Button(controls, text="Choose WAV", command=self.choose_reference).pack(side="left", padx=7)
        self.transcribe_button = ttk.Button(controls, text="Transcribe recording", command=self.transcribe)
        self.transcribe_button.pack(side="left", padx=7)
        self.preview_button = ttk.Button(controls, text="Preview", command=self.preview_reference, state="disabled")
        self.preview_button.pack(side="left", padx=7)
        ttk.Label(controls, textvariable=self.reference_name, style="StudioMuted.TLabel").pack(side="left", padx=6)

        language_row = ttk.Frame(record_frame)
        language_row.pack(fill="x", pady=(2, 6))
        ttk.Label(language_row, text="Recording language", style="StudioBody.TLabel").pack(side="left")
        self.reference_language_box = ttk.Combobox(
            language_row, textvariable=self.reference_language, state="readonly", width=14,
            values=tuple(REFERENCE_LANGUAGES),
        )
        self.reference_language_box.pack(side="left", padx=8)
        ttk.Label(language_row, text=f"ASR: {self.asr_backend.upper()} · context follows the main window setting",
                  style="StudioMuted.TLabel").pack(side="left", padx=6)

        ttk.Label(record_frame, text="Transcript (automatically filled; edit if needed)",
                  style="StudioBody.TLabel").pack(anchor="w")
        self.transcript_box = tk.Text(record_frame, height=5, wrap="word", font=("Avenir Next", 10),
                                      background="#f5f8f7", foreground="#152c2a", relief="flat",
                                      padx=8, pady=6, undo=True)
        self.transcript_box.pack(fill="x", pady=(4, 8))
        name_row = ttk.Frame(record_frame)
        name_row.pack(fill="x")
        ttk.Label(name_row, text="Name this voice", style="StudioBody.TLabel").pack(side="left")
        self.name_entry = ttk.Entry(name_row, textvariable=self.voice_name)
        self.name_entry.pack(side="left", padx=8, fill="x", expand=True)
        self.save_profile_button = ttk.Button(name_row, text="Save and use voice", command=self.save_profile)
        self.save_profile_button.pack(side="left")

        audition = ttk.LabelFrame(outer, text="Try this voice", padding=(10, 8))
        audition.pack(fill="x", pady=(0, 10))
        ttk.Label(audition, text="Text to speak", style="StudioBody.TLabel").pack(anchor="w")
        self.text_box = tk.Text(audition, height=2, wrap="word", font=("Avenir Next", 10),
                                background="#f5f8f7", foreground="#152c2a", relief="flat",
                                padx=8, pady=6, undo=True)
        self.text_box.insert("1.0", self.text.get())
        self.text_box.pack(fill="x", pady=(4, 8))
        settings_row = ttk.Frame(audition)
        settings_row.pack(fill="x")
        ttk.Label(settings_row, text="Speech language", style="StudioBody.TLabel").pack(side="left")
        self.target_language_box = ttk.Combobox(
            settings_row, textvariable=self.target_language, state="readonly", width=16,
            values=tuple(SUPPORTED_TTS_LANGUAGES.values()),
        )
        self.target_language_box.pack(side="left", padx=8)
        self.target_language_box.bind("<<ComboboxSelected>>", self._target_language_selected)
        ttk.Label(settings_row, text="Speech variation", style="StudioBody.TLabel").pack(side="left", padx=(14, 3))
        self.temperature_scale = ttk.Scale(settings_row, from_=0.1, to=1.5, variable=self.temperature)
        self.temperature_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.temperature_label = ttk.Label(settings_row, text=f"{self.temperature.get():.1f}", width=4)
        self.temperature_label.pack(side="left")
        self.temperature_scale.configure(command=self._temperature_changed)
        ttk.Label(audition,
                  text="The text is spoken as written; Voice Studio does not translate it. This checkpoint supports speech variation, not emotion or exaggeration controls.",
                  style="StudioMuted.TLabel", wraplength=740).pack(anchor="w", pady=(5, 8))
        self.consent_check = ttk.Checkbutton(
            audition, text="I have permission to reproduce this speaker’s voice.", variable=self.consent
        )
        self.consent_check.pack(anchor="w", pady=(0, 7))
        action = ttk.Frame(audition)
        action.pack(fill="x")
        self.generate_button = ttk.Button(action, text="Generate sample", command=self.generate)
        self.generate_button.pack(side="left")
        ttk.Label(action, textvariable=self.status, style="StudioMuted.TLabel",
                  wraplength=620).pack(side="left", padx=12, fill="x", expand=True)

        samples = ttk.LabelFrame(outer, text="Recent samples", padding=(8, 5))
        samples.pack(fill="both", expand=True)
        self.sample_list = ttk.Frame(samples, style="Studio.TFrame")
        self.sample_list.pack(fill="both", expand=True)
        self._refresh_voice_choices()

    def _refresh_voice_choices(self) -> None:
        self._profile_labels.clear()
        labels = []
        for voice_id, name in BUILT_IN_VOICES.items():
            label = f"{name} · preset"
            self._profile_labels[label] = None
            labels.append(label)
        for profile in self.profiles:
            label = f"{profile.name} · saved voice"
            self._profile_labels[label] = profile.profile_id
            labels.append(label)
        self.voice_box.configure(values=labels)

    def _select_initial_voice(self) -> None:
        profile = self.store.get(self.settings.tts_profile_id) if self.settings.tts_profile_id else None
        if profile is not None:
            label = next((key for key, value in self._profile_labels.items()
                          if value == profile.profile_id), "")
        else:
            label = f"{BUILT_IN_VOICES.get(self.settings.tts_voice_id, 'Agent')} · preset"
        self.voice.set(label)
        if profile:
            self._activate_profile(profile)
        else:
            self._activate_builtin(self.settings.tts_voice_id)

    def _voice_selected(self, _event: object = None) -> None:
        profile_id = self._profile_labels.get(self.voice.get())
        if profile_id:
            profile = self.store.get(profile_id)
            if profile is not None:
                self._activate_profile(profile)
        else:
            voice_id = next((key for key, value in BUILT_IN_VOICES.items()
                             if self.voice.get() == f"{value} · preset"), "voice_1")
            self._activate_builtin(voice_id)

    def _activate_profile(self, profile: VoiceProfile) -> None:
        self.selected_profile_id = profile.profile_id
        self._reference_path = profile.audio_path
        self.reference_name.set(f"{profile.name} · saved voice")
        self.voice_name.set(profile.name)
        self.reference_language.set(next(
            (name for name, code in REFERENCE_LANGUAGES.items() if code == profile.language), "English"
        ))
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", profile.transcript)
        self.preview_button.configure(state="normal")
        self.consent.set(False)
        self.synthesizer.set_custom_reference(profile.audio_path, profile.transcript)
        self._apply_options("custom", profile.profile_id)
        self._set_status(f'Using saved voice "{profile.name}".')

    def _activate_builtin(self, voice_id: str) -> None:
        self.current_builtin_voice_id = voice_id
        self.selected_profile_id = None
        self._reference_path = None
        self.reference_name.set("No recording selected")
        self.voice_name.set("")
        self.transcript_box.delete("1.0", "end")
        self.preview_button.configure(state="disabled")
        self.consent.set(True)
        self.synthesizer.clear_custom_reference()
        self._apply_options(voice_id, None)
        self._set_status(f"Using the {BUILT_IN_VOICES[voice_id]} preset voice.")

    def _apply_options(self, voice_id: str, profile_id: str | None) -> None:
        options = self._options(voice_id)
        self.synthesizer.set_default_options(options)
        self.on_options(options, profile_id)

    def _options(self, voice_id: str) -> SynthesisOptions:
        return SynthesisOptions(
            target_lang=self._selected_target_language(),
            voice_id=voice_id,
            temperature=float(self.temperature.get()),
        ).validate()

    def toggle_recording(self) -> None:
        if self._record_worker is not None and self._record_worker.is_alive():
            self._record_stop.set()
            self.record_button.configure(state="disabled")
            self._set_status("Stopping recording and saving the audio…")
            return
        self._remove_recording()
        self._reference_path = None
        self.reference_name.set("Recording…")
        directory = Path(tempfile.mkdtemp(prefix="local-translator-recording-"))
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        self._recording_directory = directory
        path = directory / "voice-reference.wav"
        self._record_stop.clear()
        self._record_cancel.clear()
        self.record_button.configure(text="Stop recording")
        self._set_status("Recording… speak naturally, then press Stop recording.")
        self._record_worker = threading.Thread(
            target=self._record_audio, args=(path, directory), name="voice-profile-recording", daemon=True
        )
        self._record_worker.start()

    def _record_audio(self, path: Path, directory: Path) -> None:
        try:
            capture = AudioCapture(
                sample_rate=24_000,
                frame_ms=20,
                device=self.input_device,
                queue_size=500,
            )
            duration = record_capture_to_wav(
                capture,
                path,
                self._record_stop,
                self._record_cancel,
                max_seconds=MAX_RECORDING_SECONDS,
                on_progress=lambda seconds: self.events.put(("recording_progress", seconds)),
            )
            if self._record_cancel.is_set():
                shutil.rmtree(directory, ignore_errors=True)
                return
            info = validate_reference_audio(path)
            self.events.put(("recorded", (path, directory, max(duration, info.duration_seconds))))
        except Exception as error:
            shutil.rmtree(directory, ignore_errors=True)
            self.events.put(("error", _safe_error(error)))

    def choose_reference(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.window, title="Choose an authorized voice recording",
            filetypes=(("PCM WAV audio", "*.wav"),),
        )
        if not selected:
            return
        try:
            info = validate_reference_audio(selected)
        except (OSError, ValueError) as error:
            self._set_status(str(error))
            return
        self._remove_recording()
        self._reference_path = Path(selected)
        self.selected_profile_id = None
        self.voice_name.set("")
        self.transcript_box.delete("1.0", "end")
        self.reference_name.set(f"{self._reference_path.name} · {info.duration_seconds:.1f}s")
        self.preview_button.configure(state="normal")
        self.consent.set(False)
        self._set_status("Recording selected. Transcribe it, name it, then save it as a voice profile.")

    def transcribe(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self._reference_path is None:
            self._set_status("Record or choose a WAV recording first.")
            return
        if self.transcriber is None:
            self._set_status("The selected local ASR model is not available. Set up ASR models first.")
            return
        self.transcribe_button.configure(state="disabled")
        path = self._reference_path
        language = REFERENCE_LANGUAGES[self.reference_language.get()]
        backend = self.asr_backend if language in {"en", "fr"} else "qwen"
        self._set_status(f"Transcribing locally with {backend.upper()}…")

        def work() -> None:
            try:
                text = self.transcriber(path, language, backend, self.asr_context)
                self.events.put(("transcribed", text))
            except Exception as error:
                logger.info("Voice recording transcription failed: %s", error)
                self.events.put(("error", _safe_error(error)))

        self._worker = threading.Thread(target=work, name="voice-profile-transcription", daemon=True)
        self._worker.start()

    def save_profile(self) -> None:
        if self._reference_path is None:
            self._set_status("Record or choose a WAV recording first.")
            return
        try:
            profile = self.store.save(
                self.voice_name.get(), self._reference_path,
                self.transcript_box.get("1.0", "end-1c"),
                REFERENCE_LANGUAGES[self.reference_language.get()],
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._set_status(str(error))
            return
        self._remove_recording()
        self.profiles = self.store.list_profiles()
        self._refresh_voice_choices()
        label = next(key for key, value in self._profile_labels.items() if value == profile.profile_id)
        self.voice.set(label)
        self._activate_profile(profile)

    def delete_profile(self) -> None:
        if not self.selected_profile_id:
            self._set_status("Choose one of your saved voice profiles first.")
            return
        profile = self.store.get(self.selected_profile_id)
        if profile is None or not messagebox.askyesno(
            "Delete voice profile", f'Delete "{profile.name}" and its saved recording?', parent=self.window
        ):
            return
        self.store.delete(profile.profile_id)
        self.profiles = self.store.list_profiles()
        self._refresh_voice_choices()
        self.voice.set(f"{BUILT_IN_VOICES[self.current_builtin_voice_id]} · preset")
        self._activate_builtin(self.current_builtin_voice_id)

    def preview_reference(self) -> None:
        if self._reference_path is None:
            return
        try:
            self.playback.play(_read_reference(self._reference_path))
        except Exception as error:
            self._set_status("Could not preview the selected recording.")
            logger.info("Voice recording preview failed: %s", error)

    def generate(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        phrase = self.text_box.get("1.0", "end-1c").strip()
        if not phrase:
            self._set_status("Enter a phrase to speak first.")
            return
        if self.selected_profile_id is None and self.voice.get().endswith("saved voice"):
            self._set_status("Choose a saved profile or a preset voice first.")
            return
        if self.selected_profile_id and not self.consent.get():
            self._set_status("Confirm that you have permission to reproduce this speaker.")
            return
        voice_id = "custom" if self.selected_profile_id else self._selected_builtin_id()
        try:
            options = self._options(voice_id)
        except ValueError as error:
            self._set_status(str(error))
            return
        self.generate_button.configure(state="disabled")
        self._set_status("Generating sample with the local Qwen3-TTS model…")

        def work() -> None:
            try:
                result = self.synthesizer.synthesize(phrase, options)
                self.playback.play(result)
                self.events.put(("success", (result, options)))
            except Exception as error:
                logger.info("Voice sample generation failed: %s", error)
                self.events.put(("error", _safe_error(error)))

        self._worker = threading.Thread(target=work, name="voice-studio-inference", daemon=True)
        self._worker.start()

    def _selected_builtin_id(self) -> str:
        return next((voice_id for voice_id, name in BUILT_IN_VOICES.items()
                     if self.voice.get() == f"{name} · preset"), "voice_1")

    def _selected_target_language(self) -> str:
        return next((code for code, name in SUPPORTED_TTS_LANGUAGES.items()
                     if name == self.target_language.get()), "en")

    def _target_language_selected(self, _event: object = None) -> None:
        """Keep the untouched starter phrase in the selected spoken language."""
        current_text = self.text_box.get("1.0", "end-1c").strip()
        starter_phrases = set(AUDITION_TEXT.values())
        if current_text in starter_phrases:
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", AUDITION_TEXT[self._selected_target_language()])

    def _temperature_changed(self, value: str) -> None:
        self.temperature_label.configure(text=f"{float(value):.1f}")
        if self.selected_profile_id is not None:
            self._apply_options("custom", self.selected_profile_id)
        elif self.voice.get().endswith("preset"):
            self._apply_options(self._selected_builtin_id(), None)

    def _drain_events(self) -> None:
        if self.closed:
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "recording_progress":
                self._set_status(f"Recording… {float(value):.0f} / {MAX_RECORDING_SECONDS} seconds. Press Stop when ready.")
            elif kind == "recorded":
                path, directory, seconds = value
                if self._recording_directory is not None and self._recording_directory != directory:
                    shutil.rmtree(self._recording_directory, ignore_errors=True)
                self._recording_directory = directory
                self._reference_path = path
                self.reference_name.set(f"Recording · {seconds:.1f}s")
                self.voice_name.set("")
                self.transcript_box.delete("1.0", "end")
                self.preview_button.configure(state="normal")
                self.consent.set(False)
                self._set_status("Recording saved temporarily. Transcribe it, name it, and save the profile.")
            elif kind == "transcribed":
                self.transcribe_button.configure(state="normal")
                self.transcript_box.delete("1.0", "end")
                self.transcript_box.insert("1.0", str(value))
                self._set_status("Transcription complete. Review it, name the voice, then save the profile.")
            elif kind == "success":
                result, options = value
                self._samples.insert(0, (result, self._settings_label(options)))
                self._samples = self._samples[:4]
                self._render_samples()
                self._apply_options(options.voice_id, self.selected_profile_id)
                self._set_status(f"Generated {result.target_lang.upper()} audio in {result.synthesis_time_ms:.0f} ms.")
            elif kind == "error":
                self.record_button.configure(text="Start recording", state="normal")
                self.transcribe_button.configure(state="normal")
                self.generate_button.configure(state="normal")
                self._set_status(str(value))
        if self._record_worker is None or not self._record_worker.is_alive():
            self.record_button.configure(text="Start recording", state="normal")
        if self._worker is None or not self._worker.is_alive():
            self.transcribe_button.configure(state="normal")
            self.generate_button.configure(state="normal")
        self.window.after(100, self._drain_events)

    def _render_samples(self) -> None:
        for child in self.sample_list.winfo_children():
            child.destroy()
        for index, (_result, label) in enumerate(self._samples, 1):
            row = ttk.Frame(self.sample_list, padding=(3, 2))
            row.pack(fill="x")
            ttk.Button(row, text=f"Play sample {index}",
                       command=lambda i=index - 1: self._play_sample(i)).pack(side="left")
            ttk.Label(row, text=label).pack(side="left", padx=9)

    def _play_sample(self, index: int) -> None:
        try:
            self.playback.play(self._samples[index][0])
        except Exception as error:
            self._set_status("Could not play that sample.")
            logger.info("Voice sample playback failed: %s", error)

    def _settings_label(self, options: SynthesisOptions) -> str:
        voice = self.voice.get().split(" · ", 1)[0]
        return f"{options.target_lang.upper()} · {voice} · variation {options.temperature:.1f}"

    def _set_status(self, message: str) -> None:
        self.status.set(message)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._record_cancel.set()
        self._record_stop.set()
        self._remove_recording()
        def cleanup() -> None:
            try:
                self.playback.close()
            except Exception:
                logger.info("Voice Studio cleanup failed", exc_info=True)
        threading.Thread(target=cleanup, name="voice-studio-close", daemon=True).start()
        self.window.destroy()

    def _remove_recording(self) -> None:
        if self._recording_directory is not None:
            shutil.rmtree(self._recording_directory, ignore_errors=True)
            self._recording_directory = None


def _read_reference(path: Path) -> SpeechResult:
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        channels, rate, width = audio.getnchannels(), audio.getframerate(), audio.getsampwidth()
        raw = audio.readframes(audio.getnframes())
    if width == 2:
        pcm = raw
    elif width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        wide = (values[:, 0].astype(np.int32) |
                (values[:, 1].astype(np.int32) << 8) |
                (values[:, 2].astype(np.int32) << 16))
        wide = np.where(wide & 0x800000, wide - 0x1000000, wide)
        pcm = (wide >> 8).astype("<i2").tobytes()
    else:
        values = np.frombuffer(raw, dtype="<i4")
        pcm = (values >> 16).astype("<i2").tobytes()
    return SpeechResult(pcm, rate, channels, 2, 0.0, "voice reference", "en", "local")


def _safe_error(error: Exception) -> str:
    if isinstance(error, (ValueError, FileNotFoundError)):
        return str(error)
    if isinstance(error, RuntimeError) and str(error).startswith(("Qwen3-TTS", "Qwen3-ASR")):
        return str(error)
    return "Voice Studio could not finish the request. Check the selected local models and try again."
