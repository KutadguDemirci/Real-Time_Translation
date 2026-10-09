"""Tk desktop interface for local French/English speech translation.

The UI owns presentation only. A coordinator factory supplied by the application creates the
direction-specific pipeline; domain events cross onto the Tk thread through a bounded queue.
"""

from __future__ import annotations

from collections.abc import Callable
import base64
import ctypes
from dataclasses import replace
import logging
import os
from pathlib import Path
import queue
import subprocess
import struct
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Any
import zlib
from PIL import Image, ImageTk

from real_time_translation.audio.capture import AudioCapture
from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.config import Settings, save_settings
from real_time_translation.domain.events import (
    DomainEvent, LatencyRecorded, PhraseCommitted, RecoverableError, SessionStatus, StatusChanged,
    SynthesizedAudio, TranscriptUpdated, TranslationCompleted,
)
from real_time_translation.domain.languages import SPEECH_LANGUAGES, SUPPORTED_LANGUAGES
from real_time_translation.ui.view_state import language_name, session_view_state
from real_time_translation.ui.voice_studio import VoiceStudio
from real_time_translation.tts.base import BUILT_IN_VOICES, SynthesisOptions

logger = logging.getLogger(__name__)
CoordinatorFactory = Callable[[str, str, Callable[[DomainEvent], None], Settings], Any]
VoiceTranscriber = Callable[[Path, str, str, str], str]

LATENCY_STAGES = (
    "Audio backlog", "VAD compute", "Endpoint silence", "Streaming ASR",
    "ASR flush", "Phrase commit", "Phrase queue", "MT inference", "MT total",
    "TTS inference", "TTS total", "Playback enqueue", "Playback queue",
    "Speech → playback", "Playback duration",
)
ASR_BACKENDS = {"Sherpa-ONNX": "sherpa", "Qwen3-ASR": "qwen"}


class DesktopApp:
    """Build and run the desktop window around a direction-specific coordinator factory."""

    def __init__(
        self,
        root: tk.Tk | None = None,
        *,
        coordinator_factory: CoordinatorFactory | None = None,
        model_status: str = "Model setup has not been checked.",
        setup_error: str = "",
        device_label: str = "CPU",
        settings: Settings | None = None,
        synthesizer: Any | None = None,
        voice_transcriber: VoiceTranscriber | None = None,
    ) -> None:
        if root is None:
            _enable_dpi_awareness()
            root = tk.Tk()
        self.root = root
        self.coordinator_factory = coordinator_factory
        self.model_status = model_status
        self.setup_error = setup_error
        self.device_label = device_label
        self.settings = settings or Settings()
        self.synthesizer = synthesizer
        self.voice_transcriber = voice_transcriber
        self.voice_studio: VoiceStudio | None = None
        self.events: queue.Queue[DomainEvent] = queue.Queue(maxsize=200)
        self.coordinator: Any | None = None
        self._worker: threading.Thread | None = None
        self._closing = False
        self.source = tk.StringVar(value=self.settings.source_language)
        self.source_display = tk.StringVar(value=language_name(self.settings.source_language))
        self.target = tk.StringVar(value=language_name(self.settings.target_language))
        self.status_text = tk.StringVar(value="Ready")
        self.status_detail = tk.StringVar(value="Choose a language and start listening.")
        self.input_device = tk.StringVar(value="Default microphone")
        self.output_device = tk.StringVar(value="Default speakers")
        self.endpoint_pause = tk.StringVar(value=f"{self.settings.endpoint_pause_ms} ms")
        self.asr_backend = tk.StringVar(value=_asr_backend_label(self.settings.asr_backend))
        self.asr_context = tk.StringVar(value=self.settings.asr_context)
        self.bubble_opacity = tk.DoubleVar(value=self.settings.bubble_opacity)
        self.bubble_opacity_label = tk.StringVar(value=f"{self.settings.bubble_opacity:.0%}")
        self._phrase_latencies: dict[str, dict[str, float]] = {}
        self._phrase_latency_times: dict[str, dict[str, float]] = {}
        self._phrase_text: dict[str, str] = {}
        self._phrase_started: dict[str, float] = {}
        self._expanded_phrases: set[str] = set()
        self._configuration_expanded = True
        self._configuration_height = 282
        self._conversation_render_after: str | None = None
        self._last_conversation_width = 0
        self._wallpaper_position_after: str | None = None
        self._wallpaper_image_size: tuple[int, int] | None = None
        self._surface_layout_after: str | None = None
        self._glass_resize_after: str | None = None
        self._glass_photo_cache: dict[tuple[int, int, int, int], tk.PhotoImage] = {}
        self.error_text = tk.StringVar(value=setup_error)
        self._input_devices: list[Any] = []
        self._output_devices: list[dict[str, Any]] = []
        self._build()
        self._language_changed()
        self._asr_backend_changed()
        self._refresh_devices(silent=True)
        self._select_saved_device(self.input_device, self._input_devices, self.settings.input_device_id)
        self._select_saved_device(self.output_device, self._output_devices, self.settings.output_device_id)
        self.root.after(80, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        self.root.title("Local Translator")
        self.root.geometry("1440x810")
        self.root.minsize(760, 600)
        self.root.configure(bg="#172b2b")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#e5eceb")
        style.configure("Sub.TLabel", background="#e5eceb", foreground="#344a49", font=("Avenir Next", 10))
        style.configure("CardTitle.TLabel", background="#e5eceb", foreground="#152c2a",
                        font=("Avenir Next", 11, "bold"))
        style.configure("Muted.TLabel", background="#e5eceb", foreground="#344a49",
                        font=("Avenir Next", 9))
        style.configure("Status.TLabel", background="#e5eceb", foreground="#173e35",
                        font=("Avenir Next", 10, "bold"), padding=(8, 4))
        style.configure("Primary.TButton", font=("Avenir Next", 10, "bold"), padding=(16, 9),
                        background="#244c48", foreground="#ffffff", borderwidth=0)
        style.map("Primary.TButton", background=[("disabled", "#849a96"),
                                                   ("pressed", "#173b38"), ("active", "#32635d")])
        style.configure("Secondary.TButton", font=("Avenir Next", 10), padding=(12, 7),
                        background="#dbe5e3", foreground="#203c39", borderwidth=0)
        style.configure("Stop.TButton", font=("Avenir Next", 10), padding=(12, 7),
                        background="#f0e2dd", foreground="#783e36", borderwidth=0)
        style.configure("TButton", font=("Avenir Next", 10), padding=(10, 7), borderwidth=0)
        style.configure("TCombobox", padding=6, fieldbackground="#f5f8f7", background="#f5f8f7")
        style.configure("Bubble.Horizontal.TScale", background="#dfe8e6", troughcolor="#c9d8d4")

        from pathlib import Path
        image_path = Path(__file__).resolve().parents[3] / "public" / "images" / "conversation-background.png"
        with Image.open(image_path) as source_image:
            self._background_source = source_image.convert("RGB")
        self._background_image = ImageTk.PhotoImage(self._background_source, master=self.root)
        self.surface = tk.Canvas(self.root, bd=0, highlightthickness=0, background="#172b2b")
        self.surface.pack(fill="both", expand=True)
        self._background_item = self.surface.create_image(0, 0, image=self._background_image,
                                                          anchor="center", tags="wallpaper")

        self.config_panel = tk.Canvas(self.surface, bd=0, highlightthickness=0,
                                      background="#173331", height=282, takefocus=True)
        self._config_background = self.config_panel.create_image(
            0, 0, image=self._background_image, anchor="center", tags="wallpaper")
        self._config_glass = self.config_panel.create_image(
            1, 1, anchor="nw", tags="glass")
        self._config_glass_image: tk.PhotoImage | None = None
        self.config_header = self.config_panel.create_text(
            0, 17, text="Configuration", anchor="n", fill="#173735",
            font=("Avenir Next", 13, "bold"), tags=("config-header",))
        self.config_panel.tag_bind("config-header", "<Enter>",
                                   lambda _event: self.config_panel.configure(cursor="hand2"))
        self.config_panel.tag_bind("config-header", "<Leave>",
                                   lambda _event: self.config_panel.configure(cursor=""))
        self.config_panel.bind("<Return>", lambda _event: self._toggle_configuration())
        self.config_panel.bind("<space>", lambda _event: self._toggle_configuration())
        self.config_panel.bind("<Button-1>", self._click_configuration_header)
        self._config_content_items: list[int] = []

        def config_label(text: str, x: float, y: float) -> int:
            item = self.config_panel.create_text(
                x, y, text=text, anchor="w", fill="#233e3b", font=("Avenir Next", 10),
                tags=("config-content",))
            self._config_content_items.append(item)
            return item

        self.language_box = ttk.Combobox(
            self.config_panel, textvariable=self.source_display, state="readonly",
            values=tuple(language_name(code) for code in SUPPORTED_LANGUAGES), width=18,
        )
        self.language_box.bind("<<ComboboxSelected>>", self._language_changed)
        self._source_window = self.config_panel.create_window(0, 0, window=self.language_box,
                                                              anchor="nw", tags="config-content")
        config_label("I’m speaking", .03, 48)
        config_label("Output language", .29, 48)
        self.target_box = ttk.Combobox(
            self.config_panel, textvariable=self.target, state="readonly",
            values=tuple(language_name(code) for code in SPEECH_LANGUAGES), width=18,
        )
        self.target_box.bind("<<ComboboxSelected>>", self._target_language_changed)
        self._target_window = self.config_panel.create_window(0, 0, window=self.target_box,
                                                              anchor="nw", tags="config-content")
        config_label("ASR engine", .55, 48)

        self.asr_box = ttk.Combobox(self.config_panel, textvariable=self.asr_backend, state="readonly",
                                    values=tuple(ASR_BACKENDS), width=18)
        self._asr_window = self.config_panel.create_window(0, 0, window=self.asr_box,
                                                           anchor="nw", tags="config-content")
        self.asr_box.bind("<<ComboboxSelected>>", self._asr_backend_changed)
        self.switch_button = ttk.Button(self.config_panel, text="Switch", style="Secondary.TButton",
                                        command=self._switch_language)
        self._switch_window = self.config_panel.create_window(0, 0, window=self.switch_button,
                                                              anchor="nw", tags="config-content")
        self.asr_context_entry = ttk.Entry(self.config_panel, textvariable=self.asr_context)
        self._context_window = self.config_panel.create_window(0, 0, window=self.asr_context_entry,
                                                              anchor="nw", tags="config-content")
        config_label("Vocabulary / context (optional)", .03, 113)

        config_label("Conversation bubble opacity", .03, 150)
        self._opacity_value_item = config_label(self.bubble_opacity_label.get(), .25, 150)
        self.bubble_opacity_scale = ttk.Scale(
            self.config_panel, from_=0.0, to=1.0, variable=self.bubble_opacity,
            command=self._bubble_opacity_changed, style="Bubble.Horizontal.TScale")
        self._opacity_window = self.config_panel.create_window(
            0, 0, window=self.bubble_opacity_scale, anchor="nw", tags="config-content")
        config_label("Audio devices", .03, 197)
        config_label("Microphone", .03, 219)
        config_label("Speakers", .35, 219)
        config_label("End of speech wait", .67, 219)
        self.input_box = ttk.Combobox(self.config_panel, textvariable=self.input_device, state="readonly")
        self.output_box = ttk.Combobox(self.config_panel, textvariable=self.output_device, state="readonly")
        pauses = [
            "200 ms", "300 ms", "400 ms", "500 ms", "600 ms", "800 ms",
            "1000 ms", "1500 ms", "2000 ms", "3000 ms",
        ]
        saved_pause = self.endpoint_pause.get()
        try:
            saved_pause_ms = int(saved_pause.removesuffix(" ms"))
        except ValueError:
            saved_pause_ms = 800
        if not 100 <= saved_pause_ms <= 5000:
            saved_pause = "800 ms"
            self.endpoint_pause.set(saved_pause)
        if saved_pause not in pauses:
            pauses.append(saved_pause)
        self.endpoint_box = ttk.Combobox(self.config_panel, textvariable=self.endpoint_pause, state="readonly",
                                         values=pauses, width=12)
        self._input_window = self.config_panel.create_window(0, 0, window=self.input_box,
                                                             anchor="nw", tags="config-content")
        self._output_window = self.config_panel.create_window(0, 0, window=self.output_box,
                                                              anchor="nw", tags="config-content")
        self._endpoint_window = self.config_panel.create_window(0, 0, window=self.endpoint_box,
                                                                anchor="nw", tags="config-content")
        self.refresh_button = ttk.Button(self.config_panel, text="Refresh", command=self._refresh_devices)
        self._refresh_window = self.config_panel.create_window(0, 0, window=self.refresh_button,
                                                               anchor="nw", tags="config-content")
        self.config_panel.bind("<Configure>", self._layout_configuration)

        self.conversation = tk.Canvas(self.surface, bd=0, highlightthickness=0, background="#173331",
                                      takefocus=True)
        self._conversation_background = self.conversation.create_image(
            0, 0, image=self._background_image, anchor="center", tags="wallpaper")
        self.conversation.bind("<Configure>", self._conversation_resized)
        self._user_scrolling = False
        self.conversation.bind("<MouseWheel>", self._on_mousewheel)
        self.conversation.bind("<Button-4>", lambda event: self._scroll_canvas(-1))
        self.conversation.bind("<Button-5>", lambda event: self._scroll_canvas(1))
        self.conversation.bind("<Up>", lambda event: self._scroll_canvas(-1))
        self.conversation.bind("<Down>", lambda event: self._scroll_canvas(1))
        self.conversation.bind("<Home>", lambda event: self._scroll_conversation_to(0))
        self.conversation.bind("<End>", lambda event: self._scroll_conversation_to(1))
        self.conversation.bind("<Tab>", lambda event: self._move_message_focus(1))
        self.conversation.bind("<Shift-Tab>", lambda event: self._move_message_focus(-1))
        self.conversation.bind("<Return>", lambda event: self._activate_focused_message())
        self.conversation.bind("<space>", lambda event: self._activate_focused_message())

        self.start_button = ttk.Button(self.surface, text="Start listening", style="Primary.TButton",
                                       command=self.start)
        self.stop_button = ttk.Button(self.surface, text="Stop", style="Stop.TButton",
                                      command=self.stop, state="disabled")
        self.voice_button = ttk.Button(self.surface, text="Voice Studio", style="Secondary.TButton",
                                       command=self.open_voice_studio)
        self._start_window = self.surface.create_window(0, 0, window=self.start_button,
                                                        anchor="e", width=140, height=38)
        self._stop_window = self.surface.create_window(0, 0, window=self.stop_button,
                                                       anchor="e", width=72, height=38)
        self._voice_window = self.surface.create_window(0, 0, window=self.voice_button,
                                                        anchor="e", width=112, height=38)
        self._status_item = self.surface.create_text(
            0, 0, text="", anchor="w", fill="#ffffff",
            font=("Avenir Next", 15, "bold"), tags="footer-text")
        self._status_detail_item = self.surface.create_text(
            0, 0, text="", anchor="w", fill="#f1f7f5",
            font=("Avenir Next", 11), tags="footer-text")
        self._error_item = self.surface.create_text(
            0, 0, text="", anchor="w", fill="#fff1ed",
            font=("Avenir Next", 9), tags="footer-text")
        self.error_text.trace_add("write", lambda *_args: self._sync_error())
        self.status_text.trace_add("write", lambda *_args: self._sync_status())
        self.status_detail.trace_add("write", lambda *_args: self._sync_status())

        self._config_window = self.surface.create_window(36, 28, window=self.config_panel, anchor="nw")
        self._conversation_window = self.surface.create_window(36, 0, window=self.conversation, anchor="nw")
        self.clear_log_button = ttk.Button(
            self.surface, text="Clear messages", style="Secondary.TButton",
            command=self._clear_message_log,
        )
        self.clear_log_button.state(["disabled"])
        self._clear_log_window = self.surface.create_window(
            0, 0, window=self.clear_log_button, anchor="se", width=116, height=32,
        )
        self.surface.bind("<Configure>", self._schedule_surface_layout)
        self._message_data: dict[str, dict[str, Any]] = {}
        self._message_layout: list[tuple[str, float, float]] = []
        self._content_height = 0
        self._focused_phrase: str | None = None
        self._target_conversation_height = 48
        self._wrapped_text_cache: dict[tuple[str, int, str], str] = {}
        self._reduced_motion = _prefers_reduced_motion()
        self._translation_font = tkfont.Font(self.root, family="Avenir Next", size=16, weight="bold")
        self._source_font = tkfont.Font(self.root, family="Avenir Next", size=11, slant="italic")
        self.root.after_idle(self._layout_surface)

    def _toggle_configuration(self) -> None:
        self._configuration_expanded = not self._configuration_expanded
        self.config_panel.itemconfigure(
            self.config_header,
            text="Configuration",
        )
        if self._configuration_expanded:
            self.config_panel.itemconfigure("config-content", state="normal")
        else:
            self.config_panel.itemconfigure("config-content", state="hidden")
        self._animate_configuration_height(282 if self._configuration_expanded else 40)

    def _click_configuration_header(self, event: tk.Event) -> str | None:
        if event.y <= 38:
            self._toggle_configuration()
            return "break"
        return None

    def _animate_configuration_height(self, target: int) -> None:
        current = max(40, self.config_panel.winfo_height())
        if self._reduced_motion:
            current = target
        elif current < target:
            current = min(target, current + max(12, (target - current) // 4))
        elif current > target:
            current = max(target, current - max(12, (current - target) // 4))
        self._configuration_height = current
        self.config_panel.configure(height=current)
        self._layout_surface()
        if current != target:
            self.root.after(16, lambda: self._animate_configuration_height(target))

    def _layout_configuration(self, event: tk.Event | None = None) -> None:
        width = event.width if event else self.config_panel.winfo_width()
        height = event.height if event else self.config_panel.winfo_height()
        if width < 2 or height < 2:
            return
        self.config_panel.coords(
            self._config_glass, 1, 1)
        self.config_panel.coords(self.config_header, width / 2, 19)
        if self._glass_resize_after is not None:
            self.root.after_cancel(self._glass_resize_after)
        self._glass_resize_after = self.root.after(120, self._render_config_glass)
        label_positions = {
            "I’m speaking": (.03, 49), "Output language": (.29, 49),
            "ASR engine": (.55, 49),
            "Vocabulary / context (optional)": (.03, 112),
            "Conversation bubble opacity": (.03, 151),
            self.bubble_opacity_label.get(): (.25, 151),
            "Audio devices": (.03, 197), "Microphone": (.03, 219),
            "Speakers": (.35, 219), "End of speech wait": (.67, 219),
        }
        for item in self._config_content_items:
            if item == self._opacity_value_item:
                self.config_panel.coords(item, .25 * width, 151)
            else:
                text = self.config_panel.itemcget(item, "text")
                fraction, y = label_positions[text]
                self.config_panel.coords(item, fraction * width, y)
        col = max(115, width * .23)
        for window, fraction, wid in (
            (self._source_window, .03, col), (self._target_window, .29, col),
            (self._asr_window, .55, col),
            (self._switch_window, .82, max(75, width * .14)),
            (self._context_window, .29, max(130, width * .68)),
            (self._opacity_window, .03, max(135, width * .24)),
            (self._input_window, .03, max(125, width * .29)),
            (self._output_window, .35, max(125, width * .29)),
            (self._endpoint_window, .67, max(125, width * .29)),
            (self._refresh_window, .89, max(76, width * .10)),
        ):
            y = 155 if window == self._refresh_window else (
                170 if window == self._opacity_window else
                241 if window in {self._input_window, self._output_window, self._endpoint_window} else
                126 if window == self._context_window else 66)
            height = (27 if window == self._refresh_window else
                      20 if window == self._opacity_window else
                      29 if window == self._context_window else 30)
            self.config_panel.coords(window, fraction * width, y)
            self.config_panel.itemconfigure(window, width=wid, height=height)

    def _sync_error(self) -> None:
        if hasattr(self, "_error_item"):
            self.surface.itemconfigure(self._error_item, text=self.error_text.get())

    def _sync_status(self) -> None:
        if hasattr(self, "_status_item"):
            label = "" if self.status_text.get() in {"Ready", "Stopped"} else self.status_text.get()
            self.surface.itemconfigure(self._status_item, text=label)
            detail = self.status_detail.get() if label else ""
            self.surface.itemconfigure(self._status_detail_item, text=detail)

    def _layout_surface(self, _event: object = None) -> None:
        width, height = max(1, self.surface.winfo_width()), max(1, self.surface.winfo_height())
        if self._wallpaper_position_after is None:
            self._wallpaper_position_after = self.root.after(120, self._position_wallpaper)
        margin = min(36, max(16, width * 0.04))
        panel_width = max(300, width - margin * 2)
        self.surface.coords(self._config_window, margin, 24)
        self.surface.itemconfigure(self._config_window, width=panel_width)
        config_height = self._configuration_height
        self.config_panel.configure(width=panel_width, height=config_height)
        config_bottom = 24 + config_height
        footer_height = 58
        bottom = height - footer_height - 18
        maximum = max(48, bottom - config_bottom - 16)
        requested = self._content_height if self._message_data else 48
        minimum = 96 if self._message_data else 48
        self._target_conversation_height = min(maximum, max(minimum, requested))
        conversation_height = self._target_conversation_height
        conversation_width = panel_width
        self.conversation.configure(width=conversation_width, height=conversation_height)
        if self._message_data:
            y = max(config_bottom + 16, bottom - conversation_height)
        else:
            empty_center = height * 0.56
            y = max(config_bottom + 16, min(bottom - conversation_height,
                                            empty_center - conversation_height / 2))
        self.surface.coords(self._conversation_window, margin, y)
        self.surface.itemconfigure(self._conversation_window, width=conversation_width, height=conversation_height)
        self.surface.coords(self._clear_log_window, margin + panel_width - 12,
                            y + conversation_height - 10)
        self.surface.coords(self._status_item, margin + 4, height - 39)
        self.surface.coords(self._status_detail_item, margin + 4, height - 16)
        self.surface.coords(self._error_item, margin + 4, height - 62)
        self.surface.itemconfigure(self._error_item, width=max(120, panel_width - 260))
        self.surface.coords(self._stop_window, margin + panel_width - 148, height - 28)
        self.surface.coords(self._voice_window, margin + panel_width - 276, height - 28)
        self.surface.coords(self._start_window, margin + panel_width - 8, height - 28)

    def _schedule_surface_layout(self, _event: object = None) -> None:
        """Limit resize-driven layout to 20 updates per second while tracking the latest size."""
        if self._surface_layout_after is None and not self._closing:
            self._surface_layout_after = self.root.after(50, self._run_scheduled_surface_layout)

    def _run_scheduled_surface_layout(self) -> None:
        self._surface_layout_after = None
        self._layout_surface()

    def _position_wallpaper(self) -> None:
        self._wallpaper_position_after = None
        width, height = self.surface.winfo_width(), self.surface.winfo_height()
        self._resize_wallpaper(width, height)
        self.surface.coords(self._background_item, width / 2, height / 2)
        config_x, config_y = self.surface.coords(self._config_window)
        self.config_panel.coords(self._config_background, width / 2 - config_x,
                                 height / 2 - config_y)
        self._position_conversation_wallpaper()

    def _resize_wallpaper(self, width: int, height: int) -> None:
        """Scale and center-crop the photo so it covers the full application surface."""
        width, height = max(1, width), max(1, height)
        bucket = (((width + 63) // 64) * 64, ((height + 63) // 64) * 64)
        if bucket == self._wallpaper_image_size:
            return

        source_width, source_height = self._background_source.size
        scale = max(bucket[0] / source_width, bucket[1] / source_height)
        resized_size = (
            max(bucket[0], round(source_width * scale)),
            max(bucket[1], round(source_height * scale)),
        )
        resized = self._background_source.resize(resized_size, Image.Resampling.LANCZOS)
        left = (resized.width - bucket[0]) // 2
        top = (resized.height - bucket[1]) // 2
        covered = resized.crop((left, top, left + bucket[0], top + bucket[1]))
        self._background_image = ImageTk.PhotoImage(covered, master=self.root)
        self._wallpaper_image_size = bucket
        for canvas, item in (
            (self.surface, self._background_item),
            (self.config_panel, self._config_background),
            (self.conversation, self._conversation_background),
        ):
            canvas.itemconfigure(item, image=self._background_image)

    def _conversation_resized(self, event: tk.Event) -> None:
        if event.width != self._last_conversation_width:
            self._last_conversation_width = event.width
            self._schedule_message_render()

    def _position_conversation_wallpaper(self) -> None:
        width, height = self.surface.winfo_width(), self.surface.winfo_height()
        x, y = self.surface.coords(self._conversation_window)
        scroll_offset = self.conversation.canvasy(0)
        self.conversation.coords(
            self._conversation_background,
            width / 2 - x,
            height / 2 - y + scroll_offset,
        )

    def _scroll_canvas(self, direction: int) -> None:
        self._user_scrolling = True
        self.conversation.yview_scroll(direction * 3, "units")
        self._check_scroll_position()

    def _scroll_conversation_to(self, fraction: float) -> str:
        self.conversation.yview_moveto(fraction)
        self._check_scroll_position()
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str:
        delta = -1 if event.delta > 0 else 1
        self._scroll_canvas(delta)
        return "break"

    def _check_scroll_position(self) -> None:
        first, last = self.conversation.yview()
        self._user_scrolling = first > 0.005 and last < 0.995
        self._position_conversation_wallpaper()

    def _add_bubble(self, phrase_id: str, original: str, translated: str) -> None:
        self._message_data[phrase_id] = {"original": original, "translated": translated}
        self.clear_log_button.state(["!disabled"])
        self._content_height = self._measure_message_content()
        self._cancel_scheduled_message_render()
        self._render_messages()

    def _clear_message_log(self) -> None:
        """Clear the visible conversation and its cached per-phrase diagnostics."""
        self._message_data.clear()
        self._message_layout.clear()
        self._phrase_latencies.clear()
        self._phrase_latency_times.clear()
        self._phrase_text.clear()
        self._phrase_started.clear()
        self._expanded_phrases.clear()
        self._focused_phrase = None
        self._user_scrolling = False
        self.clear_log_button.state(["disabled"])
        self._content_height = 48
        self._cancel_scheduled_message_render()
        self._render_messages()

    def _toggle_diagnostics(self, phrase_id: str) -> None:
        if phrase_id in self._expanded_phrases:
            self._expanded_phrases.remove(phrase_id)
        else:
            self._expanded_phrases.add(phrase_id)
        self._focused_phrase = phrase_id
        old_bottom = self.conversation.yview()[1] >= 0.995
        self._content_height = self._measure_message_content()
        self._layout_surface()
        self._cancel_scheduled_message_render()
        self._render_messages()
        self.conversation.focus_set()
        if old_bottom and not self._user_scrolling:
            self.conversation.yview_moveto(1.0)

    def _measure_message_content(self) -> int:
        width = max(260, self.conversation.winfo_width() - 64)
        y = 20
        for phrase_id, data in self._message_data.items():
            y += self._bubble_height(phrase_id, data, width) + 12
        return max(48, y + 8)

    def _bubble_height(self, phrase_id: str, data: dict[str, Any], width: int) -> int:
        available = max(160, width - 38)
        primary_lines = len(self._wrap_message_text(
            data["translated"], self._translation_font, available).splitlines())
        secondary_lines = len(self._wrap_message_text(
            data["original"], self._source_font, available).splitlines())
        height = 24 + primary_lines * 24 + secondary_lines * 18 + 18
        if phrase_id in self._expanded_phrases:
            height += 14 + self._diagnostic_line_count(phrase_id) * 17
        return height

    def _wrap_message_text(self, text: str, font: tkfont.Font, width: int) -> str:
        key = (text, width, str(font))
        wrapped = self._wrapped_text_cache.get(key)
        if wrapped is None:
            if len(self._wrapped_text_cache) >= 512:
                self._wrapped_text_cache.clear()
            wrapped = _wrap_visual_text(text, font, width)
            self._wrapped_text_cache[key] = wrapped
        return wrapped

    def _diagnostic_line_count(self, phrase_id: str) -> int:
        metrics = self._phrase_latencies.get(phrase_id, {})
        return max(1, len(metrics))

    def _render_messages(self) -> None:
        self._conversation_render_after = None
        if not hasattr(self, "_message_data"):
            return
        first, last = self.conversation.yview()
        follow_bottom = last >= 0.995 and not self._user_scrolling
        self.conversation.delete("message")
        width = max(260, self.conversation.winfo_width() - 64)
        y = 20
        self._message_layout.clear()
        for phrase_id, data in self._message_data.items():
            height = self._bubble_height(phrase_id, data, width)
            x1, x2 = 22, self.conversation.winfo_width() - 22
            bubble_glass = self._glass_photo(x2 - x1, height, 17)
            self.conversation.create_image(
                (x1 + x2) / 2, y + height / 2, image=bubble_glass, anchor="center",
                tags=("message", f"phrase:{phrase_id}"))
            translated_display = self._wrap_message_text(
                data["translated"], self._translation_font, width - 38)
            original_display = self._wrap_message_text(
                data["original"], self._source_font, width - 38)
            primary_height = len(translated_display.splitlines()) * 24
            self.conversation.create_text(
                x1 + 18, y + 13, anchor="nw",
                text=translated_display,
                width=width - 38,
                fill="#102827", font=("Avenir Next", 16, "bold"), justify="left",
                tags=("message", f"phrase:{phrase_id}"))
            self.conversation.create_text(
                x1 + 18, y + 18 + primary_height, anchor="nw",
                text=original_display,
                width=width - 38, fill="#344a49", font=("Avenir Next", 11, "italic"),
                justify="left", tags=("message", f"phrase:{phrase_id}"))
            if phrase_id in self._expanded_phrases:
                diag_y = y + 30 + primary_height + len(original_display.splitlines()) * 18
                for stage, duration in self._phrase_latencies.get(phrase_id, {}).items():
                    timestamp = self._phrase_latency_times.get(phrase_id, {}).get(stage)
                    stamp_text = f"  {timestamp:.3f} s" if timestamp is not None else ""
                    self.conversation.create_text(
                        x1 + 18, diag_y, anchor="nw", text=f"{stage}{stamp_text}",
                        fill="#526663", font=("Avenir Next", 9), tags=("message", f"phrase:{phrase_id}"))
                    self.conversation.create_text(
                        x2 - 18, diag_y, anchor="ne", text=_format_duration(duration),
                        fill="#203c39", font=("Avenir Next", 9, "bold"), tags=("message", f"phrase:{phrase_id}"))
                    diag_y += 17
                if not self._phrase_latencies.get(phrase_id):
                    self.conversation.create_text(
                        x1 + 18, diag_y, anchor="nw", text="No latency measurements recorded yet",
                        fill="#526663", font=("Avenir Next", 9), tags=("message", f"phrase:{phrase_id}"))
            self._message_layout.append((phrase_id, y, y + height))
            y += height + 12
        self._content_height = max(48, y + 8)
        self.conversation.configure(scrollregion=(0, 0, self.conversation.winfo_width(), self._content_height))
        if follow_bottom:
            self.conversation.yview_moveto(1.0)
        else:
            self.conversation.yview_moveto(min(first, 1.0))
        self._position_conversation_wallpaper()
        for phrase_id, _, _ in self._message_layout:
            tag = f"phrase:{phrase_id}"
            self.conversation.tag_bind(tag, "<Button-1>",
                lambda _event, pid=phrase_id: self._toggle_diagnostics(pid))
            self.conversation.tag_bind(tag, "<Enter>", lambda _event: self.conversation.configure(cursor="hand2"))
            self.conversation.tag_bind(tag, "<Leave>", lambda _event: self.conversation.configure(cursor=""))
        self._layout_surface()

    def _schedule_message_render(self) -> None:
        if self._closing:
            return
        if self._conversation_render_after is not None:
            self.root.after_cancel(self._conversation_render_after)
        self._conversation_render_after = self.root.after(160, self._render_messages)

    def _glass_photo(
        self, width: int, height: int, radius: int, opacity: float | None = None,
    ) -> tk.PhotoImage:
        if opacity is None:
            opacity = self.bubble_opacity.get()
        opacity = max(0.0, min(1.0, opacity))
        key = (width, height, radius, round(opacity * 100))
        photo = self._glass_photo_cache.get(key)
        if photo is None:
            photo_data = _alpha_glass_png(width, height, radius, opacity)
            photo = tk.PhotoImage(data=base64.b64encode(photo_data).decode("ascii"))
            if len(self._glass_photo_cache) >= 64:
                self._glass_photo_cache.pop(next(iter(self._glass_photo_cache)))
            self._glass_photo_cache[key] = photo
        return photo

    def _bubble_opacity_changed(self, value: str) -> None:
        opacity = round(max(0.0, min(1.0, float(value))), 2)
        if self.bubble_opacity.get() != opacity:
            self.bubble_opacity.set(opacity)
        percent = f"{opacity:.0%}"
        self.bubble_opacity_label.set(percent)
        self.config_panel.itemconfigure(self._opacity_value_item, text=percent)
        self._schedule_message_render()

    def _render_config_glass(self) -> None:
        self._glass_resize_after = None
        width, height = self.config_panel.winfo_width(), self.config_panel.winfo_height()
        if width < 3 or height < 3:
            return
        self._config_glass_image = self._glass_photo(width - 2, height - 2, 18, 0.6)
        self.config_panel.coords(self._config_glass, 1, 1)
        self.config_panel.itemconfigure(self._config_glass, image=self._config_glass_image)

    def _cancel_scheduled_message_render(self) -> None:
        if self._conversation_render_after is not None:
            self.root.after_cancel(self._conversation_render_after)
            self._conversation_render_after = None

    def _move_message_focus(self, direction: int) -> str:
        if not self._message_layout:
            return "break"
        phrases = [entry[0] for entry in self._message_layout]
        try:
            index = phrases.index(self._focused_phrase) + direction
        except ValueError:
            index = 0 if direction > 0 else len(phrases) - 1
        index = max(0, min(len(phrases) - 1, index))
        self._focused_phrase = phrases[index]
        self._cancel_scheduled_message_render()
        self._render_messages()
        self.conversation.focus_set()
        return "break"

    def _activate_focused_message(self) -> str:
        if self._focused_phrase is not None:
            self._toggle_diagnostics(self._focused_phrase)
        return "break"


    def _language_changed(self, _event: object = None) -> None:
        self.source.set(next(code for code in SUPPORTED_LANGUAGES
                             if language_name(code) == self.source_display.get()))
        source = self.source.get()
        target_code = next((code for code in SPEECH_LANGUAGES
                            if language_name(code) == self.target.get()), "fr")
        if target_code == source:
            target_code = "en" if source != "en" else "fr"
        self.target.set(language_name(target_code))
        if self.error_text.get() == "Choose a different output language from the spoken language.":
            self.error_text.set("")
        if source not in {"en", "fr"} and ASR_BACKENDS.get(self.asr_backend.get()) != "qwen":
            self.asr_backend.set("Qwen3-ASR")
            self._asr_backend_changed()
        self.switch_button.configure(state="disabled" if source == "tr" else "normal")

    def _target_language_changed(self, _event: object = None) -> None:
        target = next(code for code in SPEECH_LANGUAGES
                      if language_name(code) == self.target.get())
        if target == self.source.get():
            self.error_text.set("Choose a different output language from the spoken language.")
        else:
            self.error_text.set("")

    def _asr_backend_changed(self, _event: object = None) -> None:
        state = "normal" if ASR_BACKENDS.get(self.asr_backend.get()) == "qwen" else "disabled"
        self.asr_context_entry.configure(state=state)

    def _switch_language(self) -> None:
        current_source = self.source.get()
        current_target = next(code for code in SPEECH_LANGUAGES
                              if language_name(code) == self.target.get())
        if current_source not in SPEECH_LANGUAGES:
            self.error_text.set("Turkish speech output is not available to switch into.")
            return
        self.source.set(current_target)
        self.source_display.set(language_name(current_target))
        self.target.set(language_name(current_source))
        self._language_changed()

    def _refresh_devices(self, silent: bool = False) -> None:
        try:
            self._input_devices = AudioCapture.enumerate_devices()
            self._output_devices = AudioPlayback.enumerate_devices()
            self.input_box.configure(values=["Default microphone"] + [d.name for d in self._input_devices])
            self.output_box.configure(values=["Default speakers"] + [d["name"] for d in self._output_devices])
            if self.input_device.get() not in self.input_box.cget("values"):
                self.input_device.set("Default microphone")
            if self.output_device.get() not in self.output_box.cget("values"):
                self.output_device.set("Default speakers")
            if not silent:
                self.error_text.set("")
        except Exception as error:
            self.input_box.configure(values=["Default microphone"])
            self.output_box.configure(values=["Default speakers"])
            self.input_device.set("Default microphone")
            self.output_device.set("Default speakers")
            if not silent:
                self.error_text.set(str(error))

    def _selected_device(self, selected: str, devices: list[Any]) -> int | None:
        if selected.startswith("Default"):
            return None
        for device in devices:
            name = device.name if hasattr(device, "name") else device["name"]
            if name == selected:
                return int(device.id if hasattr(device, "id") else device["id"])
        return None

    @staticmethod
    def _select_saved_device(variable: tk.StringVar, devices: list[Any], device_id: int | None) -> None:
        if device_id is None:
            return
        for device in devices:
            identifier = device.id if hasattr(device, "id") else device["id"]
            name = device.name if hasattr(device, "name") else device["name"]
            if identifier == device_id:
                variable.set(name)
                return

    def _emit_event(self, event: DomainEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            logger.warning("Desktop event queue full; dropping %s", type(event).__name__)

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self.coordinator_factory is None:
            self.error_text.set(self.setup_error or
                "Translation models are not configured yet. Add verified local models to enable listening.")
            self.status_text.set("Setup needed")
            return
        self._set_status(SessionStatus.STARTING, "Preparing local models and audio…")
        self.error_text.set("")
        source_language = self.source.get()
        target_language = next(code for code in SPEECH_LANGUAGES
                               if language_name(code) == self.target.get())
        if source_language == target_language:
            self.error_text.set("Choose a different output language from the spoken language.")
            self._set_status(SessionStatus.ERROR, "Choose different input and output languages")
            return
        input_id = self._selected_device(self.input_device.get(), self._input_devices)
        output_id = self._selected_device(self.output_device.get(), self._output_devices)
        try:
            endpoint_pause_ms = int(self.endpoint_pause.get().removesuffix(" ms"))
            session_settings = replace(
                self.settings,
                source_language=source_language,
                target_language=target_language,
                input_device_id=input_id,
                output_device_id=output_id,
                endpoint_pause_ms=endpoint_pause_ms,
                asr_backend=ASR_BACKENDS[self.asr_backend.get()],
                asr_context=self.asr_context.get(),
            )
        except (TypeError, ValueError) as error:
            self.error_text.set(f"Invalid end-of-speech wait: {error}")
            self._set_status(SessionStatus.ERROR, "Check the session settings")
            return

        def launch() -> None:
            try:
                coordinator = self.coordinator_factory(
                    source_language, target_language, self._emit_event, session_settings
                )
                self.coordinator = coordinator
                coordinator.start()
            except Exception as error:
                self._emit_event(RecoverableError("startup", str(error), 0.0))
                self._emit_event(StatusChanged(SessionStatus.ERROR, "Could not start translation", 0.0))

        self._worker = threading.Thread(target=launch, name="desktop-session-start", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        coordinator = self.coordinator
        if coordinator is None:
            return
        self._set_status(SessionStatus.STOPPING, "Stopping translation…")

        def stop_session() -> None:
            try:
                coordinator.stop()
            except Exception as error:
                self._emit_event(RecoverableError("shutdown", str(error), 0.0))

        self._worker = threading.Thread(target=stop_session, name="desktop-session-stop", daemon=True)
        self._worker.start()

    def open_voice_studio(self) -> None:
        if self.synthesizer is None:
            self.error_text.set(self.setup_error or "Set up the local Qwen3-TTS MLX model files first.")
            return
        if self.voice_studio is not None and not self.voice_studio.closed:
            self.voice_studio.window.lift()
            return

        def save_options(options: SynthesisOptions, profile_id: str | None) -> None:
            self.synthesizer.set_default_options(options)
            voice_id = options.voice_id if options.voice_id in BUILT_IN_VOICES else self.settings.tts_voice_id
            self.settings = replace(
                self.settings,
                tts_voice_id=voice_id,
                tts_profile_id=profile_id,
                tts_temperature=float(options.temperature),
            )

        self.voice_studio = VoiceStudio(
            self.root,
            self.synthesizer,
            self.settings,
            output_device=self._selected_device(self.output_device.get(), self._output_devices),
            input_device=self._selected_device(self.input_device.get(), self._input_devices),
            source_language=self.source.get(),
            asr_backend=ASR_BACKENDS.get(self.asr_backend.get(), "sherpa"),
            asr_context=self.asr_context.get(),
            transcriber=self.voice_transcriber,
            on_options=save_options,
        )

    def _set_status(self, status: SessionStatus, detail: str | None = None) -> None:
        view = session_view_state(status)
        self.status_text.set(view.label)
        self.status_detail.set(detail or view.detail)
        self.start_button.configure(state="disabled" if view.busy or view.running else "normal")
        self.stop_button.configure(state="normal" if view.running else "disabled")
        self.language_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.target_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.input_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.output_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.endpoint_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.asr_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.asr_context_entry.configure(
            state="disabled" if view.running or view.busy or ASR_BACKENDS.get(self.asr_backend.get()) != "qwen" else "normal"
        )
        switch_unavailable = view.running or view.busy or self.source.get() == "tr"
        self.switch_button.configure(state="disabled" if switch_unavailable else "normal")

    def _drain_events(self) -> None:
        if self._closing:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, StatusChanged):
                self._set_status(event.status, event.message)
            elif isinstance(event, TranscriptUpdated):
                pass
            elif isinstance(event, PhraseCommitted):
                phrase_id = event.phrase.phrase_id
                self._phrase_latencies[phrase_id] = {}
                self._phrase_latency_times[phrase_id] = {}
                self._phrase_text[phrase_id] = event.phrase.text
                self._phrase_started[phrase_id] = event.phrase.started_at
                if self._configuration_expanded:
                    self._toggle_configuration()
            elif isinstance(event, LatencyRecorded):
                if event.stage not in LATENCY_STAGES:
                    continue
                phrase_metrics = self._phrase_latencies.setdefault(event.phrase_id, {})
                phrase_metrics[event.stage] = event.duration_ms
                self._phrase_latency_times.setdefault(event.phrase_id, {})[event.stage] = event.recorded_at
                if event.phrase_id in self._expanded_phrases:
                    self._schedule_message_render()
            elif isinstance(event, TranslationCompleted):
                started = self._phrase_started.get(event.phrase_id)
                if started is not None:
                    self._phrase_latencies.setdefault(event.phrase_id, {})[
                        "Speech to translation"] = max(0.0, (event.completed_at - started) * 1000.0)
                self._add_bubble(event.phrase_id, self._phrase_text.get(event.phrase_id, ""), event.text)
            elif isinstance(event, SynthesizedAudio):
                self.status_detail.set(f"Speaking in {language_name(event.target_language)}")
            elif isinstance(event, RecoverableError):
                self.error_text.set(event.message)
                logger.warning("%s: %s", event.component, event.message)
        self.root.after(80, self._drain_events)

    def close(self) -> None:
        self._closing = True
        try:
            save_settings(replace(
                self.settings,
                source_language=self.source.get(),
                target_language=next(code for code in SPEECH_LANGUAGES
                                     if language_name(code) == self.target.get()),
                input_device_id=self._selected_device(self.input_device.get(), self._input_devices),
                output_device_id=self._selected_device(self.output_device.get(), self._output_devices),
                endpoint_pause_ms=int(self.endpoint_pause.get().removesuffix(" ms")),
                asr_backend=ASR_BACKENDS.get(self.asr_backend.get(), "sherpa"),
                asr_context=self.asr_context.get(),
                bubble_opacity=self.bubble_opacity.get(),
                tts_voice_id=self.settings.tts_voice_id,
            ))
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("Could not save user settings: %s", error)
        if self.coordinator is not None:
            def shutdown() -> None:
                try:
                    self.coordinator.stop()
                finally:
                    if self.synthesizer is not None:
                        self.synthesizer.unload()
            self._worker = threading.Thread(target=shutdown, name="desktop-session-close", daemon=True)
            self._worker.start()
        elif self.synthesizer is not None:
            threading.Thread(target=self.synthesizer.unload, name="tts-close", daemon=True).start()
        if self.voice_studio is not None:
            self.voice_studio.close()
        self.root.destroy()

    def run(self) -> None:
        """Enter Tk's event loop."""
        self.root.mainloop()


def run_desktop(**kwargs: Any) -> None:
    """Create and run the desktop translation window."""
    DesktopApp(**kwargs).run()


def _asr_backend_label(value: str) -> str:
    return next((label for label, backend in ASR_BACKENDS.items() if backend == value), "Sherpa-ONNX")


def _enable_dpi_awareness() -> None:
    """Ask Windows to scale the window and text crisply on high-DPI displays."""
    if not hasattr(ctypes, "windll"):
        return
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        logger.debug("Windows DPI awareness could not be enabled", exc_info=True)


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000.0:
        return f"{duration_ms / 1000.0:.2f} s"
    if duration_ms < 10.0:
        return f"{duration_ms:.1f} ms"
    return f"{duration_ms:.0f} ms"


def _wrap_visual_text(text: str, font: tkfont.Font, width: int) -> str:
    """Wrap transcript text using Tk font metrics so bubbles size to visible content."""
    width = max(1, width)
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            if font.measure(word) > width:
                if line:
                    wrapped.append(line)
                    line = ""
                fragment = ""
                for char in word:
                    if fragment and font.measure(fragment + char) > width:
                        wrapped.append(fragment)
                        fragment = char
                    else:
                        fragment += char
                line = fragment
                continue
            candidate = word if not line else f"{line} {word}"
            if line and font.measure(candidate) > width:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        wrapped.append(line)
    return "\n".join(wrapped or [""])


def _rounded_rect_points(x1: float, y1: float, x2: float, y2: float,
                        radius: float) -> tuple[float, ...]:
    """Return control points for a softly rounded canvas polygon."""
    r = min(radius, max(1.0, (x2 - x1) / 2), max(1.0, (y2 - y1) / 2))
    return (
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    )


def _alpha_glass_png(width: int, height: int, radius: int, opacity: float = 0.6) -> bytes:
    """Create a rounded, pale glass overlay with true PNG alpha and no blur pass."""
    width, height = max(1, width), max(1, height)
    radius = min(max(0, radius), width // 2, height // 2)
    alpha = round(255 * max(0.0, min(1.0, opacity)))
    pixel = bytes((234, 243, 240, alpha))
    raw = bytearray()
    for y in range(height):
        row = bytearray(pixel * width)
        if radius and (y < radius or y >= height - radius):
            dy = y - radius if y < radius else y - (height - radius)
            for x in range(radius):
                if (x - radius) ** 2 + dy ** 2 > radius ** 2:
                    row[x * 4 + 3] = 0
                    row[(width - 1 - x) * 4 + 3] = 0
        raw.append(0)
        raw.extend(row)
    image_header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", image_header) +
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=1)) + _png_chunk(b"IEND", b""))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def _prefers_reduced_motion() -> bool:
    explicit = os.environ.get("REAL_TIME_TRANSLATION_REDUCE_MOTION", "").lower()
    if explicit in {"1", "true", "yes"}:
        return True
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "com.apple.reduceMotion"],
                capture_output=True, text=True, timeout=.5, check=False)
            return result.stdout.strip() == "1"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False
