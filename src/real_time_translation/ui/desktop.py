"""Tk desktop interface for local French/English speech translation.

The UI owns presentation only. A coordinator factory supplied by the application creates the
direction-specific pipeline; domain events cross onto the Tk thread through a bounded queue.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from real_time_translation.audio.capture import AudioCapture
from real_time_translation.audio.playback import AudioPlayback
from real_time_translation.config import Settings, save_settings
from real_time_translation.domain.events import (
    DomainEvent, PhraseCommitted, RecoverableError, SessionStatus, StatusChanged,
    SynthesizedAudio, TranscriptUpdated, TranslationCompleted,
)
from real_time_translation.domain.languages import target_for
from real_time_translation.ui.view_state import language_name, session_view_state

logger = logging.getLogger(__name__)
CoordinatorFactory = Callable[[str, Callable[[DomainEvent], None]], Any]


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
    ) -> None:
        self.root = root or tk.Tk()
        self.coordinator_factory = coordinator_factory
        self.model_status = model_status
        self.device_label = device_label
        self.settings = settings or Settings()
        self.events: queue.Queue[DomainEvent] = queue.Queue(maxsize=200)
        self.coordinator: Any | None = None
        self._worker: threading.Thread | None = None
        self._closing = False
        self.source = tk.StringVar(value=self.settings.source_language)
        self.status_text = tk.StringVar(value="Ready")
        self.status_detail = tk.StringVar(value="Choose a language and start listening.")
        self.input_device = tk.StringVar(value="Default microphone")
        self.output_device = tk.StringVar(value="Default speakers")
        self.error_text = tk.StringVar(value=setup_error)
        self._input_devices: list[Any] = []
        self._output_devices: list[dict[str, Any]] = []
        self._build()
        self._refresh_devices(silent=True)
        self._select_saved_device(self.input_device, self._input_devices, self.settings.input_device_id)
        self._select_saved_device(self.output_device, self._output_devices, self.settings.output_device_id)
        self.root.after(80, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        self.root.title("Local Translator")
        self.root.geometry("900x720")
        self.root.minsize(680, 600)
        self.root.configure(bg="#f4f6f3")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f4f6f3")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Eyebrow.TLabel", background="#f4f6f3", foreground="#64736b",
                        font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", background="#f4f6f3", foreground="#14251d",
                        font=("Segoe UI", 25, "bold"))
        style.configure("Sub.TLabel", background="#f4f6f3", foreground="#64736b",
                        font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#1b2c24",
                        font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#718078",
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#e3f2e9", foreground="#1f6a43",
                        font=("Segoe UI", 10, "bold"), padding=(11, 6))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 10))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 7))
        style.configure("TCombobox", padding=5)

        shell = ttk.Frame(self.root, style="App.TFrame", padding=(34, 25, 34, 22))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell, style="App.TFrame")
        header.pack(fill="x", pady=(0, 20))
        ttk.Label(header, text="LOCAL · PRIVATE · LIVE", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(header, text="Say it. Understand it.", style="Title.TLabel").pack(anchor="w", pady=(5, 3))
        ttk.Label(header, text="Real-time French ↔ English translation, processed on this device.",
                  style="Sub.TLabel").pack(anchor="w")

        setup = ttk.Frame(shell, style="Card.TFrame", padding=18)
        setup.pack(fill="x", pady=(0, 14))
        setup.columnconfigure(0, weight=1)
        setup.columnconfigure(1, weight=1)
        setup.columnconfigure(2, weight=1)
        ttk.Label(setup, text="I’m speaking", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.language_box = ttk.Combobox(setup, textvariable=self.source, state="readonly",
                                         values=("en", "fr"), width=18)
        self.language_box.grid(row=1, column=0, sticky="ew", padx=(0, 18), pady=(6, 0))
        self.language_box.bind("<<ComboboxSelected>>", self._language_changed)
        self.direction = ttk.Label(setup, text="English  →  French", style="CardTitle.TLabel")
        self.direction.grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))
        ttk.Label(setup, text="OUTPUT LANGUAGE", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Button(setup, text="⇄  Switch", command=self._switch_language).grid(
            row=1, column=2, sticky="e", pady=(6, 0))

        devices = ttk.Frame(shell, style="Card.TFrame", padding=18)
        devices.pack(fill="x", pady=(0, 14))
        devices.columnconfigure(0, weight=1)
        devices.columnconfigure(1, weight=1)
        ttk.Label(devices, text="AUDIO DEVICES", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 7))
        ttk.Button(devices, text="Refresh", command=self._refresh_devices).grid(
            row=0, column=1, sticky="e", pady=(0, 7))
        ttk.Label(devices, text="Microphone", style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(devices, text="Speakers", style="Muted.TLabel").grid(row=1, column=1, sticky="w")
        self.input_box = ttk.Combobox(devices, textvariable=self.input_device, state="readonly")
        self.output_box = ttk.Combobox(devices, textvariable=self.output_device, state="readonly")
        self.input_box.grid(row=2, column=0, sticky="ew", padx=(0, 12), pady=(5, 0))
        self.output_box.grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=(5, 0))

        transcript_row = ttk.Frame(shell, style="App.TFrame")
        transcript_row.pack(fill="both", expand=True, pady=(0, 14))
        transcript_row.columnconfigure(0, weight=1)
        transcript_row.columnconfigure(1, weight=1)
        transcript_row.rowconfigure(0, weight=1)
        self.source_card = self._text_card(transcript_row, 0, "YOU SAID", "Waiting for speech…")
        self.target_card = self._text_card(transcript_row, 1, "TRANSLATION", "Your translation will appear here.")

        footer = ttk.Frame(shell, style="App.TFrame")
        footer.pack(fill="x")
        state = ttk.Frame(footer, style="App.TFrame")
        state.pack(side="left", fill="x", expand=True)
        ttk.Label(state, textvariable=self.status_text, style="Status.TLabel").pack(anchor="w")
        ttk.Label(state, textvariable=self.status_detail, style="Sub.TLabel").pack(anchor="w", pady=(5, 0))
        self.start_button = ttk.Button(footer, text="Start listening", style="Primary.TButton",
                                       command=self.start)
        self.start_button.pack(side="right", padx=(8, 0))
        self.stop_button = ttk.Button(footer, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")

        meta = ttk.Frame(shell, style="App.TFrame")
        meta.pack(fill="x", pady=(16, 0))
        ttk.Label(meta, text=f"DEVICE  {self.device_label}", style="Eyebrow.TLabel").pack(side="left")
        ttk.Label(meta, text=self.model_status, style="Eyebrow.TLabel").pack(side="right")
        ttk.Label(shell, textvariable=self.error_text, style="Sub.TLabel", wraplength=820).pack(
            anchor="w", pady=(8, 0))

    def _text_card(self, parent: ttk.Frame, column: int, title: str, placeholder: str) -> tk.Text:
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=title, style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 9))
        text = tk.Text(card, height=7, wrap="word", relief="flat", borderwidth=0,
                       background="#ffffff", foreground="#26372e", font=("Segoe UI", 15),
                       padx=0, pady=2, state="disabled", cursor="arrow")
        text.grid(row=1, column=0, sticky="nsew")
        text.placeholder = placeholder  # type: ignore[attr-defined]
        self._set_text(text, placeholder)
        return text

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _language_changed(self, _event: object = None) -> None:
        source = self.source.get()
        self.direction.configure(text=f"{language_name(source)}  →  {language_name(target_for(source))}")

    def _switch_language(self) -> None:
        self.source.set("fr" if self.source.get() == "en" else "en")
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
            self.error_text.set("Translation models are not configured yet. Add verified local models to enable listening.")
            self.status_text.set("Setup needed")
            return
        self._set_status(SessionStatus.STARTING, "Preparing local models and audio…")
        self.error_text.set("")
        source_language = self.source.get()
        input_id = self._selected_device(self.input_device.get(), self._input_devices)
        output_id = self._selected_device(self.output_device.get(), self._output_devices)

        def launch() -> None:
            try:
                coordinator = self.coordinator_factory(source_language, self._emit_event)
                self.coordinator = coordinator
                coordinator.capture.device = input_id
                coordinator.playback.device = output_id
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

    def _set_status(self, status: SessionStatus, detail: str | None = None) -> None:
        view = session_view_state(status)
        self.status_text.set(view.label)
        self.status_detail.set(detail or view.detail)
        self.start_button.configure(state="disabled" if view.busy or view.running else "normal")
        self.stop_button.configure(state="normal" if view.running else "disabled")
        self.language_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.input_box.configure(state="disabled" if view.running or view.busy else "readonly")
        self.output_box.configure(state="disabled" if view.running or view.busy else "readonly")

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
                self._set_text(self.source_card, event.segment.text or "Listening…")
            elif isinstance(event, PhraseCommitted):
                self._set_text(self.source_card, event.phrase.text)
                self._set_text(self.target_card, "Translating…")
            elif isinstance(event, TranslationCompleted):
                self._set_text(self.target_card, event.text)
            elif isinstance(event, SynthesizedAudio):
                self.status_detail.set(f"Speaking in {language_name(event.target_language)}")
            elif isinstance(event, RecoverableError):
                self.error_text.set(event.message)
                logger.warning("%s: %s", event.component, event.message)
        self.root.after(80, self._drain_events)

    def close(self) -> None:
        self._closing = True
        try:
            save_settings(Settings(
                source_language=self.source.get(),
                input_device_id=self._selected_device(self.input_device.get(), self._input_devices),
                output_device_id=self._selected_device(self.output_device.get(), self._output_devices),
                endpoint_pause_ms=self.settings.endpoint_pause_ms,
                cpu_threads=self.settings.cpu_threads,
                model_profile=self.settings.model_profile,
                diagnostic_capture=self.settings.diagnostic_capture,
            ))
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("Could not save user settings: %s", error)
        if self.coordinator is not None:
            self._worker = threading.Thread(target=self.coordinator.stop, name="desktop-session-close", daemon=True)
            self._worker.start()
        self.root.destroy()

    def run(self) -> None:
        """Enter Tk's event loop."""
        self.root.mainloop()


def run_desktop(**kwargs: Any) -> None:
    """Create and run the desktop translation window."""
    DesktopApp(**kwargs).run()
