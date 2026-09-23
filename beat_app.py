"""A lightweight desktop front end for Vocal Beat Builder."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
APP_NAME = "Vocal Beat Builder"
AUDIO_TYPES = [("Audio files", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.aif *.aiff *.opus *.wma"),
               ("All files", "*.*")]


def _number(text, label, low, high, whole=False):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"Enter a number for {label}.") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}.")
    if whole and not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(value) if whole else value


def _summary_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{str(key).replace('_', ' ').capitalize()}: {item}"
                         for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value) if value is not None else "Your vocal, drum MIDI, and preview are ready."


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("750x700")
        self.minsize(700, 650)
        self.configure(bg="#edf1f5")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(APP_DIR / "Exports"))
        self.input_type_var = tk.StringVar(value="Full song")
        self.target_bpm_var = tk.StringVar(value="140")
        self.source_bpm_var = tk.StringVar()
        self.style_var = tk.StringVar(value="Half-time")
        self.sensitivity_var = tk.StringVar(value="60")
        self.min_slice_ms_var = tk.StringVar(value="80")
        self.section_bars_var = tk.StringVar(value="8")
        self.start_bar_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="Choose a song, then click Go.")
        self._busy = False
        self._close_when_done = False
        self._cancel_event = None
        self._queue = queue.Queue()
        self._controls = []
        self._last_result = None
        self._active_job = "go"
        self._last_error_path = None
        self._style()
        self._layout()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.option_add("*Font", "{Segoe UI} 10")
        style.configure("TFrame", background="#edf1f5")
        style.configure("TLabel", background="#edf1f5", foreground="#253448")
        style.configure("Sub.TLabel", foreground="#566476")
        style.configure("TLabelframe", background="#edf1f5")
        style.configure("TLabelframe.Label", background="#edf1f5", foreground="#253448",
                        font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", background="#245dcc", foreground="white",
                        font=("Segoe UI", 11, "bold"), padding=(22, 8))
        style.map("Accent.TButton", background=[("active", "#174bb1"), ("disabled", "#98a9c5")])

    def _control(self, widget, normal_state="normal"):
        self._controls.append((widget, normal_state))
        return widget

    def _layout(self):
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Vocal Beat Builder", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(outer, text="A tight vocal, changing drum patterns, and MIDI ready for FL Studio.",
                  style="Sub.TLabel").pack(anchor="w", pady=(3, 10))

        files = ttk.Frame(outer)
        files.pack(fill="x")
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="Song").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self._control(ttk.Entry(files, textvariable=self.input_var)).grid(row=0, column=1, sticky="ew")
        self._control(ttk.Button(files, text="Browse…", command=self._browse_input)).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(files, text="Input").grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.input_type_combo = self._control(ttk.Combobox(files, textvariable=self.input_type_var,
                                                         values=("Full song", "Already isolated vocal"),
                                                         state="readonly", width=26), "readonly")
        self.input_type_combo.grid(row=1, column=1, sticky="w", pady=(7, 0))
        ttk.Label(files, text="Save to").grid(row=2, column=0, sticky="w", pady=(7, 0))
        self._control(ttk.Entry(files, textvariable=self.output_var)).grid(row=2, column=1, sticky="ew", pady=(7, 0))
        self._control(ttk.Button(files, text="Browse…", command=self._browse_output)).grid(
            row=2, column=2, padx=(8, 0), pady=(7, 0))

        settings = ttk.LabelFrame(outer, text="Timing and beat", padding=8)
        settings.pack(fill="x", pady=(12, 9))
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._field(settings, 0, 0, "Target BPM", self.target_bpm_var, 40, 240)
        self._field(settings, 0, 2, "Source BPM", self.source_bpm_var, 30, 300)
        self._field(settings, 1, 0, "Min. cut (ms)", self.min_slice_ms_var, 40, 500)
        ttk.Label(settings, text="Leave source BPM blank to estimate it.", style="Sub.TLabel").grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(2, 6))
        ttk.Label(settings, text="Beat style").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self._control(ttk.Combobox(settings, textvariable=self.style_var, values=("Half-time", "Straight"),
                                   state="readonly", width=13), "readonly").grid(row=2, column=1, sticky="ew", padx=(0, 18))
        self._field(settings, 2, 2, "Sensitivity %", self.sensitivity_var, 0, 100)
        ttk.Label(settings, text="Section cap (bars)").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        self._control(ttk.Combobox(settings, textvariable=self.section_bars_var, values=(4, 8, 16),
                                   state="readonly", width=9), "readonly").grid(
            row=3, column=1, sticky="ew", padx=(0, 18), pady=(5, 0))
        self._field(settings, 3, 2, "First FL bar", self.start_bar_var, 2, 32, increment=2)
        ttk.Label(settings, text="Vocal cuts follow drum hits; sections can end before the bar cap.",
                  style="Sub.TLabel", wraplength=660).grid(row=4, column=0, columnspan=4, sticky="w", pady=(7, 0))

        ttk.Label(outer, text="First use downloads local audio models. Audio stays on your PC.\n"
                             "Section and instrument labels are estimates. The export includes an FL Studio import guide.",
                  style="Sub.TLabel", wraplength=700).pack(anchor="w", pady=(0, 10))
        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        self.go_button = ttk.Button(actions, text="Go", style="Accent.TButton", command=self._go)
        self.go_button.pack(side="right")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(0, 8))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=190)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 18))
        ttk.Label(outer, textvariable=self.status_var, wraplength=690, style="Sub.TLabel").pack(
            anchor="w", fill="x", pady=(7, 9))

        # Pack the result actions first so they remain reachable in a small window.
        result_actions = ttk.Frame(outer)
        result_actions.pack(side="bottom", fill="x", pady=(6, 0))
        self.open_button = ttk.Button(result_actions, text="Open output folder", command=self._open_output, state="disabled")
        self.open_button.pack(side="left")
        self.preview_button = ttk.Button(result_actions, text="Play preview", command=self._play_preview, state="disabled")
        self.preview_button.pack(side="left", padx=8)
        self.recheck_button = self._control(ttk.Button(result_actions, text="Recheck instruments…",
                                                       command=self._recheck_instruments))
        self.recheck_button.pack(side="left")
        self.error_button = ttk.Button(result_actions, text="Error details", command=self._open_error, state="disabled")
        self.error_button.pack(side="right")
        results = ttk.LabelFrame(outer, text="Your export", padding=8)
        results.pack(fill="both", expand=True)
        self.result_text = tk.Text(results, height=4, wrap="word", relief="flat", bg="white", fg="#253448",
                                   padx=8, pady=6, state="disabled")
        scroll = ttk.Scrollbar(results, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._set_result_text("The export will include your aligned vocal, an audio preview, and a combined drum MIDI.\n\n"
                              "The five drum lanes use MIDI notes 60–64 (C5–E5 in FL Studio). Instrument names come "
                              "with one-bar starter-note MIDI files for importing into FL Studio.")

    def _field(self, parent, row, column, label, variable, low, high, increment=1):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 10), pady=(5, 0))
        self._control(ttk.Spinbox(parent, from_=low, to=high, increment=increment,
                                  textvariable=variable, width=9)).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 18) if column == 0 else 0, pady=(5, 0))

    def _browse_input(self):
        if self._busy:
            return
        current = Path(self.input_var.get()).expanduser() if self.input_var.get().strip() else None
        path = filedialog.askopenfilename(parent=self, title="Choose a full song or isolated vocal",
                                          filetypes=AUDIO_TYPES,
                                          initialdir=str(current.parent) if current and current.parent.is_dir() else None)
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        if self._busy:
            return
        current = Path(self.output_var.get()).expanduser()
        path = filedialog.askdirectory(parent=self, title="Choose where to save the exports",
                                       initialdir=str(current) if current.is_dir() else str(APP_DIR))
        if path:
            self.output_var.set(path)

    def _settings(self):
        raw_input = self.input_var.get().strip()
        if not raw_input:
            raise ValueError("Choose a song or isolated vocal first.")
        source = Path(raw_input).expanduser().resolve()
        if not source.is_file():
            raise ValueError("That audio file could not be found. Choose it again with Browse.")
        raw_output = self.output_var.get().strip()
        if not raw_output:
            raise ValueError("Choose a folder for the exports.")
        output = Path(raw_output).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise ValueError("The export location is a file. Choose a folder instead.")
        source_bpm = self.source_bpm_var.get().strip()
        start_bar = _number(self.start_bar_var.get(), "First FL bar", 2, 32, whole=True)
        if start_bar % 2:
            raise ValueError("Choose an even first FL bar, such as 2, 4, or 8.")
        section_bars = _number(self.section_bars_var.get(), "Section cap", 4, 16, whole=True)
        if section_bars not in (4, 8, 16):
            raise ValueError("Choose a section cap of 4, 8, or 16 bars.")
        style = self.style_var.get()
        if style not in ("Half-time", "Straight"):
            raise ValueError("Choose Half-time or Straight for the beat style.")
        input_type = self.input_type_var.get()
        if input_type not in ("Full song", "Already isolated vocal"):
            raise ValueError("Choose whether the input is a full song or an isolated vocal.")
        config = {"target_bpm": _number(self.target_bpm_var.get(), "Target BPM", 40, 240),
                  "source_bpm": _number(source_bpm, "Source BPM", 30, 300) if source_bpm else None,
                  "sensitivity": _number(self.sensitivity_var.get(), "Sensitivity", 0, 100) / 100,
                  "min_slice_ms": _number(self.min_slice_ms_var.get(), "Minimum cut", 40, 500),
                  "section_bars": section_bars, "start_bar": start_bar,
                  "style": style, "input_is_vocal": input_type == "Already isolated vocal", "seed": 0}
        return str(source), str(output), config

    def _set_busy(self, busy):
        self._busy = busy
        for widget, normal_state in self._controls:
            widget.configure(state="disabled" if busy else normal_state)
        self.go_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled", text="Cancel")
        self.open_button.configure(state="normal" if self._last_result and not busy else "disabled")
        self.preview_button.configure(state="normal" if self._last_result and self._last_result.get("preview_path") and not busy else "disabled")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _go(self):
        if self._busy:
            return
        try:
            input_path, output_parent, settings = self._settings()
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error), parent=self)
            return
        def job(progress, cancel_event):
            from beat_engine import BeatConfig, process_song
            return process_song(input_path, output_parent, BeatConfig(**settings),
                                progress=progress, cancel_event=cancel_event)

        message = ("Starting vocal analysis…" if settings["input_is_vocal"] else
                   "Starting… The first full-song run may need to download the separation models.")
        self._start_job(job, message, "go")

    def _recheck_instruments(self):
        if self._busy:
            return
        initial = (self._last_result or {}).get("output_dir") or str(APP_DIR / "Exports")
        folder = filedialog.askdirectory(parent=self, title="Choose an existing Vocal Beat Builder export",
                                         initialdir=initial)
        if not folder:
            return
        export_dir = Path(folder).expanduser().resolve()
        if not export_dir.is_dir():
            messagebox.showerror(APP_NAME, "That export folder could not be found. Choose an existing export folder.", parent=self)
            return

        def job(progress, cancel_event):
            from beat_engine import refresh_instruments
            return refresh_instruments(str(export_dir), progress=progress, cancel_event=cancel_event)

        self._start_job(job, "Rechecking the instruments in this export…", "recheck")

    def _start_job(self, job, message, job_type):
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event
        self._active_job = job_type
        self._set_busy(True)
        self.status_var.set(message)

        def worker():
            try:
                result = job(lambda message: self._queue.put(("progress", str(message), None)), cancel_event)
                self._queue.put(("done", result, None))
            except Exception as error:
                self._queue.put(("cancelled" if cancel_event.is_set() else "error", str(error), traceback.format_exc()))

        try:
            threading.Thread(target=worker, daemon=True).start()
            self.after(100, self._poll)
        except Exception as error:
            self._set_busy(False)
            self._show_error(str(error), traceback.format_exc())

    def _poll(self):
        while True:
            try:
                kind, value, detail = self._queue.get_nowait()
            except queue.Empty:
                self.after(100, self._poll)
                return
            if kind != "progress":
                break
            if not self._cancel_event.is_set():
                self.status_var.set(value)
        self._set_busy(False)
        if kind == "done":
            self._last_result = value
            self._set_busy(False)
            summary = _summary_text(value.get("summary"))
            warnings = value.get("warnings") or []
            text = summary + "\n\nSaved to: " + str(value.get("output_dir", ""))
            if "instrument_names" in value:
                names = value.get("instrument_names") or []
                if isinstance(names, str):
                    names = [names]
                instruments = "\n".join(f"• {name}" for name in names) if names else "No instrument labels available."
                text = "Detected instruments (estimates):\n" + instruments + "\n\n" + text
            if warnings:
                text += "\n\nNotes to review:\n" + "\n".join(str(item) for item in warnings)
            self._set_result_text(text)
            self.status_var.set("Instrument labels updated. Open the output folder to use the revised files." if self._active_job == "recheck"
                                else "Ready. Play the preview or open the output folder to use the files in FL Studio.")
        elif kind == "cancelled":
            self.status_var.set("Stopped. You can recheck the instruments again when you're ready." if self._active_job == "recheck"
                                else "Stopped. You can choose new settings and click Go again.")
        else:
            self._show_error(value, detail)
        if self._close_when_done:
            self.destroy()

    def _cancel(self):
        if not self._busy or self._cancel_event is None:
            return
        self._cancel_event.set()
        self.cancel_button.configure(state="disabled", text="Stopping…")
        self.status_var.set("Stopping the instrument check…" if self._active_job == "recheck" else
                            "Stopping safely… This may take a moment if separation is finishing a step.")

    def _set_result_text(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _write_error(self, detail):
        folders = [APP_DIR, Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Vocal Beat Builder"]
        for folder in folders:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "Last error.txt"
                path.write_text(f"{APP_NAME}\n\n{detail}", encoding="utf-8")
                return path
            except OSError:
                continue
        return None

    def _show_error(self, message, detail):
        self._last_error_path = self._write_error(detail or message)
        self.error_button.configure(state="normal" if self._last_error_path else "disabled")
        self.status_var.set("The run could not finish. Check the error message, then try again.")
        log_note = "\n\nUse Error details to open the full report." if self._last_error_path else ""
        if not self._close_when_done:
            messagebox.showerror(APP_NAME, f"The run could not finish.\n\n{message}{log_note}", parent=self)

    def _open_path(self, path, label):
        try:
            if not path or not Path(path).exists():
                raise ValueError(f"The {label} could not be found. It may have been moved or deleted.")
            os.startfile(str(path))
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error), parent=self)

    def _open_output(self):
        if not self._busy and self._last_result:
            self._open_path(self._last_result.get("output_dir"), "output folder")

    def _play_preview(self):
        if not self._busy and self._last_result:
            self._open_path(self._last_result.get("preview_path"), "preview")

    def _open_error(self):
        if self._last_error_path:
            self._open_path(self._last_error_path, "error report")

    def _close(self):
        if self._busy:
            self._close_when_done = True
            self._cancel()
            return
        self.destroy()


def main():
    try:
        app = App()
        app.mainloop()
    except Exception:
        detail = traceback.format_exc()
        try:
            (APP_DIR / "Last error.txt").write_text(detail, encoding="utf-8")
        except OSError:
            pass
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "Vocal Beat Builder could not start.\n\n"
                                             "Open Last error.txt in the app folder, or try the .bat launcher.",
                                             APP_NAME, 0x10)
        raise


if __name__ == "__main__":
    main()
