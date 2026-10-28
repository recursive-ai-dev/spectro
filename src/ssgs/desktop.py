"""Spectro's desktop audio workbench.

Tk owns the interface; numerical jobs run on a daemon thread and communicate
through a queue. Importing this module never starts a window.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from datetime import datetime
from typing import Callable

import numpy as np
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal

from . import __version__
from . import workflows

BG = "#101820"
PANEL = "#18242f"
FIELD = "#223340"
TEXT = "#e8f0f4"
MUTED = "#a0b3c2"
TEAL = "#60dfca"
AMBER = "#f0c27b"


class _QueueHandler(logging.Handler):
    def __init__(self, events: queue.Queue):
        super().__init__()
        self.events = events
        self.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))

    def emit(self, record):
        self.events.put(("log", self.format(record)))


class SpectroApp:
    """A reusable window controller, also usable in display smoke tests."""

    _logging_clients = 0
    _original_log_level = logging.NOTSET

    def __init__(self, root: tk.Tk):
        self.root = root
        self.model = None
        self.model_path = None
        self.output_path = None
        self.inputs: list[Path] = []
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self._closed = False
        self._buttons: list[ttk.Button] = []
        self._config_path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "spectro" / "preferences.json"
        self._preferences = self._load_preferences()
        self.root.title("Spectro — Spectral audio studio")
        width = min(1220, max(760, self.root.winfo_screenwidth() - 48))
        height = min(850, max(560, self.root.winfo_screenheight() - 96))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(920, width), min(640, height))
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._variables()
        self._build()
        self._logger = logging.getLogger("ssgs")
        self._handler = _QueueHandler(self.events)
        self._logger.addHandler(self._handler)
        if SpectroApp._logging_clients == 0:
            SpectroApp._original_log_level = self._logger.level
        SpectroApp._logging_clients += 1
        self._logger.setLevel(logging.INFO)
        self.root.bind("<Control-o>", lambda _: self.open_model())
        self.root.bind("<Control-s>", lambda _: self.save_model())
        self.root.bind("<Control-g>", lambda _: self.generate())
        self.root.bind("<Control-d>", lambda _: self.demo())
        self._poll_id = self.root.after(80, self._poll)
        self._log("Welcome to Spectro. Add audio to train a model, open a saved model, or try the demo.")

    def _load_preferences(self):
        try:
            value = json.loads(self._config_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _variables(self):
        defaults = {"states": "16", "iterations": "8", "train_rate": "16000", "seed": "0", "duration": "5", "output_rate": "22050", "method": "stochastic", "fidelity": "0.0", "temperature": "1.0", "excitation": "Karplus-Strong"}
        for key, default in defaults.items():
            value = self._preferences.get(key, default)
            setattr(self, key, tk.StringVar(self.root, value=str(value)))
        self.recursive = tk.BooleanVar(self.root, value=True)
        self.status = tk.StringVar(self.root, value="Ready to create")
        self.model_name = tk.StringVar(self.root, value="No model loaded")
        self.model_detail = tk.StringVar(self.root, value="Train from your audio or open an existing model.")
        self.audio_detail = tk.StringVar(self.root, value="Your sound, in a new direction")
        self.input_count = tk.StringVar(self.root, value="No audio selected")

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.option_add("*Listbox.selectBackground", "#306a69")
        style.configure(".", background=PANEL, foreground=TEXT, borderwidth=0, bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD, font=("DejaVu Sans", 10))
        style.configure("TFrame", background=PANEL)
        style.configure("Outer.TFrame", background=BG)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", background=BG, font=("DejaVu Sans", 25, "bold"), foreground=TEXT)
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED)
        style.configure("Section.TLabel", font=("DejaVu Sans", 12, "bold"))
        style.configure("Metric.TLabel", font=("DejaVu Sans", 16, "bold"), foreground=TEAL)
        style.configure("TButton", background=FIELD, foreground=TEXT, padding=(12, 8), borderwidth=0)
        style.map("TButton", background=[("active", "#345060"), ("disabled", "#1c2a35")], foreground=[("disabled", "#687c8c")])
        style.configure("Accent.TButton", background=TEAL, foreground=BG, font=("DejaVu Sans", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#92f1e0"), ("disabled", "#304c4d")], foreground=[("disabled", "#74918d")])
        style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, padding=7)
        style.configure("TSpinbox", fieldbackground=FIELD, foreground=TEXT, arrowcolor=TEXT, padding=6)
        style.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT, arrowcolor=TEXT, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)], selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        style.configure("TCheckbutton", background=PANEL, foreground=MUTED, indicatorbackground=FIELD)
        style.map("TCheckbutton", background=[("active", PANEL)], indicatorbackground=[("selected", TEAL)])
        style.configure("TNotebook", background=PANEL, borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL, tabmargins=(0, 0, 0, 12))
        style.configure("TNotebook.Tab", background=FIELD, foreground=MUTED, borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL, padding=(15, 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEAL)])
        style.configure("Horizontal.TProgressbar", background=TEAL, troughcolor=FIELD, borderwidth=0, thickness=5)
        style.configure("TPanedwindow", background=BG)
        style.configure("Vertical.TScrollbar", background=FIELD, troughcolor=PANEL, bordercolor=PANEL, lightcolor=FIELD, darkcolor=FIELD, arrowcolor=MUTED, width=10)
        style.configure("TSeparator", background=FIELD)

    def _button(self, parent, text, command, accent=False, **pack):
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        button.pack(**pack)
        self._buttons.append(button)
        return button

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, style="Outer.TFrame", padding=(24, 20, 24, 18))
        header.grid(row=0, column=0, sticky="ew")
        title = ttk.Frame(header, style="Outer.TFrame")
        title.pack(side="left")
        ttk.Label(title, text="spectro", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title, text="SPECTRAL AUDIO STUDIO   /   " + __version__, style="Subtitle.TLabel", font=("DejaVu Sans", 9)).pack(anchor="w", pady=(3, 0))
        self._button(header, "Try a demo", self.demo, side="right", padx=(8, 0))
        self._button(header, "Open model…", self.open_model, side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=24)
        left = ttk.Frame(body, padding=16, width=350)
        body.add(left, weight=0)
        right = ttk.Frame(body, padding=(18, 14, 12, 10))
        body.add(right, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)
        train = ttk.Frame(self.notebook, padding=(0, 4, 0, 0))
        generate = ttk.Frame(self.notebook, padding=(0, 4, 0, 0))
        self.notebook.add(train, text="01  Train")
        self.notebook.add(generate, text="02  Generate")
        self._scroll_canvases = []
        self._train_panel(self._scroll_panel(train))
        self._generate_panel(self._scroll_panel(generate))
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(sequence, self._scroll_controls, add="+")

        model_label = ttk.Label(right, textvariable=self.model_name, style="Metric.TLabel", wraplength=650)
        model_label.grid(row=0, column=0, sticky="w")
        detail_label = ttk.Label(right, textvariable=self.model_detail, style="Muted.TLabel", wraplength=650)
        detail_label.grid(row=1, column=0, sticky="w", pady=(5, 12))
        right.bind("<Configure>", lambda event: (model_label.configure(wraplength=max(240, event.width - 40)), detail_label.configure(wraplength=max(240, event.width - 40))))
        ttk.Separator(right).grid(row=2, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(right, textvariable=self.audio_detail, style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(0, 6))
        self.figure = Figure(figsize=(7, 4), dpi=100, facecolor=PANEL)
        self.wave_axis, self.spectrum_axis = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.6]})
        self.figure.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.12, hspace=0.5)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=4, column=0, sticky="nsew")
        self._empty_plot()
        output_buttons = ttk.Frame(right)
        output_buttons.grid(row=5, column=0, sticky="ew", pady=(4, 10))
        self.play_button = self._button(output_buttons, "Listen ↗", self.play, side="left")
        self.folder_button = self._button(output_buttons, "Open output folder ↗", self.open_folder, side="left", padx=(8, 0))
        self.save_button = self._button(output_buttons, "Save model…", self.save_model, side="right")
        log_header = ttk.Frame(right)
        log_header.grid(row=6, column=0, sticky="ew")
        ttk.Label(log_header, text="SESSION ACTIVITY", style="Muted.TLabel", font=("DejaVu Sans", 8, "bold")).pack(side="left")
        self.log_widget = tk.Text(right, height=4, bg=BG, fg=MUTED, insertbackground=TEXT, relief="flat", borderwidth=0, highlightthickness=0, font=("DejaVu Sans Mono", 9), padx=10, pady=8, wrap="word", state="disabled")
        self.log_widget.grid(row=7, column=0, sticky="ew", pady=(5, 0))
        footer = ttk.Frame(self.root, style="Outer.TFrame", padding=(24, 12, 24, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        status_label = ttk.Label(footer, textvariable=self.status, style="Subtitle.TLabel", wraplength=550)
        status_label.grid(row=0, column=0, sticky="w")
        footer.bind("<Configure>", lambda event: status_label.configure(wraplength=max(180, event.width - 360)))
        self.cancel_button = ttk.Button(footer, text="Cancel", command=self.cancel, state="disabled", padding=(12, 4))
        self.cancel_button.grid(row=0, column=2, sticky="e")
        self.progress = ttk.Progressbar(footer, mode="determinate", length=220, maximum=1)
        self.progress.grid(row=0, column=1, padx=14, sticky="e")
        self._sync_buttons()

    def _scroll_panel(self, parent):
        """Keep controls reachable on small laptop screens and when resized."""
        canvas = tk.Canvas(parent, bg=PANEL, borderwidth=0, highlightthickness=0, width=310)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas, padding=(0, 2, 8, 8))
        window = canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        self._scroll_canvases.append(canvas)
        return inner

    def _scroll_controls(self, event):
        if event.widget.winfo_class() in ("TCombobox", "Listbox"):
            return
        for canvas in self._scroll_canvases:
            if str(event.widget).startswith(str(canvas)) and canvas.yview() != (0.0, 1.0):
                upward = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
                canvas.yview_scroll(-1 if upward else 1, "units")
                return "break"

    def _train_panel(self, parent):
        ttk.Label(parent, text="Start with sound", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Learn the spectral character of your recordings.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(4, 12))
        self.file_list = tk.Listbox(parent, height=5, bg=BG, fg=TEXT, relief="flat", highlightthickness=0, activestyle="none", selectmode="extended", exportselection=False, font=("DejaVu Sans", 10))
        self.file_list.pack(fill="x")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(7, 4))
        self._button(row, "Add files…", self.add_files, side="left")
        self._button(row, "Folder…", self.add_folder, side="left", padx=5)
        self._button(row, "Remove", self.remove_inputs, side="right")
        ttk.Label(parent, textvariable=self.input_count, style="Muted.TLabel").pack(anchor="w", pady=(2, 6))
        ttk.Checkbutton(parent, text="Include audio in subfolders", variable=self.recursive).pack(anchor="w", pady=(0, 10))
        fields = ttk.Frame(parent)
        fields.pack(fill="x")
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)
        self._field(fields, "Spectral states", self.states, 0, 0, ("8", "16", "34", "42"))
        self._field(fields, "Training iterations", self.iterations, 0, 1)
        self._field(fields, "Analysis rate (Hz)", self.train_rate, 1, 0, ("8000", "16000", "22050", "44100", "48000"))
        self._field(fields, "Random seed", self.seed, 1, 1)
        ttk.Label(parent, text="16 states is a useful starting point. More states capture more detail and take longer to train.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(12, 14))
        self._button(parent, "Train model", self.train, accent=True, fill="x")
        ttk.Label(parent, text="Models remain in memory until you save them.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(10, 0))

    def _generate_panel(self, parent):
        ttk.Label(parent, text="Explore a new sound", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Generate a WAV from your learned model.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(4, 16))
        fields = ttk.Frame(parent)
        fields.pack(fill="x")
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)
        self._field(fields, "Duration (seconds)", self.duration, 0, 0)
        self._field(fields, "Output rate (Hz)", self.output_rate, 0, 1, ("16000", "22050", "44100", "48000"))
        self._field(fields, "State path", self.method, 1, 0, ("stochastic", "viterbi"))
        self._field(fields, "Temperature", self.temperature, 1, 1)
        self._field(fields, "Excitation", self.excitation, 2, 0, ("Karplus-Strong", "Wavelet"))
        self._field(fields, "Source fidelity", self.fidelity, 2, 1, ("0.0", "0.25", "0.5", "0.75", "1.0"))
        self._field(fields, "Random seed", self.seed, 3, 0)
        ttk.Label(parent, text="Stochastic paths explore variations. Fidelity blends the source texture into the synthesis; it works best with saved training artifacts.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(12, 18))
        self.generate_button = self._button(parent, "Generate & save WAV…", self.generate, accent=True, fill="x")
        ttk.Label(parent, text="Listen opens the audio in your desktop's player.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(10, 0))

    def _field(self, parent, label, variable, row, column, choices=None):
        box = ttk.Frame(parent, padding=(0 if column == 0 else 6, 0, 6 if column == 0 else 0, 10))
        box.grid(row=row, column=column, sticky="ew")
        ttk.Label(box, text=label, style="Muted.TLabel", font=("DejaVu Sans", 9)).pack(anchor="w", pady=(0, 5))
        if choices:
            widget = ttk.Combobox(box, textvariable=variable, values=choices, width=13, state="readonly" if label in ("State path", "Excitation") else "normal")
        else:
            widget = ttk.Entry(box, textvariable=variable, width=13)
        widget.pack(fill="x")

    def _sync_buttons(self):
        for button in self._buttons:
            button.configure(state="disabled" if self.busy else "normal")
        if not self.busy:
            self.generate_button.configure(state="normal" if self.model is not None else "disabled")
            self.save_button.configure(state="normal" if self.model is not None else "disabled")
            self.play_button.configure(state="normal" if self.output_path is not None else "disabled")
            self.folder_button.configure(state="normal" if self.output_path is not None else "disabled")
        self.cancel_button.configure(state="normal" if self.busy else "disabled")

    def _log(self, message):
        if self._closed:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", datetime.now().strftime("%H:%M:%S  ") + str(message) + "\n")
        if int(self.log_widget.index("end-1c").split(".")[0]) > 300:
            self.log_widget.delete("1.0", "50.0")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _start(self, label: str, work: Callable, complete: Callable):
        if self.busy:
            return
        self.busy = True
        self.cancel_event.clear()
        self.status.set(label)
        self.progress["value"] = 0
        self._sync_buttons()
        self._log(label)

        def report(message, fraction=0):
            self.events.put(("progress", (message, fraction)))

        def worker():
            try:
                result = work(report, self.cancel_event)
                self.events.put(("done", (result, complete)))
            except Exception as error:
                self.events.put(("error", error))

        threading.Thread(target=worker, name="spectro-job", daemon=True).start()

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._log(value)
                elif kind == "progress":
                    message, fraction = value
                    self.status.set(message)
                    self.progress["value"] = max(0, min(1, fraction))
                    self._log(message)
                elif kind in ("done", "error"):
                    self.busy = False
                    if kind == "done":
                        result, complete = value
                        try:
                            complete(result)
                            self.progress["value"] = 1
                            self.status.set("Ready")
                        except Exception as error:
                            self._show_error(error)
                    else:
                        if isinstance(value, (InterruptedError, workflows.WorkflowCancelled)) or self.cancel_event.is_set():
                            self.status.set("Cancelled")
                            self._log("Task cancelled. Previously saved files remain available.")
                        else:
                            self._show_error(value)
                    self._sync_buttons()
        except queue.Empty:
            pass
        self._poll_id = self.root.after(80, self._poll)

    def _show_error(self, error):
        self.status.set("Could not complete the task")
        self._log(f"Error: {error}")
        messagebox.showerror("Spectro", str(error), parent=self.root)

    def _guard(self, callback):
        if self.busy:
            return
        try:
            callback()
        except (ValueError, OSError) as error:
            self._show_error(error)

    def add_files(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="Choose training audio", filetypes=[("Audio files", "*.wav *.flac *.ogg *.mp3 *.aiff *.aif *.au *.opus"), ("All files", "*")])
        self._add_inputs(paths)

    def add_folder(self):
        if self.busy:
            return
        folder = filedialog.askdirectory(parent=self.root, title="Choose a training audio folder")
        if folder:
            self._add_inputs([folder])

    def _add_inputs(self, paths):
        for value in paths:
            path = Path(value)
            if path not in self.inputs:
                self.inputs.append(path)
        self._refresh_inputs()

    def remove_inputs(self):
        if self.busy:
            return
        for index in reversed(self.file_list.curselection()):
            del self.inputs[index]
        self._refresh_inputs()

    def _refresh_inputs(self):
        self.file_list.delete(0, "end")
        for path in self.inputs:
            self.file_list.insert("end", ("▸ " if path.is_dir() else "♪ ") + path.name)
        self.input_count.set(f"{len(self.inputs)} source{'s' if len(self.inputs) != 1 else ''} selected" if self.inputs else "No audio selected")

    def _accept_model(self, model, path=None):
        self.model = model
        self.model_path = Path(path) if path else None
        self.model_name.set(self.model_path.name if self.model_path else "Your spectral model")
        frames = len(model.lpc_coefficients) if model.lpc_coefficients is not None else 0
        self.model_detail.set(f"{model.n_states} states   ·   {model.sample_rate or 16000:,} Hz analysis   ·   LPC {model.lpc_order}" + (f"   ·   {frames:,} cached frames" if frames else ""))
        self._log("Model ready. Generate a sound or save the model to continue later.")
        self.notebook.select(1)

    def train(self):
        def begin():
            if not self.inputs:
                raise ValueError("Add at least one audio file or folder before training.")
            inputs = list(self.inputs)
            options = dict(n_states=int(self.states.get()), iterations=int(self.iterations.get()), sample_rate=int(self.train_rate.get()), seed=int(self.seed.get()), recursive=self.recursive.get())
            self._start("Training spectral model…", lambda progress, cancel: workflows.train_model(inputs, progress=progress, cancel=cancel, **options), lambda model: self._accept_model(model))
        self._guard(begin)

    def open_model(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="Open Spectro model", filetypes=[("Spectro models", "*.npz *.safetensors"), ("All files", "*")])
        if path:
            self._start("Opening model…", lambda *_: workflows.load_model(path), lambda model: self._accept_model(model, path))

    def save_model(self):
        if self.busy or self.model is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Save Spectro model", defaultextension=".npz", filetypes=[("Compressed model", "*.npz"), ("SafeTensors checkpoint", "*.safetensors")], initialfile=self.model_path.name if self.model_path else "spectro-model.npz")
        if path:
            model = self.model
            self._start("Saving model and training artifacts…", lambda *_: workflows.save_model(model, path, force=True, include_training_artifacts=True), self._saved_model)

    def _saved_model(self, path):
        self.model_path = Path(path)
        self.model_name.set(self.model_path.name)
        self._log(f"Saved model: {path}")

    def generate(self):
        def begin():
            if self.model is None:
                raise ValueError("Train or open a model before generating audio.")
            options = dict(duration=float(self.duration.get()), sample_rate=int(self.output_rate.get()), method=self.method.get(), seed=int(self.seed.get()), fidelity=float(self.fidelity.get()), temperature=float(self.temperature.get()), excitation=self.excitation.get())
            if not np.isfinite(options["duration"]) or options["duration"] <= 0:
                raise ValueError("Duration must be a finite number greater than zero.")
            path = filedialog.asksaveasfilename(parent=self.root, title="Save generated WAV", defaultextension=".wav", filetypes=[("WAV audio", "*.wav")], initialfile="spectro-generated.wav")
            if not path:
                return
            model = self.model
            def work(progress, cancel):
                output = workflows.generate_audio(model, path, force=True, progress=progress, cancel=cancel, **options)
                audio, rate = self._read_preview(output)
                return output, audio, rate
            self._start("Generating audio…", work, self._generated)
        self._guard(begin)

    @staticmethod
    def _read_preview(path):
        with sf.SoundFile(str(path)) as audio_file:
            audio = audio_file.read(min(len(audio_file), audio_file.samplerate * 60), dtype="float64", always_2d=True).mean(axis=1)
            return audio, audio_file.samplerate

    def _generated(self, result):
        path, audio, rate = result
        self.output_path = Path(path)
        duration = len(audio) / rate
        preview = "   ·   first 60 s" if duration >= 60 else f"   ·   {duration:.2f} s"
        self.audio_detail.set(self.output_path.name + preview)
        self._plot_audio(audio, rate)
        self._log(f"Audio saved: {self.output_path}")

    def demo(self):
        if self.busy:
            return
        data_root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "spectro" / "sessions"
        def work(progress, cancel):
            data_root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="demo-", dir=data_root))
            paths = workflows.run_demo(directory, progress=progress, cancel=cancel)
            model = workflows.load_model(paths["model"])
            audio, rate = self._read_preview(paths["audio"])
            return paths, model, audio, rate
        def complete(result):
            paths, model, audio, rate = result
            self._accept_model(model, paths["model"])
            self._generated((paths["audio"], audio, rate))
            self._log("Demo complete. Try changing the seed, temperature, or excitation and generate again.")
        self._start("Creating a demo model and audio…", work, complete)

    def _axes_style(self):
        for axis in (self.wave_axis, self.spectrum_axis):
            axis.set_facecolor(BG)
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.xaxis.label.set_color(MUTED)
            axis.yaxis.label.set_color(MUTED)
            for spine in axis.spines.values():
                spine.set_color(FIELD)
            axis.grid(color=FIELD, alpha=0.5, linewidth=0.5)

    def _empty_plot(self):
        self._axes_style()
        self.wave_axis.set_xlim(0, 5)
        self.wave_axis.set_ylim(-1, 1)
        self.wave_axis.set_ylabel("Amplitude")
        self.wave_axis.axhline(0, color=TEAL, linewidth=1, alpha=0.5)
        self.spectrum_axis.set_xlim(0, 5)
        self.spectrum_axis.set_ylim(0, 8)
        self.spectrum_axis.set_ylabel("Frequency (kHz)")
        self.spectrum_axis.set_xlabel("Time (seconds)")
        self.spectrum_axis.text(0.5, 0.5, "Sound takes shape here\nTrain or open a model, then generate audio.", transform=self.spectrum_axis.transAxes, color=MUTED, ha="center", va="center", fontsize=11, linespacing=1.7)
        self.canvas.draw_idle()

    def _plot_audio(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float64)
        self.wave_axis.clear()
        self.spectrum_axis.clear()
        self._axes_style()
        stride = max(1, len(audio) // 18000)
        self.wave_axis.plot(np.arange(0, len(audio), stride) / sample_rate, audio[::stride], color=TEAL, linewidth=0.65, marker="." if len(audio) < 2 else None)
        self.wave_axis.set_ylabel("Amplitude")
        self.wave_axis.set_xlim(0, max(len(audio) / sample_rate, 0.01))
        self.wave_axis.set_ylim(-1.05, 1.05)
        if len(audio) >= 8:
            # Compute at the original rate to preserve the full frequency range.
            # Average time bins for large previews instead of aliasing the audio.
            nperseg = min(1024, len(audio))
            frequencies, times, power = signal.spectrogram(audio, fs=sample_rate, nperseg=nperseg, noverlap=0)
            group_size = max(1, int(np.ceil(len(times) / 1200)))
            if group_size > 1:
                starts = np.arange(0, len(times), group_size)
                counts = np.minimum(group_size, len(times) - starts)
                power = np.add.reduceat(power, starts, axis=1) / counts
                times = np.add.reduceat(times, starts) / counts
            self.spectrum_axis.pcolormesh(times, frequencies / 1000, 10 * np.log10(np.maximum(power, 1e-12)), cmap="magma", shading="auto", vmin=-95, vmax=-20, rasterized=True)
            self.spectrum_axis.set_ylim(0, min(8, sample_rate / 2000))
        else:
            self.spectrum_axis.text(0.5, 0.5, "Audio is too short for a spectrogram", transform=self.spectrum_axis.transAxes, color=MUTED, ha="center", va="center")
        self.spectrum_axis.set_xlabel("Time (seconds)")
        self.spectrum_axis.set_ylabel("Frequency (kHz)")
        self.canvas.draw_idle()

    def _open_path(self, path):
        try:
            subprocess.Popen(["xdg-open", str(Path(path).resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            self._show_error(OSError(f"Could not open the desktop application: {error}"))

    def play(self):
        if self.output_path is not None:
            self._open_path(self.output_path)

    def open_folder(self):
        if self.output_path is not None:
            self._open_path(self.output_path.parent)

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.status.set("Cancelling after the current processing step…")
            self.cancel_button.configure(state="disabled")

    def close(self):
        if self._closed:
            return
        if self.busy and not messagebox.askyesno("Close Spectro?", "A task is running. Cancel it and close the studio?", parent=self.root):
            return
        self.cancel_event.set()
        self._closed = True
        self.root.after_cancel(self._poll_id)
        self._logger.removeHandler(self._handler)
        self._handler.close()
        SpectroApp._logging_clients -= 1
        if SpectroApp._logging_clients == 0 and self._logger.level == logging.INFO:
            self._logger.setLevel(SpectroApp._original_log_level)
        preferences = {key: getattr(self, key).get() for key in ("states", "iterations", "train_rate", "seed", "duration", "output_rate", "method", "fidelity", "temperature", "excitation")}
        preference_temp = None
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._config_path.parent, delete=False) as output:
                json.dump(preferences, output, indent=2)
                preference_temp = Path(output.name)
            os.replace(preference_temp, self._config_path)
        except OSError:
            pass
        finally:
            if preference_temp is not None:
                try:
                    preference_temp.unlink(missing_ok=True)
                except OSError:
                    pass
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spectro desktop audio studio")
    parser.add_argument("--smoke-test", action="store_true", help="initialize a hidden window and exit")
    args = parser.parse_args(argv)
    try:
        root = tk.Tk(className="Spectro")
    except tk.TclError as error:
        parser.exit(1, f"Spectro could not open a display: {error}\nUse spectro --help for command-line workflows.\n")
    if args.smoke_test:
        root.withdraw()
    app = SpectroApp(root)
    if args.smoke_test:
        root.update_idletasks()
        root.update()
        app.close()
        print("Spectro desktop initialized successfully")
    else:
        root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
