"""Application workflows shared by the command line and desktop workbench.

Every output is validated before expensive work starts. Numeric work remains in
the model; this module handles paths, audio files, progress and cancellation.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from . import SpectralStateGuidedSynthesis, __version__
from .model import _NATIVE_BACKEND_AVAILABLE


Progress = Optional[Callable[[str, float], None]]


class WorkflowCancelled(Exception):
    """The user cancelled a workflow between computational stages."""


def _notify(progress: Progress, message: str, fraction: float, cancel=None) -> None:
    if cancel is not None and cancel.is_set():
        raise WorkflowCancelled("Operation cancelled")
    if progress is not None:
        progress(message, float(fraction))


def supported_audio_extensions() -> set[str]:
    """Return common extensions supported by the installed libsndfile build."""
    formats = sf.available_formats()
    mapping = {
        "WAV": {".wav", ".wave"}, "WAVEX": {".wav", ".wave"},
        "FLAC": {".flac"}, "OGG": {".ogg", ".oga"},
        "AIFF": {".aif", ".aiff", ".aifc"}, "AU": {".au", ".snd"},
        "CAF": {".caf"}, "W64": {".w64"}, "RF64": {".rf64"},
        "MPEG": {".mp3"}, "MP3": {".mp3"},
    }
    return set().union(*(extensions for name, extensions in mapping.items() if name in formats))


def discover_audio(inputs, recursive: bool = False) -> list[Path]:
    """Find audio deterministically, deduplicating overlapping folders and links.

    Explicit invalid inputs fail with their path. Unrelated files in directories
    are ignored. Decoding errors are reported later with the affected filename.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    supported = supported_audio_extensions()
    discovered = set()
    for item in inputs:
        path = Path(item).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            discovered.update(
                candidate.resolve() for candidate in candidates
                if candidate.is_file() and candidate.suffix.casefold() in supported
            )
        elif path.is_file() and path.suffix.casefold() in supported:
            discovered.add(path.resolve())
        else:
            raise ValueError(f"Unsupported audio input: {path}. Supported extensions: {', '.join(sorted(supported))}")
    if not discovered:
        raise ValueError("No supported audio files found. Select audio files or a folder containing them; use recursive search for subfolders.")
    return sorted(discovered, key=lambda path: (str(path).casefold(), str(path)))


def _output_path(path, *, force: bool, suffixes) -> Path:
    target = Path(path).expanduser().absolute()
    if target.suffix not in suffixes:
        raise ValueError(f"Output must end in {' or '.join(suffixes)}: {target}")
    if target.exists() and (not force or not target.is_file()):
        raise FileExistsError(f"Output already exists: {target}. Choose another path or enable overwrite.")
    if target.is_symlink():
        raise ValueError(f"Output must not be a symbolic link: {target}")
    return target


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def _number(value, name: str, *, minimum=None, maximum=None, positive=False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def load_model(path) -> SpectralStateGuidedSynthesis:
    path = Path(path).expanduser()
    if path.suffix == ".safetensors":
        return SpectralStateGuidedSynthesis.load_checkpoint(path)
    if path.suffix == ".npz":
        return SpectralStateGuidedSynthesis.load_model(path)
    raise ValueError("Model files must end in .npz or .safetensors")


def save_model(model, path, force: bool = False, include_training_artifacts: bool = True) -> Path:
    target = _output_path(path, force=force, suffixes=(".npz", ".safetensors"))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    try:
        if target.suffix == ".npz":
            model.export_model(temporary, include_training_artifacts=include_training_artifacts)
        else:
            model.save_checkpoint(temporary, include_training_artifacts=include_training_artifacts)
        if not force:
            os.link(temporary, target)
        else:
            os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def train_model(
    inputs, output=None, *, n_states: int = 16, iterations: int = 10,
    sample_rate: int = 16000, lpc_order: int = 12, frame_size: int = 1024,
    hop_size: int = 256, seed: int = 0, recursive: bool = False,
    force: bool = False, progress: Progress = None, cancel=None,
) -> SpectralStateGuidedSynthesis:
    iterations = _integer(iterations, "iterations")
    seed = _integer(seed, "seed")
    sample_rate = SpectralStateGuidedSynthesis._coerce_sample_rate(sample_rate)
    target = None if output is None else _output_path(output, force=force, suffixes=(".npz", ".safetensors"))
    model = SpectralStateGuidedSynthesis(
        n_states=n_states, lpc_order=lpc_order, frame_size=frame_size,
        hop_size=hop_size, random_state=seed,
    )
    files = discover_audio(inputs, recursive=recursive)
    _notify(progress, f"Analyzing {len(files)} audio file(s)…", 0.02, cancel)
    model.extract_features_from_files(files, sample_rate)
    _notify(progress, f"Initializing {n_states} spectral states…", 0.25, cancel)
    model.initialize_hmm_parameters()
    for iteration in range(iterations):
        _notify(progress, f"Learning iteration {iteration + 1} of {iterations}…", 0.30 + 0.60 * iteration / max(iterations, 1), cancel)
        likelihood, _, _ = model.run_em_iteration()
        _notify(progress, f"Iteration {iteration + 1}/{iterations} · log likelihood {likelihood:.2f}", 0.30 + 0.60 * (iteration + 1) / max(iterations, 1), cancel)
    _notify(progress, "Refining state transitions…", 0.92, cancel)
    model.identify_graph_constraints()
    if target is not None:
        _notify(progress, "Saving trained model…", 0.97, cancel)
        save_model(model, target, force=force)
    _notify(progress, "Training complete", 1.0)
    return model


def adapt_model(
    model_path, inputs, output, *, adaptation_rate: float = 0.25,
    stability_bias: float = 0.05, memory_blend: float = 0.7, memory_limit: int = 6000,
    iterations: int = 1, seed: int = 0, recursive: bool = False,
    force: bool = False, progress: Progress = None, cancel=None,
) -> tuple[SpectralStateGuidedSynthesis, list[dict]]:
    """Adapt a saved model to new recordings, then publish a separate model.

    Each recording forms its own adaptive sequence. All updates happen in a new
    in-memory model, and no output is published until every recording succeeds.
    """
    adaptation_rate = _number(adaptation_rate, "adaptation_rate", minimum=0.0, maximum=1.0)
    stability_bias = _number(stability_bias, "stability_bias", minimum=0.0, maximum=1.0)
    memory_blend = _number(memory_blend, "memory_blend", minimum=0.0, maximum=1.0)
    memory_limit = _integer(memory_limit, "memory_limit", 1)
    iterations = _integer(iterations, "iterations")
    seed = _integer(seed, "seed")
    target = _output_path(output, force=force, suffixes=(".npz", ".safetensors"))
    source = Path(model_path).expanduser().absolute()
    if target.resolve() == source.resolve():
        raise ValueError("Adaptation output must differ from the input model; choose a new model filename")
    files = discover_audio(inputs, recursive=recursive)
    _notify(progress, "Loading model for adaptation…", 0.02, cancel)
    model = load_model(source)
    metrics = []
    for index, path in enumerate(files):
        _notify(progress, f"Adapting to {path.name} ({index + 1}/{len(files)})…", 0.1 + 0.8 * index / len(files), cancel)
        try:
            audio, rate = sf.read(path, dtype="float32", always_2d=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Failed to load audio file {path}: {exc}") from exc
        values = model.adapt_to_audio(
            audio, sample_rate=int(rate), adaptation_rate=adaptation_rate,
            stability_bias=stability_bias, memory_blend=memory_blend,
            memory_limit=memory_limit, n_adaptation_iterations=iterations, seed=seed + index,
        )
        metrics.append({"source": str(path), **values})
    _notify(progress, "Saving adapted model…", 0.95, cancel)
    save_model(model, target, force=force)
    _notify(progress, "Adaptation complete", 1.0)
    return model, metrics


def write_audio(path, audio, sample_rate: int, *, force: bool = False) -> Path:
    """Write 24-bit WAV through a temporary file; preserve an existing output on failure."""
    target = _output_path(path, force=force, suffixes=(".wav",))
    sample_rate = SpectralStateGuidedSynthesis._coerce_sample_rate(sample_rate)
    samples = np.asarray(audio)
    if samples.ndim not in (1, 2) or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Audio must contain finite, non-empty mono or samples-by-channels data")
    if np.max(np.abs(samples)) > 1.0:
        raise ValueError("Audio amplitude exceeds the WAV range [-1, 1]")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".wav", dir=target.parent)
    os.close(descriptor)
    try:
        sf.write(temporary, samples, sample_rate, format="WAV", subtype="PCM_24")
        if not force:
            # link is an atomic create-if-absent on the output filesystem.
            os.link(temporary, target)
        else:
            os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def generate_audio(
    model_or_path, output, *, duration: float = 3.0, sample_rate: int = 22050,
    method: str = "stochastic", seed: int = 0, fidelity: float = 0.0,
    temperature: float = 1.0, excitation: str = "Karplus-Strong", force: bool = False,
    progress: Progress = None, cancel=None,
) -> Path:
    target = _output_path(output, force=force, suffixes=(".wav",))
    duration = _number(duration, "duration", positive=True)
    seed = _integer(seed, "seed")
    fidelity = _number(fidelity, "fidelity", minimum=0.0, maximum=1.0)
    temperature = _number(temperature, "temperature", positive=True)
    if method not in {"stochastic", "viterbi"}:
        raise ValueError("method must be 'stochastic' or 'viterbi'")
    if excitation not in {"Karplus-Strong", "Wavelet"}:
        raise ValueError("excitation must be 'Karplus-Strong' or 'Wavelet'")
    sample_rate = SpectralStateGuidedSynthesis._coerce_sample_rate(sample_rate)
    _notify(progress, "Loading model…", 0.05, cancel)
    model = load_model(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    _notify(progress, "Generating audio…", 0.15, cancel)
    samples = model.synthesize_audio(
        duration, sample_rate, method=method, seed=seed, fidelity=fidelity,
        temperature=temperature, excitation_type=excitation,
    )
    _notify(progress, "Writing 24-bit WAV…", 0.90, cancel)
    write_audio(target, samples, sample_rate, force=force)
    _notify(progress, "Audio ready", 1.0)
    return target


def model_info(model_or_path) -> dict:
    model = load_model(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    frames = 0 if model.lpc_coefficients is None else len(model.lpc_coefficients)
    transitions = model.transition_matrix
    summary = {
        "version": __version__, "states": int(model.n_states),
        "lpc_order": int(model.lpc_order), "frame_size": int(model.frame_size),
        "hop_size": int(model.hop_size), "sample_rate": model.sample_rate,
        "training_frames": int(frames),
        "training_sequences": 0 if model.sequence_starts is None else len(model.sequence_starts),
        "includes_training_artifacts": bool(frames),
        "backend": "native" if _NATIVE_BACKEND_AVAILABLE else "numpy",
        "seed": int(model.random_state),
    }
    if transitions is not None:
        summary["transition_entropy"] = float(-np.sum(transitions * np.log(np.maximum(transitions, 1e-30))) / model.n_states)
    if model._adaptive_stats is not None:
        summary["adaptive_frames"] = float(model._adaptive_stats.total_frames)
        summary["last_adaptation"] = model._adaptive_stats.last_adaptation
    return summary


def create_demo_signal(sample_rate: int = 16000, duration: float = 2.0, seed: int = 0) -> np.ndarray:
    from .signals import create_test_signal
    return create_test_signal(sample_rate=sample_rate, duration=duration, seed=seed, normalize=True)


def run_demo(output_dir, *, force: bool = False, progress: Progress = None, cancel=None) -> dict:
    directory = Path(output_dir).expanduser().absolute()
    paths = {"source": directory / "source.wav", "model": directory / "model.npz", "audio": directory / "generated.wav"}
    for name, path in paths.items():
        _output_path(path, force=force, suffixes=(".npz",) if name == "model" else (".wav",))
    _notify(progress, "Creating a synthetic training signal…", 0.0, cancel)
    write_audio(paths["source"], create_demo_signal(), 16000, force=force)
    model = train_model(
        [paths["source"]], paths["model"], n_states=8, iterations=3,
        frame_size=512, hop_size=128, force=force, cancel=cancel,
        progress=None if progress is None else lambda message, fraction: progress(message, 0.1 + 0.65 * fraction),
    )
    generate_audio(
        model, paths["audio"], duration=2.0, force=force, cancel=cancel,
        progress=None if progress is None else lambda message, fraction: progress(message, 0.75 + 0.25 * fraction),
    )
    return paths


def doctor_info() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "soundfile", "PyWavelets", "safetensors", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    try:
        import tkinter
        tkinter_version = str(tkinter.TkVersion)
    except ImportError:
        tkinter_version = "unavailable"
    return {
        "spectro": __version__, "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "backend": "native" if _NATIVE_BACKEND_AVAILABLE else "numpy",
        "tkinter": tkinter_version, "display_available": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform in ("win32", "darwin")),
        "dependencies": versions, "audio_extensions": sorted(supported_audio_extensions()),
    }
