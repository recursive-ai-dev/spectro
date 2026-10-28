"""Product workflow checks using real models and real audio files."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ssgs import workflows
from ssgs.cli import main


@pytest.fixture
def source(tmp_path):
    sample_rate = 8000
    times = np.arange(3200) / sample_rate
    audio = 0.45 * np.sin(2 * np.pi * 220 * times) + 0.1 * np.sin(2 * np.pi * 480 * times)
    path = tmp_path / "source.WAV"
    sf.write(path, audio, sample_rate)
    return path


@pytest.fixture
def trained(source, tmp_path):
    path = tmp_path / "model.npz"
    model = workflows.train_model(
        [source], path, n_states=3, iterations=1, sample_rate=8000,
        lpc_order=6, frame_size=128, hop_size=64,
    )
    return model, path


def test_real_cli_round_trip(source, tmp_path, capsys):
    path = tmp_path / "model.safetensors"
    assert main([
        "train", str(source), "--output", str(path), "--states", "3",
        "--iterations", "1", "--sample-rate", "8000", "--lpc-order", "6",
        "--frame-size", "128", "--hop-size", "64",
    ]) == 0
    capsys.readouterr()
    assert main(["inspect", str(path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["states"] == 3
    assert summary["training_frames"] > 0
    assert summary["sample_rate"] == 8000
    outputs = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for output in outputs:
        assert main(["generate", str(path), "--output", str(output), "--duration", "0.125", "--sample-rate", "12000", "--seed", "123"]) == 0
    audio_a, rate = sf.read(outputs[0])
    audio_b, _ = sf.read(outputs[1])
    assert rate == 12000
    assert audio_a.shape == (1500,)
    assert np.array_equal(audio_a, audio_b)
    assert np.any(np.abs(audio_a) > 0)
    assert sf.info(outputs[0]).subtype == "PCM_24"


def test_outputs_are_protected_and_force_is_explicit(trained, tmp_path):
    model, path = trained
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        workflows.save_model(model, path)
    assert path.read_bytes() == before
    output = tmp_path / "protected.wav"
    output.write_bytes(b"keep this")
    with pytest.raises(FileExistsError):
        workflows.generate_audio(model, output, duration=0.1)
    assert output.read_bytes() == b"keep this"
    workflows.generate_audio(model, output, duration=0.1, force=True)
    assert sf.info(output).frames == 2205


def test_discovery_is_case_insensitive_recursive_and_deduplicated(source, tmp_path):
    subdir = tmp_path / "nested"
    subdir.mkdir()
    other = subdir / "another.wav"
    other.write_bytes(source.read_bytes())
    (tmp_path / "ignore.txt").write_text("not audio")
    assert workflows.discover_audio([tmp_path, source]) == [source.resolve()]
    assert set(workflows.discover_audio([tmp_path], recursive=True)) == {source.resolve(), other.resolve()}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        workflows.discover_audio([tmp_path / "missing"])
    with pytest.raises(ValueError, match="Unsupported audio input"):
        workflows.discover_audio([tmp_path / "ignore.txt"])


def test_cancellation_before_training_creates_no_output(source, tmp_path):
    cancel = threading.Event()
    cancel.set()
    output = tmp_path / "never.npz"
    with pytest.raises(workflows.WorkflowCancelled):
        workflows.train_model([source], output, cancel=cancel)
    assert not output.exists()


def test_failures_are_readable_and_do_not_traceback(tmp_path, capsys):
    assert main(["generate", str(tmp_path / "missing.npz"), "--output", str(tmp_path / "out.wav")]) == 1
    captured = capsys.readouterr()
    assert "spectro:" in captured.err
    assert "Traceback" not in captured.err
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] in {"native", "numpy"}


def test_atomic_audio_failure_preserves_output(tmp_path, monkeypatch):
    output = tmp_path / "existing.wav"
    output.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("simulated encoder failure")

    monkeypatch.setattr(sf, "write", fail)
    with pytest.raises(OSError, match="encoder failure"):
        workflows.write_audio(output, np.zeros(20), 8000, force=True)
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]


def test_model_save_race_never_overwrites_new_file(trained, tmp_path, monkeypatch):
    model, _ = trained
    output = tmp_path / "raced.npz"
    export = model.export_model

    def create_competitor(path, **kwargs):
        export(path, **kwargs)
        output.write_bytes(b"other process")

    monkeypatch.setattr(model, "export_model", create_competitor)
    with pytest.raises(FileExistsError):
        workflows.save_model(model, output)
    assert output.read_bytes() == b"other process"
    assert not list(tmp_path.glob(".raced-*"))


def test_demo_runs_without_external_training_data(tmp_path):
    outputs = workflows.run_demo(tmp_path / "demo")
    assert set(outputs) == {"source", "model", "audio"}
    assert all(isinstance(path, Path) and path.is_file() for path in outputs.values())
    assert sf.info(outputs["audio"]).frames == 44100
    assert workflows.model_info(outputs["model"])["states"] == 8


def test_adaptation_round_trip_retains_statistics_and_original_model(trained, tmp_path, capsys):
    model, original = trained
    original_bytes = original.read_bytes()
    new_audio = tmp_path / "new.wav"
    times = np.arange(4800) / 12000
    sf.write(new_audio, 0.5 * np.sin(2 * np.pi * 550 * times), 12000)
    output = tmp_path / "adapted.safetensors"
    assert main([
        "adapt", str(original), str(new_audio), "--output", str(output),
        "--adaptation-rate", "0.6", "--memory-limit", "24", "--json",
    ]) == 0
    metrics = json.loads(capsys.readouterr().out)["recordings"]
    assert len(metrics) == 1
    assert metrics[0]["frames_used"] > 0
    assert np.isfinite(metrics[0]["log_likelihood"])
    reloaded = workflows.load_model(output)
    assert reloaded.sample_rate == 8000
    assert len(reloaded.training_frames) <= 24
    assert not np.allclose(reloaded.state_means, model.state_means)
    assert workflows.model_info(reloaded)["adaptive_frames"] > 0
    assert workflows.model_info(reloaded)["last_adaptation"]
    assert original.read_bytes() == original_bytes


def test_adaptation_cannot_replace_its_input_even_with_force(trained, source):
    _, original = trained
    before = original.read_bytes()
    with pytest.raises(ValueError, match="differ from the input"):
        workflows.adapt_model(original, [source], original, force=True)
    assert original.read_bytes() == before


def test_failed_adaptation_does_not_publish_partial_model(trained, source, tmp_path):
    _, original = trained
    before = original.read_bytes()
    bad_audio = tmp_path / "zz-invalid.wav"
    bad_audio.write_bytes(b"not audio")
    output = tmp_path / "adapted.npz"
    with pytest.raises(ValueError, match="Failed to load audio"):
        workflows.adapt_model(original, [source, bad_audio], output)
    assert not output.exists()
    assert original.read_bytes() == before


@pytest.mark.parametrize("command,kwargs", [
    ("train", {"iterations": -1}),
    ("train", {"seed": -1}),
    ("train", {"frame_size": 4}),
    ("adapt", {"adaptation_rate": float("nan")}),
    ("adapt", {"memory_blend": 1.1}),
    ("adapt", {"memory_limit": 0}),
    ("generate", {"duration": float("inf")}),
    ("generate", {"temperature": -1}),
    ("generate", {"fidelity": 2}),
])
def test_numeric_settings_fail_before_loading_audio_or_models(tmp_path, monkeypatch, command, kwargs):
    def unexpected(*args, **kw):
        pytest.fail("expensive input work started before validating settings")

    monkeypatch.setattr(workflows, "discover_audio", unexpected)
    monkeypatch.setattr(workflows, "load_model", unexpected)
    with pytest.raises(ValueError):
        if command == "train":
            workflows.train_model([tmp_path / "source.wav"], tmp_path / "out.npz", **kwargs)
        elif command == "adapt":
            workflows.adapt_model(tmp_path / "model.npz", [tmp_path / "source.wav"], tmp_path / "out.npz", **kwargs)
        else:
            workflows.generate_audio(tmp_path / "model.npz", tmp_path / "out.wav", **kwargs)
