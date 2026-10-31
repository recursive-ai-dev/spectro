"""Publication-path regressions: formats, atomic publish, demo cleanup."""

import errno
import os
from pathlib import Path

import numpy as np
import pytest

import soundfile as sf

from ssgs import SpectralStateGuidedSynthesis, workflows


SAMPLE_RATE = 8000


def _tone(seconds=0.3, frequency=220.0, sample_rate=SAMPLE_RATE):
    times = np.arange(int(sample_rate * seconds)) / sample_rate
    return (0.4 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def _model(seed=0):
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=seed
    )
    model.train(_tone(), SAMPLE_RATE, n_em_iterations=1, verbose=False)
    return model


def _leftovers(directory):
    return sorted(
        path.name
        for path in Path(directory).iterdir()
        if path.name.startswith(".") or path.name.endswith((".tmp", ".part"))
    )


# --------------------------------------------------------------------------- #
# Opus discovery and decoding
# --------------------------------------------------------------------------- #

def test_opus_is_advertised_and_discovered(tmp_path):
    if "OGG" not in sf.available_formats():
        pytest.skip("libsndfile build has no OGG container")
    assert ".opus" in workflows.supported_audio_extensions()

    opus = tmp_path / "voice.opus"
    sf.write(opus, _tone(0.4), SAMPLE_RATE, format="OGG", subtype="OPUS")
    assert workflows.discover_audio([tmp_path]) == [opus]

    model = workflows.train_model(
        [opus], n_states=2, iterations=1, frame_size=128, hop_size=64,
        sample_rate=SAMPLE_RATE,
    )
    assert len(model.lpc_coefficients) > 0
    assert model.transition_matrix.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Atomic publication
# --------------------------------------------------------------------------- #

def test_save_model_falls_back_when_hard_links_are_unavailable(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(workflows.os, "link", unsupported)
    model = _model()
    target = workflows.save_model(model, tmp_path / "model.npz")
    assert target.exists()
    assert _leftovers(tmp_path) == []
    assert len(workflows.load_model(target).lpc_coefficients) == len(model.lpc_coefficients)


def test_write_audio_falls_back_when_hard_links_are_unavailable(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(workflows.os, "link", unsupported)
    audio = _tone(0.05)
    target = workflows.write_audio(tmp_path / "out.wav", audio, SAMPLE_RATE)
    written, rate = sf.read(target, dtype="int32")
    assert rate == SAMPLE_RATE
    assert written.shape[0] == audio.size
    assert _leftovers(tmp_path) == []


def test_publish_never_clobbers_an_existing_target(tmp_path, monkeypatch):
    existing = tmp_path / "model.npz"
    existing.write_bytes(b"keep me")
    model = _model()

    with pytest.raises(FileExistsError):
        workflows.save_model(model, existing)
    assert existing.read_bytes() == b"keep me"
    assert _leftovers(tmp_path) == []

    workflows.save_model(model, existing, force=True)
    assert existing.read_bytes() != b"keep me"
    assert _leftovers(tmp_path) == []


def test_save_model_failure_leaves_no_temporary_files(tmp_path, monkeypatch):
    model = _model()

    def explode(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(model, "export_model", explode)
    with pytest.raises(RuntimeError, match="serialization failed"):
        workflows.save_model(model, tmp_path / "model.npz")

    assert list(tmp_path.iterdir()) == []


def test_write_audio_failure_preserves_the_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "out.wav"
    sf.write(target, _tone(0.05), SAMPLE_RATE)
    before = target.read_bytes()

    def explode(*args, **kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(workflows.sf, "write", explode)
    with pytest.raises(RuntimeError, match="writer failed"):
        workflows.write_audio(target, _tone(0.05), SAMPLE_RATE, force=True)

    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def test_run_demo_success_writes_all_artifacts(tmp_path):
    directory = tmp_path / "demo"
    paths = workflows.run_demo(directory)
    assert paths["source"].exists()
    assert paths["model"].exists()
    assert paths["audio"].exists()
    assert sorted(path.name for path in directory.iterdir()) == [
        "generated.wav",
        "model.npz",
        "source.wav",
    ]


def test_run_demo_failure_removes_its_own_partial_output(tmp_path, monkeypatch):
    directory = tmp_path / "demo"

    def explode(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(workflows, "train_model", explode)
    with pytest.raises(RuntimeError, match="training failed"):
        workflows.run_demo(directory)

    assert not directory.exists(), "the demo directory this call created should be gone"


def test_run_demo_failure_never_deletes_pre_existing_files(tmp_path, monkeypatch):
    directory = tmp_path / "demo"
    directory.mkdir()
    keep = directory / "source.wav"
    keep.write_bytes(b"previous demo")

    def explode(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(workflows, "train_model", explode)
    with pytest.raises(RuntimeError, match="training failed"):
        workflows.run_demo(directory, force=True)

    # ``force=True`` allowed the source to be rewritten, but cleanup must not
    # remove a file this call did not create, and must not leave new files.
    assert keep.exists()
    assert sorted(path.name for path in directory.iterdir()) == ["source.wav"]


def test_run_demo_refuses_to_overwrite_without_force(tmp_path):
    directory = tmp_path / "demo"
    directory.mkdir()
    keep = directory / "source.wav"
    keep.write_bytes(b"previous demo")

    with pytest.raises(FileExistsError):
        workflows.run_demo(directory)
    assert keep.read_bytes() == b"previous demo"
    assert sorted(path.name for path in directory.iterdir()) == ["source.wav"]
