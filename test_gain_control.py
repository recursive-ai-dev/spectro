import numpy as np
import pytest

from ssgs import SpectralStateGuidedSynthesis


def test_gain_floor_raises_quiet_audio():
    model = SpectralStateGuidedSynthesis()
    quiet = np.full(128, 1e-4, dtype=np.float32)

    boosted = model._apply_gain_floor(quiet, minimum_peak=0.75, target_peak=0.9)

    boosted_peak = np.max(np.abs(boosted))

    assert boosted.dtype == np.float32
    assert boosted_peak == pytest.approx(0.75, rel=0.0, abs=1e-3)


def test_synthesis_applies_gain_floor(monkeypatch):
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=4, frame_size=16, hop_size=4)
    model.transition_matrix = np.array([[1.0]], dtype=np.float64)
    model.initial_probabilities = np.array([1.0], dtype=np.float64)
    model.state_means = np.zeros((1, model.lpc_order), dtype=np.float64)
    model.state_covariances = np.eye(model.lpc_order, dtype=np.float64)[None, :, :]

    call_count = {"count": 0}

    def fake_gain(arr, minimum_peak=0.75, target_peak=0.9):
        call_count["count"] += 1
        return np.full_like(arr, 0.8, dtype=np.float32)

    monkeypatch.setattr(model, "_apply_gain_floor", fake_gain)

    audio = model.synthesize_audio(0.05, sample_rate=8000, fidelity=0.0)

    assert call_count["count"] == 1
    assert np.max(np.abs(audio)) == pytest.approx(0.8, rel=0.0, abs=1e-3)
