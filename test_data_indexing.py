import numpy as np
import pytest

from ssgs import SpectralStateGuidedSynthesis


def _test_signal(sample_rate=8000, duration=0.5):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    # Stack a few harmonics to give LPC something meaningful
    signal = 0.6 * np.sin(2 * np.pi * 220 * t)
    signal += 0.3 * np.sin(2 * np.pi * 330 * t)
    signal += 0.2 * np.sin(2 * np.pi * 440 * t)
    signal += 0.05 * np.random.randn(len(t))
    return signal


def test_frame_metadata_and_indexing_round_trip():
    sample_rate = 8000
    signal = _test_signal(sample_rate=sample_rate)

    model = SpectralStateGuidedSynthesis(
        n_states=6,
        lpc_order=10,
        frame_size=256,
        hop_size=64,
    )

    coeffs, residuals = model.extract_features(signal, sample_rate)

    assert coeffs.shape[0] == model.training_frames.shape[0]
    assert residuals.shape[0] == coeffs.shape[0]
    assert model.sample_rate == sample_rate

    metadata = model.get_frame_metadata()
    assert metadata is not None
    assert "energy" in metadata
    assert metadata["energy"].shape[0] == coeffs.shape[0]
    assert "spectral_centroid" in metadata

    # Ensure ranking works and returns consistent metadata batches
    top = model.top_frames("energy", k=5)
    assert top.indices.ndim == 1
    assert top.metadata["energy"].shape[0] == top.indices.shape[0]
    assert top.metadata["spectral_centroid"].shape == top.metadata["energy"].shape

    # Build and reuse LPC-only index
    lpc_index = model.build_feature_index("lpc")
    assert lpc_index.tree.n == coeffs.shape[0]

    query = model.query_similar_frames(0, k=3, feature_space="lpc")
    assert query.indices.shape[0] == min(3, coeffs.shape[0] - 1)
    assert all(idx != 0 for idx in query.indices)

    direct_query = model.query_frames_by_features(
        coeffs[0], k=4, feature_space="lpc", include_metadata=False
    )
    assert direct_query.metadata == {}
    assert direct_query.indices.shape[0] == min(4, coeffs.shape[0])

    # Extended feature index uses metadata fusion
    extended_index = model.build_feature_index("extended")
    assert extended_index.tree.m == coeffs.shape[1] + 3

    extended_vector = {
        "lpc": coeffs[0],
        "energy": model.frame_metadata["energy"][0],
        "spectral_centroid": model.frame_metadata["spectral_centroid"][0],
        "spectral_bandwidth": model.frame_metadata["spectral_bandwidth"][0],
    }

    extended_query = model.query_frames_by_features(
        extended_vector, feature_space="extended", k=2
    )
    assert extended_query.indices.shape[0] == min(2, coeffs.shape[0])

    with pytest.raises(ValueError):
        model.query_frames_by_features({}, feature_space="extended")


def test_frame_metadata_export_import(tmp_path):
    sample_rate = 8000
    signal = _test_signal(sample_rate=sample_rate)

    model = SpectralStateGuidedSynthesis(n_states=4, lpc_order=8, frame_size=256, hop_size=64)
    model.extract_features(signal, sample_rate)
    model.initialize_hmm_parameters()

    path = tmp_path / "model.npz"
    model.export_model(path, include_training_artifacts=True)

    restored = SpectralStateGuidedSynthesis.load_model(path)
    assert restored.sample_rate == sample_rate
    if model.frame_metadata:
        for key, values in model.frame_metadata.items():
            assert key in restored.frame_metadata
            assert np.allclose(values, restored.frame_metadata[key])

    restored_index = restored.build_feature_index("lpc")
    assert restored_index.tree.n == model.lpc_coefficients.shape[0]
