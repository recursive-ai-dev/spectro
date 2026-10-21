"""Regression tests for SSGS validation, numerical safety, and persistence contracts."""

from pathlib import Path

import numpy as np
import pytest

from ssgs import SpectralStateGuidedSynthesis
from ssgs.core import forward_backward, gaussian_log_probs


def _manual_model(n_states=2, lpc_order=2):
    model = SpectralStateGuidedSynthesis(
        n_states=n_states,
        lpc_order=lpc_order,
        frame_size=max(8, lpc_order + 2),
        hop_size=4,
        smoothness_weight=0.0,
        random_state=23,
    )
    model.transition_matrix = np.full((n_states, n_states), 1.0 / n_states)
    model.initial_probabilities = np.full(n_states, 1.0 / n_states)
    model.state_means = np.zeros((n_states, lpc_order))
    model.state_covariances = np.tile(np.eye(lpc_order), (n_states, 1, 1))
    return model


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_states": 0},
        {"n_states": 1.5},
        {"lpc_order": 0},
        {"frame_size": 8, "lpc_order": 8},
        {"frame_size": 64, "hop_size": 65},
        {"smoothness_weight": -0.1},
        {"smoothness_weight": float("nan")},
        {"covariance_floor": 0},
        {"transition_pseudocount": -1},
        {"random_state": "not a seed"},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SpectralStateGuidedSynthesis(**kwargs)


def test_audio_preprocessing_handles_channels_integer_pcm_and_resampling():
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=2, frame_size=16, hop_size=8)
    pcm = np.array([[32767, 32767], [-32768, -32768]], dtype=np.int16)
    mono = model._prepare_audio(pcm, 8000)
    assert mono.shape == (2,)
    assert np.max(np.abs(mono)) <= 1.0
    unsigned = model._prepare_audio(np.array([0, 128, 255], dtype=np.uint8), 8000)
    assert unsigned[0] == pytest.approx(-1.0)
    assert unsigned[1] == pytest.approx(0.0)
    upsampled = model._prepare_audio(np.ones(80, dtype=np.float32), 8000, 16000)
    assert len(upsampled) == 160
    assert np.all(np.isfinite(upsampled))


@pytest.mark.parametrize(
    "bad_audio",
    [np.array([], dtype=np.float32), np.array([0.0, np.nan]), np.array([0.0, np.inf])],
)
def test_audio_preprocessing_rejects_empty_or_nonfinite_data(bad_audio):
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=2, frame_size=16, hop_size=8)
    with pytest.raises(ValueError):
        model._prepare_audio(bad_audio, 8000)


def test_frame_signal_covers_partial_tail_without_unsafe_strides():
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=4, frame_size=128, hop_size=64)
    frames = model._frame_signal(np.arange(513, dtype=np.float32))
    assert frames.shape == (8, 128)
    assert frames.flags.c_contiguous
    assert frames[-1, 0] == 448.0
    assert frames[-1, -1] == 0.0
    with pytest.raises(ValueError, match="empty"):
        model._frame_signal(np.array([], dtype=np.float32))


def test_feature_extraction_rejects_nonfinite_input_and_invalid_frequency_bounds():
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=4, frame_size=128, hop_size=32)
    with pytest.raises(ValueError, match="NaN"):
        model.extract_features(np.array([0.0, np.nan]), 8000)
    with pytest.raises(ValueError, match="Nyquist"):
        model._mel_filter_bank(8000, 128, fmin=20, fmax=5000)
    with pytest.raises(ValueError, match="Nyquist"):
        model._erb_filter_bank(8000, 128, fmin=30, fmax=5000)


def test_forward_backward_supports_single_frame_and_rejects_empty_input():
    gamma, xi, likelihood = forward_backward(
        np.array([[-0.2, -1.3]]),
        np.log(np.array([0.6, 0.4])),
        np.log(np.array([[0.9, 0.1], [0.2, 0.8]])),
    )
    assert gamma.shape == (1, 2)
    assert xi.shape == (0, 2, 2)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.isfinite(likelihood)
    with np.errstate(divide="ignore"):
        log_transition = np.log(np.eye(2))
    with pytest.raises(ValueError):
        forward_backward(np.empty((0, 2)), np.log([0.5, 0.5]), log_transition)


def test_singular_gaussian_covariance_is_regularized_without_nonfinite_values():
    means = np.zeros((2, 3))
    covariances = np.zeros((2, 3, 3))
    observations = np.array([[0.0, 0.2, -0.1], [0.5, -0.4, 0.1]])
    result = gaussian_log_probs(means, covariances, observations)
    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))


def test_forward_backward_respects_independent_sequence_boundaries():
    model = _manual_model()
    observations = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    starts = np.array([0, 2])
    gamma, xi, likelihood = model._expectation_step(observations, sequence_starts=starts)
    left_gamma, left_xi, left_ll = model._expectation_step(observations[:2], sequence_starts=[0])
    right_gamma, right_xi, right_ll = model._expectation_step(observations[2:], sequence_starts=[0])
    assert xi.shape == (2, 2, 2)
    assert np.allclose(gamma, np.concatenate((left_gamma, right_gamma)))
    assert np.allclose(xi, np.concatenate((left_xi, right_xi)))
    assert likelihood == pytest.approx(left_ll + right_ll)


def test_m_step_uses_only_within_sequence_transitions_and_positive_covariances():
    model = _manual_model()
    observations = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.2, 5.1]])
    starts = np.array([0, 2])
    gamma, xi, _ = model._expectation_step(observations, sequence_starts=starts)
    assert xi.shape[0] == 2  # no transition across the recording boundary
    model._maximization_step(observations, gamma, xi, sequence_starts=starts)
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    assert np.all(np.linalg.eigvalsh(model.state_covariances) > 0.0)


def test_cache_replacement_invalidates_emission_and_feature_indexes():
    model = _manual_model()
    model.lpc_coefficients = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    original_index = model.build_feature_index("lpc")
    model.lpc_coefficients = np.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])
    replacement_index = model.build_feature_index("lpc")
    assert replacement_index is not original_index
    assert replacement_index.tree.data.mean() == pytest.approx(0.0, abs=1e-7)

    model._compute_and_cache_cholesky_factors()
    model.state_covariances = np.tile(2.0 * np.eye(2), (2, 1, 1))
    assert model._caches.chol_factors is None
    emissions = model._compute_emission_log_probs(np.zeros((1, 2)))
    assert np.all(np.isfinite(emissions))


def test_kmeans_initialization_is_reproducible_from_random_state():
    sr = 8000
    t = np.arange(sr // 4) / sr
    audio = np.sin(2 * np.pi * 220 * t) + 0.01 * np.sin(2 * np.pi * 440 * t)
    models = [
        SpectralStateGuidedSynthesis(
            n_states=3, lpc_order=4, frame_size=64, hop_size=16, random_state=91
        )
        for _ in range(2)
    ]
    for model in models:
        model.extract_features(audio, sr)
        model.initialize_hmm_parameters()
    assert np.allclose(models[0].state_means, models[1].state_means)
    assert np.allclose(models[0].transition_matrix, models[1].transition_matrix)


def test_multi_file_training_records_independent_sequences():
    sr = 8000
    t = np.arange(256) / sr
    first = 0.5 * np.sin(2 * np.pi * 220 * t)
    second = 0.5 * np.sin(2 * np.pi * 440 * t)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=4, frame_size=64, hop_size=16, random_state=3
    )
    model.extract_features_from_files([first, second], sample_rate=sr)
    original_frames = len(model.lpc_coefficients)
    assert model.sequence_starts[0] == 0
    assert model.sequence_starts[1] > 0
    model.initialize_hmm_parameters()
    assert len(model.sequence_starts) == 10  # original plus four independent augmentations
    gamma, xi, _ = model._expectation_step(model.lpc_coefficients)
    assert gamma.shape[0] == len(model.lpc_coefficients)
    assert len(xi) == len(model.lpc_coefficients) - len(model.sequence_starts)
    assert len(model.lpc_coefficients) == 5 * original_frames


def test_lpc_stabilizer_projects_unstable_filters_inside_unit_circle():
    unstable = np.array([3.0, -2.0, 0.5], dtype=np.float64)
    stable = SpectralStateGuidedSynthesis._stabilize_lpc(unstable)
    poles = np.roots(np.concatenate(([1.0], stable)))
    assert np.all(np.isfinite(stable))
    assert np.max(np.abs(poles)) < 1.0


def test_gain_stage_sanitizes_nonfinite_samples_and_preserves_silence():
    model = SpectralStateGuidedSynthesis(n_states=1, lpc_order=2, frame_size=8, hop_size=4)
    cleaned = model._apply_gain_floor(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    assert np.all(np.isfinite(cleaned))
    assert np.array_equal(model._apply_gain_floor(np.zeros(8)), np.zeros(8))


def test_markov_sampler_and_state_sequence_validation_are_strict():
    model = _manual_model()
    with pytest.raises(ValueError, match="temperature"):
        model.sample_state_sequence(4, temperature=0)
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_state_sequence(0)
    with pytest.raises(ValueError, match="out-of-range"):
        model._validate_state_sequence([0, 2], 2)
    with pytest.raises(ValueError, match="integer"):
        model._validate_state_sequence([0, 1.5], 2)
    assert len(model.sample_state_sequence(7, seed=1, temperature=0.8)) == 7


def test_sequence_boundaries_and_numeric_gain_settings_reject_bad_types():
    model = _manual_model()
    with pytest.raises(ValueError, match="integer"):
        model._normalize_sequence_starts(4, [0, "bad"])
    with pytest.raises(ValueError, match="minimum_peak"):
        model._apply_gain_floor(np.ones(4), minimum_peak="loud")
    with pytest.raises(ValueError, match="min_state_probability"):
        model.identify_graph_constraints(min_state_probability="many")


def test_bic_candidate_validation_and_overlarge_candidate_fallback():
    model = _manual_model()
    model.lpc_coefficients = np.array(
        [[0.0, 0.0], [0.1, 0.2], [2.0, 2.1], [2.2, 2.0]], dtype=np.float32
    )
    with pytest.raises(ValueError, match="candidate_counts"):
        model._select_optimal_n_states([])
    with pytest.raises(ValueError, match="positive integers"):
        model._select_optimal_n_states([2, 0])
    selected = model._select_optimal_n_states([99])
    assert selected == 2


def _small_trained_model():
    sr = 8000
    t = np.arange(2400) / sr
    audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=4, frame_size=64, hop_size=16, random_state=13
    )
    model.train(audio, sr, n_em_iterations=1)
    return model


def test_generation_is_exact_length_repeatable_and_rate_correct():
    model = _small_trained_model()
    output = model.generate(
        0.123,
        sample_rate=11025,
        method="stochastic",
        seed=44,
    )
    duplicate = model.generate(
        0.123,
        sample_rate=11025,
        method="stochastic",
        seed=44,
    )
    assert len(output) == round(0.123 * 11025)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert np.array_equal(output, duplicate)


def test_preview_does_not_reuse_or_corrupt_emission_workspace():
    model = _small_trained_model()
    sentinel = np.full((model.lpc_order, 5), 17.0)
    model._caches.emission_workspace = sentinel
    audio, sr = model.render_preview(8, seed=9)
    assert sr == model.sample_rate
    assert len(audio) == 8 * model.hop_size
    assert model._caches.emission_workspace is sentinel
    assert np.all(sentinel == 17.0)


def test_training_artifact_export_preserves_sequence_boundaries(tmp_path):
    sr = 8000
    t = np.arange(256) / sr
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=4, frame_size=64, hop_size=16, random_state=17
    )
    model.extract_features_from_files(
        [np.sin(2 * np.pi * 180 * t), np.sin(2 * np.pi * 420 * t)], sr
    )
    model.initialize_hmm_parameters()
    path = tmp_path / "multi.npz"
    model.export_model(path, include_training_artifacts=True)
    restored = SpectralStateGuidedSynthesis.load_model(path)
    assert np.array_equal(restored.sequence_starts, model.sequence_starts)
    assert restored.random_state == model.random_state
    assert restored.lpc_coefficients.shape == model.lpc_coefficients.shape
    assert not list(tmp_path.glob(".*.npz"))


def test_public_training_rejects_invalid_samples_and_em_count():
    model = SpectralStateGuidedSynthesis(n_states=2, lpc_order=2, frame_size=16, hop_size=4)
    with pytest.raises(ValueError, match="non-negative|positive integer"):
        model.iterative_refinement(0)
    with pytest.raises(ValueError, match="positive integer"):
        model._coerce_sample_rate(0)
    with pytest.raises(ValueError, match="positive integer"):
        model._prepare_audio([0.0], 8000.0)


def test_case_insensitive_audio_discovery_deduplicates_paths(tmp_path):
    from train_on_folder import find_audio_files

    (tmp_path / "voice.WAV").write_bytes(b"")
    (tmp_path / "VOICE.wav").write_bytes(b"")
    (tmp_path / "other.Mp3").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.wav").write_bytes(b"")
    direct = find_audio_files([str(tmp_path), str(tmp_path)], ["wav", ".MP3"])
    assert [path.name for path in direct] == ["other.Mp3", "VOICE.wav", "voice.WAV"]
    recursive = find_audio_files([str(tmp_path)], [".wav"], recursive=True)
    assert len(recursive) == 3
