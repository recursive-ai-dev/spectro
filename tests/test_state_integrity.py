"""State-integrity regressions: archives, re-training, augmentation, lookups.

Each test here reproduces a failure that was observed before the corresponding
fix and asserts the corrected contract, not merely that a call succeeds.
"""

import json
import warnings

import numpy as np
import pytest

from ssgs import SpectralStateGuidedSynthesis


SAMPLE_RATE = 8000


def _tone(seconds, frequency=220.0, amplitude=0.45):
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * frequency * times)


def _trained_model(seed=0):
    audio = _tone(0.35) + 0.2 * np.sin(2 * np.pi * 660 * np.arange(2800) / SAMPLE_RATE)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=seed
    )
    model.extract_features(audio, SAMPLE_RATE)
    model.initialize_hmm_parameters()
    model.iterative_refinement(1, verbose=False)
    model.identify_graph_constraints()
    return model, audio


def _adapted_archive(model, audio, tmp_path):
    model.adapt_to_audio(audio, SAMPLE_RATE, adaptation_rate=0.5, seed=1)
    path = tmp_path / "adapted.npz"
    model.export_model(path, include_training_artifacts=False)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    return path, payload


def _write(tmp_path, payload, name="tampered.npz"):
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


TAMPER_CASES = {
    "short_counts": lambda p: p.update(
        {"adaptive_state_frame_counts": p["adaptive_state_frame_counts"][:1]}
    ),
    "nan_counts": lambda p: p.update(
        {"adaptive_state_frame_counts": np.full(3, np.nan)}
    ),
    "bad_transition_shape": lambda p: p.update(
        {"adaptive_transition_counts": np.zeros((2, 2))}
    ),
    "bad_mean_shape": lambda p: p.update(
        {"adaptive_mean_accumulator": np.zeros((3, 5))}
    ),
    "nan_second_moment": lambda p: p.update(
        {"adaptive_second_moment": np.full_like(p["adaptive_second_moment"], np.nan)}
    ),
    "negative_initial_counts": lambda p: p.update(
        {"adaptive_initial_counts": np.array([-1.0, 0.0, 0.0])}
    ),
    "negative_total": lambda p: p.update(
        {"adaptive_total_frames": np.array(-1.0)}
    ),
    "nan_total": lambda p: p.update(
        {"adaptive_total_frames": np.array(np.nan)}
    ),
}


@pytest.mark.parametrize("case", sorted(TAMPER_CASES))
def test_npz_adaptive_statistics_tampering_is_rejected(tmp_path, case):
    model, audio = _trained_model()
    _, payload = _adapted_archive(model, audio, tmp_path)
    TAMPER_CASES[case](payload)

    with pytest.raises(ValueError, match="Adaptive"):
        SpectralStateGuidedSynthesis.load_model(_write(tmp_path, payload))


def test_npz_adaptive_statistics_round_trip_is_exact(tmp_path):
    model, audio = _trained_model()
    path, _ = _adapted_archive(model, audio, tmp_path)

    reloaded = SpectralStateGuidedSynthesis.load_model(path)
    original = model._adaptive_stats
    restored = reloaded._adaptive_stats
    assert restored is not None
    assert np.array_equal(restored.state_frame_counts, original.state_frame_counts)
    assert np.array_equal(restored.transition_counts, original.transition_counts)
    assert np.array_equal(restored.mean_accumulator, original.mean_accumulator)
    assert np.array_equal(
        restored.second_moment_accumulator, original.second_moment_accumulator
    )
    assert np.array_equal(restored.initial_counts, original.initial_counts)
    assert restored.total_frames == original.total_frames

    # The reloaded statistics must be usable, not merely present.
    reloaded.adapt_to_audio(audio, SAMPLE_RATE, adaptation_rate=0.5, seed=2)
    assert np.all(np.isfinite(reloaded.state_means))
    assert np.all(np.isfinite(reloaded.state_covariances))


def test_checkpoint_adaptive_statistics_tampering_is_rejected(tmp_path):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    model, audio = _trained_model()
    model.adapt_to_audio(audio, SAMPLE_RATE, adaptation_rate=0.5, seed=1)
    path = tmp_path / "model.safetensors"
    model.save_checkpoint(path, include_training_artifacts=False)
    tensors = safetensors_numpy.load_file(path)
    raw = path.read_bytes()
    header = json.loads(raw[8 : 8 + int.from_bytes(raw[:8], "little")])
    metadata = {
        str(key): str(value) for key, value in header.get("__metadata__", {}).items()
    }

    broken = dict(tensors)
    broken["adaptive_state_frame_counts"] = tensors["adaptive_state_frame_counts"][:1]
    tampered = tmp_path / "broken.safetensors"
    safetensors_numpy.save_file(
        {key: np.ascontiguousarray(value) for key, value in broken.items()}, tampered, metadata=metadata
    )
    with pytest.raises(ValueError, match="Adaptive"):
        SpectralStateGuidedSynthesis.load_checkpoint(tampered)


def test_checkpoint_adaptive_statistics_round_trip_is_exact(tmp_path):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    model, audio = _trained_model()
    model.adapt_to_audio(audio, SAMPLE_RATE, adaptation_rate=0.5, seed=1)
    path = tmp_path / "model.safetensors"
    model.save_checkpoint(path, include_training_artifacts=False)

    tensors = safetensors_numpy.load_file(path)
    assert tensors["adaptive_state_frame_counts"].dtype == np.float64
    reloaded = SpectralStateGuidedSynthesis.load_checkpoint(path)
    np.testing.assert_array_equal(
        reloaded._adaptive_stats.state_frame_counts,
        model._adaptive_stats.state_frame_counts,
    )
    np.testing.assert_array_equal(
        reloaded._adaptive_stats.transition_counts,
        model._adaptive_stats.transition_counts,
    )
    assert reloaded._adaptive_stats.total_frames == model._adaptive_stats.total_frames


def test_retraining_resets_per_run_state():
    audio_one = _tone(0.4, 220.0)
    audio_two = _tone(0.3, 330.0)

    reference = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    reference.extract_features(audio_two, SAMPLE_RATE)
    raw_frames = len(reference.lpc_coefficients)

    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    model.train(audio_one, SAMPLE_RATE, n_em_iterations=1, verbose=False)
    assert model.augmentation_applied
    assert len(model.training_frames) > raw_frames

    model.iterative_refinement(1, preview_every=1, verbose=False)
    model.adapt_to_audio(audio_one, SAMPLE_RATE, adaptation_rate=0.5, seed=3)
    assert model._adaptive_stats is not None
    assert model.preview_snapshots

    model.train(audio_two, SAMPLE_RATE, n_em_iterations=1, verbose=False)

    # The new training set is augmented again instead of inheriting the flag,
    # the previous run's sufficient statistics are gone, and previews from the
    # old run are not reused.
    assert model.augmentation_applied
    assert len(model.training_frames) == len(model.lpc_coefficients)
    assert len(model.lpc_coefficients) == 5 * raw_frames
    assert model._adaptive_stats is None
    assert model.preview_snapshots == []


def test_failed_extraction_leaves_the_model_usable():
    model, _ = _trained_model()
    frames_before = model.training_frames.copy()
    lpc_before = model.lpc_coefficients.copy()

    with pytest.raises(ValueError, match="No valid frames"):
        model.extract_features(np.zeros(4000), SAMPLE_RATE)

    assert np.array_equal(model.training_frames, frames_before)
    assert np.array_equal(model.lpc_coefficients, lpc_before)
    audio = model.synthesize_audio(0.05, SAMPLE_RATE, seed=1)
    assert np.all(np.isfinite(audio)) and audio.size == 400


def test_fidelity_warns_only_when_residuals_are_missing(tmp_path):
    model, audio = _trained_model()
    assert model.residual_signals is not None

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.synthesize_audio(0.05, SAMPLE_RATE, fidelity=0.5, seed=1)
    assert not [item for item in recorded if "fidelity" in str(item.message)]

    stripped = tmp_path / "stripped.npz"
    model.export_model(stripped, include_training_artifacts=False)
    reloaded = SpectralStateGuidedSynthesis.load_model(stripped)
    assert reloaded.residual_signals is None

    with pytest.warns(RuntimeWarning, match="fidelity > 0 was requested"):
        blended = reloaded.synthesize_audio(0.05, SAMPLE_RATE, fidelity=0.5, seed=1)
    assert blended.size == int(round(0.05 * SAMPLE_RATE))
    assert np.all(np.isfinite(blended))


def test_augmentation_boundaries_follow_retained_frames():
    gap_audio = np.concatenate(
        [_tone(0.35), np.zeros(int(SAMPLE_RATE * 0.25)), _tone(0.35, 330.0)]
    )
    second = _tone(0.4, 440.0)

    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    model.extract_features_from_files([gap_audio, second], SAMPLE_RATE)
    original_slices = model._sequence_slices(len(model.training_frames))

    # Simulate a batch processor that drops one frame per batch.  Boundary
    # derivation must follow the frames that actually survived instead of
    # assuming a 1:1 correspondence with the input.
    original_batch = model._process_frame_batch
    original_mask = model._retained_frame_mask

    def dropping_batch(frames, *args, **kwargs):
        output = original_batch(frames, *args, **kwargs)
        return tuple(item[1:] for item in output) if len(output[0]) > 1 else output

    def dropping_mask(frames, *args, **kwargs):
        mask = original_mask(frames, *args, **kwargs)
        return mask[1:] if len(mask) > 1 else mask

    model._process_frame_batch = dropping_batch
    model._retained_frame_mask = dropping_mask
    model.initialize_hmm_parameters()

    starts = model.sequence_starts
    total = len(model.training_frames)
    assert starts[0] == 0
    assert np.all(np.diff(starts) > 0)
    assert starts[-1] < total
    # Two source recordings, augmented into five batches, no empty sequence.
    assert len(starts) == len(original_slices) * 5
    counts = np.diff(np.append(starts, total))
    assert counts.min() > 0
    assert counts.sum() == total


def test_augmentation_boundaries_are_exact_without_drops():
    audio = _tone(0.4)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    model.extract_features(audio, SAMPLE_RATE)
    raw = len(model.training_frames)
    model.initialize_hmm_parameters()
    assert list(model.sequence_starts) == [0, raw, 2 * raw, 3 * raw, 4 * raw]
    assert len(model.training_frames) == 5 * raw


def test_frame_metadata_indices_are_validated():
    model, _ = _trained_model()
    total = len(model.frame_metadata["rms"])

    with pytest.raises(IndexError):
        model.get_frame_metadata("rms", indices=[-1])
    with pytest.raises(IndexError):
        model.get_frame_metadata(indices=[total])
    with pytest.raises(ValueError):
        model.get_frame_metadata("rms", indices=[1.5])
    with pytest.raises(ValueError):
        model.get_frame_metadata("rms", indices=[True])

    expected = model.frame_metadata["rms"][[2, 4]]
    np.testing.assert_array_equal(model.get_frame_metadata("rms", indices=[2, 4]), expected)
    np.testing.assert_array_equal(
        model.get_frame_metadata(indices=np.array([2, 4]))["rms"], expected
    )
    scalar = model.get_frame_metadata("rms", indices=3)
    assert scalar == model.frame_metadata["rms"][3]


def test_graph_component_count_describes_retained_states():
    model, _ = _trained_model()
    model.transition_matrix = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    model.initial_probabilities = np.array([0.5, 0.5, 0.0])

    pruned = model.identify_graph_constraints(1e-4)
    # State 2 is dormant (zero stationary probability) and must not be counted
    # as its own strongly connected component.
    assert list(pruned["active_states"]) == [0, 1]
    assert pruned["n_components"] == 1
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)

    connected = _trained_model()[0]
    connected.transition_matrix = np.full((3, 3), 0.1) + np.diag([0.7, 0.7, 0.7])
    connected.initial_probabilities = np.full(3, 1.0 / 3.0)
    fully_active = connected.identify_graph_constraints(0.0)
    assert len(fully_active["active_states"]) == connected.n_states
    assert fully_active["n_components"] == 1
