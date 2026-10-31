"""Resource-bound regressions: render limits, streamed metadata, batching.

The memory assertions compare transient allocation against the size of the
data being processed, so they stay meaningful on machines with plenty of RAM.
Thresholds are set roughly four times above the measured post-fix peak and
below the measured pre-fix peak.
"""

import tracemalloc

import numpy as np
import pytest
from scipy import signal

from ssgs import SpectralStateGuidedSynthesis, cli, workflows
from ssgs.signals import create_simple_test_signal


SAMPLE_RATE = 8000


def _tiny_model(seed=0):
    audio = create_simple_test_signal(sample_rate=SAMPLE_RATE, duration=0.4, seed=42)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=seed
    )
    model.train(audio, SAMPLE_RATE, n_em_iterations=1, verbose=False)
    return model


# --------------------------------------------------------------------------- #
# Render limits
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "duration, sample_rate, message",
    [
        (0.5, 768_001, "sample_rate must not exceed"),
        (SpectralStateGuidedSynthesis.MAX_DURATION_SECONDS + 1, SAMPLE_RATE, "limit per render"),
        (3600, 768_000, "sample limit"),
    ],
)
def test_synthesize_audio_rejects_oversized_renders(duration, sample_rate, message):
    model = _tiny_model()
    with pytest.raises(ValueError, match=message):
        model.synthesize_audio(duration, sample_rate)


def test_render_limits_are_not_applied_to_valid_requests():
    model = _tiny_model()
    audio = model.synthesize_audio(0.05, SpectralStateGuidedSynthesis.MAX_SAMPLE_RATE)
    expected = int(round(0.05 * SpectralStateGuidedSynthesis.MAX_SAMPLE_RATE))
    assert audio.size == expected
    assert np.all(np.isfinite(audio))


def test_workflow_fails_before_loading_an_missing_model(tmp_path):
    output = tmp_path / "out.wav"
    with pytest.raises(ValueError, match="limit per render"):
        workflows.generate_audio(
            tmp_path / "does-not-exist.npz",
            output,
            duration=SpectralStateGuidedSynthesis.MAX_DURATION_SECONDS + 5,
            sample_rate=SAMPLE_RATE,
        )
    assert not output.exists()


def test_cli_reports_render_limits_without_traceback(tmp_path, capsys):
    model = _tiny_model()
    model_path = tmp_path / "model.npz"
    model.export_model(model_path)
    output = tmp_path / "out.wav"

    exit_code = cli.main(
        ["generate", str(model_path), "--output", str(output), "--duration", "100000"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "duration" in captured.err and "limit per render" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.npz"]


# --------------------------------------------------------------------------- #
# Frame metadata: equivalence and bounded allocation
# --------------------------------------------------------------------------- #

def _reference_metadata(frames, sample_rate):
    """The pre-optimisation whole-matrix formulation, kept as an oracle."""
    count, frame_len = frames.shape
    energy = np.sum(frames * frames, axis=1)
    zero_crossings = np.count_nonzero(
        np.signbit(frames[:, :-1]) != np.signbit(frames[:, 1:]), axis=1
    )
    metadata = {
        "energy": energy.astype(np.float32, copy=False),
        "rms": np.sqrt(energy / frame_len).astype(np.float32, copy=False),
        "zero_crossings": zero_crossings.astype(np.int32, copy=False),
    }
    if sample_rate:
        window = np.hanning(frame_len)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
        power = spectrum + 1e-12
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sample_rate))
        centroid = (power * freqs).sum(axis=1) / power.sum(axis=1)
        metadata["spectral_centroid"] = centroid.astype(np.float32, copy=False)
        metadata["spectral_bandwidth"] = np.sqrt(
            ((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / power.sum(axis=1)
        ).astype(np.float32, copy=False)
        metadata["dominant_frequency"] = freqs[np.argmax(power, axis=1)].astype(
            np.float32, copy=False
        )
    return metadata


@pytest.mark.parametrize("frame_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("sample_rate", [0, 8000, 22050])
@pytest.mark.parametrize("count", [1, 3, 255, 1025])
def test_frame_metadata_matches_reference_bit_for_bit(frame_size, sample_rate, count):
    rng = np.random.default_rng(frame_size + count)
    frames = (rng.standard_normal((count, frame_size)) * 0.2).astype(np.float32)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=frame_size, hop_size=frame_size // 2
    )
    model.training_frames = frames
    model.lpc_coefficients = None
    model.residual_signals = None

    model._compute_frame_metadata(sample_rate)
    expected = _reference_metadata(frames, sample_rate)

    for key, values in expected.items():
        actual = model.frame_metadata[key]
        assert actual.dtype == np.asarray(values).dtype, key
        assert np.array_equal(actual, values), key


def test_frame_metadata_transient_allocation_is_chunk_bounded():
    frame_size = 2048
    count = 3000
    rng = np.random.default_rng(7)
    frames = (rng.standard_normal((count, frame_size)) * 0.2).astype(np.float32)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=frame_size, hop_size=1024
    )
    model.training_frames = frames
    model.lpc_coefficients = None
    model.residual_signals = None

    tracemalloc.start()
    try:
        model._compute_frame_metadata(SAMPLE_RATE)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Whole-matrix processing needed about 7x the frame buffer; chunked
    # processing stays well below the frame buffer itself.
    assert peak < 4 * frames.nbytes, (
        f"frame metadata peak {peak / 2**20:.1f} MB exceeds 4x the "
        f"{frames.nbytes / 2**20:.1f} MB frame buffer"
    )


# --------------------------------------------------------------------------- #
# Frame batching: equivalence and bounded allocation
# --------------------------------------------------------------------------- #

def test_frame_batch_matches_scalar_reference():
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=256, hop_size=128, random_state=0
    )
    mel = model._mel_filter_bank(SAMPLE_RATE, model.frame_size)
    erb = model._erb_filter_bank(SAMPLE_RATE, model.frame_size)
    window = np.hanning(model.frame_size).astype(np.float32)
    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((40, model.frame_size)) * 0.2).astype(np.float32)
    frames[7] = 0.0  # gated row, exercising the retained mask
    frames[21] = 0.0

    batch = model._process_frame_batch(frames, mel, erb, window)

    kept_frames, kept_lpc, kept_residual, kept_perceptual = [], [], [], []
    previous = None
    for frame in frames:
        if np.dot(frame, frame) < 1e-10:
            continue
        windowed = frame * window
        coeffs = model._compute_lpc(windowed)
        residual = signal.lfilter(coeffs, [1.0], windowed)
        magnitude = np.abs(np.fft.rfft(windowed))
        vector, previous = model._perceptual_feature_bundle(
            magnitude, mel, erb, previous
        )
        kept_frames.append(frame)
        kept_lpc.append(coeffs[1:])
        kept_residual.append(residual[: model.hop_size])
        kept_perceptual.append(vector)

    expected = (
        np.asarray(kept_frames, dtype=np.float32),
        np.asarray(kept_lpc, dtype=np.float32),
        np.asarray(kept_residual, dtype=np.float32),
        np.asarray(kept_perceptual, dtype=np.float32),
    )
    for index, (actual, reference) in enumerate(zip(batch[:4], expected)):
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference, err_msg=f"output {index}")
    np.testing.assert_array_equal(batch[4], expected[3][:, -1])

    # A quiet batch returns empty, correctly-shaped arrays.
    empty = model._process_frame_batch(np.zeros((3, model.frame_size), np.float32), mel, erb, window)
    assert empty[0].shape == (0, model.frame_size)
    assert empty[4].shape == (0,)


def test_frame_batch_transient_allocation_is_bounded():
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=256, hop_size=128, random_state=0
    )
    mel = model._mel_filter_bank(SAMPLE_RATE, model.frame_size)
    erb = model._erb_filter_bank(SAMPLE_RATE, model.frame_size)
    window = np.hanning(model.frame_size).astype(np.float32)
    rng = np.random.default_rng(3)
    frames = (rng.standard_normal((1200, model.frame_size)) * 0.2).astype(np.float32)

    tracemalloc.start()
    try:
        outputs = model._process_frame_batch(frames, mel, erb, window)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    output_bytes = sum(array.nbytes for array in outputs)
    assert peak < 2 * output_bytes, (
        f"batch peak {peak / 2**20:.1f} MB exceeds 2x the "
        f"{output_bytes / 2**20:.1f} MB of results"
    )
