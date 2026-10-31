"""Backend regressions: native/reference equivalence, reporting, silence.

These tests require the compiled ``libssgs_native`` kernel (built by ``make``);
they skip on a checkout without it instead of silently passing.
"""

import re

import numpy as np
import pytest

from ssgs import SpectralStateGuidedSynthesis, core, native
from ssgs.signals import create_simple_test_signal


pytestmark = pytest.mark.skipif(
    native._lib is None, reason="libssgs_native.so is not built (run `make`)"
)


SAMPLE_RATE = 8000


def _decode_objective(path, transition_cost, smoothness, initial_cost, weight):
    path = np.asarray(path, dtype=np.int64)
    smooth_sum = float(np.sum(smoothness[path[:-1], path[1:]])) if path.size > 1 else 0.0
    total = float(initial_cost[path[0]]) + float(
        np.sum(transition_cost[path[:-1], path[1:]])
    )
    return smooth_sum, total


@pytest.mark.parametrize("seed", range(8))
def test_native_kernels_match_the_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    lpc_order = int(rng.choice([1, 2, 5, 12]))
    states = int(rng.choice([1, 2, 4, 7]))
    frames = int(rng.choice([1, 3, 17, 64]))

    autocorr = rng.random(lpc_order + 1) + 0.05
    autocorr[0] = autocorr[0] + lpc_order
    np.testing.assert_allclose(
        native.lpc_levinson(autocorr, lpc_order),
        core.lpc_coefficients(autocorr, lpc_order),
        rtol=1e-10,
        atol=1e-12,
    )

    means = rng.standard_normal((states, lpc_order))
    basis = rng.standard_normal((states, lpc_order, lpc_order))
    covariances = np.einsum("sij,skj->sik", basis, basis) + np.eye(lpc_order) * 0.5
    observations = rng.standard_normal((frames, lpc_order))
    np.testing.assert_allclose(
        native.gaussian_log_probs(means, covariances, observations),
        core.gaussian_log_probs(means, covariances, observations),
        rtol=1e-9,
        atol=1e-9,
    )

    log_init = np.log(rng.random(states) + 0.05)
    log_trans = np.log(rng.random((states, states)) + 0.05)
    log_emit = -rng.random((frames, states))
    gamma_ref, xi_ref, ll_ref = core.forward_backward(log_emit, log_init, log_trans)
    gamma, xi, loglik = native.forward_backward(log_emit, log_init, log_trans)
    np.testing.assert_allclose(gamma, gamma_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xi, xi_ref, rtol=1e-9, atol=1e-12)
    assert loglik == pytest.approx(ll_ref, rel=1e-10, abs=1e-10)

    weight = float(rng.choice([0.0, 0.25, 1.5]))
    transition_cost = rng.random((states, states))
    smoothness = rng.random((states, states))
    initial_cost = rng.random(states)
    decode_frames = int(rng.choice([1, 2, 25]))

    reference_path = core.decode_sequence(
        transition_cost, smoothness, initial_cost, weight, decode_frames
    )
    native_path = native.decode_sequence(
        transition_cost, smoothness, initial_cost, weight, decode_frames
    )
    assert len(native_path) == decode_frames
    assert all(0 <= state < states for state in native_path)
    assert _decode_objective(
        native_path, transition_cost, smoothness, initial_cost, weight
    ) == pytest.approx(
        _decode_objective(
            reference_path, transition_cost, smoothness, initial_cost, weight
        ),
        rel=1e-12,
        abs=1e-12,
    )


def test_benchmark_reports_two_distinct_backends(capsys):
    native.benchmark()
    output = capsys.readouterr().out

    assert "%d" not in output, "benchmark must not print raw format specifiers"
    assert "reference" in output and "native" in output
    rows = [
        line
        for line in output.splitlines()
        if "frame" in line or "emission" in line or "decode" in line
    ]
    assert len(rows) == 3
    for line in rows:
        match = re.search(
            r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)x", line
        )
        assert match, f"unparsable benchmark row: {line!r}"
        reference, native_time, speedup = (float(value) for value in match.groups())
        assert reference > 0 and native_time > 0
        # Times are printed with four decimals, so millisecond kernels carry
        # up to a few percent of display rounding before the ratio is formed.
        assert speedup == pytest.approx(reference / native_time, rel=0.2)


def test_library_calls_are_silent(capsys):
    audio = create_simple_test_signal(sample_rate=SAMPLE_RATE, duration=0.4, seed=42)
    model = SpectralStateGuidedSynthesis(
        n_states=6, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    model.extract_features(audio, SAMPLE_RATE)
    model.initialize_hmm_parameters()
    model.iterative_refinement(1, verbose=False)
    model.identify_graph_constraints()
    model._select_optimal_n_states([2, 3])
    model.synthesize_audio(0.05, SAMPLE_RATE, seed=1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_progress_information_is_logged(caplog):
    audio = create_simple_test_signal(sample_rate=SAMPLE_RATE, duration=0.3, seed=42)
    model = SpectralStateGuidedSynthesis(
        n_states=3, lpc_order=6, frame_size=128, hop_size=64, random_state=0
    )
    model.extract_features(audio, SAMPLE_RATE)
    model.initialize_hmm_parameters()
    with caplog.at_level("INFO", logger="ssgs.model"):
        model._select_optimal_n_states([2, 3])
    assert any("optimal n_states" in record.message.lower() for record in caplog.records)
