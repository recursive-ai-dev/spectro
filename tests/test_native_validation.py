"""Exercise the public C boundary without handing unsafe buffers to ctypes."""

import numpy as np
import pytest

from ssgs import native
from ssgs.core import forward_backward, gaussian_log_probs


@pytest.mark.parametrize("values,order", [([1.0], 4), ([[1.0, 0.2]], 1), ([1.0, np.nan], 1), ([1.0], 0)])
def test_lpc_rejects_invalid_buffer(values, order):
    with pytest.raises(ValueError):
        native.lpc_levinson(values, order)


def test_gaussian_rejects_misaligned_buffers():
    with pytest.raises(ValueError, match="covariances"):
        native.gaussian_log_probs(np.zeros((2, 3)), np.eye(3), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="observations"):
        native.gaussian_log_probs(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), np.zeros((4, 2)))


def test_gaussian_repairs_indefinite_covariances_via_reference_backend():
    means = np.zeros((1, 2))
    covariances = np.array([[[1.0, 2.0], [2.0, 1.0]]])
    observations = np.array([[0.1, -0.2]])
    result = native.gaussian_log_probs(means, covariances, observations)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, gaussian_log_probs(means, covariances, observations))


def test_forward_backward_handles_unreachable_states_and_exact_zero_transitions():
    with np.errstate(divide="ignore"):
        initial = np.log([1.0, 0.0])
        transitions = np.log(np.eye(2))
    emissions = np.array([[-0.2, -0.1], [-0.4, -0.3]])
    actual = native.forward_backward(emissions, initial, transitions)
    expected = forward_backward(emissions, initial, transitions)
    for observed, reference in zip(actual, expected):
        assert np.allclose(observed, reference)


@pytest.mark.parametrize("emissions,initial,transitions", [
    (np.zeros((0, 2)), np.zeros(2), np.zeros((2, 2))),
    (np.zeros((3, 2)), np.zeros(3), np.zeros((2, 2))),
    (np.full((1, 2), np.nan), np.zeros(2), np.zeros((2, 2))),
])
def test_forward_backward_rejects_invalid_buffers(emissions, initial, transitions):
    with pytest.raises(ValueError):
        native.forward_backward(emissions, initial, transitions)


def test_forward_backward_reports_impossible_observations():
    with pytest.raises(FloatingPointError):
        native.forward_backward(np.full((1, 2), -np.inf), np.log([0.5, 0.5]), np.log(np.full((2, 2), 0.5)))


@pytest.mark.parametrize("frames", [0, -1, 1.5, True, 2**32])
def test_decoder_rejects_invalid_frame_count(frames):
    with pytest.raises(ValueError):
        native.decode_sequence(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), 0.5, frames)


def test_decoder_rejects_wrong_shapes_before_native_call():
    with pytest.raises(ValueError):
        native.decode_sequence(np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), 0.5, 4)
    with pytest.raises(ValueError):
        native.decode_sequence(np.zeros((2, 2)), np.zeros((3, 3)), np.zeros(2), 0.5, 4)


def test_decoder_rejects_workspace_index_overflow_before_allocation():
    with pytest.raises(ValueError, match="indexing limit"):
        native.decode_sequence(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), 0.5, 2**30)
