#!/usr/bin/env python3
"""Benchmark regression guard: native/core equivalence + full pipeline integration."""
import numpy as np


def _assert_equivalence():
    from ssgs_core import _self_test
    print("[regression] ssgs_core._self_test ...")
    _self_test()


def _assert_native_integration():
    from ssgs import SpectralStateGuidedSynthesis
    from ssgs_core import (
        lpc_coefficients, gaussian_log_probs,
        decode_sequence, forward_backward,
    )
    import ssgs_native

    print("[regression] full pipeline integration (native vs core) ...")
    sr = 12000
    sig = np.random.default_rng(42).standard_normal(sr * 2).astype(np.float32)
    m = SpectralStateGuidedSynthesis(
        n_states=8, lpc_order=10, frame_size=512, hop_size=128
    )
    m.extract_features(sig, sr)
    m.initialize_hmm_parameters()
    m.iterative_refinement(2)
    m.identify_graph_constraints()

    obs = m.lpc_coefficients
    # 1) LPC equivalence
    frame = m.training_frames[0]
    autocorr = m._autocorrelation(frame)
    core_lpc = lpc_coefficients(autocorr, m.lpc_order)
    native_lpc = m._compute_lpc(frame)
    assert np.allclose(core_lpc, native_lpc, atol=1e-5), "LPC regression"

    # 2) Emission equivalence
    core_em = gaussian_log_probs(
        m.state_means, m.state_covariances, obs
    )
    native_em = m._compute_emission_log_probs(obs)
    assert np.allclose(core_em, native_em, atol=1e-4), "Emission regression"

    # 3) Decode equivalence (cached path for stability)
    tc = m._transition_cost_matrix()
    sm = m._spectral_smoothness_matrix()
    init = -np.log(m.initial_probabilities + 1e-12)
    core_path = decode_sequence(tc, sm, init, m.smoothness_weight, 20)
    native_path = m._decode_state_sequence_cached(20)
    assert core_path == native_path, "Decode regression"

    # 4) Forward-backward equivalence
    emit = m._compute_emission_log_probs(obs[:20])
    init_arr = np.log(m.initial_probabilities + 1e-12)
    trans_arr = np.log(m.transition_matrix + 1e-12)
    core_gamma, core_xi, core_ll = forward_backward(
        emit, init_arr, trans_arr
    )
    native_gamma, native_xi, native_ll = m._expectation_step(obs[:20])
    assert np.allclose(core_gamma, native_gamma, atol=1e-3), \
        "Gamma regression"
    assert np.allclose(core_xi, native_xi, atol=1e-3), \
        "Xi regression"
    assert abs(core_ll - native_ll) < 1e-3, "Log-likelihood regression"

    print("[regression] all native/core assertions passed.")


def _assert_native_load():
    print("[regression] native library load check ...")
    import ssgs_native
    assert ssgs_native._lib is not None, \
        "libssgs_native.so missing or unloadable"
    print("[regression] native library loaded.")


def test_native_load():
    _assert_native_load()


def test_core_equivalence():
    _assert_equivalence()


def test_native_integration():
    _assert_native_integration()


if __name__ == "__main__":
    _assert_native_load()
    _assert_equivalence()
    _assert_native_integration()
    print("[regression] ALL GUARDS PASSED.")
