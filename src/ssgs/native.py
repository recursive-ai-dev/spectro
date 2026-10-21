# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ctypes wrapper + benchmark for the compiled SSGS kernels (ssgs_native.so).

Exposes the same call signatures as ``ssgs_core`` so it can drop in as the
fast backend. Run ``python ssgs_native.py`` to (1) verify numeric equivalence
with the SciPy/NumPy path and (2) benchmark the speedup.
"""

import ctypes
import os
import time

import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libssgs_native.so")
_lib = ctypes.CDLL(_LIB_PATH)


def _ptr(arr):
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(ctypes.c_void_p)


# --- signatures ---
_lib.ssgs_lpc_levinson.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_lib.ssgs_gaussian_logprobs.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
_lib.ssgs_forward_backward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.ssgs_decode.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]


def lpc_levinson(autocorr, lpc_order):
    r = np.ascontiguousarray(autocorr, dtype=np.float64)
    a = np.zeros(lpc_order + 1, dtype=np.float64)
    _lib.ssgs_lpc_levinson(_ptr(r), int(lpc_order), a.ctypes.data_as(ctypes.c_void_p))
    return a


def gaussian_log_probs(state_means, state_covariances, observations):
    means = np.ascontiguousarray(state_means, dtype=np.float64)
    covs = np.ascontiguousarray(state_covariances, dtype=np.float64)
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    S, p = means.shape
    T = obs.shape[0]
    out = np.empty((T, S), dtype=np.float64)
    _lib.ssgs_gaussian_logprobs(
        _ptr(means), _ptr(covs), _ptr(obs), S, p, T,
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out


def forward_backward(log_emissions, log_init, log_trans):
    emit = np.ascontiguousarray(log_emissions, dtype=np.float64)
    init = np.ascontiguousarray(log_init, dtype=np.float64)
    trans = np.ascontiguousarray(log_trans, dtype=np.float64)
    T, S = emit.shape
    gamma = np.empty((T, S), dtype=np.float64)
    xi = np.empty((T - 1, S, S), dtype=np.float64)
    loglik = ctypes.c_double()
    _lib.ssgs_forward_backward(
        _ptr(emit), _ptr(init), _ptr(trans), T, S,
        gamma.ctypes.data_as(ctypes.c_void_p),
        xi.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(loglik),
    )
    return gamma, xi, loglik.value


def decode_sequence(transition_cost, smoothness, initial_cost, smoothness_weight, n_frames):
    trans = np.ascontiguousarray(transition_cost, dtype=np.float64)
    smooth = np.ascontiguousarray(smoothness, dtype=np.float64)
    init = np.ascontiguousarray(initial_cost, dtype=np.float64)
    S = trans.shape[0]
    path = np.empty(n_frames, dtype=np.int32)
    _lib.ssgs_decode(
        _ptr(trans), _ptr(smooth), _ptr(init), float(smoothness_weight), S, int(n_frames),
        path.ctypes.data_as(ctypes.c_void_p),
    )
    return path.tolist()


# ---------------------------------------------------------------------------
def verify():
    from ssgs import SpectralStateGuidedSynthesis
    from .signals import create_simple_test_signal

    sr = 12000
    sig = create_simple_test_signal(sample_rate=sr, duration=1.5, seed=42)
    m = SpectralStateGuidedSynthesis(n_states=8, lpc_order=10, frame_size=512, hop_size=128)
    m.extract_features(sig, sr)
    m.initialize_hmm_parameters()
    m.iterative_refinement(3)
    m.identify_graph_constraints()

    frame = m.training_frames[0]
    ref = m._compute_lpc(frame)
    core = lpc_levinson(m._autocorrelation(frame), m.lpc_order)
    assert np.allclose(ref, core, atol=1e-5), "C LPC mismatch"
    print("  [ok] C LPC matches ssgs")

    obs = m.lpc_coefficients
    ref_lp = m._compute_emission_log_probs(obs)
    core_lp = gaussian_log_probs(m.state_means, m.state_covariances, obs)
    assert np.allclose(ref_lp, core_lp, atol=1e-4), "C emission mismatch"
    print("  [ok] C Gaussian emission matches ssgs")

    emit = m._compute_emission_log_probs(obs)
    init = np.log(m.initial_probabilities + 1e-12)
    trans = np.log(m.transition_matrix + 1e-12)
    g, xi, ll = forward_backward(emit, init, trans)
    g2, xi2, ll2 = m._expectation_step(obs)
    dg = np.max(np.abs(g - g2))
    dxi = np.max(np.abs(xi - xi2))
    dll = abs(ll - ll2)
    print(f"  [fwd-bwd] max|dgamma|={dg:.2e}  max|dxi|={dxi:.2e}  |dloglik|={dll:.2e}")
    assert dg < 1e-3 and dxi < 1e-3, "C forward-backward mismatch too large"
    print("  [ok] C forward-backward matches ssgs")

    tc = m._transition_cost_matrix()
    sm = m._spectral_smoothness_matrix()
    ref_path = m._decode_state_sequence_cached(20)
    core_path = decode_sequence(tc, sm, -np.log(m.initial_probabilities + 1e-12), m.smoothness_weight, 20)
    assert ref_path == core_path, "C decode mismatch"
    print("  [ok] C decode matches ssgs")
    print("ALL C KERNEL VERIFICATIONS PASSED")


def benchmark():
    from ssgs import SpectralStateGuidedSynthesis
    from .signals import create_simple_test_signal

    sr = 12000
    sig = create_simple_test_signal(sample_rate=sr, duration=1.5, seed=7)
    m = SpectralStateGuidedSynthesis(n_states=34, lpc_order=12, frame_size=1024, hop_size=256)
    m.extract_features(sig, sr)
    m.initialize_hmm_parameters()
    m.iterative_refinement(2)
    m.identify_graph_constraints()

    obs = m.lpc_coefficients
    tc = m._transition_cost_matrix()
    sm = m._spectral_smoothness_matrix()
    init = -np.log(m.initial_probabilities + 1e-12)
    frames = m.training_frames
    N_FRAMES = min(400, frames.shape[0])

    def timeit(fn, reps=20):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    # LPC over many frames
    t_scipy_lpc = timeit(lambda: [m._compute_lpc(frames[i]) for i in range(N_FRAMES)])
    t_c_lpc = timeit(lambda: [lpc_levinson(m._autocorrelation(frames[i]), m.lpc_order) for i in range(N_FRAMES)])

    # Emission
    t_scipy_em = timeit(lambda: m._compute_emission_log_probs(obs))
    t_c_em = timeit(lambda: gaussian_log_probs(m.state_means, m.state_covariances, obs))

    # Decode (large path - the Python triple loop is the worst offender)
    t_scipy_dec = timeit(lambda: m._decode_state_sequence_cached(2000))
    t_c_dec = timeit(lambda: decode_sequence(tc, sm, init, m.smoothness_weight, 2000))

    print(f"\n{'kernel':<22}{'scipy/numpy':>14}{'C':>14}{'speedup':>12}")
    print(f"{'-'*62}")
    print(f"{'LPC x%d frames':<22}{t_scipy_lpc:>14.4f}{t_c_lpc:>14.4f}{t_scipy_lpc/t_c_lpc:>11.1f}x")
    print(f"{'emission (T x S)':<22}{t_scipy_em:>14.4f}{t_c_em:>14.4f}{t_scipy_em/t_c_em:>11.1f}x")
    print(f"{'decode 2000 frames':<22}{t_scipy_dec:>14.4f}{t_c_dec:>14.4f}{t_scipy_dec/t_c_dec:>11.1f}x")


if __name__ == "__main__":
    print("Verifying C kernels against ssgs.py ...")
    verify()
    print("\nBenchmarking (compiled C vs SciPy/NumPy) ...")
    benchmark()
