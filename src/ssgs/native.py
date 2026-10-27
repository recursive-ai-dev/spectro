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
# Load lazily-tolerantly: the .so is a build artifact (``make``) and is not
# shipped in the repo, so importing this module on a fresh clone must not
# crash -- callers (e.g. ssgs.model) check ``_lib is not None`` and fall back
# to the pure-NumPy core.  Previously the unconditional CDLL raised OSError
# at import time, defeating that fallback for direct importers.
try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError:
    _lib = None


def _require_lib():
    if _lib is None:
        raise RuntimeError(
            "libssgs_native.so is not available; build it with `make` "
            "or use the pure-NumPy ssgs.core backend"
        )
    return _lib


def _ptr(arr):
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(ctypes.c_void_p)


def _finite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


if _lib is not None:
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
    # Validate before crossing the ctypes boundary: the C kernel trusts its
    # arguments, so a shape mismatch would read out of bounds and return
    # garbage (or segfault) instead of raising.  These checks mirror
    # ssgs.core.lpc_coefficients.
    lpc_order = _positive_int("lpc_order", lpc_order)
    r = np.ascontiguousarray(autocorr, dtype=np.float64)
    if r.ndim != 1 or r.shape[0] != lpc_order + 1:
        raise ValueError("autocorr must be 1-D with length lpc_order + 1")
    _finite("autocorr", r)
    lib = _require_lib()
    a = np.zeros(lpc_order + 1, dtype=np.float64)
    lib.ssgs_lpc_levinson(_ptr(r), lpc_order, a.ctypes.data_as(ctypes.c_void_p))
    return a


def gaussian_log_probs(state_means, state_covariances, observations):
    means = np.ascontiguousarray(state_means, dtype=np.float64)
    covs = np.ascontiguousarray(state_covariances, dtype=np.float64)
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    if means.ndim != 2 or means.shape[0] <= 0 or means.shape[1] <= 0:
        raise ValueError("state_means must have shape (S, p) with S, p > 0")
    S, p = means.shape
    if covs.shape != (S, p, p):
        raise ValueError("state_covariances must have shape (S, p, p)")
    if obs.ndim != 2 or obs.shape[1] != p:
        raise ValueError("observations must have shape (T, p)")
    _finite("state_means", means); _finite("state_covariances", covs); _finite("observations", obs)
    T = obs.shape[0]
    out = np.empty((T, S), dtype=np.float64)
    if T == 0:
        return out
    lib = _require_lib()
    lib.ssgs_gaussian_logprobs(
        _ptr(means), _ptr(covs), _ptr(obs), S, p, T,
        out.ctypes.data_as(ctypes.c_void_p),
    )
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("native Gaussian emission kernel returned non-finite values")
    return out


def forward_backward(log_emissions, log_init, log_trans):
    emit = np.ascontiguousarray(log_emissions, dtype=np.float64)
    init = np.ascontiguousarray(log_init, dtype=np.float64)
    trans = np.ascontiguousarray(log_trans, dtype=np.float64)
    if emit.ndim != 2 or emit.shape[0] == 0 or emit.shape[1] == 0:
        raise ValueError("log_emissions must have shape (T, S) with T, S > 0")
    T, S = emit.shape
    if init.shape != (S,) or trans.shape != (S, S):
        raise ValueError("log_init/log_trans shapes do not match log_emissions")
    for name, arr in (("log_emissions", emit), ("log_init", init), ("log_trans", trans)):
        if np.any(np.isnan(arr)) or np.any(np.isposinf(arr)):
            raise ValueError(f"{name} may contain -inf but not NaN or +inf")
    if np.all(np.isneginf(init)) or np.any(np.all(np.isneginf(trans), axis=1)):
        raise ValueError("initial and transition probabilities must permit a valid path")
    lib = _require_lib()
    gamma = np.empty((T, S), dtype=np.float64)
    xi = np.empty((max(T - 1, 0), S, S), dtype=np.float64)
    loglik = ctypes.c_double()
    lib.ssgs_forward_backward(
        _ptr(emit), _ptr(init), _ptr(trans), T, S,
        gamma.ctypes.data_as(ctypes.c_void_p),
        xi.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(loglik),
    )
    if not (np.all(np.isfinite(gamma)) and np.all(np.isfinite(xi)) and np.isfinite(loglik.value)):
        raise FloatingPointError("HMM sequence has zero or non-finite likelihood")
    return gamma, xi, loglik.value


def decode_sequence(transition_cost, smoothness, initial_cost, smoothness_weight, n_frames):
    n_frames = _positive_int("n_frames", n_frames)
    trans = np.ascontiguousarray(transition_cost, dtype=np.float64)
    smooth = np.ascontiguousarray(smoothness, dtype=np.float64)
    init = np.ascontiguousarray(initial_cost, dtype=np.float64)
    if trans.ndim != 2 or trans.shape[0] == 0 or trans.shape[0] != trans.shape[1]:
        raise ValueError("transition_cost must be a non-empty square matrix")
    S = trans.shape[0]
    if smooth.shape != (S, S) or init.shape != (S,):
        raise ValueError("smoothness and initial_cost shapes must match transition_cost")
    if not np.isfinite(smoothness_weight) or smoothness_weight < 0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    _finite("transition_cost", trans); _finite("smoothness", smooth); _finite("initial_cost", init)
    lib = _require_lib()
    path = np.empty(n_frames, dtype=np.int32)
    lib.ssgs_decode(
        _ptr(trans), _ptr(smooth), _ptr(init), float(smoothness_weight), S, n_frames,
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

    # Compare on a single sequence only: after EM, residual augmentation has
    # appended extra sequences (m.sequence_starts has >1 entries), and
    # _expectation_step correctly drops cross-sequence transitions from xi,
    # so comparing against the whole observation set mismatches xi shapes.
    emit = m._compute_emission_log_probs(obs[:20])
    init = np.log(m.initial_probabilities + 1e-12)
    trans = np.log(m.transition_matrix + 1e-12)
    g, xi, ll = forward_backward(emit, init, trans)
    g2, xi2, ll2 = m._expectation_step(obs[:20])
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
