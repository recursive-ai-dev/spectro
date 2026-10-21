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
"""
ssgs_core: a rudimentary, dependency-light numeric foundation for SSGS.

Purpose
-------
The production ``ssgs.py`` leans on SciPy for LPC (``solve_toeplitz``), Gaussian
emissions (``scipy.linalg.cholesky`` / ``solve_triangular``), and the EM
forward/backward + smoothness decode. Those calls carry object/validation
overhead and, more importantly, hide numerical behavior (eig sign flips,
state-carry quirks) that produced real bugs.

This module re-implements the *same math* with the simplest possible
algorithms so the results are auditable and the hot loops have a clean,
flat-array ABI that a compiled (C/numba) kernel can replace 1:1.

Scope of the "rudimentary" core
--------------------------------
* ``lpc_coefficients``       - Levinson-Durbin (no Toeplitz solve)
* ``gaussian_log_probs``     - hand Cholesky + forward substitution (no SciPy)
* ``forward_backward``       - log-space forward/backward (NumPy reference)
* ``decode_sequence``        - smoothness-first lexicographic Viterbi

The compiled drop-in (owned by a separate agent) must expose the C ABI
documented at the bottom of this file and pass the same ``__main__`` self-test.

Only NumPy (and optionally SciPy for logsumexp) is required.
"""

import numpy as np


# ---------------------------------------------------------------------------
# LPC via Levinson-Durbin
# ---------------------------------------------------------------------------
def lpc_coefficients(autocorr, lpc_order):
    """Prediction-error-filter coefficients from an autocorrelation sequence.

    ``autocorr`` must have length ``lpc_order + 1`` and ``autocorr[0] > 0``.
    Returns ``coeffs`` of length ``lpc_order + 1`` with ``coeffs[0] == 1.0``,
    matching the convention ``[1.0, -solution...]`` produced by the SciPy
    Toeplitz solver used previously.
    """
    autocorr = np.asarray(autocorr, dtype=np.float64)
    if isinstance(lpc_order, (bool, np.bool_)) or not isinstance(lpc_order, (int, np.integer)) or lpc_order <= 0:
        raise ValueError("lpc_order must be a positive integer")
    if autocorr.ndim != 1 or autocorr.shape[0] != int(lpc_order) + 1:
        raise ValueError("autocorr must be 1-D with length lpc_order + 1")
    if not np.all(np.isfinite(autocorr)):
        raise ValueError("autocorr must contain only finite values")
    if autocorr[0] <= 1e-8:
        coeffs = np.zeros(lpc_order + 1, dtype=np.float64)
        coeffs[0] = 1.0
        return coeffs

    # a[0] is always 1; a[1..order] are the recursion coefficients.
    a = np.zeros(lpc_order + 1, dtype=np.float64)
    a[0] = 1.0
    error = float(autocorr[0])

    for i in range(1, lpc_order + 1):
        acc = autocorr[i]
        for j in range(1, i):
            acc += a[j] * autocorr[i - j]
        k = -acc / error

        # Update a[1..i] symmetrically: a'_j = a_j + k * a_{i-j}.
        updated = a.copy()
        for j in range(1, i):
            updated[j] = a[j] + k * a[i - j]
        updated[i] = k

        a = updated
        error *= 1.0 - k * k
        if error <= 1e-12:
            break

    return a


# ---------------------------------------------------------------------------
# Gaussian emission log-probabilities (hand Cholesky + forward solve)
# ---------------------------------------------------------------------------
def _cholesky_lower(cov):
    """Lower-triangular Cholesky factor L with L @ L.T == cov (p x p)."""
    p = cov.shape[0]
    L = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1):
            s = cov[i, j] - np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, i] = np.sqrt(max(s, 1e-12))
            else:
                L[i, j] = s / L[j, j]
    return L


def _forward_solve(L, d):
    """Solve L @ y = d for lower-triangular L (non-unit diagonal)."""
    p = L.shape[0]
    y = np.zeros(p, dtype=np.float64)
    for i in range(p):
        s = d[i] - np.dot(L[i, :i], y[:i])
        y[i] = s / L[i, i]
    return y


def gaussian_log_probs(state_means, state_covariances, observations):
    """Per-frame, per-state Gaussian log-likelihoods with robust SPD repair."""
    state_means = np.asarray(state_means, dtype=np.float64)
    state_covariances = np.asarray(state_covariances, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    if state_means.ndim != 2:
        raise ValueError("state_means must have shape (S, p)")
    S, p = state_means.shape
    if S <= 0 or p <= 0:
        raise ValueError("state_means must contain at least one state and one feature")
    if state_covariances.shape != (S, p, p):
        raise ValueError("state_covariances must have shape (S, p, p)")
    if observations.ndim != 2 or observations.shape[1] != p:
        raise ValueError("observations must have shape (T, p)")
    if not all(np.all(np.isfinite(array)) for array in (state_means, state_covariances, observations)):
        raise ValueError("Gaussian parameters and observations must be finite")
    T = observations.shape[0]
    log_probs = np.empty((T, S), dtype=np.float64)
    if T == 0:
        return log_probs
    const = p * np.log(2.0 * np.pi)
    eye = np.eye(p, dtype=np.float64)

    for state in range(S):
        cov = 0.5 * (state_covariances[state] + state_covariances[state].T)
        jitter = 0.0
        for attempt in range(9):
            try:
                L = np.linalg.cholesky(cov + eye * jitter)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if attempt == 0 else jitter * 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            cov = (eigvecs * np.maximum(eigvals, 1e-8)) @ eigvecs.T
            L = np.linalg.cholesky(0.5 * (cov + cov.T))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        solved = np.linalg.solve(L, (observations - state_means[state]).T)
        quad = np.sum(solved * solved, axis=0)
        log_probs[:, state] = -0.5 * (quad + const + log_det)
    return log_probs


# ---------------------------------------------------------------------------
# Forward-backward (EM expectation step) - NumPy reference
# ---------------------------------------------------------------------------
def forward_backward(log_emissions, log_init, log_trans):
    """Validated log-space forward/backward for one non-empty HMM sequence."""
    try:
        from scipy.special import logsumexp
    except Exception:  # pragma: no cover - dependency-light fallback
        def logsumexp(values, axis=None, keepdims=False):
            values = np.asarray(values, dtype=np.float64)
            maximum = np.max(values, axis=axis, keepdims=True)
            finite_max = np.isfinite(maximum)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                shifted = np.where(finite_max, values - maximum, -np.inf)
                total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
                result = np.where(finite_max, maximum + np.log(total), maximum)
            if axis is None and not keepdims:
                return float(result.reshape(-1)[0])
            if not keepdims:
                result = np.squeeze(result, axis=axis)
            return result

    log_emissions = np.asarray(log_emissions, dtype=np.float64)
    log_init = np.asarray(log_init, dtype=np.float64)
    log_trans = np.asarray(log_trans, dtype=np.float64)
    if log_emissions.ndim != 2 or log_emissions.shape[0] == 0 or log_emissions.shape[1] == 0:
        raise ValueError("log_emissions must have shape (T, S) with T,S > 0")
    T, S = log_emissions.shape
    if log_init.shape != (S,) or log_trans.shape != (S, S):
        raise ValueError("log_init/log_trans shapes do not match log_emissions")
    if any(np.any(np.isnan(array)) or np.any(np.isposinf(array)) for array in (log_emissions, log_init, log_trans)):
        raise ValueError("log probabilities may contain -inf but not NaN or +inf")
    if np.all(np.isneginf(log_init)) or np.any(np.all(np.isneginf(log_trans), axis=1)):
        raise ValueError("initial and transition probabilities must permit a valid path")

    alpha = np.empty((T, S), dtype=np.float64)
    alpha[0] = log_init + log_emissions[0]
    for t in range(1, T):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
        alpha[t] += log_emissions[t]

    log_likelihood = float(logsumexp(alpha[-1]))
    if not np.isfinite(log_likelihood):
        raise FloatingPointError("HMM sequence has zero or non-finite likelihood")
    beta = np.zeros((T, S), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        temp = log_trans + log_emissions[t + 1] + beta[t + 1]
        beta[t] = logsumexp(temp, axis=1)

    log_gamma = alpha + beta
    normalizers = logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - normalizers)
    if T == 1:
        xi = np.empty((0, S, S), dtype=np.float64)
    else:
        log_xi = (
            alpha[:-1, :, None]
            + log_trans[None, :, :]
            + log_emissions[1:, None, :]
            + beta[1:, None, :]
        )
        log_xi -= logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
        xi = np.exp(log_xi)
    return gamma, xi, log_likelihood


# ---------------------------------------------------------------------------
# Smoothness-first (lexicographic) Viterbi decode
# ---------------------------------------------------------------------------
def decode_sequence(transition_cost, smoothness, initial_cost, smoothness_weight, n_frames):
    """Smoothness-first lexicographic decode.

    ``transition_cost`` already includes the psychoacoustic term and
    ``smoothness_weight * smoothness`` (mirrors ``ssgs._transition_cost_matrix``),
    so we recover the base transition cost before applying the lexicographic
    ordering, exactly like the corrected ``ssgs._decode_state_sequence``.

    Returns a list of ``n_frames`` state indices.
    """
    transition_cost = np.asarray(transition_cost, dtype=np.float64)
    smoothness = np.asarray(smoothness, dtype=np.float64)
    initial_cost = np.asarray(initial_cost, dtype=np.float64)
    if isinstance(n_frames, (bool, np.bool_)) or not isinstance(n_frames, (int, np.integer)) or n_frames <= 0:
        raise ValueError("n_frames must be a positive integer")
    if transition_cost.ndim != 2 or transition_cost.shape[0] == 0 or transition_cost.shape[0] != transition_cost.shape[1]:
        raise ValueError("transition_cost must be a non-empty square matrix")
    S = transition_cost.shape[0]
    if smoothness.shape != (S, S) or initial_cost.shape != (S,):
        raise ValueError("smoothness and initial_cost shapes must match transition_cost")
    if not np.isfinite(smoothness_weight) or smoothness_weight < 0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    if not all(np.all(np.isfinite(array)) for array in (transition_cost, smoothness, initial_cost)):
        raise ValueError("decode costs must be finite")
    if smoothness_weight <= 0.0:
        dp = np.empty((n_frames, S), dtype=np.float64)
        back = np.zeros((n_frames, S), dtype=np.int32)
        dp[0] = initial_cost
        for t in range(1, n_frames):
            costs = dp[t - 1][:, None] + transition_cost
            back[t] = np.argmin(costs, axis=0)
            dp[t] = np.min(costs, axis=0)
        end = int(np.argmin(dp[-1]))
    else:
        base = transition_cost - smoothness_weight * smoothness
        dp_smooth = np.empty((n_frames, S), dtype=np.float64)
        dp_total = np.empty((n_frames, S), dtype=np.float64)
        back = np.zeros((n_frames, S), dtype=np.int32)
        dp_smooth[0] = 0.0
        dp_total[0] = initial_cost
        tol = 1e-9
        for t in range(1, n_frames):
            for j in range(S):
                cand_smooth = dp_smooth[t - 1] + smoothness[:, j]
                min_smooth = float(np.min(cand_smooth))
                mask = cand_smooth <= (min_smooth + tol)
                total = dp_total[t - 1] + base[:, j] + smoothness_weight * smoothness[:, j]
                idxs = np.where(mask)[0]
                best = idxs[int(np.argmin(total[mask]))]
                dp_smooth[t, j] = min_smooth
                dp_total[t, j] = total[best]
                back[t, j] = best
        final_smooth = dp_smooth[-1]
        min_smooth = float(np.min(final_smooth))
        final_mask = final_smooth <= (min_smooth + tol)
        end = int(np.where(final_mask)[0][np.argmin(dp_total[-1][final_mask])])

    path = [0] * n_frames
    path[-1] = end
    for t in range(n_frames - 1, 0, -1):
        end = int(back[t, end])
        path[t - 1] = end
    return path


# ---------------------------------------------------------------------------
# C ABI contract for the compiled drop-in (agent-owned)
# ---------------------------------------------------------------------------
"""
The compiled kernel must expose (C linkage, row-major float64, no allocation
inside the hot loop):

  void ssgs_lpc_levinson(const double* autocorr, int lpc_order, double* coeffs);
      // coeffs length lpc_order+1, coeffs[0]=1

  void ssgs_gaussian_logprobs(const double* means,       // S*p
                              const double* covs,        // S*p*p
                              const double* obs,         // T*p
                              int S, int p, int T,
                              double* out);              // T*S

  void ssgs_forward_backward(const double* log_emit,     // T*S
                             const double* log_init,     // S
                             const double* log_trans,    // S*S
                             int T, int S,
                             double* gamma,              // T*S
                             double* xi,                  // (T-1)*S*S
                             double* loglik);            // scalar

  void ssgs_decode(const double* transition_cost,        // S*S
                   const double* smoothness,             // S*S
                   const double* initial_cost,           // S
                   double weight, int S, int n_frames,
                   int* path);                           // n_frames

All functions are pure and reentrant. The Python wrapper (`ssgs_core` or a
thin Cython/numba shim) passes NumPy arrays with ``copy=False``. The reference
implementations above are the correctness oracle for the compiled versions.
"""
# ---------------------------------------------------------------------------
# Self-test: the rudimentary core must reproduce ssgs.py's numbers
# ---------------------------------------------------------------------------
def _self_test():
    from ssgs import SpectralStateGuidedSynthesis
    from .signals import create_simple_test_signal

    sr = 12000
    sig = create_simple_test_signal(sample_rate=sr, duration=1.5, seed=42)
    m = SpectralStateGuidedSynthesis(n_states=8, lpc_order=10, frame_size=512, hop_size=128)
    m.extract_features(sig, sr)
    m.initialize_hmm_parameters()
    m.iterative_refinement(3)
    m.identify_graph_constraints()

    # 1) LPC equivalence on a frame
    frame = m.training_frames[0]
    ref = m._compute_lpc(frame)
    autocorr = m._autocorrelation(frame)
    core = lpc_coefficients(autocorr, m.lpc_order)
    assert np.allclose(ref, core, atol=1e-5), "LPC mismatch vs SciPy"
    print("  [ok] LPC (Levinson-Durbin) matches ssgs._compute_lpc")

    # 2) Emission equivalence
    obs = m.lpc_coefficients
    ref_lp = m._compute_emission_log_probs(obs)
    core_lp = gaussian_log_probs(m.state_means, m.state_covariances, obs)
    assert np.allclose(ref_lp, core_lp, atol=1e-4), "Emission mismatch"
    print("  [ok] Gaussian emission matches ssgs emission")

    # 3) Decode equivalence (cached path, which uses corrected ordering)
    tc = m._transition_cost_matrix()
    sm = m._spectral_smoothness_matrix()
    init = -np.log(m.initial_probabilities + 1e-12)
    ref_path = m._decode_state_sequence_cached(20)
    core_path = decode_sequence(tc, sm, init, m.smoothness_weight, 20)
    assert ref_path == core_path, "Decode mismatch"
    print("  [ok] Smoothness decode matches ssgs (cached)")

    # 4) Forward-backward equivalence
    emit = m._compute_emission_log_probs(obs[:20])
    init = np.log(m.initial_probabilities + 1e-12)
    trans = np.log(m.transition_matrix + 1e-12)
    core_gamma, core_xi, core_ll = forward_backward(emit, init, trans)
    native_gamma, native_xi, native_ll = m._expectation_step(obs[:20])
    assert np.allclose(core_gamma, native_gamma, atol=1e-3), "Gamma mismatch"
    assert np.allclose(core_xi, native_xi, atol=1e-3), "Xi mismatch"
    assert abs(core_ll - native_ll) < 1e-3, "Log-likelihood mismatch"
    print("  [ok] Forward-backward (core) matches ssgs._expectation_step")

    print("ALL CORE SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
