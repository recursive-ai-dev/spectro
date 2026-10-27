# Adversarial Evaluation & Alternative-Method Hunt

**Question asked:** bug-hunt, logic-test and adversarially test the codebase, and
evaluate whether *another method for a particular function* could improve it.

**Method:** (1) baseline — full suite green (59 tests, after `make` builds the
native lib); (2) logic checks of the pure-NumPy oracle in `ssgs/core.py`
against independent references (SciPy, exhaustive search); (3) adversarial
probes of the ctypes native boundary, each run in a subprocess so an
out-of-bounds read/segfault could not kill the harness; (4) benchmark of
three candidate implementations for the target function. One harness bug was
found and fixed during the work (a wrong polynomial-root convention in the
LPC stability check — the library was correct), which is why every claim
below is tied to a reproducible test in
`tests/test_autocorrelation_and_native_hardening.py`.

---

## Verdict up front

| Function | Current method | Better method found? | Action |
|---|---|---|---|
| `model._autocorrelation` | full `np.correlate(mode="full")`, O(N²), all 2N−1 lags | **Yes — lag-limited dot products, O(N·order); FFT for very high orders** | ✅ Implemented |
| `core.decode_sequence` | lexicographic DP | No — matches exhaustive search exactly | Kept |
| `core.gaussian_log_probs` | Cholesky + jitter/eigen repair | No — matches `scipy.stats.multivariate_normal` to 1e-8 | Kept |
| `core.lpc_coefficients` | Levinson–Durbin | No — matches `solve_toeplitz` to ~3e-16, poles stable | Kept |
| `core.forward_backward` | log-space forward/backward | No — posteriors normalise, finite log-likelihood | Kept |
| `ssgs.native` wrappers | raw ctypes, no validation, import-time `CDLL` | **Yes — validate-then-call + tolerant import (bugs below)** | ✅ Implemented |
| C `ssgs_gaussian_logprobs` | single 1e-4 jitter, failure ignored | **Yes — symmetrise + escalating jitter + eigen-floor fallback (mirrors core)** | ✅ Implemented |

---

## The improvement: `_autocorrelation`

The docstring already said *"lag-limited autocorrelation"*, but the code
computed **all** lags and sliced out the `lpc_order + 1` (≈13) that
`_compute_lpc` actually consumes — verified by grep: `_autocorrelation` has
exactly one caller, and it uses only those lags.

Benchmark (order 12, per call, this machine; correctness vs the old method
verified to <1e-9 relative error at every size):

| frame N | current (full correlate) | lag-limited direct | FFT |
|---|---|---|---|
| 512 | 0.040 ms | **0.018 ms** | 0.026 ms |
| 4 096 | 2.51 ms | **0.027 ms (≈93×)** | 0.130 ms |
| 16 384 | 65 ms | **0.061 ms (≈1 070×)** | 0.91 ms |
| 65 536 | 702 ms | **0.133 ms (≈5 300×)** | 4.7 ms |

End-to-end `extract_features` (2 s signal): 0.044→0.024 s (frame 1024),
0.085→0.013 s (4096), 0.319→0.008 s (16384).

**Decision:** direct lag-limited dots win everywhere at realistic LPC
orders because only ~13 lags are needed; FFT only wins when the *order*
is large. Implemented as a hybrid: direct for `lpc_order <= 64`, FFT
(`rfft`, next-power-of-two) above that. Both branches are tested against
the old full-correlate reference on noise/zero/constant/impulse/sine/
alternating frames at orders 1–200, and LPC output is unchanged end-to-end.

## Bugs found by adversarial probing (all now fixed + regression-tested)

1. **Fresh-clone crash (C1).** `ssgs/native.py` ran `ctypes.CDLL(...)`
   unconditionally at import; the `.so` is git-ignored and built by `make`,
   so `import ssgs.native` raised `OSError` on a clean checkout — defeating
   the graceful core fallback `model.py` was written around (it checks
   `native._lib is not None`). → Import is now tolerant (`_lib = None`),
   kernel calls raise a clear `RuntimeError`, and `import ssgs` works
   without the `.so` (subprocess regression test).
2. **Silent garbage from out-of-bounds reads (C4).** Native wrappers did no
   shape validation; e.g. covariances shaped `(1,2,2)` with `(2,3)` means
   returned plausible-looking garbage (`[[11.06, 1071.47], …]`) from memory
   past the buffer — a segfault with less lucky shapes. → Every wrapper now
   validates shapes/finiteness *before* ctypes, mirroring `ssgs.core`
   (invalid input raises `ValueError`, proven by parametrized tests).
3. **NaNs on non-PSD covariances (C2).** The C Gaussian kernel made one
   1e-4 jitter attempt and ignored Cholesky failure, dividing by a zero
   pivot → `[[nan], …]`, while the core oracle repairs and returns finite
   values. → C now symmetrises, escalates jitter 1e-10→1e-3, and falls back
   to the core's 1e-8 eigen-floor; output now matches core exactly on the
   probe case (16.5828…).
4. **Silent NaNs on impossible sequences (C3).** All `-inf` emissions made
   native forward-backward return NaN gamma/xi/log-likelihood; core raises
   `FloatingPointError`. → Wrapper validates inputs and checks outputs,
   raising `FloatingPointError` identically.
5. **Silent empty decode (C5).** `decode_sequence(..., n_frames=0)`
   returned `[]`; core raises `ValueError`. → Positive-int validation.
6. **Broken self-verification script.** `python -m ssgs.native` (its
   `verify()`) failed even on the original code: it compared single-sequence
   native forward-backward over *all* observations against
   `_expectation_step`, which — after residual augmentation appends extra
   sequences (`sequence_starts = [0, 138, 276, …]`) — correctly drops
   cross-sequence transitions, so the xi shapes mismatch (689 vs 685).
   → `verify()` now compares on one 20-frame sequence, like the regression
   guard; the module's verify + benchmark now run to completion
   (decode: ~130× native speedup confirmed).

## Logic checks that passed (no change warranted)

- **Decode optimality:** `core.decode_sequence` equals exhaustive search
  over all 3⁶ paths on 30 random problems under the lexicographic
  (smoothness, total-cost) objective.
- **Emissions:** `core.gaussian_log_probs` vs SciPy `multivariate_normal`
  on 20 random SPD problems, max error <1e-8.
- **LPC:** Levinson–Durbin vs `solve_toeplitz` on 50 AR-style signals,
  max coefficient error ~3e-16, all poles inside the unit circle.
- **Forward/backward:** gamma and xi normalise to 1; log-likelihood finite.

## Notes / not changed

- `core._cholesky_lower` / `core._forward_solve` are dead code
  (`gaussian_log_probs` uses `np.linalg` directly). Harmless, left in place;
  deleting or wiring them in is a cleanup decision for the maintainers.
- Validation added to the native wrappers costs O(input size), negligible
  next to the kernels themselves; the synthesis hot path in `model.py`
  passes already-validated arrays and is unaffected in behaviour.
