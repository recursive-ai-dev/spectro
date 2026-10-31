# TESTING — Adaptive Persistence Coverage

## Purpose
These tests are designed to validate adaptive learning with real audio signals, confirm persistence through export/load, and stress input boundaries across the full expected range of outcomes.

## Test Coverage Strategy
1. **Core adaptation pipeline:** Use real synthetic audio (harmonic + noise) to drive adaptation and verify non-trivial parameter updates.
2. **Persistence checks:** Export the model with adaptive statistics and reload it to ensure continuity of learned state.
3. **Boundary sweep (-1..12):** Validate adaptation-rate constraints across the full numeric range to cover edge conditions and consumer hardware variability.

## Tests Implemented
### 1) Adaptive Persistence Integration
**Script:** `test_adaptive_persistence.py`
- Trains on a real, generated harmonic signal.
- Adapts to a different audio signal with new spectral content.
- Asserts that the transition matrix changes (learning occurred).
- Confirms training memory is bounded by the configured limit.
- Exports and reloads the model to ensure adaptive statistics persist.

### 2) Boundary Sweep for Adaptation Rate (-1..12)
**Script:** `test_adaptive_persistence.py`
- Runs `adapt_to_audio` with adaptation_rate values in **[-1, 0, 1, 2, ..., 12]**.
- Valid values (0, 1) complete adaptation on real audio.
- Invalid values (<0 or >1) are required to raise `ValueError`.

## How to Run
```bash
python tests/test_adaptive_persistence.py
```

## Expected Outcomes
- **Valid rates (0, 1)** adapt without errors and update adaptive statistics.
- **Invalid rates (-1, 2..12)** must raise `ValueError`.
- Adaptive statistics are preserved across export/load.

---

# TESTING — Smoothness-First Decoder Coverage

## Purpose
Validate that the smoothness-aware decoder produces paths that are at least as spectrally smooth as a transition-only Viterbi baseline when decoding noisy signals.

## Tests Implemented
### 1) A* vs Viterbi Smoothness Comparison
**Script:** `test_ssgs.py` (`test_astar_vs_viterbi_decoding`)
- Trains on a real synthetic signal with multiple frequency regimes plus noise.
- Decodes a short sequence using the smoothness-aware A* path and a transition-only Viterbi path.
- Computes spectral smoothness cost on both paths and asserts the A* path is less than or equal to Viterbi.

## How to Run
```bash
pytest -q test_ssgs.py::test_astar_vs_viterbi_decoding
```

## Expected Outcomes
- A* decoding returns a path with spectral smoothness cost **≤** Viterbi.
- The decoder remains deterministic and reproducible for the same trained model.

---

# TESTING — AR(4) Recovery Coverage

## Purpose
Validate the LPC/HMM estimation pipeline by recovering known AR(4) coefficients from a synthetic process.

## Tests Implemented
### 1) AR(4) Coefficient Recovery
**Script:** `test_ar_process.py` (`test_ar4_recovery_with_single_state`)
- Generates a stable AR(4) signal with Gaussian noise.
- Trains a single-state HMM with `lpc_order=4`.
- Asserts the recovered LPC mean matches `-a` within ±0.05.

## How to Run
```bash
pytest -q test_ar_process.py
```

## Expected Outcomes
- The recovered LPC mean is within ±0.05 of the ground-truth coefficients.

---

# Robustness Regression Coverage

## Purpose
Exercise public and internal contracts for validation, numerical stability,
sequence segmentation, reproducibility, synthesis, and serialization.

## Tests
**Script:** `test_robustness.py`

Coverage includes:
- Invalid model settings, sample rates, audio arrays, frequency bounds, sequence starts, state paths, and BIC candidates fail explicitly.
- Integer PCM, unsigned PCM, multichannel input, sample-rate conversion, and padded tail frames behave consistently.
- Forward/backward inference handles one-frame and segmented observations without adding transitions between recordings.
- Singular Gaussian covariances and unstable LPC filters are repaired without non-finite results.
- Cache invalidation, seeded k-means/stochastic generation, exact output length, and preview-buffer isolation are checked.
- Multi-recording boundaries survive training-artifact export/load; folder discovery is case-insensitive, deterministic, and duplicate-free.

## How to Run
```bash
python -m pytest -q test_robustness.py
python -m pytest -q
```

The full suite should complete without failures; the robustness module currently
contains focused regression cases for the newly hardened paths.
# Production Hardening Tests

## Purpose
Ensure deterministic test signals, explicit error paths, and checkpoint validation work as intended.

## Tests Implemented
### 1) Deterministic Signal Generation
**Scripts:** `test_generative_capabilities.py`, `test_checkpoints.py`, `test_adaptive_persistence.py`, `test_ssgs.py`, `test_data_indexing.py`
- All test signal generators now accept seeded RNGs.
- Each test passes a fixed seed to guarantee reproducibility.

### 2) Empty-Frame Guard for HMM Initialization
**Script:** `test_data_indexing.py`
- Creates a model with empty LPC coefficients and asserts `initialize_hmm_parameters` raises a descriptive `ValueError`.

### 3) Missing Tensor Validation for Checkpoints
**Script:** `test_checkpoints.py`
- Saves a safetensors checkpoint missing `state_covariances`.
- Confirms `load_checkpoint` rejects the file with an explicit error.

## How to Run
```bash
python tests/test_data_indexing.py
python tests/test_checkpoints.py
python tests/test_generative_capabilities.py
```

## Expected Outcomes
- Signal generation is repeatable across runs when the same seeds are used.
- Empty feature inputs fail fast with a clear error.
- Missing checkpoint tensors are rejected before model reconstruction begins.

---

# README Alignment Validation

## Purpose
Ensure the documentation references executable scripts and valid APIs so onboarding steps remain reliable.

## Validation Checklist
1. **Quick Start scripts exist:** `test_ssgs.py`, `demo_enhanced_capabilities.py`, `train_on_folder.py`.
2. **Checkpoint APIs are present:** `save_checkpoint` and `load_checkpoint` in `ssgs.py`.
3. **Adaptive learning API is present:** `adapt_to_audio` in `ssgs.py`.
4. **Feature indexing APIs are present:** `build_feature_index`, `query_similar_frames`, and `search_by_feature_vector` in `ssgs.py`.
5. **Test scripts listed in README exist:** all `test_*.py` files referenced by name.

## How to Run (Spot Checks)
```bash
python tests/test_ssgs.py
python tests/test_checkpoints.py
```

## Expected Outcomes
- Scripts complete without import errors when dependencies are installed.
- Checkpoint and adaptive APIs execute with real signals, confirming documented behavior.

---

# TESTING — Adversarial Rust-Proofing Suites (2025 sprint)

Five modules reproduce failures that were observed on real inputs and assert the
corrected contract (not merely that a call returns). They run under pytest and
need no display; the native-backend module skips when `libssgs_native.so` has not
been built with `make`.

| Module | Covers | Representative cases |
|---|---|---|
| `tests/test_state_integrity.py` | archive validation (T01, T17), re-training state (T03), augmentation boundaries (T07), fidelity warning (T08), metadata indices (T11), graph components (T13) | 8-case `.npz` tamper matrix, `.safetensors` tamper, bit-exact float64 round trip, injected frame-dropping processor, boolean/negative/out-of-range indices |
| `tests/test_resource_bounds.py` | render limits (T02), streamed metadata (T04), frame batching (T04) | rate 768 000/768 001, duration 3600/3601, 2**28 sample cap, CLI exit code 1 without traceback, 48 bit-exact metadata cases, tracemalloc bounds |
| `tests/test_publishing_and_formats.py` | Opus input (T05), atomic publish (T06), demo cleanup (T12) | `.opus` discovery + training, `os.link` forced to `EXDEV`, no-clobber/forced overwrite, temp-file cleanup, failing demo rollback |
| `tests/test_native_backend.py` | C/Python parity (T10), benchmark honesty and stdout silence (T09) | 8 randomised shape/parameter trials across all four kernels, parsed benchmark table, empty stdout for library calls |
| `tests/test_python_support.py` | declared 3.9 floor (T15) | 3.9 grammar parse of every module, rejected runtime PEP 585/604 annotations, CI matrix covers `requires-python` |

Run everything (builds the C kernels first):

```bash
make test PYTHON=.venv/bin/python
```

Measured on the development machine (Python 3.11.2, NumPy 2.4.6, SciPy 1.17.1):
**290 passed, 6 skipped** (the skips are Tk/desktop checks that require a
display). CI runs the same suite on Python 3.9, 3.10, 3.11 and 3.12.

## Sanitizer verification

The C kernels are additionally exercised under AddressSanitizer and
UndefinedBehaviorSanitizer:

```bash
gcc -shared -fPIC -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o /tmp/libssgs_asan.so src/ssgs/ssgs_native.c
cp /tmp/libssgs_asan.so src/ssgs/libssgs_native.so
LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 \
    .venv/bin/python -m pytest tests/test_native_backend.py -q
make    # restore the optimised library afterwards
```
