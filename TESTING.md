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
python test_adaptive_persistence.py
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
python test_data_indexing.py
python test_checkpoints.py
python test_generative_capabilities.py
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
python test_ssgs.py
python test_checkpoints.py
```

## Expected Outcomes
- Scripts complete without import errors when dependencies are installed.
- Checkpoint and adaptive APIs execute with real signals, confirming documented behavior.
