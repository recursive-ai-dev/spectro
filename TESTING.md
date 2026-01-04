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
