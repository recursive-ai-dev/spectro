# LOGIC-MAP — SSGS Adaptive Persistence Update

## Purpose
Ensure the model excels at learning audio by introducing persistent adaptive statistics, memory blending, and mathematically grounded parameter updates that are directly tied to real audio observations.

## Logic Chains (Steps 1–3)

### Chain A — Audio Ingestion → Feature Statistics (Step 1)
**Why:** Adaptation requires consistent, high-quality statistics derived from the same DSP pipeline used in initial training.

**How:**
1. **Frame extraction** (`_extract_features_for_adaptation` in `ssgs.py`) uses the existing LPC + perceptual feature pipeline (Hanning window, mel/ERB filters) to guarantee identical feature space semantics.
2. **Signal energy screening** stays intact via `_process_frame_batch`, preventing low-energy frames from corrupting adaptive statistics.
3. **Frame metadata** (energy, RMS, spectral flux) is derived directly from the audio, enabling measurable adaptation metrics (avg_energy, avg_rms).

**Mathematical rigor:** Feature vectors remain in the original LPC domain, so adaptation uses the same Gaussian emission model. This preserves the HMM likelihood function and prevents drift from mismatched feature spaces.

---

### Chain B — Adaptive Statistics Accumulation (Step 2)
**Why:** Persisting learning means the model must retain sufficient statistics so new audio updates are blended, not replaced.

**How:**
1. **Posterior inference** uses the current HMM parameters to compute `gamma` and `xi` over the new observations (`_expectation_step`).
2. **Sufficient statistics** are computed as:
   - `counts = Σ_t γ_t(state)`
   - `mean_accumulator = Σ_t γ_t(state) * x_t`
   - `second_moment = Σ_t γ_t(state) * (x_t x_tᵀ)`
   - `transition_counts = Σ_t ξ_t`
3. **Exponential blending** uses `adaptation_rate` to merge new statistics with persistent statistics:
   - `stats := (1 - α) * stats_old + α * stats_new`
   - `stability_bias` adds a positive floor to avoid degeneracy.

**Mathematical rigor:** The above is the standard online EM sufficient-statistics update (EMA form). It maintains a consistent maximum-likelihood estimator under streaming updates with a controlled forgetting factor.

---

### Chain C — Parameter Reconstruction + Covariance Safety (Step 3)
**Why:** Persistent learning must yield stable, valid parameters (positive-definite covariances, valid transition probabilities).

**How:**
1. **Means:** `μ_state = mean_accumulator / counts`.
2. **Covariances:** `Σ_state = E[x xᵀ] - μ μᵀ` with diagonal regularization.
3. **Transition matrix:** `A = transition_counts / row_sums`, clipped and normalized.
4. **Initial probabilities:** `π = initial_counts / Σ initial_counts`.
5. **Positive-definite enforcement:** eigenvalue checks ensure all covariances are valid; negative or near-zero eigenvalues are corrected via diagonal shifts.

**Mathematical rigor:** Covariance reconstruction uses the standard second-moment identity. Regularization plus eigenvalue floor ensures Σ ≻ 0, preserving Gaussian log-likelihood calculations.

---

## Result
The update introduces persistent, mathematically justified adaptive learning without breaking the original training pipeline, enabling the model to learn continually from new audio while protecting prior knowledge.

---

# LOGIC-MAP — README Alignment Refresh

## Purpose
Ensure the README reflects the current runnable workflows (checkpointing, adaptive learning, indexing, and expanded test suite) without altering core implementation.

## Logic Chains (Steps 1–3)

### Chain A — Source-of-Truth Audit (Step 1)
**Why:** Documentation must mirror real entry points to avoid user misconfiguration and silent failures.

**How:**
1. **Inventory live scripts** (`train_on_folder.py`, `demo_enhanced_capabilities.py`, `example_checkpoints.py`, `example_fidelity.py`) to confirm current usage patterns.
2. **Inspect core APIs** in `ssgs.py` (checkpointing, adaptive learning, feature indexing) to align documentation with existing method signatures.
3. **Cross-check dependencies** against `requirements.txt` to keep install steps exact.

**Mathematical rigor:** Matching dependency versions and callable entry points preserves operational correctness; the README now references only functions that exist, ensuring reproducibility of the algorithmic pipeline.

---

### Chain B — Workflow Mapping (Step 2)
**Why:** Users need a coherent progression from install → demo → training → persistence.

**How:**
1. **Quick Start** mirrors executable scripts for training, demos, and folder-based training.
2. **Persistence section** now includes `.npz` export and `.safetensors` checkpoints to match available serialization formats.
3. **Adaptive learning and indexing** are documented with existing public methods, keeping API usage consistent.

**Mathematical rigor:** The README traces the same execution order used by the model’s probabilistic training loop and persistence flow, preserving state integrity from data ingestion to export.

---

### Chain C — Test Surface Alignment (Step 3)
**Why:** Tests are part of the executable contract; missing them in docs creates coverage blind spots.

**How:**
1. Enumerate all current test scripts in the README.
2. Keep the testing section aligned to real file names to avoid run-time resolution errors.

**Mathematical rigor:** The testing list maps directly to implemented verification paths, ensuring each stochastic/EM component is exercised by actual tests rather than undocumented assumptions.

---

## Result
The README now reflects the real APIs and scripts in the repository while preserving the existing model behavior and reproducibility guarantees.
