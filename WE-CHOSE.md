# WE-CHOSE — Multi-Perspective Design Rationale

## Perspective Mapping

### CEO Perspective (Outcome Quality + Marketability)
**Goal:** Deliver a model that keeps improving over time, retains its learned knowledge, and shows measurable adaptation.

**Choice:** Persistent adaptive statistics and memory blending were chosen because they enable streaming learning without catastrophic forgetting.
- **Mapping:** Logic Chain B and C in `LOGIC-MAP.md`.
- **Why:** These add a durable competitive edge (continuous learning) and provable stability (valid HMM parameters).

---

### Junior Developer Perspective (Maintainability + Clarity)
**Goal:** Keep changes additive and consistent with existing code structure.

**Choice:**
- New helpers (`_extract_features_for_adaptation`, `_compute_statistics_from_observations`, `_apply_adaptive_statistics`) reuse existing pipelines instead of replacing them.
- Adaptive state is encapsulated in `_AdaptiveStatistics`, avoiding scattered globals.
- Export/load pathways are extended without breaking previous interfaces.

**Mapping:** Logic Chain A (feature extraction consistency) and Chain B (statistics update).

---

### End-Customer Perspective (Usability + Audio Quality)
**Goal:** The model should adapt to new audio quickly without losing prior expressiveness.

**Choice:** Memory blending (`memory_blend`) ensures some portion of the original training distribution remains present while new audio updates are integrated.
- **Mapping:** Logic Chain B and C.
- **Why:** Users can feed new audio and keep the original “voice” intact, resulting in a persistent and adaptable generator rather than a fragile one.

---

## Decision Summary
We selected the blended adaptive-statistics path because it satisfies all three perspectives simultaneously:
- **CEO:** Demonstrates persistent learning and measurable adaptation.
- **Junior Dev:** Adds clear, compartmentalized logic that is easy to reason about.
- **End Customer:** Maintains creative continuity while still evolving from new audio.

---

# README Alignment Update — Perspective Mapping

### CEO Perspective (Credibility + Adoption)
**Goal:** Ensure first-run success and reduced onboarding friction.

**Choice:** Documented the current scripts, checkpoint formats, and dependencies so new users can reproduce results without guesswork.
- **Mapping:** LOGIC-MAP README Chain A and B.

---

### Junior Developer Perspective (Accuracy + Maintenance)
**Goal:** Keep documentation aligned with real APIs so changes do not drift.

**Choice:** Tied Quick Start and API examples directly to existing methods (`save_checkpoint`, `adapt_to_audio`, `build_feature_index`) and current test scripts.
- **Mapping:** LOGIC-MAP README Chain A and C.

---

### End-Customer Perspective (Usability + Trust)
**Goal:** Offer a clear, runnable path to generate audio and persist models.

**Choice:** Added explicit run commands and checkpoint examples so users can immediately validate output on their own data.
- **Mapping:** LOGIC-MAP README Chain B.

---

## Decision Summary
We updated the README to reduce ambiguity, reflect real entry points, and preserve the operational path from training to persistence.

---

# Smoothness-First Decoder — Perspective Mapping

### CEO Perspective (Reliability + Claims)
**Goal:** Ensure the decoder’s smoothness heuristic matches the product claim of stable, globally coherent paths.

**Choice:** A lexicographic smoothness-first decode ensures spectral smoothness is always minimized before likelihood trade-offs.
- **Mapping:** LOGIC-MAP Smoothness-First Chains A–C.

---

### Junior Developer Perspective (Debuggability + Intent)
**Goal:** Make the decode objective explicit so it is easy to reason about test expectations.

**Choice:** Split the transition, psychoacoustic, and smoothness objectives into separate DP tracks, then tie-break.
- **Mapping:** LOGIC-MAP Smoothness-First Chain B.

---

### End-Customer Perspective (Audio Quality + Consistency)
**Goal:** Produce sequences that avoid erratic state jumps, matching the smoother audio output users expect.

**Choice:** Prioritize smoothness at every decoding step so the resulting path avoids unnecessary spectral jumps.
- **Mapping:** LOGIC-MAP Smoothness-First Chain C.
# Production Hardening — Perspective Mapping

### CEO Perspective (Reliability + Risk Reduction)
**Goal:** Prevent production failures from invalid inputs or corrupted checkpoints.

**Choice:** Enforce strict validation of required checkpoint tensors and add early guards for empty training frames.
- **Mapping:** LOGIC-MAP Production Chain A and B.
- **Why:** Deterministic failure modes reduce operational risk and support predictable recovery workflows.

---

### Junior Developer Perspective (Debuggability + Test Stability)
**Goal:** Make test outcomes repeatable and errors actionable.

**Choice:** Add seeded RNG support in `test_utils` and update tests to use fixed seeds instead of global RNG state.
- **Mapping:** LOGIC-MAP Production Chain C.
- **Why:** Deterministic signals isolate regressions and allow direct comparison of spectral statistics across runs.

---

### End-Customer Perspective (Trust + Usability)
**Goal:** Provide clear feedback when inputs are invalid and preserve confidence in exported models.

**Choice:** Explicit error messages for empty inputs and missing checkpoint tensors, plus improved training file discovery warnings.
- **Mapping:** LOGIC-MAP Production Chain A and B.
- **Why:** Clear failure reasons shorten support cycles and improve user trust in the model pipeline.

---

## Decision Summary
We chose lexicographic decoding to honor smoothness claims, keep the logic readable, and deliver the most coherent output for listeners.
We chose strict validation and deterministic test signals because they simultaneously improve operational reliability, developer productivity, and end-user confidence.
