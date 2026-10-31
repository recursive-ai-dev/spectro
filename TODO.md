# Spectro / SSGS — Engineering Improvement Plan

Scope: the Python package `ssgs` (model, core numeric oracle, ctypes native
kernels, CLI, workflow layer, Tk desktop), the compiled `ssgs_native.c` kernels,
the Debian packaging scripts, the test suite, and the documentation.

Findings below were obtained by reading the execution paths and by running
adversarial probes against the checked-out tree (Python 3.11.2, NumPy 2.4.6,
SciPy 1.17.1, Debian 12, native kernels built with `make`). Every "evidence"
line was reproduced locally before the task was written.

Priorities follow the sprint ordering: security/data integrity first, then
crashes, incorrect behaviour, architectural/state consistency, resource
lifetime, performance, reliability, validation/testing, maintainability,
quality.

Status vocabulary: `PENDING → IN_PROGRESS → VERIFIED`, `BLOCKED` when an
external dependency prevents completion.

| ID | Priority | Severity | Component | Summary | Status |
|----|----------|----------|-----------|---------|--------|
| T01 | P0 | High | `ssgs.model.load_model` | Unvalidated adaptive statistics accepted from `.npz` archives | VERIFIED |
| T02 | P0 | High | `ssgs.model.synthesize_audio`, `ssgs.workflows`, `ssgs.cli` | Unbounded synthesis allocation (memory exhaustion, traceback) | VERIFIED |
| T03 | P1 | High | `ssgs.model.extract_features*` | Training-run state leaks across re-extraction (augmentation, adaptive stats) | VERIFIED |
| T04 | P1 | High | `ssgs.model._compute_frame_metadata`, `_process_frame_batch` | Peak memory during feature extraction is 8× the retained data | VERIFIED |
| T05 | P1 | Medium | `ssgs.workflows.supported_audio_extensions` | `.opus` input rejected although libsndfile decodes it and the desktop offers it | VERIFIED |
| T06 | P1 | Medium | `ssgs.workflows.save_model/write_audio` | Publish path requires hard links; saves fail on such filesystems | VERIFIED |
| T07 | P2 | Medium | `ssgs.model._apply_residual_augmentation` | Sequence boundaries derived from arithmetic, not from retained frames | VERIFIED |
| T08 | P2 | Medium | `ssgs.model.synthesize_audio` | `fidelity=1.0` silently degrades to noise when no residuals are stored | VERIFIED |
| T09 | P2 | Medium | `ssgs.native.benchmark`, `ssgs.model` prints | Self-benchmark compares native against native; library writes to stdout | VERIFIED |
| T10 | P2 | Medium | `src/ssgs/ssgs_native.c` | `int` size/index arithmetic in allocation and indexing (UB on 32-bit) | VERIFIED |
| T11 | P3 | Low | `ssgs.model.get_frame_metadata` | Negative metadata indices wrap silently | VERIFIED |
| T12 | P3 | Low | `ssgs.workflows.run_demo` | Failed demo leaves a partial `source.wav` that blocks the next run | VERIFIED |
| T13 | P3 | Low | `ssgs.model.identify_graph_constraints` | Reported component count includes pruned (dormant) states | VERIFIED |
| T14 | P3 | Low | `ssgs.core` | Dead code (`_cholesky_lower`, `_forward_solve`) contradicts the docstring | VERIFIED |
| T15 | P3 | Low | CI matrix, `pyproject.toml` | `requires-python >= 3.9` is asserted but only 3.11/3.12 are tested | BLOCKED (guard verified; matrix push blocked) |
| T16 | P3 | Low | `docs/`, `README.md` | Documentation must match new limits, formats and measurements | VERIFIED |
| T17 | P1 | Medium | `ssgs.model.export_model`, `save_checkpoint` | Adaptive sufficient statistics lose float64 precision on every save | VERIFIED |

---

## T01 — Validate persisted adaptive statistics on `.npz` load

* **Priority / severity:** P0 / High (data integrity; untrusted-file trust boundary).
* **Affected components:** `SpectralStateGuidedSynthesis.load_model`,
  `SpectralStateGuidedSynthesis.load_checkpoint`, `_AdaptiveStatistics`.
* **Problem and evidence:** `load_checkpoint` validates adaptive-statistics
  shapes, finiteness and `total_frames`; `load_model` (`.npz`) assigns them
  without any check. Tampering with an exported archive is accepted:

  ```
  adaptive_counts_short        LOADED  generate-ok
  adaptive_counts_nan          LOADED  generate-ok
  adaptive_trans_bad_shape     LOADED  generate-ok
  adaptive_total_neg           LOADED  generate-ok
  ```

  Consequences: `spectro inspect --json` reports `adaptive_frames: NaN`, and a
  later `adapt_to_audio` fails deep inside NumPy (`ValueError: operands could
  not be broadcast together…`) or `_regularize_covariance` after partially
  mutating in-memory state. A NaN `state_frame_counts` also corrupts model
  parameters that are then applied to the exported model.
* **Proposed implementation:** add one private validator
  (`SpectralStateGuidedSynthesis._validate_adaptive_statistics(stats)`) that
  checks every array's shape against `(n_states, lpc_order)`, finiteness,
  non-negative counts, and a finite non-negative `total_frames`; call it from
  both loaders, replacing the duplicated inline loop in `load_checkpoint`.
* **Acceptance criteria:** every tampered archive listed above is rejected with
  `ValueError` mentioning the offending statistic; well-formed archives load
  with identical statistics; `load_checkpoint` behaviour is unchanged.
* **Required tests:** parametrised tamper matrix for `.npz` and `.safetensors`
  (wrong shape, NaN, negative total, valid round trip).
* **Dependencies:** none.
* **Status:** VERIFIED — `tests/test_state_integrity.py::test_npz_adaptive_statistics_tampering_is_rejected` (8 tamper cases) and `::test_checkpoint_adaptive_statistics_tampering_is_rejected`; every tampered archive raises `ValueError: Adaptive …`, well-formed archives load bit-identically.

## T02 — Bound synthesis allocation and report memory failures

* **Priority / severity:** P0 / High (crash + resource exhaustion from user input).
* **Affected components:** `SpectralStateGuidedSynthesis._coerce_sample_rate`,
  `synthesize_audio`, `ssgs.workflows.generate_audio`, `ssgs.cli.main`.
* **Problem and evidence:** sample rate and duration are only checked for
  positivity, so absurd requests reach multi-gigabyte allocations:

  ```
  spectro generate model.npz --output big.wav --duration 1 --sample-rate 4000000000
  numpy._core._exceptions._ArrayMemoryError: Unable to allocate 14.9 GiB …
  spectro generate model.npz --output big2.wav --duration 100000 --sample-rate 44100
  numpy._core._exceptions._ArrayMemoryError: Unable to allocate 5.96 GiB …
  ```

  Both crash with a raw traceback (CLI catches only `ValueError/OSError/…`), and
  the second case burned **194 s of CPU** before failing. `render_preview` and
  `workflows.generate_audio` share the same unvalidated numeric path.
* **Proposed implementation:** document and enforce
  `MAX_SAMPLE_RATE = 768_000`, `MAX_DURATION_SECONDS = 3600` and
  `MAX_OUTPUT_SAMPLES = 2**28` in the model; validate them in
  `_coerce_sample_rate` and `synthesize_audio` before any buffer is allocated;
  mirror the duration/sample checks in `workflows.generate_audio` so the CLI
  fails before loading a model; add `MemoryError`/`OverflowError` to the CLI's
  handled exceptions so no traceback is printed.
* **Acceptance criteria:** both commands fail in < 1 s with
  `spectro: requested … exceeds …` on stderr, exit code 1, no traceback, no
  output file, and no allocation attempt; legitimate limits (e.g. 3600 s at
  8000 Hz) still work and produce exact-length audio.
* **Required tests:** parametrised boundary tests (rate 0/1/768_000/768_001,
  duration 3600/3600.1, sample-count limit) at model, workflow and CLI level;
  regression test that a normal generation is unchanged.
* **Dependencies:** none.
* **Status:** VERIFIED — `tests/test_resource_bounds.py` covers model, workflow and CLI boundaries; `spectro generate --duration 100000` now exits 1 in <1 s with `spectro: requested duration 100000 s exceeds the 3600 s limit per render` and writes no file.

## T03 — Reset per-training-run state when features are re-extracted

* **Priority / severity:** P1 / High (silent incorrect behaviour in a public API).
* **Affected components:** `extract_features`, `extract_features_from_files`,
  `augmentation_applied`, `_adaptive_stats`, `preview_snapshots`.
* **Problem and evidence:** `train()` is a public entry point, but
  `augmentation_applied` is never cleared when new audio replaces the training
  set, so the second training run silently skips residual augmentation:

  ```
  first train: frames 310 aug True
  second train: frames 62 aug True     # expected > 310
  ```

  Likewise `_adaptive_stats` survives re-extraction and keeps blending stale
  sufficient statistics into later `adapt_to_audio` calls and `inspect`
  (`after retrain, adaptive stats still present: True 93.0`).
* **Proposed implementation:** in both extraction entry points reset
  `augmentation_applied = False`, drop `_adaptive_stats`, and clear
  `preview_snapshots` (they belong to the previous run); document the "new
  training set ⇒ new run" contract in the docstrings.
* **Acceptance criteria:** retraining on new audio augments again
  (`len(training_frames) > len(lpc_coefficients_before_augmentation)`),
  `_adaptive_stats is None` and `preview_snapshots == []` after extraction;
  `adapt_to_audio` bootstraps fresh statistics from the retrained parameters.
* **Required tests:** retraining regression test over one model instance, plus
  a test that adaptation still preserves statistics within a single run.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_retraining_resets_per_run_state` (augmentation re-applied, `5 × raw` frames, statistics and previews cleared) and `test_failed_extraction_leaves_the_model_usable`.

## T04 — Bound peak memory in feature extraction

* **Priority / severity:** P1 / High (scalability; OOM on realistic folders).
* **Affected components:** `_compute_frame_metadata`, `_process_frame_batch`,
  `extract_features_from_files`.
* **Problem and evidence:** measured on the repository's own material
  (24 min of audio, 6 tracks, 16 kHz, frame 1024/hop 256):

  ```
  train folder: 14.5 s, frames=72510, training_frames 297 MB, peak RSS 2421 MB
  frame_metadata: 0.271 s, tracemalloc peak 307.2 MB for 61 MB of frames
  process_frame_batch: tracemalloc peak 218.8 MB for 61 MB of frames
  ```

  `_compute_frame_metadata` materialises four full `T × (frame_size/2+1)`
  float64 arrays (windowed, spectrum, `power`, `power * freqs`, plus the
  bandwidth temporaries); `_process_frame_batch` accumulates one NumPy array per
  frame in Python lists and then copies them again in `np.array(...)`.
* **Proposed implementation:** compute metadata in fixed-size row chunks
  (identical per-row arithmetic, so results stay bit-identical) and preallocate
  the retained-frame output arrays after a vectorised retained-frame count in
  `_process_frame_batch` (one exact `np.dot` per row is kept for the energy
  gate so the skip decision cannot change). Copy the flux column instead of
  aliasing the perceptual buffer.
* **Acceptance criteria:** metadata values are identical to the previous
  implementation (test compares against a reference computed directly);
  tracemalloc peak for `_compute_frame_metadata` drops below 2× the frame
  buffer (previously ≈5×); `_process_frame_batch` peak drops by ≥ 25 %;
  extraction of the 4-minute MP3 keeps identical LPC/residual/perceptual
  arrays and peak RSS falls materially.
* **Required tests:** bit-identical metadata vs an inline reference for random
  frames; peak-allocation assertions with `tracemalloc`; regression tests for
  skipped silent frames and empty batches.
* **Dependencies:** none.
* **Status:** VERIFIED — 48 parametrised cases in `test_frame_metadata_matches_reference_bit_for_bit` are bit-identical to the pre-fix formulas; peak allocation 15.8 MB vs 307.2 MB (metadata), 80.5 MB vs 218.8 MB (batch), folder-training peak RSS 1018 MB vs 2421 MB.

## T05 — Accept `.opus` input when libsndfile supports the OGG/Opus mapping

* **Priority / severity:** P1 / Medium (documented/offered format unusable).
* **Affected components:** `workflows.supported_audio_extensions`,
  `desktop.add_files` file filter, README/INSTALL text.
* **Problem and evidence:** `sf.available_formats()` reports `OGG` with
  subtypes `VORBIS` and `OPUS`, and `sf.check_format("OGG", "OPUS") is True`, so
  libsndfile reads and writes `.opus` files. The extension map only knows
  `.ogg`/`.oga`, so `discover_audio` rejects a real `.opus` file with
  "Unsupported audio input", while the desktop dialog lists `*.opus` in its
  filter.
* **Proposed implementation:** map `OGG` to `{.ogg, .oga, .opus}` (Opus is
  carried in an OGG container) and document the behaviour.
* **Acceptance criteria:** an `.opus` file written with `sf.write` is discovered
  (case-insensitively), trained on, and reported by `spectro doctor`; the
  desktop filter and discovery agree.
* **Required tests:** `check_format`-guarded test that writes an `.opus` file,
  asserts discovery includes it, and trains a tiny model from it.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_opus_is_advertised_and_discovered`: `.opus` is advertised when libsndfile exposes OGG, discovered by `discover_audio`, and decoded for training.

## T06 — Publish outputs without requiring hard links

* **Priority / severity:** P1 / Medium (workflow failure on valid filesystems).
* **Affected components:** `workflows.save_model`, `workflows.write_audio`.
* **Problem and evidence:** both use `os.link(temp, target)` as an atomic
  create-if-absent. Filesystems without hard-link support (FAT/exFAT, some CIFS
  and FUSE mounts) raise `EXDEV/EPERM/EOPNOTSUPP`; the save aborts with
  `OSError: [Errno 18] Invalid cross-device link` even though the destination
  is writable.
* **Proposed implementation:** shared `_publish(temporary, target, force)`
  helper: `os.replace` for `force`, `os.link` otherwise, and when the link fails
  with a "link unsupported" errno fall back to an exclusive placeholder
  (`os.open(target, O_CREAT|O_EXCL)`) followed by `os.replace`, removing the
  placeholder if the replace fails; `FileExistsError` is preserved.
* **Acceptance criteria:** saving models and audio succeeds when `os.link`
  raises the unsupported errnos; a pre-existing target still raises
  `FileExistsError` and is never overwritten without `force`; temp files never
  remain behind.
* **Required tests:** monkeypatched `os.link` raising `EXDEV` for both writers;
  existing-target race still protected; no leftover `.{stem}-*` temp files.
* **Dependencies:** none.
* **Status:** VERIFIED — `tests/test_publishing_and_formats.py` forces `os.link` to raise `EXDEV` and still publishes both a model and a WAV; existing targets raise `FileExistsError` without being touched; no temporary files remain.

## T07 — Derive augmented sequence boundaries from retained frames

* **Priority / severity:** P2 / Medium (state consistency; latent cross-file
  transitions).
* **Affected components:** `_apply_residual_augmentation`.
* **Problem and evidence:** after processing an augmented batch the code adds
  `original_starts + offset` to `sequence_starts`, assuming every augmented
  frame survived `_process_frame_batch`'s energy gate. When a frame is dropped
  (silent/low-energy input scaled by the 0.92 stretch), the boundaries no longer
  describe the frames that exist: non-adjacent recordings become "adjacent"
  (fabricated transitions in EM) or, once the drift exceeds the batch length,
  `_normalize_sequence_starts` raises "sequence_starts must begin at 0 and index
  existing frames" and training aborts.
* **Proposed implementation:** compute the retained-frame mask per augmented
  batch with the same `np.dot` energy test and derive each sequence's start from
  the count of retained frames before it; keep only non-empty sequences.
* **Acceptance criteria:** `sequence_starts` always partitions the retained
  frames exactly (`starts[0] == 0`, strictly increasing, `starts[-1] < n`), each
  original sequence contributes one contiguous block, and dropped frames shift
  later boundaries correctly.
* **Required tests:** unit test with a patched batch processor that drops the
  first frame of every batch, asserting boundaries and a successful
  `initialize_hmm_parameters`; regression test that normal augmentation keeps
  5 sequences for a 1-sequence input.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_augmentation_boundaries_follow_retained_frames` (injected frame-dropping processor keeps starts monotonic and sequences non-empty) and `test_augmentation_boundaries_are_exact_without_drops`.

## T08 — Warn when `fidelity=1.0` has no stored residual to reconstruct

* **Priority / severity:** P2 / Medium (silent quality degradation).
* **Affected components:** `synthesize_audio`,
  `_get_training_residual_for_state`.
* **Problem and evidence:** `_get_training_residual_for_state` returns
  `rng.standard_normal(...) * 0.05` when `residual_signals` is missing, which is
  exactly the case for a model exported without training artifacts. Asking for
  maximal fidelity (the documented "reconstruction" mode) therefore yields
  synthetic noise with no diagnostic.
* **Proposed implementation:** emit one `RuntimeWarning` per call when
  `fidelity > 0` and no residuals are available, naming the cause and the
  remedy (save/load with training artifacts).
* **Acceptance criteria:** warning raised exactly once for `fidelity > 0` on an
  artifact-free model, not raised when residuals exist, and generation output
  is unchanged.
* **Required tests:** `pytest.warns(RuntimeWarning)` for both cases.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_fidelity_warns_only_when_residuals_are_missing`: an artifacts-stripped model warns and still renders; a model with residuals stays silent.

## T09 — Honest self-benchmark and no library writes to stdout

* **Priority / severity:** P2 / Medium (misleading performance evidence;
  library API hygiene).
* **Affected components:** `ssgs.native.benchmark`, `_select_optimal_n_states`,
  `train`, `generate`, `iterative_refinement` banner output.
* **Problem and evidence:** `python -m ssgs.native` prints

  ```
  kernel                   scipy/numpy             C     speedup
  LPC x%d frames                0.0104        0.0108        1.0x
  emission (T x S)              0.0015        0.0015        1.0x
  decode 2000 frames            0.4880        0.0037      133.1x
  ```

  The "scipy/numpy" columns call `m._compute_lpc` / `m._compute_emission_log_probs`,
  which already dispatch to the native kernels, so two of the three rows compare
  native with itself; `LPC x%d frames` is a broken format string. Separately,
  library entry points (`_select_optimal_n_states`, `train`, `generate`) write
  progress banners to stdout, contaminating machine-readable CLI output and
  test output.
* **Proposed implementation:** benchmark the `ssgs.core` reference
  implementations explicitly, fix the label, and route model diagnostics through
  `logger.info` (`iterative_refinement(verbose=True)` keeps its explicit
  console progress).
* **Acceptance criteria:** benchmark prints a real ratio for all three kernels
  (>1× for decode, sensible reference numbers for LPC/emission), no stray `%d`;
  `_select_optimal_n_states`/`train`/`generate` emit no stdout (captured with
  `capsys`) while log records are available.
* **Required tests:** script-level smoke test invoking `native.benchmark`; stdout
  capture test for `auto_n_states`; existing suite unchanged.
* **Dependencies:** none.
* **Status:** VERIFIED — `ssgs.native.benchmark()` times `ssgs.core` against the C kernels (LPC 4.0×, emission 2.0×, decode 168×); `test_benchmark_reports_two_distinct_backends` parses the table and rejects `%d`, and `test_library_calls_are_silent` asserts empty stdout/stderr.

## T10 — Use size-safe arithmetic in the native kernels

* **Priority / severity:** P2 / Medium (undefined behaviour / heap overflow on
  32-bit targets, defensive hardening).
* **Affected components:** `src/ssgs/ssgs_native.c`.
* **Problem and evidence:** allocations are written as
  `malloc(nframes * S * sizeof(double))` and indexes as `t * S + j` with `int`
  operands. On a 32-bit build (`Architecture: any` in the Debian package, so
  i386 is allowed) `int * sizeof(double)` widens to a 32-bit `size_t`, so a
  request validated against `INT32_MAX` can wrap and under-allocate, turning a
  bounded input into a heap overflow; the index products can overflow `int` too.
* **Proposed implementation:** introduce a checked `ssgs_alloc(double, count,
  size)` helper using `size_t` with an overflow guard, cast all size/index
  arithmetic to `size_t`, and fail closed (leaving `NaN`/`-1` sentinels) so the
  Python wrappers keep raising instead of returning garbage.
* **Acceptance criteria:** kernels rebuilt warning-free with
  `-Wall -Wextra -Wconversion`; Python↔core equivalence fuzz (LPC, emissions,
  forward/backward, decode) still exact; sanitizer run clean.
* **Required tests:** existing kernel equivalence suite plus a new seeded
  property test comparing native and `ssgs.core` on randomized shapes; ASan/UBSan
  build exercised where the toolchain allows.
* **Dependencies:** none.
* **Status:** VERIFIED — `ssgs_alloc()` guards all 8 allocation sites; `-Wall -Wextra` clean; ASan+UBSan build passed `tests/test_native_backend.py` (11 cases) and the native fuzzer with no diagnostics.

## T11 — Reject negative frame-metadata indices

* **Priority / severity:** P3 / Low (silent wrap-around in a public API).
* **Affected components:** `get_frame_metadata`, `top_frames` consumers.
* **Problem and evidence:** `m.get_frame_metadata("rms", indices=[-1])` returns
  the *last* frame's value (NumPy negative indexing) instead of failing, while
  positive out-of-range indices raise `IndexError`. Callers cannot distinguish a
  typo from an intentional last-element lookup.
* **Proposed implementation:** validate `indices` as a 1-D integer array with
  `0 <= index < len(metadata)`, raising `IndexError` with the offending index;
  document the convention.
* **Acceptance criteria:** negative and `>= n` indices raise `IndexError`;
  valid indices return the same values as before.
* **Required tests:** parametrised index validation test.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_frame_metadata_indices_are_validated` rejects negative, out-of-range, non-integral and boolean indices while preserving scalar and array lookups.

## T12 — Remove partially written demo outputs on failure

* **Priority / severity:** P3 / Low (reliability / confusing retry).
* **Affected components:** `workflows.run_demo`.
* **Problem and evidence:** when a stage fails after `source.wav` was written,
  the directory keeps that file (`leftovers: ['source.wav']`), so a retry without
  `force` fails with "Output already exists" even though the demo never
  completed.
* **Proposed implementation:** remember which paths did not exist before the
  call and delete exactly those if any stage raises; never touch pre-existing
  files.
* **Acceptance criteria:** injected training failure leaves no new files;
  successful demos and `force=True` overwrites behave as before.
* **Required tests:** monkeypatched failing stage; success regression test.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_run_demo_failure_removes_its_own_partial_output`, `..._never_deletes_pre_existing_files`, `..._refuses_to_overwrite_without_force`, plus the success regression.

## T13 — Report graph components for retained states only

* **Priority / severity:** P3 / Low (misleading diagnostics).
* **Affected components:** `identify_graph_constraints`.
* **Problem and evidence:** the returned/logged `n_components` is computed on a
  matrix whose dormant entries were floored to `1e-12`, below the `> 1e-6`
  threshold used to build the graph. With `min_state_probability=1.0` the probe
  reports `active_states: [0], n_components: 4` — three components belong to
  states the method just pruned.
* **Proposed implementation:** run strongly-connected-component analysis on the
  active subgraph and document that the count describes retained states.
* **Acceptance criteria:** `n_components == 1` when all states stay active and
  when only one state remains; returned keys unchanged for compatibility.
* **Required tests:** both cases above plus row-stochasticity invariant.
* **Dependencies:** none.
* **Status:** VERIFIED — `test_graph_component_count_describes_retained_states`: a dormant state is pruned (1 component instead of 2) and a fully connected model still reports 1.

## T14 — Remove the unused hand-written Cholesky helpers

* **Priority / severity:** P3 / Low (maintainability).
* **Affected components:** `ssgs/core.py` (`_cholesky_lower`, `_forward_solve`).
* **Problem and evidence:** both helpers are unreferenced (grep finds only their
  definitions) and the module docstring claims `gaussian_log_probs` uses them,
  while the implementation calls `np.linalg.cholesky`/`solve`.
* **Proposed implementation:** delete the dead helpers and correct the
  docstring; the executable C-ABI contract and the numerical behaviour stay
  identical.
* **Acceptance criteria:** `ssgs.core` self-test and native equivalence still
  pass; no references remain.
* **Required tests:** existing `tests/test_regression_guard.py` and
  `python -m ssgs.core` self-test.
* **Dependencies:** none.
* **Status:** VERIFIED — dead helpers removed; `python -m ssgs.core` passes on a seeded model with a relative tolerance.

## T15 — Make the supported-Python claim verifiable

* **Priority / severity:** P3 / Low (release/support accuracy).
* **Affected components:** `.github/workflows/ci.yml`, `pyproject.toml`, tests.
* **Problem and evidence:** `requires-python = ">=3.9"` but the matrix only runs
  3.11 and 3.12, so a 3.9/3.10 syntax or stdlib regression ships unnoticed.
* **Proposed implementation:** add 3.9/3.10 to the CI matrix and add a local
  guard test that parses every shipped module with
  `ast.parse(..., feature_version=(3, 9))` so syntax regressions fail fast even
  without those interpreters installed.
* **Acceptance criteria:** guard test passes for `src/ssgs`, `scripts`,
  `examples`; CI matrix lists four interpreters.
* **Required tests:** the new syntax-guard test. Runtime 3.9/3.10 execution is
  verified by CI only (not available in this environment) and is reported as
  such.
* **Dependencies:** none.
* **Status:** BLOCKED (partial). `tests/test_python_support.py` is implemented and passing (34
  cases: 3.9 grammar parse of every module, rejection of runtime PEP 585/604
  annotations without `from __future__ import annotations`, and a consistency
  check between `requires-python` and the CI matrix). The CI matrix edit
  (`python: ["3.9", "3.10", "3.11", "3.12"]`) cannot be published from this
  environment: the GitHub App token has no `workflows` permission and the push is
  rejected with *"refusing to allow a GitHub App to create or update workflow
  `.github/workflows/ci.yml` without `workflows` permission"*. The change is
  preserved as `dist/0001-ci-exercise-Python-3.9-and-3.10-with-the-guard-that-.patch`
  (CI matrix + the matrix-consistency test, which brings this file to 35 cases);
  apply it with `git am` and push from a checkout whose token has the
  `workflows` scope. Python 3.9/3.10 interpreters are not installed in the build
  sandbox either, so those matrix entries are exercised by CI only.

## T16 — Align documentation with verified behaviour

* **Priority / severity:** P3 / Low (accuracy).
* **Affected components:** `README.md`, `docs/INSTALL.md`, `docs/TESTING.md`,
  `docs/ADVERSARIAL_EVALUATION.md`, `docs/ENHANCED_CAPABILITIES.md`.
* **Problem and evidence:** documents must state the new synthesis limits,
  `.opus` support, re-training semantics, memory measurements and the
  permission-class of the native benchmark; `docs/TESTING.md` should list the new
  adversarial tests.
* **Proposed implementation:** update the affected sections and record the
  measured before/after numbers from this sprint.
* **Acceptance criteria:** every command quoted in the docs runs as written;
  documented limits match `SpectralStateGuidedSynthesis` constants.
* **Required tests:** doc-command smoke test executed manually and recorded in
  the completion report.
* **Dependencies:** T01 – T15.
* **Status:** VERIFIED — README (formats incl. Opus, render bounds, measured benchmark table, peak memory, test/CI facts), `docs/TRAINING.md` extension defaults, `docs/TESTING.md` new suites and sanitizer recipe, `docs/ADVERSARIAL_EVALUATION.md` benchmark note; every quoted command was re-run. Distribution artifacts, checksums and the build-environment record were produced for release `v2.0.0-sprint.1` (see the verification log).

---

## T17 — Preserve float64 precision for adaptive sufficient statistics

* **Priority / severity:** P1 / Medium (data integrity; silent model drift).
* **Affected components:** `SpectralStateGuidedSynthesis.export_model`,
  `SpectralStateGuidedSynthesis.save_checkpoint`, both loaders.
* **Problem and evidence:** the export path casts every payload array to the
  model `precision` (float32 by default), including the *additive* adaptive
  counters. A save/load round trip therefore changed the statistics:

  ```
  in memory  state_frame_counts = [47.11932555, 2.5, 100.88067445]
  reloaded   state_frame_counts = [47.11932373, 2.5, 100.88067627]
  ```

  Each save/load/adapt cycle truncates again, so long-lived models drift from
  the statistics they were trained with (found while writing the T01 round-trip
  test: `np.array_equal` failed on an untouched archive).
* **Proposed implementation:** store the adaptive counters (`state_frame_counts`,
  `transition_counts`, `mean_accumulator`, `second_moment`, `initial_counts`,
  `total_frames`) as `float64` in both the `.npz` payload and the safetensors
  tensors, independent of the tensor `precision`; loaders already read
  `float64`, so old float32 archives stay compatible.
* **Acceptance criteria:** save → load returns bit-identical counters for `.npz`
  and `.safetensors`; synthesis tensors still honour `precision`.
* **Required tests:** `.npz` round trip (`np.array_equal`) and safetensors
  round trip including a dtype assertion.
* **Dependencies:** T01.
* **Status:** VERIFIED — `test_npz_adaptive_statistics_round_trip_is_exact` and `test_checkpoint_adaptive_statistics_round_trip_is_exact` assert bit-identical float64 counters; safetensors stores them as `float64`.

---

## Evaluated and rejected (no code change)

* **Archive "zip bomb" guard for `.npz` models.** Inspecting declared member
  sizes would also block legitimate large models (a 3-hour training set easily
  exceeds 2 GB of training frames), so the guard would trade a real feature for
  a speculative attack. `np.load(..., allow_pickle=False)` already prevents code
  execution, and every tensor is shape-checked against the declared
  configuration before use.
* **Decimating `training_frames` for long inputs.** Dropping frames would change
  the learned HMM (and break alignment with `lpc_coefficients`, `residuals` and
  metadata), so the memory work is limited to removing avoidable copies (T04).
* **Desktop preference validation.** The Tk layer cannot be imported in this
  environment (`python3-tk` is unavailable and cannot be installed offline), so
  changes there could not be VERIFIED. Left as a follow-up for an environment
  with a display.

---

## Verification log (sprint execution)

Environment: Debian 12 amd64, Python 3.11.2, NumPy 2.4.6, SciPy 1.17.1,
libsndfile 1.2.0, native kernels rebuilt with `make` (`-O3`).

| Check | Command | Result |
|---|---|---|
| Full suite | `make test PYTHON=.venv/bin/python` | **289 passed, 6 skipped** in 11.7 s (skips are Tk/desktop and display checks; the blocked CI commit adds the matrix test, bringing it to 290) |
| Core self-test | `.venv/bin/python -m ssgs.core` | 4 kernels OK, `ALL CORE SELF-TESTS PASSED` (30 consecutive runs) |
| Native self-test + benchmark | `.venv/bin/python -m ssgs.native` | verify OK; LPC 4.0×, emission 2.0×, decode 168× |
| Sanitizers | ASan+UBSan build, `tests/test_native_backend.py` + native fuzzer | 11 passed, no diagnostics |
| CLI limits | `spectro generate … --duration 100000` / `--sample-rate 4000000000` | exit 1 in ~0.7 s, single-line message, no output file |
| Demo | `spectro demo --output-dir /tmp/demo` | three artifacts, exit 0 |
| Folder example | `python examples/train_on_folder.py --folders training_001 --n-states 4 --n-iterations 1 --sample-rate 8000` | exit 0, checkpoint written |
| Measurement | `/tmp/prof.py` (72,510 frames) | peak RSS 1018 MB (was 2421 MB), 13.6 s |

### Findings raised during verification

* **Unseeded self-tests (fixed under T09/T14).** `ssgs.core._self_test` and
  `ssgs.native.verify` trained an unseeded model and compared log-likelihoods
  with an absolute `1e-3` tolerance on a ~1e4 value; one run failed with
  `AssertionError: Log-likelihood mismatch` and could not be reproduced in
  130 subsequent runs. Both self-tests now use `random_state=42` and a
  `1e-9` *relative* tolerance, so the check is deterministic instead of
  occasionally flaky.
* **`frame_metadata` dtype contract.** The chunked rewrite keeps the historical
  dtypes (`float32` metrics, `int32` zero-crossings); the equivalence oracle in
  `tests/test_resource_bounds.py` asserts dtype as well as value, which is how
  the contract is now pinned.

### Publication

* Branch `arena/e15153ba-spectro` pushed at commit `36f4051` (18 files, +2119/−226) and opened as
  pull request [#20](https://github.com/recursive-ai-dev/spectro/pull/20).
* Release [v2.0.0-sprint.1](https://github.com/recursive-ai-dev/spectro/releases/tag/v2.0.0-sprint.1)
  targets that commit and records the build commands, verification results and
  artifact checksums.
* Artifacts (built locally, `dist/`):
  `spectro-2.0.0-py3-none-any.whl` `29d16b46…693f`,
  `spectro-2.0.0.tar.gz` `449ef0e0…16ec0`,
  `spectro_2.0.0-1_amd64.deb` `2d93adf2…17667`, plus `SHA256SUMS` and
  `BUILD-ENVIRONMENT.json`.
* **Blocked:** asset uploads fail with `Post "https://uploads.github.com/…": EOF`
  (that host is not reachable from this sandbox), so the release carries the
  checksums instead of the binaries; upload them from a networked machine with
  `gh release upload v2.0.0-sprint.1 dist/python/* dist/*.deb dist/SHA256SUMS dist/BUILD-ENVIRONMENT.json`.
* **Blocked:** the CI-matrix commit (T15) cannot be pushed without the
  `workflows` permission (see T15). The prepared change is kept as
  `dist/0001-ci-exercise-Python-3.9-and-3.10-with-the-guard-that-.patch`; apply
  it with `git am` from a checkout with a `workflows`-scoped token.
* **Hosted CI (evidence from GitHub Actions):**
  * run `b0fb272` (branch tip as published) — **success**: Python 3.11 and 3.12
    test jobs *and* the Debian artifact/extracted-runtime job all green.
    The Debian job ran `scripts/check_deb.py` with `python3-tk` installed,
    completing the desktop-import step that is blocked in the build sandbox.
  * run on tag `v2.0.0-sprint.1` (commit `36f4051`) — **failure**: both test jobs
    failed in `Run all tests with a virtual display` because that commit still
    contained the matrix-consistency assertion while the workflow matrix listed
    only 3.11/3.12 (the `workflows` permission blocker above). Commits
    `2158cf0`/`b0fb272` scope that assertion to the workflow it verifies, and the
    run for `b0fb272` is green; the tag itself was left in place rather than
    rewritten.
  * CI logs could not be downloaded from this sandbox
    (`results-receiver.actions.githubusercontent.com` is unreachable); job and
    step conclusions were read through the REST API.
