
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

import json
import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

__version__ = "1.1.0"

from scipy import signal
from scipy.cluster import vq
from scipy.linalg import LinAlgError, cholesky, solve_toeplitz, solve_triangular
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

try:
    from scipy.sparse.csgraph import strongly_connected_components as _scipy_scc

    def _strong_components(graph):
        return _scipy_scc(graph, directed=True, connection="strong")

except ImportError:  # pragma: no cover - older SciPy fallback
    from scipy.sparse.csgraph import connected_components as _scipy_cc

    def _strong_components(graph):
        return _scipy_cc(graph, directed=True, connection="strong")

# Optional native/core backend: compiled kernels (ssgs_native.so) or pure-NumPy core
try:
    import ssgs_core
    _CORE_BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover
    ssgs_core = None  # type: ignore
    _CORE_BACKEND_AVAILABLE = False

try:
    import ssgs_native
    # Ensure the native .so is actually loadable (ctypes already loaded it)
    if ssgs_native._lib is not None:
        _NATIVE_BACKEND_AVAILABLE = True
    else:
        _NATIVE_BACKEND_AVAILABLE = False
except Exception:  # pragma: no cover - .so missing or ABI mismatch
    ssgs_native = None  # type: ignore
    _NATIVE_BACKEND_AVAILABLE = False

# Note: Warning filtering has been removed for production safety.
# If specific warnings need suppression, use context managers:
# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', category=SpecificWarning)
#     # your code here

_ERR_HMM_NOT_INITIALIZED = "HMM parameters are not initialized"


@dataclass
class FeatureIndex:
    """Container for reusable nearest-neighbor structures built on frame features."""

    feature_space: str
    tree: cKDTree
    mean: Union[np.ndarray, None]
    std: Union[np.ndarray, None]
    metadata_keys: Tuple[str, ...]


@dataclass
class FrameQueryResult:
    """Structured return type for frame-based searches and rankings."""

    indices: np.ndarray
    distances: Union[np.ndarray, None]
    metadata: Dict[str, np.ndarray]
    feature_space: str


@dataclass
class _ModelParameters:
    transition_matrix: Optional[np.ndarray] = None
    initial_probabilities: Optional[np.ndarray] = None
    state_means: Optional[np.ndarray] = None
    state_covariances: Optional[np.ndarray] = None


@dataclass
class _TrainingArtifacts:
    training_frames: Optional[np.ndarray] = None
    lpc_coefficients: Optional[np.ndarray] = None
    residual_signals: Optional[np.ndarray] = None
    frame_metadata: Dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate: Optional[int] = None
    perceptual_features: Optional[np.ndarray] = None
    spectral_flux: Optional[np.ndarray] = None
    augmentation_applied: bool = False
    preview_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    sequence_starts: Optional[np.ndarray] = None


@dataclass
class _ComputationCaches:
    smoothness: Optional[np.ndarray] = None
    transition_cost: Optional[np.ndarray] = None
    chol_factors: Optional[List[np.ndarray]] = None
    log_determinants: Optional[np.ndarray] = None
    emission_workspace: Optional[np.ndarray] = None
    emission_observations: Optional[np.ndarray] = None
    psychoacoustic_weight: Optional[np.ndarray] = None
    spectral_prototypes: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    state_assignments: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.smoothness = None
        self.transition_cost = None
        self.chol_factors = None
        self.log_determinants = None
        self.emission_workspace = None
        self.emission_observations = None
        self.psychoacoustic_weight = None
        self.spectral_prototypes = None
        self.state_assignments = None


@dataclass
class _AdaptiveStatistics:
    state_frame_counts: np.ndarray
    transition_counts: np.ndarray
    mean_accumulator: np.ndarray
    second_moment_accumulator: np.ndarray
    initial_counts: np.ndarray
    total_frames: float
    last_adaptation: Optional[str] = None


class SpectralStateGuidedSynthesis:
    """
    Spectral-State Guided Synthesis (SSGS) Algorithm
    A modular two-stage generative model that decouples high-level musical/phonetic 
    structure from low-level spectral content using HMM and LPC synthesis.
    """
    
    # Class constants for configuration
    DEFAULT_BIC_CANDIDATES = [8, 16, 24, 32]
    WAVELET_VARIANCE_HIGH = 0.1
    WAVELET_VARIANCE_MED = 0.05
    EXCITATION_NORMALIZATION_SCALE = 0.5
    # Minimum eigenvalue threshold for positive definite covariance matrices
    # after loading from checkpoints (precision loss during save/load)
    COVARIANCE_MIN_EIGENVALUE = 1e-4
    COVARIANCE_REGULARIZATION_EPSILON = 1e-5
    
    def __init__(
        self,
        n_states=34,
        lpc_order=12,
        frame_size=1024,
        hop_size=256,
        smoothness_weight=0.5,
        auto_n_states=False,
        adaptive_memory_limit=6000,
        random_state=None,
        covariance_floor=1e-4,
        transition_pseudocount=1e-6,
    ):
        """
        Initialize SSGS parameters
        
        Args:
            n_states: Number of HMM states (default: 34 for enhanced generative capabilities)
                     NOTE: This parameter is ignored when auto_n_states=True. In that case,
                     the optimal state count will be selected automatically during training.
            lpc_order: Order of Linear Prediction coefficients
            frame_size: Size of analysis frames
            hop_size: Hop size between frames
            smoothness_weight: Weight for spectral smoothness heuristic
            auto_n_states: If True, automatically select optimal n_states using BIC.
                          When enabled, the n_states parameter is ignored and the optimal
                          value is determined during initialize_hmm_parameters().
        """
        def positive_int(name, value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return int(value)

        self.n_states = positive_int("n_states", n_states)
        self.lpc_order = positive_int("lpc_order", lpc_order)
        self.frame_size = positive_int("frame_size", frame_size)
        self.hop_size = positive_int("hop_size", hop_size)
        self.adaptive_memory_limit = positive_int("adaptive_memory_limit", adaptive_memory_limit)
        if self.frame_size <= self.lpc_order:
            raise ValueError("frame_size must be greater than lpc_order")
        if self.hop_size > self.frame_size:
            raise ValueError("hop_size must be less than or equal to frame_size")
        for name, value, allow_zero in (
            ("smoothness_weight", smoothness_weight, True),
            ("covariance_floor", covariance_floor, False),
            ("transition_pseudocount", transition_pseudocount, False),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} must be a finite real number")
            if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                bound = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {bound}")
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None")

        self.smoothness_weight = float(smoothness_weight)
        self.auto_n_states = bool(auto_n_states)
        self.covariance_floor = float(covariance_floor)
        self.transition_pseudocount = float(transition_pseudocount)
        self.random_state = int(random_state) if random_state is not None else int(np.random.SeedSequence().entropy)
        self._rng = np.random.default_rng(self.random_state)
        if self.auto_n_states:
            logger.info("auto_n_states enabled: n_states parameter will be determined automatically")
        
        # Internal state containers keep responsibilities isolated yet accessible.
        self._params = _ModelParameters()
        self._artifacts = _TrainingArtifacts()
        self._caches = _ComputationCaches()
        self._feature_indexes: Dict[Tuple[str, bool], FeatureIndex] = {}
        self._max_preview_snapshots = 6
        self._adaptive_stats: Optional[_AdaptiveStatistics] = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    # Properties expose internal containers without changing external API.
    @property
    def transition_matrix(self):
        return self._params.transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, matrix):
        self._params.transition_matrix = matrix
        self._invalidate_model_caches()

    @property
    def initial_probabilities(self):
        return self._params.initial_probabilities

    @initial_probabilities.setter
    def initial_probabilities(self, probabilities):
        self._params.initial_probabilities = probabilities
        self._invalidate_model_caches()

    @property
    def state_means(self):
        return self._params.state_means

    @state_means.setter
    def state_means(self, means):
        self._params.state_means = means
        self._invalidate_model_caches()

    @property
    def state_covariances(self):
        return self._params.state_covariances

    @state_covariances.setter
    def state_covariances(self, covariances):
        self._params.state_covariances = covariances
        self._invalidate_model_caches()

    @property
    def training_frames(self):
        return self._artifacts.training_frames

    @training_frames.setter
    def training_frames(self, frames):
        self._artifacts.training_frames = frames
        self._invalidate_model_caches()

    @property
    def lpc_coefficients(self):
        return self._artifacts.lpc_coefficients

    @lpc_coefficients.setter
    def lpc_coefficients(self, coeffs):
        self._artifacts.lpc_coefficients = coeffs
        self._invalidate_model_caches()

    @property
    def residual_signals(self):
        return self._artifacts.residual_signals

    @residual_signals.setter
    def residual_signals(self, residuals):
        self._artifacts.residual_signals = residuals
        self._invalidate_model_caches()

    @property
    def perceptual_features(self):
        return self._artifacts.perceptual_features

    @perceptual_features.setter
    def perceptual_features(self, features):
        self._artifacts.perceptual_features = features
        self._invalidate_model_caches()

    @property
    def spectral_flux(self):
        return self._artifacts.spectral_flux

    @spectral_flux.setter
    def spectral_flux(self, values):
        self._artifacts.spectral_flux = values
        self._invalidate_model_caches()

    @property
    def frame_metadata(self):
        return self._artifacts.frame_metadata

    @frame_metadata.setter
    def frame_metadata(self, metadata):
        if metadata is None:
            self._artifacts.frame_metadata = {}
        else:
            self._artifacts.frame_metadata = metadata
        self._invalidate_model_caches()

    @property
    def sequence_starts(self):
        return self._artifacts.sequence_starts

    @sequence_starts.setter
    def sequence_starts(self, starts):
        if starts is None:
            self._artifacts.sequence_starts = None
        else:
            values = np.asarray(starts)
            if values.ndim != 1 or values.size == 0 or values.dtype.kind == "b":
                raise ValueError("sequence_starts must be a non-empty 1-D integer array")
            if values.dtype.kind not in "iu":
                if values.dtype.kind != "f" or not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
                    raise ValueError("sequence_starts must contain integer indices")
            self._artifacts.sequence_starts = values.astype(np.int64, copy=True)
        self._invalidate_model_caches()

    @property
    def sample_rate(self):
        return self._artifacts.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value is None:
            self._artifacts.sample_rate = None
        else:
            self._artifacts.sample_rate = self._coerce_sample_rate(value)

    @property
    def augmentation_applied(self):
        return self._artifacts.augmentation_applied

    @augmentation_applied.setter
    def augmentation_applied(self, applied):
        self._artifacts.augmentation_applied = bool(applied)

    @property
    def preview_snapshots(self):
        return self._artifacts.preview_snapshots

    def clear_preview_snapshots(self):
        self._artifacts.preview_snapshots.clear()

    def _invalidate_model_caches(self):
        """Invalidate derived data whenever a public model/artifact array is replaced."""
        if hasattr(self, "_caches"):
            self._reset_caches()

    def _reset_caches(self):
        self._caches.reset()
        self._reset_feature_indexes()

    def _reset_feature_indexes(self):
        """Drop any cached nearest-neighbor structures tied to frame features."""
        self._feature_indexes = {}

    def _normalize_sequence_starts(self, n_frames, sequence_starts=None):
        """Return validated starts for concatenated independent sequences."""
        if isinstance(n_frames, (bool, np.bool_)) or int(n_frames) <= 0:
            raise ValueError("observations must contain at least one frame")
        n_frames = int(n_frames)
        if sequence_starts is None:
            if (
                self.sequence_starts is not None
                and self.lpc_coefficients is not None
                and len(self.lpc_coefficients) == n_frames
            ):
                sequence_starts = self.sequence_starts
            else:
                sequence_starts = np.array([0], dtype=np.int64)
        starts = np.asarray(sequence_starts)
        if starts.ndim != 1 or starts.size == 0 or starts.dtype.kind == "b":
            raise ValueError("sequence_starts must be a non-empty 1-D integer array")
        if starts.dtype.kind not in "iu":
            if starts.dtype.kind != "f" or not np.all(np.isfinite(starts)) or not np.all(starts == np.floor(starts)):
                raise ValueError("sequence_starts must contain integer frame indices")
        starts = starts.astype(np.int64, copy=False)
        if starts[0] != 0 or np.any(starts < 0) or np.any(starts >= n_frames):
            raise ValueError("sequence_starts must begin at 0 and index existing frames")
        if np.any(np.diff(starts) <= 0):
            raise ValueError("sequence_starts must be strictly increasing")
        return starts

    def _sequence_slices(self, n_frames, sequence_starts=None):
        starts = self._normalize_sequence_starts(n_frames, sequence_starts)
        ends = np.concatenate((starts[1:], np.array([int(n_frames)], dtype=np.int64)))
        return [(int(start), int(end)) for start, end in zip(starts, ends)]

    def _regularize_covariance(self, covariance, *, floor=None):
        """Symmetrize and make a covariance matrix numerically positive definite."""
        floor = self.covariance_floor if floor is None else float(floor)
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (self.lpc_order, self.lpc_order):
            raise ValueError(f"covariance must have shape ({self.lpc_order}, {self.lpc_order})")
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance contains NaN or infinite values")
        cov = 0.5 * (cov + cov.T)
        minimum = float(np.min(np.linalg.eigvalsh(cov)))
        if minimum < floor:
            cov = cov + np.eye(self.lpc_order) * (floor - minimum)
        return 0.5 * (cov + cov.T)

    @staticmethod
    def _normalize_probabilities(probabilities, *, axis=None, fallback=None):
        """Normalize non-negative finite probabilities, recovering zero-mass rows."""
        probs = np.asarray(probabilities, dtype=np.float64)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if axis is None:
            total = float(probs.sum())
            if total <= 0:
                if fallback is None:
                    return np.full(probs.shape, 1.0 / max(probs.size, 1), dtype=np.float64)
                probs = np.asarray(fallback, dtype=np.float64).copy()
                total = float(probs.sum())
                if total <= 0 or not np.isfinite(total):
                    return np.full(probs.shape, 1.0 / max(probs.size, 1), dtype=np.float64)
            return probs / total
        totals = probs.sum(axis=axis, keepdims=True)
        zero = totals <= 0
        if np.any(zero):
            if fallback is None:
                probs = np.where(zero, 1.0, probs)
            else:
                fallback = np.asarray(fallback, dtype=np.float64)
                probs = np.where(zero, fallback, probs)
            totals = probs.sum(axis=axis, keepdims=True)
        return probs / np.maximum(totals, np.finfo(np.float64).tiny)

    def _validated_model_parameters(self):
        """Return shape-checked, normalized model arrays without mutating public state."""
        if any(value is None for value in (
            self.transition_matrix,
            self.initial_probabilities,
            self.state_means,
            self.state_covariances,
        )):
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)
        transition = np.asarray(self.transition_matrix, dtype=np.float64)
        initial = np.asarray(self.initial_probabilities, dtype=np.float64)
        means = np.asarray(self.state_means, dtype=np.float64)
        covariances = np.asarray(self.state_covariances, dtype=np.float64)
        if transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition_matrix shape does not match n_states")
        if initial.shape != (self.n_states,):
            raise ValueError("initial_probabilities shape does not match n_states")
        if means.shape != (self.n_states, self.lpc_order):
            raise ValueError("state_means shape does not match model configuration")
        if covariances.shape != (self.n_states, self.lpc_order, self.lpc_order):
            raise ValueError("state_covariances shape does not match model configuration")
        for name, array in (("transition_matrix", transition), ("initial_probabilities", initial),
                            ("state_means", means), ("state_covariances", covariances)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")
        if np.any(transition < 0) or np.any(initial < 0):
            raise ValueError("transition and initial probabilities must be non-negative")
        transition = self._normalize_probabilities(transition, axis=1)
        initial = self._normalize_probabilities(initial)
        return transition, initial, means, covariances

    def _record_preview_snapshot(
        self,
        iteration: int,
        frames: int,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        snapshot = {
            "iteration": int(iteration),
            "frames": int(frames),
            "sample_rate": int(sample_rate),
            "audio": np.asarray(audio, dtype=np.float32).copy(),
        }
        container = self.preview_snapshots
        container.append(snapshot)
        if len(container) > self._max_preview_snapshots:
            del container[0]
        return snapshot

    def _compute_and_cache_cholesky_factors(self):
        """Validate covariance dimensions and cache robust lower Cholesky factors."""
        if self.state_covariances is None:
            return
        if np.shape(self.state_covariances) != (self.n_states, self.lpc_order, self.lpc_order):
            raise ValueError(
                "state_covariances must have shape "
                f"({self.n_states}, {self.lpc_order}, {self.lpc_order})"
            )

        caches = self._caches
        if caches.chol_factors is not None and len(caches.chol_factors) == self.n_states:
            return

        covariances = np.asarray(self.state_covariances, dtype=np.float64).copy()
        chol_cache: List[np.ndarray] = []
        log_det_cache = np.empty(self.n_states, dtype=np.float64)
        eye = np.eye(self.lpc_order, dtype=np.float64)

        for state in range(self.n_states):
            cov = self._regularize_covariance(covariances[state])
            jitter = 0.0
            for attempt in range(8):
                try:
                    chol = cholesky(
                        cov + eye * jitter,
                        lower=True,
                        overwrite_a=False,
                        check_finite=False,
                    )
                    if jitter:
                        cov = cov + eye * jitter
                    break
                except LinAlgError:
                    jitter = self.covariance_floor if attempt == 0 else jitter * 10.0
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                cov = (eigenvectors * np.maximum(eigenvalues, self.covariance_floor)) @ eigenvectors.T
                cov = 0.5 * (cov + cov.T)
                chol = cholesky(cov, lower=True, check_finite=False)
            covariances[state] = cov
            chol_cache.append(chol)
            log_det_cache[state] = 2.0 * np.sum(np.log(np.diag(chol)))

        # Keep any repaired matrix in the model, but avoid triggering cache reset here.
        self._params.state_covariances = covariances
        caches.chol_factors = chol_cache
        caches.log_determinants = log_det_cache

    # ------------------------------------------------------------------
    # Data-centric helpers
    # ------------------------------------------------------------------

    def _compute_frame_metadata(self, sample_rate):
        """Compute frame-level metrics useful for indexing and retrieval."""
        if self.training_frames is None or len(self.training_frames) == 0:
            self.frame_metadata = {}
            return

        frames = self.training_frames
        frame_len = frames.shape[1]

        energy = np.sum(frames * frames, axis=1)
        rms = np.sqrt(energy / frame_len)
        zero_crossings = np.count_nonzero(
            np.signbit(frames[:, :-1]) != np.signbit(frames[:, 1:]), axis=1
        )

        metadata: Dict[str, np.ndarray] = {
            "energy": energy.astype(np.float32, copy=False),
            "rms": rms.astype(np.float32, copy=False),
            "zero_crossings": zero_crossings.astype(np.int32, copy=False),
        }

        if self.lpc_coefficients is not None and len(self.lpc_coefficients) == len(frames):
            metadata["lpc_norm"] = np.linalg.norm(self.lpc_coefficients, axis=1).astype(  # pylint: disable=no-member
                np.float32, copy=False
            )

        if (
            self.residual_signals is not None
            and len(self.residual_signals) == len(frames)
            and self.residual_signals.shape[1] > 0
        ):
            residual_energy = np.sum(self.residual_signals * self.residual_signals, axis=1)
            residual_rms = np.sqrt(residual_energy / self.residual_signals.shape[1])
            metadata["residual_rms"] = residual_rms.astype(np.float32, copy=False)  # pylint: disable=no-member

        if sample_rate:
            window = np.hanning(frame_len)
            windowed = frames * window
            spectrum = np.abs(np.fft.rfft(windowed, axis=1))
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sample_rate))
            power = spectrum + 1e-12

            centroid = (power * freqs).sum(axis=1) / power.sum(axis=1)
            metadata["spectral_centroid"] = centroid.astype(np.float32, copy=False)  # pylint: disable=no-member

            bandwidth = np.sqrt(
                ((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / power.sum(axis=1)
            )
            metadata["spectral_bandwidth"] = bandwidth.astype(np.float32, copy=False)  # pylint: disable=no-member

            dominant_idx = np.argmax(power, axis=1)
            metadata["dominant_frequency"] = freqs[dominant_idx].astype(np.float32, copy=False)  # pylint: disable=no-member

        self.frame_metadata = metadata

    def _prepare_feature_matrix(self, feature_space):
        if self.lpc_coefficients is None or len(self.lpc_coefficients) == 0:
            raise ValueError("No LPC coefficients available; call extract_features first")

        if feature_space == "lpc":
            matrix = self.lpc_coefficients.astype(np.float64, copy=False)
            metadata_keys: Tuple[str, ...] = ()
        elif feature_space == "extended":
            required_keys = ("energy", "spectral_centroid", "spectral_bandwidth")
            missing = [key for key in required_keys if key not in self.frame_metadata]
            if missing:
                raise ValueError(
                    "Missing frame metadata for extended feature space: " + ", ".join(missing)
                )
            extras = [
                self.frame_metadata[key][:, None].astype(np.float64, copy=False)  # pylint: disable=no-member
                for key in required_keys
            ]
            matrix = np.hstack([self.lpc_coefficients.astype(np.float64, copy=False)] + extras)
            metadata_keys = required_keys
        elif feature_space == "residual":
            if self.residual_signals is None or len(self.residual_signals) == 0:
                raise ValueError("Residual signals not available; call extract_features first")
            matrix = self.residual_signals.astype(np.float64, copy=False)
            metadata_keys = ()
        else:
            raise ValueError(f"Unknown feature_space '{feature_space}'")

        if matrix.ndim != 2:
            raise ValueError("Feature matrix must be 2-D")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Feature matrix must be non-empty")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Feature matrix contains NaN or infinite values")

        return matrix, tuple(metadata_keys)

    def build_feature_index(self, feature_space="lpc", *, normalize=True):
        """Build or retrieve a cached nearest-neighbor index over frame features."""
        key = (feature_space, bool(normalize))
        if key in self._feature_indexes:
            return self._feature_indexes[key]

        matrix, metadata_keys = self._prepare_feature_matrix(feature_space)

        mean = matrix.mean(axis=0) if normalize else None
        std = matrix.std(axis=0) + 1e-12 if normalize else None
        working = matrix if not normalize else (matrix - mean) / std

        tree = cKDTree(working)
        index = FeatureIndex(
            feature_space=feature_space,
            tree=tree,
            mean=mean if normalize else None,
            std=std if normalize else None,
            metadata_keys=metadata_keys,
        )
        self._feature_indexes[key] = index
        return index

    def _encode_feature_vector(self, vector, feature_space, index: FeatureIndex):
        if feature_space == "extended" and isinstance(vector, dict):
            if "lpc" not in vector:
                raise ValueError("Extended queries require an 'lpc' entry in the vector dictionary")
            parts: List[np.ndarray] = [np.asarray(vector["lpc"], dtype=np.float64)]
            for key in index.metadata_keys:
                if key not in vector:
                    raise ValueError(f"Vector dictionary missing required key '{key}'")
                parts.append(np.atleast_1d(np.asarray(vector[key], dtype=np.float64)))
            arr = np.concatenate(parts, axis=0)
        else:
            arr = np.asarray(vector, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError("Feature query vector must be 1-D")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Feature query vector must contain only finite values")

        expected = index.tree.m
        if arr.shape[0] != expected:
            raise ValueError(f"Expected vector of length {expected}, got {arr.shape[0]}")

        if index.mean is not None and index.std is not None:
            arr = (arr - index.mean) / index.std

        return arr

    def query_similar_frames(
        self,
        frame_idx,
        *,
        k=5,
        feature_space="lpc",
        normalize=True,
        include_metadata=True,
    ):
        """Return the nearest frames to the provided frame index in feature space."""
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if isinstance(frame_idx, (bool, np.bool_)) or not isinstance(frame_idx, (int, np.integer)):
            raise IndexError("frame_idx must be an integer")

        index = self.build_feature_index(feature_space, normalize=normalize)

        total_frames = index.tree.n
        if total_frames == 0:
            raise ValueError("Feature index is empty")
        if frame_idx < 0 or frame_idx >= total_frames:
            raise IndexError("frame_idx out of range")

        query_vec = index.tree.data[frame_idx]
        max_k = min(total_frames, k + 1)
        distances, candidate_indices = index.tree.query(query_vec, k=max_k)

        candidate_indices = np.atleast_1d(np.asarray(candidate_indices, dtype=np.int64))
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))

        mask = candidate_indices != frame_idx
        filtered_indices = candidate_indices[mask][:k]
        filtered_distances = distances[mask][:k]

        metadata = {}
        if include_metadata and self.frame_metadata:
            for key, values in self.frame_metadata.items():
                metadata[key] = values[filtered_indices]

        return FrameQueryResult(
            indices=filtered_indices,
            distances=filtered_distances,
            metadata=metadata,
            feature_space=feature_space,
        )

    def query_frames_by_features(
        self,
        vector,
        *,
        k=5,
        feature_space="lpc",
        normalize=True,
        include_metadata=True,
    ):
        """Nearest-neighbor query using an explicit feature vector."""
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")

        index = self.build_feature_index(feature_space, normalize=normalize)
        encoded = self._encode_feature_vector(vector, feature_space, index)

        total_frames = index.tree.n
        if total_frames == 0:
            raise ValueError("Feature index is empty")

        effective_k = min(k, total_frames)
        distances, candidate_indices = index.tree.query(encoded, k=effective_k)

        candidate_indices = np.atleast_1d(np.asarray(candidate_indices, dtype=np.int64))
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))

        metadata = {}
        if include_metadata and self.frame_metadata:
            for key, values in self.frame_metadata.items():
                metadata[key] = values[candidate_indices]

        return FrameQueryResult(
            indices=candidate_indices,
            distances=distances,
            metadata=metadata,
            feature_space=feature_space,
        )

    def get_frame_metadata(self, key=None, indices=None):
        """Retrieve frame metadata, optionally filtered by key or by frame indices."""
        if not self.frame_metadata:
            return None if key is None else np.array([], dtype=np.float32)

        if key is None:
            if indices is None:
                return {k: v.copy() for k, v in self.frame_metadata.items()}
            idx = np.asarray(indices, dtype=np.int64)
            return {k: v[idx] for k, v in self.frame_metadata.items()}

        if key not in self.frame_metadata:
            raise KeyError(f"Unknown metadata key '{key}'")

        values = self.frame_metadata[key]
        if indices is None:
            return values.copy()

        idx = np.asarray(indices, dtype=np.int64)
        return values[idx]

    def top_frames(self, metric, *, k=10, descending=True):
        """Rank frames by a metadata metric and return the top results."""
        if metric not in self.frame_metadata:
            raise KeyError(f"Unknown metadata metric '{metric}'")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")

        values = self.frame_metadata[metric]
        total = values.shape[0]
        if total == 0:
            raise ValueError("No frame metadata available")

        effective_k = min(k, total)
        order = np.argsort(values)
        if descending:
            order = order[::-1]

        selected = order[:effective_k]
        metadata = {key: val[selected] for key, val in self.frame_metadata.items()}

        return FrameQueryResult(
            indices=selected,
            distances=None,
            metadata=metadata,
            feature_space=f"rank:{metric}",
        )

    @staticmethod
    def _coerce_sample_rate(sample_rate, name="sample_rate"):
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return sample_rate

    def _prepare_audio(self, audio_signal, source_sample_rate, target_sample_rate=None):
        """Validate audio, downmix channels, and band-limited-resample if needed."""
        source_sample_rate = self._coerce_sample_rate(source_sample_rate, "source_sample_rate")
        target_sample_rate = self._coerce_sample_rate(
            source_sample_rate if target_sample_rate is None else target_sample_rate,
            "target_sample_rate",
        )
        raw = np.asarray(audio_signal)
        if raw.dtype.kind not in "fiu":
            raise ValueError("audio_signal must contain real numeric samples")
        integer_input = raw.dtype.kind in "iu"
        integer_scale = None
        integer_offset = 0.0
        if integer_input:
            limits = np.iinfo(raw.dtype)
            if raw.dtype.kind == "u":
                integer_offset = float(limits.max + 1) / 2.0
                integer_scale = integer_offset
            else:
                integer_scale = float(max(abs(limits.min), abs(limits.max)))
        if raw.ndim == 2:
            if raw.shape[0] == 0 or raw.shape[1] == 0:
                raise ValueError("audio_signal must contain at least one sample and channel")
            # soundfile's standard layout is (samples, channels); average avoids
            # phase-dependent selection of a single channel.
            raw = raw.astype(np.float64, copy=False).mean(axis=1)
        elif raw.ndim != 1:
            raise ValueError("audio_signal must be mono (1-D) or samples-by-channels (2-D)")
        if integer_input:
            audio = (raw.astype(np.float64) - integer_offset) / integer_scale
        else:
            audio = raw.astype(np.float64, copy=False)
        if audio.size == 0:
            raise ValueError("audio_signal is empty")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio_signal contains NaN or infinite samples")
        if source_sample_rate != target_sample_rate:
            divisor = math.gcd(source_sample_rate, target_sample_rate)
            audio = signal.resample_poly(
                audio,
                target_sample_rate // divisor,
                source_sample_rate // divisor,
            )
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ValueError("audio preprocessing produced no finite samples")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _frame_signal(self, audio_signal):
        """Frame mono audio with complete coverage and deterministic zero padding."""
        audio_signal = np.asarray(audio_signal, dtype=np.float32)
        if audio_signal.ndim != 1:
            raise ValueError("audio_signal must be 1-D; downmix channels before framing")
        if audio_signal.size == 0:
            raise ValueError("audio_signal is empty")
        if not np.all(np.isfinite(audio_signal)):
            raise ValueError("audio_signal contains NaN or infinite samples")

        n_samples = audio_signal.size
        if n_samples < self.frame_size:
            target_length = self.frame_size
        else:
            trailing = (n_samples - self.frame_size) % self.hop_size
            target_length = n_samples if trailing == 0 else n_samples + self.hop_size - trailing
        if target_length != n_samples:
            audio_signal = np.pad(audio_signal, (0, target_length - n_samples), mode="constant")

        # sliding_window_view is safer than hand-crafted as_strided shapes/strides;
        # slicing retains the repository's exact frame/hop convention.
        frames = np.lib.stride_tricks.sliding_window_view(audio_signal, self.frame_size)
        return np.ascontiguousarray(frames[:: self.hop_size], dtype=np.float32)

    def _autocorrelation(self, frame):
        """Compute a finite, lag-limited autocorrelation sequence for LPC."""
        frame = np.asarray(frame, dtype=np.float64)
        if frame.ndim != 1 or frame.size <= self.lpc_order:
            raise ValueError("frame must be 1-D and longer than lpc_order")
        if not np.all(np.isfinite(frame)):
            raise ValueError("frame contains NaN or infinite samples")
        corr = np.correlate(frame, frame, mode="full")
        mid = len(frame) - 1
        return corr[mid : mid + self.lpc_order + 1]

    def _compute_lpc(self, frame):
        """
        Compute LPC coefficients using autocorrelation and Levinson-Durbin recursion.
        """
        autocorr = self._autocorrelation(frame)
        if _NATIVE_BACKEND_AVAILABLE:
            coeffs = ssgs_native.lpc_levinson(autocorr, self.lpc_order)
            return coeffs.astype(np.float32, copy=False)
        if _CORE_BACKEND_AVAILABLE:
            coeffs = ssgs_core.lpc_coefficients(autocorr, self.lpc_order)
            return coeffs.astype(np.float32, copy=False)
        if autocorr[0] <= 1e-8:
            coeffs = np.zeros(self.lpc_order + 1, dtype=np.float32)
            coeffs[0] = 1.0
            return coeffs

        try:
            toeplitz_col = autocorr[:-1]
            toeplitz_row = autocorr[:-1]
            rhs = autocorr[1:]
            solution = solve_toeplitz((toeplitz_col, toeplitz_row), rhs)
            coeffs = np.concatenate(([1.0], -solution))
        except LinAlgError:
            coeffs = np.zeros(self.lpc_order + 1, dtype=np.float32)
            coeffs[0] = 1.0
        return coeffs.astype(np.float32, copy=False)
        
    def _analyze_frame(self, frame):
        """
        Analyze a single frame using Linear Prediction
        
        Args:
            frame: Audio frame samples
            
        Returns:
            lpc_coeffs: LPC coefficients
            residual: Residual error signal
        """
        windowed = frame * np.hanning(len(frame))
        lpc_coeffs = self._compute_lpc(windowed)

        # Residual through prediction error filter
        residual = signal.lfilter(lpc_coeffs, [1.0], windowed)
        return lpc_coeffs, residual

    def _mel_filter_bank(self, sample_rate, n_fft, n_mels=32, fmin=20.0, fmax=None):
        sample_rate = self._coerce_sample_rate(sample_rate)
        if isinstance(n_fft, (bool, np.bool_)) or not isinstance(n_fft, (int, np.integer)) or n_fft <= 1:
            raise ValueError("n_fft must be an integer greater than one")
        if isinstance(n_mels, (bool, np.bool_)) or not isinstance(n_mels, (int, np.integer)) or n_mels <= 0:
            raise ValueError("n_mels must be a positive integer")
        fmax = sample_rate / 2.0 if fmax is None else float(fmax)
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin < 0 or fmax <= fmin or fmax > sample_rate / 2.0:
            raise ValueError("mel frequency bounds must satisfy 0 <= fmin < fmax <= Nyquist")

        n_fft_bins = n_fft // 2 + 1

        def hz_to_mel(freq_hz):
            return 2595.0 * np.log10(1.0 + freq_hz / 700.0)

        def mel_to_hz(mel_val):
            return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        freq_bins = np.floor((n_fft_bins - 1) * hz_points / fmax).astype(int)
        freq_bins = np.clip(freq_bins, 0, n_fft_bins - 1)

        filters = np.zeros((n_mels, n_fft_bins), dtype=np.float64)
        for m in range(1, n_mels + 1):
            start, center, end = freq_bins[m - 1 : m + 2]
            if center == start:
                center = min(center + 1, n_fft_bins - 1)
            if end == center:
                end = min(end + 1, n_fft_bins - 1)
            if end <= start:
                continue

            up = np.linspace(0.0, 1.0, max(center - start, 1), endpoint=False)
            down = np.linspace(1.0, 0.0, max(end - center, 1), endpoint=False)

            filters[m - 1, start:center] = up
            filters[m - 1, center:end] = down

        row_sums = filters.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        filters /= row_sums
        return filters

    def _erb_filter_bank(self, sample_rate, n_fft, n_bands=16, fmin=30.0, fmax=None):
        sample_rate = self._coerce_sample_rate(sample_rate)
        if isinstance(n_fft, (bool, np.bool_)) or not isinstance(n_fft, (int, np.integer)) or n_fft <= 1:
            raise ValueError("n_fft must be an integer greater than one")
        if isinstance(n_bands, (bool, np.bool_)) or not isinstance(n_bands, (int, np.integer)) or n_bands <= 0:
            raise ValueError("n_bands must be a positive integer")
        fmax = sample_rate / 2.0 if fmax is None else float(fmax)
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin < 0 or fmax <= fmin or fmax > sample_rate / 2.0:
            raise ValueError("ERB frequency bounds must satisfy 0 <= fmin < fmax <= Nyquist")

        n_fft_bins = n_fft // 2 + 1

        def hz_to_erb(freq_hz):
            return 21.4 * np.log10(4.37e-3 * freq_hz + 1.0)

        def erb_to_hz(erb_val):
            return (10.0 ** (erb_val / 21.4) - 1.0) / 4.37e-3

        erb_min = hz_to_erb(fmin)
        erb_max = hz_to_erb(fmax)
        erb_points = np.linspace(erb_min, erb_max, n_bands + 2)
        hz_points = erb_to_hz(erb_points)
        freq_bins = np.floor((n_fft_bins - 1) * hz_points / fmax).astype(int)
        freq_bins = np.clip(freq_bins, 0, n_fft_bins - 1)

        filters = np.zeros((n_bands, n_fft_bins), dtype=np.float64)
        for b in range(1, n_bands + 1):
            start, center, end = freq_bins[b - 1 : b + 2]
            if center == start:
                center = min(center + 1, n_fft_bins - 1)
            if end == center:
                end = min(end + 1, n_fft_bins - 1)
            if end <= start:
                continue

            up = np.linspace(0.0, 1.0, max(center - start, 1), endpoint=False)
            down = np.linspace(1.0, 0.0, max(end - center, 1), endpoint=False)

            filters[b - 1, start:center] = up
            filters[b - 1, center:end] = down

        row_sums = filters.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        filters /= row_sums
        return filters

    def _perceptual_feature_bundle(self, magnitude, mel_filters, erb_filters, prev_magnitude):
        power = magnitude * magnitude
        mel_env = mel_filters.dot(power)
        erb_env = erb_filters.dot(power)
        mel_env = np.log(mel_env + 1e-9)
        erb_env = np.log(erb_env + 1e-9)

        if prev_magnitude is None:
            flux = 0.0
        else:
            diff = magnitude - prev_magnitude
            flux = float(np.linalg.norm(np.maximum(diff, 0.0)))

        feature = np.concatenate((mel_env, erb_env, np.array([flux], dtype=np.float64)))
        return feature.astype(np.float32, copy=False), magnitude

    def _process_frame_batch(
        self,
        frames,
        mel_filters,
        erb_filters,
        window,
        energy_threshold=1e-10,
    ):
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.frame_size:
            raise ValueError(f"frames must have shape (n, {self.frame_size})")
        if not np.all(np.isfinite(frames)):
            raise ValueError("frames contain NaN or infinite samples")
        if not np.isfinite(energy_threshold) or energy_threshold < 0:
            raise ValueError("energy_threshold must be finite and non-negative")
        lpc_coeffs_list: List[np.ndarray] = []
        residual_list: List[np.ndarray] = []
        retained_frames: List[np.ndarray] = []
        perceptual_vectors: List[np.ndarray] = []
        prev_magnitude: Optional[np.ndarray] = None

        for frame in frames:
            if np.dot(frame, frame) < energy_threshold:
                continue

            windowed = frame * window
            lpc_coeffs = self._compute_lpc(windowed)
            residual = signal.lfilter(lpc_coeffs, [1.0], windowed)
            magnitude = np.abs(np.fft.rfft(windowed))
            perceptual_vec, prev_magnitude = self._perceptual_feature_bundle(
                magnitude, mel_filters, erb_filters, prev_magnitude
            )

            lpc_coeffs_list.append(lpc_coeffs[1:])
            residual_list.append(residual[: self.hop_size])
            retained_frames.append(frame)
            perceptual_vectors.append(perceptual_vec)

        if not retained_frames:
            empty_frames = np.empty((0, self.frame_size), dtype=np.float32)
            empty_lpc = np.empty((0, self.lpc_order), dtype=np.float32)
            empty_residual = np.empty((0, self.hop_size), dtype=np.float32)
            empty_perceptual = np.empty((0, mel_filters.shape[0] + erb_filters.shape[0] + 1), dtype=np.float32)
            empty_flux = np.empty(0, dtype=np.float32)
            return empty_frames, empty_lpc, empty_residual, empty_perceptual, empty_flux

        frames_arr = np.array(retained_frames, dtype=np.float32)
        lpc_arr = np.array(lpc_coeffs_list, dtype=np.float32)
        residual_arr = np.array(residual_list, dtype=np.float32)
        perceptual_arr = np.array(perceptual_vectors, dtype=np.float32)
        flux_arr = perceptual_arr[:, -1] if perceptual_arr.size else np.empty(0, dtype=np.float32)
        return frames_arr, lpc_arr, residual_arr, perceptual_arr, flux_arr
    
    def extract_features(self, audio_signal, sample_rate):
        """
        Phase 1, Step 1: Feature Extraction using Spectral Transform and Linear Prediction
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sampling rate
            
        Returns:
            lpc_coeffs_matrix: Matrix of LPC coefficients for all frames
            residual_matrix: Matrix of residual signals
        """
        sample_rate = self._coerce_sample_rate(sample_rate)
        audio_signal = self._prepare_audio(audio_signal, sample_rate, sample_rate)
        frames = self._frame_signal(audio_signal)

        mel_filters = self._mel_filter_bank(sample_rate, self.frame_size)
        erb_filters = self._erb_filter_bank(sample_rate, self.frame_size)
        window = np.hanning(self.frame_size).astype(np.float32)
        (
            processed_frames,
            lpc_coeffs,
            residuals,
            perceptual_vectors,
            flux_values,
        ) = self._process_frame_batch(frames, mel_filters, erb_filters, window)

        self.training_frames = processed_frames
        self.lpc_coefficients = lpc_coeffs
        self.residual_signals = residuals
        self.perceptual_features = perceptual_vectors
        self.sample_rate = int(sample_rate)

        if self.lpc_coefficients.size == 0:
            self.frame_metadata = {}
            raise ValueError("No valid frames extracted; check input signal")

        self.spectral_flux = flux_values.astype(np.float32, copy=False)
        self.sequence_starts = np.array([0], dtype=np.int64)

        self._compute_frame_metadata(self.sample_rate)
        if self.spectral_flux is not None and self.spectral_flux.size:
            metadata = self.frame_metadata
            metadata["spectral_flux"] = self.spectral_flux
            self.frame_metadata = metadata

        self._reset_caches()
        self._compute_and_cache_cholesky_factors()
        return self.lpc_coefficients, self.residual_signals

    def _extract_features_for_adaptation(self, audio_signal, sample_rate):
        """Extract new audio at the model's analysis rate without replacing buffers."""
        sample_rate = self._coerce_sample_rate(sample_rate)
        analysis_rate = self.sample_rate or sample_rate
        audio_signal = self._prepare_audio(audio_signal, sample_rate, analysis_rate)
        frames = self._frame_signal(audio_signal)

        mel_filters = self._mel_filter_bank(analysis_rate, self.frame_size)
        erb_filters = self._erb_filter_bank(analysis_rate, self.frame_size)
        window = np.hanning(self.frame_size).astype(np.float32)

        (
            processed_frames,
            lpc_coeffs,
            residuals,
            perceptual_vectors,
            flux_values,
        ) = self._process_frame_batch(frames, mel_filters, erb_filters, window)

        if lpc_coeffs.size == 0:
            raise ValueError("No valid frames extracted from adaptation audio")

        frame_metadata = {}
        if processed_frames.size:
            frame_len = processed_frames.shape[1]
            energy = np.sum(processed_frames * processed_frames, axis=1)
            rms = np.sqrt(energy / frame_len)
            zero_crossings = np.count_nonzero(
                np.signbit(processed_frames[:, :-1]) != np.signbit(processed_frames[:, 1:]),
                axis=1,
            )
            frame_metadata = {
                "energy": energy.astype(np.float32, copy=False),
                "rms": rms.astype(np.float32, copy=False),
                "zero_crossings": zero_crossings.astype(np.int32, copy=False),
            }
            if flux_values.size:
                frame_metadata["spectral_flux"] = flux_values.astype(np.float32, copy=False)  # pylint: disable=no-member

        return processed_frames, lpc_coeffs, residuals, perceptual_vectors, flux_values, frame_metadata
    
    def _apply_residual_augmentation(self):
        if self.augmentation_applied:
            return
        if (
            self.training_frames is None
            or self.training_frames.size == 0
            or self.lpc_coefficients is None
            or self.lpc_coefficients.size == 0
        ):
            return
        # If we already have ample training frames (>= 2000), skip augmentation to conserve memory and speed
        if self.training_frames.shape[0] >= 2000:
            self.augmentation_applied = True
            return
        if self.sample_rate is None or self.sample_rate <= 0:
            return

        mel_filters = self._mel_filter_bank(self.sample_rate, self.frame_size)
        erb_filters = self._erb_filter_bank(self.sample_rate, self.frame_size)
        window = np.hanning(self.frame_size).astype(np.float32)

        original_starts = self._normalize_sequence_starts(len(self.training_frames))
        augmented_frames = [self.training_frames]
        augmented_lpc = [self.lpc_coefficients]
        augmented_residuals = [self.residual_signals]
        augmented_sequence_starts = original_starts.tolist()

        augmented_perceptual: List[np.ndarray] = []
        if self.perceptual_features is not None and self.perceptual_features.size:
            augmented_perceptual.append(self.perceptual_features)

        augmented_flux: List[np.ndarray] = []
        if self.spectral_flux is not None and self.spectral_flux.size:
            augmented_flux.append(self.spectral_flux)

        def append_augmented(batch):
            (
                frames_aug,
                lpc_aug,
                residual_aug,
                perceptual_aug,
                flux_aug,
            ) = self._process_frame_batch(batch, mel_filters, erb_filters, window)
            if lpc_aug.size == 0:
                return
            offset = sum(len(item) for item in augmented_frames)
            augmented_sequence_starts.extend((original_starts + offset).tolist())
            augmented_frames.append(frames_aug)
            augmented_lpc.append(lpc_aug)
            augmented_residuals.append(residual_aug)
            if perceptual_aug.size:
                augmented_perceptual.append(perceptual_aug)
            if flux_aug.size:
                augmented_flux.append(flux_aug)

        stretch_factors = (0.92, 1.08)
        for factor in stretch_factors:
            target_len = max(2, int(round(self.frame_size * factor)))
            stretched = signal.resample(self.training_frames, target_len, axis=1)
            stretched = signal.resample(stretched, self.frame_size, axis=1)
            append_augmented(stretched.astype(np.float32, copy=False))

        semitone_offsets = (-0.5, 0.5)
        for semitone in semitone_offsets:
            pitch_factor = 2.0 ** (semitone / 12.0)
            target_len = max(2, int(round(self.frame_size / pitch_factor)))
            pitched = signal.resample(self.training_frames, target_len, axis=1)
            pitched = signal.resample(pitched, self.frame_size, axis=1)
            append_augmented(pitched.astype(np.float32, copy=False))

        if len(augmented_frames) == 1:
            self.augmentation_applied = True
            return

        self.training_frames = np.concatenate(augmented_frames, axis=0)
        self.lpc_coefficients = np.concatenate(augmented_lpc, axis=0)
        self.residual_signals = np.concatenate(augmented_residuals, axis=0)
        self.sequence_starts = np.asarray(augmented_sequence_starts, dtype=np.int64)

        if augmented_perceptual:
            self.perceptual_features = np.concatenate(augmented_perceptual, axis=0)
        if augmented_flux:
            self.spectral_flux = np.concatenate(augmented_flux, axis=0)

        self._compute_frame_metadata(self.sample_rate)
        if self.spectral_flux is not None and self.spectral_flux.size:
            metadata = self.frame_metadata
            metadata["spectral_flux"] = self.spectral_flux
            self.frame_metadata = metadata

        self._reset_caches()
        self._compute_and_cache_cholesky_factors()
        self.augmentation_applied = True

    def _compute_hmm_bic(self, n_states, n_em_iterations=3):
        """
        Compute Bayesian Information Criterion (BIC) for an HMM with given state count.
        
        Args:
            n_states: Number of states to test
            n_em_iterations: Number of EM iterations for fitting
            
        Returns:
            bic: BIC score (lower is better)
        """
        if self.lpc_coefficients is None:
            raise ValueError("Must extract features first")
        if isinstance(n_states, (bool, np.bool_)) or not isinstance(n_states, (int, np.integer)) or n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if isinstance(n_em_iterations, (bool, np.bool_)) or not isinstance(n_em_iterations, (int, np.integer)) or n_em_iterations < 0:
            raise ValueError("n_em_iterations must be a non-negative integer")
        
        num_frames = self.lpc_coefficients.shape[0]
        if num_frames <= 0 or n_states > num_frames:
            raise ValueError("n_states must not exceed the number of extracted frames")
        
        # Save current state
        original_n_states = self.n_states
        original_augmentation_applied = self.augmentation_applied
        original_params = (
            self.transition_matrix,
            self.initial_probabilities,
            self.state_means,
            self.state_covariances,
        )
        
        try:
            # Temporarily set n_states
            self.n_states = n_states
            
            # Initialize HMM with this state count
            coeff_mean = np.mean(self.lpc_coefficients, axis=0)
            
            if (
                self.perceptual_features is not None
                and self.perceptual_features.shape[0] == num_frames
            ):
                clustering_features = np.hstack([
                    self.lpc_coefficients,
                    self.perceptual_features,
                ])
            else:
                clustering_features = self.lpc_coefficients
            
            clustering_features = clustering_features.astype(np.float64, copy=False)
            feature_mean = clustering_features.mean(axis=0)
            feature_std = clustering_features.std(axis=0) + 1e-6
            normalized_features = (clustering_features - feature_mean) / feature_std
            
            seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
            centroids, labels = vq.kmeans2(
                normalized_features, n_states, minit="++", iter=20, seed=seed
            )
            if isinstance(labels, tuple):
                labels = labels[0]
            labels = np.asarray(labels, dtype=np.int64)
            
            self.initial_probabilities = np.zeros(n_states, dtype=np.float64)
            self.transition_matrix = np.zeros((n_states, n_states), dtype=np.float64)
            self.state_means = np.zeros((n_states, self.lpc_order), dtype=np.float64)
            self.state_covariances = np.array(
                [np.eye(self.lpc_order, dtype=np.float64) * 0.05 for _ in range(n_states)]
            )
            
            # Initialize parameters from clustering
            for state in range(n_states):
                mask = labels == state
                if not np.any(mask):
                    self.state_means[state] = coeff_mean
                    continue
                state_coeffs = self.lpc_coefficients[mask]
                self.state_means[state] = state_coeffs.mean(axis=0)
                centered = state_coeffs - self.state_means[state]
                cov = np.dot(centered.T, centered) / (len(state_coeffs) - 1 + 1e-9)
                cov += np.eye(self.lpc_order) * 1e-4
                self.state_covariances[state] = cov
            
            starts = self._normalize_sequence_starts(num_frames)
            for start, end in self._sequence_slices(num_frames, starts):
                self.initial_probabilities[labels[start]] += 1.0
                for idx in range(start + 1, end):
                    self.transition_matrix[labels[idx - 1], labels[idx]] += 1.0
            
            self.initial_probabilities += 1e-3
            self.initial_probabilities /= self.initial_probabilities.sum()
            
            self.transition_matrix += 1e-6
            row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
            self.transition_matrix /= row_sums
            
            self._reset_caches()
            self._compute_and_cache_cholesky_factors()
            
            # Run a few EM iterations
            for _ in range(n_em_iterations):
                gamma, xi, log_likelihood = self._expectation_step(self.lpc_coefficients)
                self._maximization_step(self.lpc_coefficients, gamma, xi)
            
            # Final log-likelihood
            _, _, log_likelihood = self._expectation_step(self.lpc_coefficients)
            
            # Compute number of free parameters
            # Transition matrix: n_states * (n_states - 1) (rows sum to 1)
            # Initial probs: n_states - 1 (sum to 1)
            # State means: n_states * lpc_order
            # State covariances: n_states * lpc_order * (lpc_order + 1) / 2 (symmetric)
            n_transition_params = n_states * (n_states - 1)
            n_init_params = n_states - 1
            n_mean_params = n_states * self.lpc_order
            n_cov_params = n_states * self.lpc_order * (self.lpc_order + 1) // 2
            n_params = n_transition_params + n_init_params + n_mean_params + n_cov_params
            
            # BIC = -2 * log_likelihood + k * log(n)
            bic = -2 * log_likelihood + n_params * np.log(num_frames)
            
            return bic
            
        finally:
            # Restore original state
            self.n_states = original_n_states
            self.augmentation_applied = original_augmentation_applied
            (
                self.transition_matrix,
                self.initial_probabilities,
                self.state_means,
                self.state_covariances,
            ) = original_params
            self._reset_caches()
    
    def _select_optimal_n_states(self, candidate_counts=None):
        """
        Select optimal number of states using BIC.
        
        Args:
            candidate_counts: List of state counts to test (default: DEFAULT_BIC_CANDIDATES)
            
        Returns:
            optimal_n_states: Selected state count
        """
        if candidate_counts is None:
            candidate_counts = self.DEFAULT_BIC_CANDIDATES
        if self.lpc_coefficients is None or len(self.lpc_coefficients) == 0:
            raise ValueError("Must extract features before selecting n_states")
        if isinstance(candidate_counts, (str, bytes)):
            raise ValueError("candidate_counts must be a non-empty iterable of positive integers")
        try:
            candidate_values = list(candidate_counts)
        except TypeError as exc:
            raise ValueError("candidate_counts must be a non-empty iterable of positive integers") from exc
        candidates = []
        for value in candidate_values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError("candidate state counts must be positive integers")
            if int(value) not in candidates:
                candidates.append(int(value))
        if not candidates:
            raise ValueError("candidate_counts must contain at least one state count")
        if not any(value <= len(self.lpc_coefficients) for value in candidates):
            candidates = [min(self.n_states, len(self.lpc_coefficients))]
        print(f"Auto-selecting n_states from candidates: {candidates}")
        
        bic_scores = {}
        for n in candidates:
            if n > len(self.lpc_coefficients):
                print(f"  Skipping n_states={n} (exceeds frame count)")
                continue
            print(f"  Testing n_states={n}...", end=" ")
            bic = self._compute_hmm_bic(n, n_em_iterations=3)
            bic_scores[n] = bic
            print(f"BIC={bic:.2f}")
        
        if not bic_scores:
            raise ValueError(
                f"No valid candidate state counts found. All candidates {candidate_counts} "
                f"exceed frame count ({len(self.lpc_coefficients)}). "
                f"Please use smaller state counts or provide more training data."
            )
        
        optimal_n_states = min(bic_scores, key=bic_scores.get)
        print(f"Selected optimal n_states={optimal_n_states} (lowest BIC)")
        
        return optimal_n_states

    def initialize_hmm_parameters(self):
        """Initialize a well-conditioned Gaussian HMM from LPC/perceptual features."""
        if self.lpc_coefficients is None:
            raise ValueError("Must extract features first using extract_features()")
        if self.lpc_coefficients.ndim != 2 or self.lpc_coefficients.shape[1] != self.lpc_order:
            raise ValueError(f"lpc_coefficients must have shape (n, {self.lpc_order})")
        if self.lpc_coefficients.shape[0] == 0:
            raise ValueError("No frames extracted; check input audio content and sample_rate.")
        if not np.all(np.isfinite(self.lpc_coefficients)):
            raise ValueError("LPC features contain NaN or infinite values")

        self._apply_residual_augmentation()
        if self.auto_n_states:
            self.n_states = self._select_optimal_n_states()

        num_frames = int(self.lpc_coefficients.shape[0])
        if num_frames < self.n_states:
            warnings.warn(
                "Fewer frames than states; reducing state count for initialization",
                RuntimeWarning,
                stacklevel=2,
            )
            self.n_states = num_frames

        if (
            self.perceptual_features is not None
            and self.perceptual_features.ndim == 2
            and self.perceptual_features.shape[0] == num_frames
        ):
            clustering_features = np.hstack((self.lpc_coefficients, self.perceptual_features))
        else:
            clustering_features = self.lpc_coefficients
        clustering_features = np.asarray(clustering_features, dtype=np.float64)
        if not np.all(np.isfinite(clustering_features)):
            raise ValueError("Clustering features contain NaN or infinite values")
        feature_mean = clustering_features.mean(axis=0)
        feature_std = clustering_features.std(axis=0)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        normalized = (clustering_features - feature_mean) / feature_std

        seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        try:
            centroids, labels = vq.kmeans2(
                normalized,
                self.n_states,
                minit="++",
                iter=40,
                seed=seed,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            logger.warning("k-means++ initialization failed (%s); using deterministic point seeding", exc)
            rng = np.random.default_rng(seed)
            centers = normalized[rng.choice(num_frames, size=self.n_states, replace=False)]
            distances = np.sum((normalized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            centroids = centers
        if isinstance(labels, tuple):
            labels = labels[0]
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != num_frames or np.any(labels < 0) or np.any(labels >= self.n_states):
            raise RuntimeError("k-means returned malformed state assignments")

        # kmeans2 can leave clusters empty (especially on constant or very short data).
        counts = np.bincount(labels, minlength=self.n_states)
        current_distances = np.sum((normalized - centroids[labels]) ** 2, axis=1)
        for empty_state in np.flatnonzero(counts == 0):
            candidates = counts[labels] > 1
            if not np.any(candidates):
                candidates = np.ones(num_frames, dtype=bool)
            candidate_distances = np.where(candidates, current_distances, -np.inf)
            frame_idx = int(np.argmax(candidate_distances))
            old_state = int(labels[frame_idx])
            labels[frame_idx] = int(empty_state)
            counts[old_state] -= 1
            counts[empty_state] += 1
            centroids[empty_state] = normalized[frame_idx]
            current_distances[frame_idx] = 0.0

        global_mean = np.mean(self.lpc_coefficients, axis=0, dtype=np.float64)
        centered_global = self.lpc_coefficients.astype(np.float64) - global_mean
        global_cov = centered_global.T @ centered_global / max(num_frames - 1, 1)
        global_cov = self._regularize_covariance(global_cov)

        means = np.empty((self.n_states, self.lpc_order), dtype=np.float64)
        covariances = np.empty((self.n_states, self.lpc_order, self.lpc_order), dtype=np.float64)
        for state in range(self.n_states):
            state_coeffs = self.lpc_coefficients[labels == state].astype(np.float64)
            means[state] = state_coeffs.mean(axis=0)
            if len(state_coeffs) > 1:
                centered = state_coeffs - means[state]
                cov = centered.T @ centered / (len(state_coeffs) - 1)
            else:
                cov = global_cov.copy()
            covariances[state] = self._regularize_covariance(cov)

        starts = self._normalize_sequence_starts(num_frames)
        start_set = set(starts.tolist())
        initial_counts = np.full(self.n_states, self.transition_pseudocount, dtype=np.float64)
        transition_counts = np.full(
            (self.n_states, self.n_states), self.transition_pseudocount, dtype=np.float64
        )
        for start, end in self._sequence_slices(num_frames, starts):
            initial_counts[labels[start]] += 1.0
            for idx in range(start + 1, end):
                transition_counts[labels[idx - 1], labels[idx]] += 1.0

        self.initial_probabilities = self._normalize_probabilities(initial_counts)
        self.transition_matrix = self._normalize_probabilities(transition_counts, axis=1)
        self.state_means = means
        self.state_covariances = covariances
        self._reset_caches()
        self._caches.training_history.clear()
        self._compute_and_cache_cholesky_factors()
        self._transition_cost_matrix()

    def _validate_adaptation_inputs(
        self,
        adaptation_rate: float,
        stability_bias: float,
        memory_blend: float,
        memory_limit: int,
    ) -> None:
        for name, value in (("adaptation_rate", adaptation_rate), ("stability_bias", stability_bias), ("memory_blend", memory_blend)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be between 0.0 and 1.0")
        if not 0.0 <= stability_bias <= 1.0:
            raise ValueError("stability_bias must be between 0.0 and 1.0")
        if not 0.0 <= memory_blend <= 1.0:
            raise ValueError("memory_blend must be between 0.0 and 1.0")
        if memory_limit <= 0:
            raise ValueError("memory_limit must be > 0")

    def _compute_statistics_from_observations(
        self, observations, gamma, xi, sequence_starts=None
    ):
        observations = np.asarray(observations, dtype=np.float64)
        gamma = np.asarray(gamma, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.lpc_order:
            raise ValueError(f"observations must have shape (n, {self.lpc_order})")
        if gamma.shape != (len(observations), self.n_states):
            raise ValueError("gamma shape does not match observations and model states")
        starts = self._normalize_sequence_starts(len(observations), sequence_starts)
        expected_transitions = len(observations) - len(starts)
        if np.shape(xi) != (expected_transitions, self.n_states, self.n_states):
            raise ValueError("xi shape does not match observations and sequence boundaries")
        counts = gamma.sum(axis=0)
        mean_accumulator = gamma.T @ observations
        second_moment = np.einsum("ts,ti,tj->sij", gamma, observations, observations, optimize=True)
        transition_counts = np.asarray(xi, dtype=np.float64).sum(axis=0)
        initial_counts = gamma[starts].sum(axis=0)
        total_frames = float(observations.shape[0])
        return counts, transition_counts, mean_accumulator, second_moment, initial_counts, total_frames

    def _bootstrap_adaptive_statistics(self):
        if self.lpc_coefficients is None or self.lpc_coefficients.size == 0:
            counts = np.ones(self.n_states, dtype=np.float64)
            transition_counts = np.ones((self.n_states, self.n_states), dtype=np.float64)
            mean_accumulator = self.state_means.astype(np.float64, copy=False)
            mean_accumulator = mean_accumulator * counts[:, None]
            second_moment = np.zeros(
                (self.n_states, self.lpc_order, self.lpc_order),
                dtype=np.float64,
            )
            for state in range(self.n_states):
                mean = self.state_means[state]
                cov = self.state_covariances[state]
                second_moment[state] = cov + np.outer(mean, mean)
            initial_counts = self.initial_probabilities.astype(np.float64, copy=True)
            total_frames = float(self.n_states)
        else:
            starts = self._normalize_sequence_starts(len(self.lpc_coefficients))
            gamma, xi, _ = self._expectation_step(
                self.lpc_coefficients, sequence_starts=starts
            )
            (
                counts,
                transition_counts,
                mean_accumulator,
                second_moment,
                initial_counts,
                total_frames,
            ) = self._compute_statistics_from_observations(
                self.lpc_coefficients, gamma, xi, sequence_starts=starts
            )

        self._adaptive_stats = _AdaptiveStatistics(
            state_frame_counts=counts,
            transition_counts=transition_counts,
            mean_accumulator=mean_accumulator,
            second_moment_accumulator=second_moment,
            initial_counts=initial_counts,
            total_frames=total_frames,
            last_adaptation=None,
        )

    def _update_adaptive_statistics(
        self,
        observations,
        gamma,
        xi,
        *,
        adaptation_rate: float,
        stability_bias: float,
        sequence_starts=None,
    ) -> None:
        if self._adaptive_stats is None:
            self._bootstrap_adaptive_statistics()

        (
            new_counts,
            new_transition,
            new_mean_acc,
            new_second_moment,
            new_initial,
            new_total,
        ) = self._compute_statistics_from_observations(
            observations, gamma, xi, sequence_starts=sequence_starts
        )

        stats = self._adaptive_stats
        if stats is None:
            return

        decay = float(adaptation_rate)
        retain = 1.0 - decay

        stats.state_frame_counts = retain * stats.state_frame_counts + decay * new_counts
        stats.transition_counts = retain * stats.transition_counts + decay * new_transition
        stats.mean_accumulator = retain * stats.mean_accumulator + decay * new_mean_acc
        stats.second_moment_accumulator = (
            retain * stats.second_moment_accumulator + decay * new_second_moment
        )
        stats.initial_counts = retain * stats.initial_counts + decay * new_initial
        stats.total_frames = retain * stats.total_frames + decay * new_total

        # NOTE: ``stability_bias`` is intentionally NOT added to the running accumulators
        # here. Adding it on every update would grow the counts unboundedly and progressively
        # shrink the reconstructed means/covariances toward zero. Instead it is applied as a
        # proper Dirichlet prior (toward a uniform distribution) when the parameters are
        # reconstructed in ``_apply_adaptive_statistics``.
        stats.last_adaptation = datetime.utcnow().isoformat() + "Z"

    def _apply_adaptive_statistics(self, stability_bias: float = 0.0) -> None:
        stats = self._adaptive_stats
        if stats is None:
            return
        prior = float(max(stability_bias, 0.0))
        counts = np.clip(stats.state_frame_counts, 1e-6, None)

        initial_counts = stats.initial_counts + prior
        if float(initial_counts.sum()) <= 0:
            initial_counts = np.asarray(self.initial_probabilities, dtype=np.float64).copy()
        initial = self._normalize_probabilities(initial_counts)

        transition_counts = np.asarray(stats.transition_counts, dtype=np.float64).copy()
        empty_rows = transition_counts.sum(axis=1) <= 0
        if prior > 0.0:
            transition_counts += prior
        if np.any(empty_rows):
            transition_counts[empty_rows] = np.asarray(self.transition_matrix)[empty_rows]
        transition = self._normalize_probabilities(
            transition_counts,
            axis=1,
            fallback=self.transition_matrix,
        )
        transition = np.clip(transition, 1e-12, None)
        transition = self._normalize_probabilities(transition, axis=1)
        means = stats.mean_accumulator / counts[:, None]

        covariances = np.empty((self.n_states, self.lpc_order, self.lpc_order), dtype=np.float64)
        for state in range(self.n_states):
            second = stats.second_moment_accumulator[state] / counts[state]
            mean = means[state]
            cov = second - np.outer(mean, mean)
            covariances[state] = self._regularize_covariance(cov)

        self.initial_probabilities = initial
        self.transition_matrix = transition
        self.state_means = means
        self.state_covariances = covariances
        self._reset_caches()
        self._compute_and_cache_cholesky_factors()
        self._transition_cost_matrix()

    def _merge_training_memory(
        self,
        frames,
        lpc_coeffs,
        residuals,
        perceptual_vectors,
        flux_values,
        *,
        memory_limit: int,
        memory_blend: float,
        rng: np.random.Generator,
    ) -> None:
        if self.training_frames is None or self.training_frames.size == 0:
            self.training_frames = frames
            self.lpc_coefficients = lpc_coeffs
            self.residual_signals = residuals
            self.perceptual_features = perceptual_vectors
            self.spectral_flux = flux_values
            self.sequence_starts = np.array([0], dtype=np.int64)
        else:
            existing_frames = self.training_frames
            existing_starts = self._normalize_sequence_starts(len(existing_frames))
            existing_lpc = self.lpc_coefficients
            existing_residuals = self.residual_signals
            existing_perceptual = self.perceptual_features
            existing_flux = self.spectral_flux

            total_existing = existing_frames.shape[0]
            total_new = frames.shape[0]
            total = total_existing + total_new

            if total <= memory_limit:
                self.training_frames = np.concatenate([existing_frames, frames], axis=0)
                self.lpc_coefficients = np.concatenate([existing_lpc, lpc_coeffs], axis=0)
                self.residual_signals = np.concatenate([existing_residuals, residuals], axis=0)
                if (
                    existing_perceptual is not None
                    and existing_perceptual.size
                    and perceptual_vectors is not None
                    and perceptual_vectors.size
                ):
                    self.perceptual_features = np.concatenate(
                        [existing_perceptual, perceptual_vectors], axis=0
                    )
                if (
                    existing_flux is not None
                    and existing_flux.size
                    and flux_values is not None
                    and flux_values.size
                ):
                    self.spectral_flux = np.concatenate([existing_flux, flux_values], axis=0)
                self.sequence_starts = np.concatenate(
                    (existing_starts, np.array([total_existing], dtype=np.int64))
                )
            else:
                target_old = int(round(memory_limit * memory_blend))
                target_new = memory_limit - target_old

                target_old = min(target_old, total_existing)
                target_new = min(target_new, total_new)
                if target_old + target_new < memory_limit:
                    remainder = memory_limit - (target_old + target_new)
                    if total_new - target_new >= remainder:
                        target_new += remainder
                    else:
                        target_old = min(total_existing, target_old + remainder)

                old_indices = rng.choice(total_existing, size=target_old, replace=False)
                new_indices = rng.choice(total_new, size=target_new, replace=False)

                merged_frames = np.concatenate([existing_frames[old_indices], frames[new_indices]], axis=0)
                merged_lpc = np.concatenate([existing_lpc[old_indices], lpc_coeffs[new_indices]], axis=0)
                merged_residuals = np.concatenate(
                    [existing_residuals[old_indices], residuals[new_indices]], axis=0
                )
                self.training_frames = merged_frames
                self.lpc_coefficients = merged_lpc
                self.residual_signals = merged_residuals
                # Reservoir samples no longer form a meaningful temporal sequence.
                # Treat each retained frame independently rather than fabricating
                # transitions between randomly reordered frames.
                self.sequence_starts = np.arange(len(merged_frames), dtype=np.int64)

                if (
                    existing_perceptual is not None
                    and existing_perceptual.size
                    and perceptual_vectors is not None
                    and perceptual_vectors.size
                ):
                    merged_perceptual = np.concatenate(
                        [existing_perceptual[old_indices], perceptual_vectors[new_indices]],
                        axis=0,
                    )
                    self.perceptual_features = merged_perceptual
                if (
                    existing_flux is not None
                    and existing_flux.size
                    and flux_values is not None
                    and flux_values.size
                ):
                    merged_flux = np.concatenate(
                        [existing_flux[old_indices], flux_values[new_indices]],
                        axis=0,
                    )
                    self.spectral_flux = merged_flux
    
    def _estimate_state_assignments(self):
        if self._caches.state_assignments is not None:
            return self._caches.state_assignments
        if (
            self.lpc_coefficients is None
            or self.state_means is None
            or self.lpc_coefficients.size == 0
        ):
            return None

        distances = np.linalg.norm(
            self.lpc_coefficients[:, None, :] - self.state_means[None, :, :],
            axis=2,
        )
        assignments = np.argmin(distances, axis=1)
        self._caches.state_assignments = assignments
        return assignments

    def _compute_psychoacoustic_weight(self):
        if self._caches.psychoacoustic_weight is not None:
            return self._caches.psychoacoustic_weight
        if self.state_covariances is None:
            raise ValueError("State covariances are not initialized")

        recon_error = np.array([np.trace(cov) for cov in self.state_covariances], dtype=np.float64)
        recon_error -= recon_error.min()
        recon_max = recon_error.max()
        if recon_max > 0.0:
            recon_error /= recon_max
        else:
            recon_error.fill(0.0)

        psycho_proxy = np.zeros(self.n_states, dtype=np.float64)
        assignments = self._estimate_state_assignments()
        metadata = self.frame_metadata if self.frame_metadata else {}

        if assignments is not None and assignments.size and metadata:
            centroid = metadata.get("spectral_centroid")
            rms = metadata.get("rms")
            flux = metadata.get("spectral_flux")
            bandwidth = metadata.get("spectral_bandwidth")

            centroid_den = float(np.max(centroid)) if centroid is not None and centroid.size else 1.0
            rms_den = float(np.max(rms)) if rms is not None and rms.size else 1.0
            flux_den = float(np.max(flux)) if flux is not None and flux.size else 1.0
            bandwidth_den = float(np.max(bandwidth)) if bandwidth is not None and bandwidth.size else 1.0

            centroid_den = centroid_den if centroid_den > 0.0 else 1.0
            rms_den = rms_den if rms_den > 0.0 else 1.0
            flux_den = flux_den if flux_den > 0.0 else 1.0
            bandwidth_den = bandwidth_den if bandwidth_den > 0.0 else 1.0

            for state in range(self.n_states):
                mask = assignments == state
                if not np.any(mask):
                    continue
                bright = (float(np.mean(centroid[mask])) / centroid_den) if centroid is not None else 0.0
                loud = (float(np.mean(rms[mask])) / rms_den) if rms is not None else 0.0
                rough = (float(np.mean(flux[mask])) / flux_den) if flux is not None else 0.0
                spread = (float(np.mean(bandwidth[mask])) / bandwidth_den) if bandwidth is not None else 0.0

                bright = np.clip(bright, 0.0, 1.0)
                loud = np.clip(loud, 0.0, 1.0)
                rough = np.clip(rough, 0.0, 1.0)
                spread = np.clip(spread, 0.0, 1.0)

                psycho_proxy[state] = 0.35 * bright + 0.35 * loud + 0.2 * rough + 0.1 * spread

        combined = 0.6 * recon_error + 0.4 * psycho_proxy
        combined -= combined.min()
        combined_max = float(combined.max())
        combined /= max(combined_max, 1e-12)
        self._caches.psychoacoustic_weight = combined
        return combined

    def _expectation_step(self, observations, sequence_starts=None):
        """Run stable log-domain forward/backward independently per audio sequence."""
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.lpc_order:
            raise ValueError(f"observations must have shape (n, {self.lpc_order})")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations contain NaN or infinite values")
        slices = self._sequence_slices(len(observations), sequence_starts)
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)
        if self.state_means is None or self.state_covariances is None:
            raise ValueError("Gaussian emission parameters are not initialized")

        initial = self._normalize_probabilities(self.initial_probabilities)
        transition = self._normalize_probabilities(self.transition_matrix, axis=1)
        log_init = np.log(np.clip(initial, np.finfo(np.float64).tiny, None))
        log_trans = np.log(np.clip(transition, np.finfo(np.float64).tiny, None))
        gamma = np.empty((len(observations), self.n_states), dtype=np.float64)
        xi_parts = []
        total_log_likelihood = 0.0

        for start, end in slices:
            emissions = self._compute_emission_log_probs(observations[start:end])
            length = end - start
            if length == 1:
                log_posterior = log_init + emissions[0]
                normalizer = float(logsumexp(log_posterior))
                gamma[start] = np.exp(log_posterior - normalizer)
                total_log_likelihood += normalizer
                continue

            if _NATIVE_BACKEND_AVAILABLE:
                local_gamma, local_xi, log_likelihood = ssgs_native.forward_backward(
                    np.ascontiguousarray(emissions, dtype=np.float64),
                    np.ascontiguousarray(log_init, dtype=np.float64),
                    np.ascontiguousarray(log_trans, dtype=np.float64),
                )
            else:
                local_gamma, local_xi, log_likelihood = ssgs_core.forward_backward(
                    emissions, log_init, log_trans
                )
            gamma[start:end] = local_gamma
            xi_parts.append(np.asarray(local_xi, dtype=np.float64))
            total_log_likelihood += float(log_likelihood)

        if xi_parts:
            xi = np.concatenate(xi_parts, axis=0)
        else:
            xi = np.empty((0, self.n_states, self.n_states), dtype=np.float64)
        if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(xi)):
            raise FloatingPointError("forward/backward produced non-finite posteriors")
        return gamma, xi, float(total_log_likelihood)

    def _compute_emission_log_probs(self, observations):
        """Compute finite Gaussian log-likelihoods with cached robust Cholesky factors."""
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.lpc_order:
            raise ValueError(f"observations must have shape (n, {self.lpc_order})")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations contain NaN or infinite values")
        n_frames = observations.shape[0]
        if self.state_means is None or np.shape(self.state_means) != (self.n_states, self.lpc_order):
            raise ValueError("state_means are not initialized with the expected shape")
        if not np.all(np.isfinite(self.state_means)):
            raise ValueError("state_means contain NaN or infinite values")
        if n_frames == 0:
            return np.empty((0, self.n_states), dtype=np.float64)

        self._compute_and_cache_cholesky_factors()
        if _NATIVE_BACKEND_AVAILABLE:
            result = ssgs_native.gaussian_log_probs(
                np.ascontiguousarray(self.state_means, dtype=np.float64),
                np.ascontiguousarray(self.state_covariances, dtype=np.float64),
                np.ascontiguousarray(observations, dtype=np.float64),
            )
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("native Gaussian emission kernel returned non-finite values")
            return result

        caches = self._caches
        log_probs = np.empty((n_frames, self.n_states), dtype=np.float64)
        gaussian_const = self.lpc_order * np.log(2.0 * np.pi)
        obs_t = observations.T

        if caches.emission_observations is None or caches.emission_observations.shape != obs_t.shape:
            caches.emission_observations = np.empty(obs_t.shape, dtype=np.float64)
        np.copyto(caches.emission_observations, obs_t)
        if caches.emission_workspace is None or caches.emission_workspace.shape != obs_t.shape:
            caches.emission_workspace = np.empty(obs_t.shape, dtype=np.float64)
        workspace = caches.emission_workspace

        for state in range(self.n_states):
            np.subtract(caches.emission_observations, self.state_means[state, :, None], out=workspace)
            solved = solve_triangular(
                caches.chol_factors[state],
                workspace,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            quadratic = np.sum(solved * solved, axis=0)
            log_probs[:, state] = -0.5 * (
                quadratic + gaussian_const + caches.log_determinants[state]
            )
        return log_probs

    def _maximization_step(self, observations, gamma, xi, sequence_starts=None):
        """Update HMM sufficient statistics without inventing cross-file transitions."""
        observations = np.asarray(observations, dtype=np.float64)
        gamma = np.asarray(gamma, dtype=np.float64)
        xi = np.asarray(xi, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.lpc_order:
            raise ValueError(f"observations must have shape (n, {self.lpc_order})")
        n_frames = len(observations)
        slices = self._sequence_slices(n_frames, sequence_starts)
        if gamma.shape != (n_frames, self.n_states):
            raise ValueError("gamma must have shape (n_frames, n_states)")
        expected_transitions = sum(max(end - start - 1, 0) for start, end in slices)
        if xi.shape != (expected_transitions, self.n_states, self.n_states):
            raise ValueError("xi shape does not match within-sequence transitions")
        if not all(np.all(np.isfinite(arr)) for arr in (observations, gamma, xi)):
            raise ValueError("observations and posteriors must contain only finite values")
        if np.any(gamma < 0) or np.any(xi < 0):
            raise ValueError("posterior probabilities must be non-negative")

        initial_counts = gamma[np.asarray([start for start, _ in slices])].sum(axis=0) + 1e-6
        new_initial = self._normalize_probabilities(initial_counts)

        old_transition = self._normalize_probabilities(self.transition_matrix, axis=1)
        transition_counts = xi.sum(axis=0)
        transition_prior = self.transition_pseudocount * old_transition
        transition_counts += transition_prior
        transition_rows = transition_counts.sum(axis=1, keepdims=True)
        new_transition = np.divide(
            transition_counts,
            transition_rows,
            out=old_transition.copy(),
            where=transition_rows > 0,
        )
        new_transition = np.clip(new_transition, 1e-12, None)
        new_transition = self._normalize_probabilities(new_transition, axis=1)

        # Each transition's source posterior is counted once; sequence-final frames
        # cannot be sources because their next frame belongs to a separate recording.
        source_mask = np.ones(n_frames, dtype=bool)
        for _, end in slices:
            source_mask[end - 1] = False
        # Touching this statistic explicitly helps catch invalid transition posteriors
        # before any parameters are modified.
        if not np.all(np.isfinite(gamma[source_mask].sum(axis=0))):
            raise FloatingPointError("transition occupancy contains non-finite values")
        means = np.asarray(self.state_means, dtype=np.float64).copy()
        covariances = np.asarray(self.state_covariances, dtype=np.float64).copy()
        for state in range(self.n_states):
            weights = gamma[:, state]
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-10:
                continue
            mean = weights @ observations / weight_sum
            centered = observations - mean
            covariance = (centered * weights[:, None]).T @ centered / weight_sum
            means[state] = mean
            covariances[state] = self._regularize_covariance(covariance)

        self.initial_probabilities = new_initial
        self.transition_matrix = new_transition
        self.state_means = means
        self.state_covariances = covariances
        self._reset_caches()

    def iterative_refinement(
        self,
        n_iterations=10,
        *,
        preview_every: int = 0,
        preview_duration_frames: int = 32,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        tolerance: float = 1e-4,
        patience: int = 2,
        min_iterations: int = 1,
        verbose: bool = True,
    ):
        """Refine HMM parameters with finite checks and optional convergence stopping."""
        if isinstance(n_iterations, (bool, np.bool_)) or not isinstance(n_iterations, (int, np.integer)) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer")
        if isinstance(preview_every, (bool, np.bool_)) or not isinstance(preview_every, (int, np.integer)) or preview_every < 0:
            raise ValueError("preview_every must be a non-negative integer")
        if isinstance(preview_duration_frames, (bool, np.bool_)) or not isinstance(preview_duration_frames, (int, np.integer)) or preview_duration_frames <= 0:
            raise ValueError("preview_duration_frames must be a positive integer")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer)) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if isinstance(min_iterations, (bool, np.bool_)) or not isinstance(min_iterations, (int, np.integer)) or not 1 <= min_iterations <= n_iterations:
            raise ValueError("min_iterations must be between 1 and n_iterations")
        if self.lpc_coefficients is None or self.transition_matrix is None:
            raise ValueError("Must initialize HMM parameters first")

        if verbose:
            print(f"Starting EM training with at most {n_iterations} iterations...")
        self._caches.training_history.clear()
        final_log_likelihood = 0.0
        final_entropy = 0.0
        final_smooth_penalty = 0.0
        previous_likelihood = None
        stable_iterations = 0
        starts = self._normalize_sequence_starts(len(self.lpc_coefficients))

        for iteration in range(int(n_iterations)):
            gamma, xi, log_likelihood = self._expectation_step(
                self.lpc_coefficients, sequence_starts=starts
            )
            occupancy = gamma.mean(axis=0)
            occupancy = occupancy / max(float(occupancy.sum()), 1e-12)
            entropy = float(-np.sum(occupancy * np.log(occupancy + 1e-12)))
            smooth_matrix = self._spectral_smoothness_matrix()
            smooth_penalty = (
                float(np.sum(xi * smooth_matrix[None, :, :]) / max(xi.shape[0], 1))
                if xi.shape[0]
                else 0.0
            )
            history_entry = {
                "iteration": iteration + 1,
                "log_likelihood": float(log_likelihood),
                "occupancy_entropy": entropy,
                "avg_smoothness_penalty": smooth_penalty,
            }
            self._caches.training_history.append(history_entry)
            self._maximization_step(
                self.lpc_coefficients, gamma, xi, sequence_starts=starts
            )

            if verbose:
                print(
                    "Iteration {idx}: Log-likelihood = {ll:.2f} | Entropy = {ent:.3f} | "
                    "Smoothness = {smooth:.3f}".format(
                        idx=iteration + 1,
                        ll=log_likelihood,
                        ent=entropy,
                        smooth=smooth_penalty,
                    )
                )
            final_log_likelihood = float(log_likelihood)
            final_entropy = entropy
            final_smooth_penalty = smooth_penalty

            if previous_likelihood is not None and tolerance > 0:
                relative_improvement = (log_likelihood - previous_likelihood) / max(
                    1.0, abs(previous_likelihood)
                )
                if -tolerance <= relative_improvement <= tolerance:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
            previous_likelihood = float(log_likelihood)

            if preview_every and ((iteration + 1) % int(preview_every) == 0):
                preview_frames = int(preview_duration_frames)
                preview_sr_arg = int(self.sample_rate or 16000)
                preview_audio, preview_sr = self.render_preview(
                    n_frames=preview_frames,
                    sample_rate=preview_sr_arg,
                    seed=self.random_state + iteration,
                )
                preview_snapshot = self._record_preview_snapshot(
                    iteration=iteration + 1,
                    frames=preview_frames,
                    audio=preview_audio,
                    sample_rate=preview_sr,
                )
                history_entry["preview_index"] = len(self.preview_snapshots) - 1
                if verbose:
                    print(f"  Preview stored ({preview_frames} frames @ {preview_sr} Hz)")
                if preview_callback is not None:
                    try:
                        preview_callback(preview_snapshot)
                    except Exception as exc:  # pragma: no cover - defensive log
                        warnings.warn(
                            f"Preview callback raised an exception: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            if stable_iterations >= patience and iteration + 1 >= min_iterations:
                if verbose:
                    print(f"Early stopping: relative likelihood improvement stayed within {tolerance:g}.")
                break

        return final_log_likelihood, final_entropy, final_smooth_penalty

    def run_em_iteration(self):
        """Run one complete E/M step and return likelihood, entropy, smoothness."""
        if self.lpc_coefficients is None or self.transition_matrix is None:
            raise ValueError("Must initialize HMM parameters first")
        starts = self._normalize_sequence_starts(len(self.lpc_coefficients))
        gamma, xi, log_likelihood = self._expectation_step(
            self.lpc_coefficients, sequence_starts=starts
        )
        occupancy = gamma.mean(axis=0)
        occupancy = occupancy / max(float(occupancy.sum()), 1e-12)
        entropy = float(-np.sum(occupancy * np.log(occupancy + 1e-12)))
        smooth_matrix = self._spectral_smoothness_matrix()
        smooth_penalty = (
            float(np.sum(xi * smooth_matrix[None, :, :]) / max(xi.shape[0], 1))
            if xi.shape[0]
            else 0.0
        )
        self._maximization_step(
            self.lpc_coefficients, gamma, xi, sequence_starts=starts
        )
        self._caches.training_history.append({
            "iteration": len(self._caches.training_history) + 1,
            "log_likelihood": float(log_likelihood),
            "occupancy_entropy": entropy,
            "avg_smoothness_penalty": smooth_penalty,
        })
        return float(log_likelihood), entropy, smooth_penalty

    def identify_graph_constraints(self, min_state_probability=1e-4):
        """Rehome dormant states and guarantee a finite row-stochastic transition graph.

        State dimensions are retained for checkpoint/API compatibility. Low-occupancy
        states are made dormant: their outgoing transitions route to active states and
        incoming probability is redistributed instead of silently deleting matrix mass.
        """
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)
        if (
            isinstance(min_state_probability, (bool, np.bool_))
            or not isinstance(min_state_probability, (int, float, np.number))
            or not np.isfinite(min_state_probability)
            or not 0.0 <= min_state_probability <= 1.0
        ):
            raise ValueError("min_state_probability must be a number between 0.0 and 1.0")
        transition = np.asarray(self.transition_matrix, dtype=np.float64)
        if transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition_matrix shape does not match n_states")
        if not np.all(np.isfinite(transition)) or np.any(transition < 0):
            raise ValueError("transition_matrix must be finite and non-negative")
        transition = self._normalize_probabilities(transition, axis=1)
        stationary = self._normalize_probabilities(self.initial_probabilities)
        # Damping makes power iteration converge for periodic (e.g. alternating) chains.
        for _ in range(2000):
            next_stationary = transition.T @ stationary
            next_stationary = self._normalize_probabilities(next_stationary)
            updated = 0.5 * stationary + 0.5 * next_stationary
            if np.max(np.abs(updated - stationary)) < 1e-12:
                stationary = updated
                break
            stationary = updated

        active = stationary >= float(min_state_probability)
        if not np.any(active):
            active[int(np.argmax(stationary))] = True
        active_distribution = stationary[active]
        if active_distribution.sum() <= 0:
            active_distribution = np.ones(np.count_nonzero(active), dtype=np.float64)
        active_distribution = self._normalize_probabilities(active_distribution)

        adjusted = transition.copy()
        for state in range(self.n_states):
            if not active[state]:
                adjusted[state] = 0.0
                adjusted[state, active] = active_distribution
            else:
                adjusted[state, ~active] = 0.0
                row_sum = adjusted[state].sum()
                if row_sum <= 0:
                    adjusted[state, active] = active_distribution
                else:
                    adjusted[state] /= row_sum
        # A tiny floor preserves reachability and prevents log(0) in decoders.
        adjusted = np.clip(adjusted, 1e-12, None)
        adjusted = self._normalize_probabilities(adjusted, axis=1)
        initial = np.asarray(self.initial_probabilities, dtype=np.float64).copy()
        dormant_mass = float(initial[~active].sum())
        initial[~active] = 0.0
        if np.any(active):
            initial[active] += dormant_mass * active_distribution
        initial = np.clip(initial, 1e-12, None)
        initial = self._normalize_probabilities(initial)

        graph = csr_matrix((adjusted > 1e-6).astype(np.float64))
        n_components, _ = _strong_components(graph)
        self.transition_matrix = adjusted
        self.initial_probabilities = initial
        self._reset_caches()
        self._transition_cost_matrix()
        logger.info(
            "Graph constraints retained %d active states and %d strongly connected components",
            int(np.count_nonzero(active)),
            int(n_components),
        )
        return {
            "active_states": np.flatnonzero(active),
            "stationary_distribution": stationary,
            "n_components": int(n_components),
        }

    def _spectral_smoothness_cost(self, state1, state2):
        """
        Compute spectral smoothness cost between two states
        
        Args:
            state1: First HMM state
            state2: Second HMM state
            
        Returns:
            Smoothness cost (lower is smoother)
        """
        return self._spectral_smoothness_matrix()[state1, state2]

    def _transition_cost_matrix(self):
        transition, _, _, _ = self._validated_model_parameters()
        if self._caches.transition_cost is None:
            transition_cost = -np.log(transition + 1e-12)
            smoothness = self._spectral_smoothness_matrix()
            psycho_weight = self._compute_psychoacoustic_weight()
            self._caches.transition_cost = (
                transition_cost
                + self.smoothness_weight * smoothness
                + 0.25 * psycho_weight[np.newaxis, :]
            )
        return self._caches.transition_cost

    def _spectral_smoothness_matrix(self):
        if self._caches.smoothness is None:
            diffs = self.state_means[:, None, :] - self.state_means[None, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            scale = np.std(self.state_means) + 1e-10
            self._caches.smoothness = distances / scale
        return self._caches.smoothness
    
    def _a_star_search(self, target_duration_frames, max_expanded=10000):
        """
        Phase 2, Step 5: State sequence decoding using smoothness-aware dynamic programming
        
        Note: This is implemented as a Viterbi-style DP with spectral smoothness costs.
        The "A*" naming reflects the heuristic-guided nature of the smoothness penalty,
        though technically this is a constrained Viterbi decoder.
        
        Args:
            target_duration_frames: Target number of frames
            max_expanded: Maximum number of nodes to expand (reserved for future use)
            
        Returns:
            optimal_state_sequence: Most probable state sequence
        """
        if isinstance(target_duration_frames, (bool, np.bool_)) or not isinstance(target_duration_frames, (int, np.integer)) or target_duration_frames <= 0:
            raise ValueError("target_duration_frames must be a positive integer")
        # Use the compiled native decode when available (identical output to the
        # Python decoders, but ~170x faster on the synthesis hot path).
        if _NATIVE_BACKEND_AVAILABLE:
            transition_cost = self._transition_cost_matrix()
            smoothness = self._spectral_smoothness_matrix()
            initial_cost = -np.log(self.initial_probabilities + 1e-12)
            return ssgs_native.decode_sequence(
                transition_cost,
                smoothness,
                initial_cost,
                self.smoothness_weight,
                target_duration_frames,
            )

        # Fallback: cached Python decoder (used only when the native .so is absent)
        if self._caches.transition_cost is not None:
            return self._decode_state_sequence_cached(target_duration_frames)
        return self._decode_state_sequence(target_duration_frames)

    def _decode_state_sequence(self, target_frames):
        """Deterministic dynamic-programming decode with smoothness-first objective."""
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)

        initial_cost = -np.log(self.initial_probabilities + 1e-12)
        transition_cost = -np.log(self.transition_matrix + 1e-12)
        psycho_weight = self._compute_psychoacoustic_weight()
        transition_cost = transition_cost + 0.25 * psycho_weight[np.newaxis, :]
        smoothness = self._spectral_smoothness_matrix()

        if _NATIVE_BACKEND_AVAILABLE:
            path = ssgs_native.decode_sequence(
                transition_cost,
                smoothness,
                initial_cost,
                self.smoothness_weight,
                target_frames,
            )
            return path

        if self.smoothness_weight <= 0.0:
            dp = np.empty((target_frames, self.n_states), dtype=np.float64)
            backpointer = np.zeros((target_frames, self.n_states), dtype=np.int32)
            dp[0] = initial_cost

            for t in range(1, target_frames):
                costs = dp[t - 1][:, None] + transition_cost
                backpointer[t] = np.argmin(costs, axis=0)
                dp[t] = np.min(costs, axis=0)

            end_state = int(np.argmin(dp[-1]))
        else:
            dp_smooth = np.empty((target_frames, self.n_states), dtype=np.float64)
            dp_total = np.empty((target_frames, self.n_states), dtype=np.float64)
            backpointer = np.zeros((target_frames, self.n_states), dtype=np.int32)
            dp_smooth[0] = 0.0
            dp_total[0] = initial_cost

            tolerance = 1e-9

            for t in range(1, target_frames):
                for j in range(self.n_states):
                    smooth_candidates = dp_smooth[t - 1] + smoothness[:, j]
                    min_smooth = float(np.min(smooth_candidates))
                    candidate_mask = smooth_candidates <= (min_smooth + tolerance)
                    total_candidates = (
                        dp_total[t - 1]
                        + transition_cost[:, j]
                        + self.smoothness_weight * smoothness[:, j]
                    )
                    candidate_indices = np.where(candidate_mask)[0]
                    best_local = candidate_indices[
                        int(np.argmin(total_candidates[candidate_mask]))
                    ]
                    dp_smooth[t, j] = min_smooth
                    dp_total[t, j] = total_candidates[best_local]
                    backpointer[t, j] = best_local

            final_smooth = dp_smooth[-1]
            min_smooth = float(np.min(final_smooth))
            final_mask = final_smooth <= (min_smooth + tolerance)
            end_state = int(
                np.where(final_mask)[0][np.argmin(dp_total[-1][final_mask])]
            )

        path = [0] * target_frames
        path[-1] = end_state

        for t in range(target_frames - 1, 0, -1):
            end_state = backpointer[t, end_state]
            path[t - 1] = end_state

        return path
    
    def _decode_state_sequence_cached(self, target_frames):
        """Cached version of state sequence decoding using pre-computed transition costs.

        This must produce results identical to :meth:`_decode_state_sequence`; caching is
        only a performance optimization. The cached ``transition_cost`` folds in the term
        ``smoothness_weight * smoothness``, so we recover the base transition cost (without
        that term) to preserve the exact smoothness-first lexicographic ordering.
        """
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)

        transition_cost = self._caches.transition_cost
        smoothness = self._spectral_smoothness_matrix()
        initial_cost = -np.log(self.initial_probabilities + 1e-12)

        if self.smoothness_weight <= 0.0:
            dp = np.empty((target_frames, self.n_states), dtype=np.float64)
            backpointer = np.zeros((target_frames, self.n_states), dtype=np.int32)
            dp[0] = initial_cost

            for t in range(1, target_frames):
                costs = dp[t - 1][:, None] + transition_cost
                backpointer[t] = np.argmin(costs, axis=0)
                dp[t] = np.min(costs, axis=0)

            end_state = int(np.argmin(dp[-1]))
        else:
            # Recover the base transition cost (transition + psychoacoustic only) so the
            # total cost used for tie-breaking matches _decode_state_sequence exactly.
            base_transition = transition_cost - self.smoothness_weight * smoothness

            dp_smooth = np.empty((target_frames, self.n_states), dtype=np.float64)
            dp_total = np.empty((target_frames, self.n_states), dtype=np.float64)
            backpointer = np.zeros((target_frames, self.n_states), dtype=np.int32)
            dp_smooth[0] = 0.0
            dp_total[0] = initial_cost

            tolerance = 1e-9

            for t in range(1, target_frames):
                for j in range(self.n_states):
                    smooth_candidates = dp_smooth[t - 1] + smoothness[:, j]
                    min_smooth = float(np.min(smooth_candidates))
                    candidate_mask = smooth_candidates <= (min_smooth + tolerance)
                    total_candidates = (
                        dp_total[t - 1]
                        + base_transition[:, j]
                        + self.smoothness_weight * smoothness[:, j]
                    )
                    candidate_indices = np.where(candidate_mask)[0]
                    best_local = candidate_indices[
                        int(np.argmin(total_candidates[candidate_mask]))
                    ]
                    dp_smooth[t, j] = min_smooth
                    dp_total[t, j] = total_candidates[best_local]
                    backpointer[t, j] = best_local

            final_smooth = dp_smooth[-1]
            min_smooth = float(np.min(final_smooth))
            final_mask = final_smooth <= (min_smooth + tolerance)
            end_state = int(
                np.where(final_mask)[0][np.argmin(dp_total[-1][final_mask])]
            )

        path = [0] * target_frames
        path[-1] = end_state

        for t in range(target_frames - 1, 0, -1):
            end_state = backpointer[t, end_state]
            path[t - 1] = end_state

        return path

    def _state_spectral_prototypes(self, rng=None, seed=None):
        """Build state LPC/residual prototypes; cache only data-derived results."""
        if self._caches.spectral_prototypes is not None:
            return self._caches.spectral_prototypes
        if self.state_means is None or np.shape(self.state_means) != (self.n_states, self.lpc_order):
            raise ValueError("State means are not initialized")

        residuals = (
            self.residual_signals
            if self.residual_signals is not None and self.residual_signals.size
            else None
        )
        assignments = self._estimate_state_assignments()
        rng = rng if rng is not None else np.random.default_rng(seed)
        prototypes: List[Tuple[np.ndarray, np.ndarray]] = []
        cacheable = residuals is not None and assignments is not None and len(residuals) == len(assignments)
        global_residual = residuals.mean(axis=0) if residuals is not None else None

        for state in range(self.n_states):
            if cacheable:
                mask = assignments == state
                # An unoccupied state uses the robust global residual template instead
                # of injecting and then permanently caching seed-dependent random noise.
                residual_template = residuals[mask].mean(axis=0) if np.any(mask) else global_residual.copy()
            else:
                residual_template = rng.standard_normal(self.hop_size).astype(np.float32) * 0.05
            residual_template = np.asarray(residual_template, dtype=np.float32).reshape(-1)
            if residual_template.size < self.hop_size:
                residual_template = np.pad(
                    residual_template,
                    (0, self.hop_size - residual_template.size),
                    mode="constant",
                )
            elif residual_template.size > self.hop_size:
                residual_template = residual_template[: self.hop_size]
            prototypes.append((
                np.asarray(self.state_means[state], dtype=np.float32).copy(),
                residual_template.astype(np.float32, copy=False),
            ))

        if cacheable:
            self._caches.spectral_prototypes = prototypes
        return prototypes

    def _karplus_strong_excitation(self, state, duration_samples, rng=None):
        """Generate a bounded, state-conditioned Karplus-Strong excitation block."""
        if not isinstance(state, (int, np.integer)) or not 0 <= int(state) < self.n_states:
            raise ValueError("state index is out of range")
        if isinstance(duration_samples, (bool, np.bool_)) or not isinstance(duration_samples, (int, np.integer)) or duration_samples <= 0:
            raise ValueError("duration_samples must be a positive integer")
        mean_coeffs = np.asarray(self.state_means[state], dtype=np.float64)
        spectral_centroid = float(np.mean(np.abs(mean_coeffs)))
        base_delay = int(20 + 80 * (1 - np.tanh(spectral_centroid)))
        rng = rng if rng is not None else self._rng
        delay_line = max(2, base_delay + int(rng.integers(-5, 6)))
        duration_samples = int(duration_samples)

        excitation = np.zeros(duration_samples, dtype=np.float32)
        noise_burst = rng.standard_normal(delay_line).astype(np.float32) * 0.5
        damping = float(np.clip(0.98 - 0.1 * spectral_centroid, 0.95, 0.99))
        if delay_line >= duration_samples:
            excitation[:] = noise_burst[:duration_samples]
        else:
            excitation[:delay_line] = noise_burst
            for idx in range(delay_line, duration_samples):
                excitation[idx] = damping * 0.5 * (
                    excitation[idx - delay_line] + excitation[idx - delay_line + 1]
                )
        excitation *= np.exp(-np.linspace(0, 5, duration_samples)).astype(np.float32)
        return excitation

    def _wavelet_excitation(self, state, duration_samples, rng=None):
        """
        Wavelet-based excitation generation as an alternative to Karplus-Strong.
        
        Uses inverse wavelet transform of coefficients derived from the current
        spectral state to generate excitation signals with different textural
        characteristics.
        
        Args:
            state: HMM state for parameterization
            duration_samples: Number of samples to generate
            
        Returns:
            excitation_signal: Wavelet-based excitation signal
        """
        if not isinstance(state, (int, np.integer)) or not 0 <= int(state) < self.n_states:
            raise ValueError("state index is out of range")
        if isinstance(duration_samples, (bool, np.bool_)) or not isinstance(duration_samples, (int, np.integer)) or duration_samples <= 0:
            raise ValueError("duration_samples must be a positive integer")
        duration_samples = int(duration_samples)
        try:
            import pywt
        except ImportError:
            raise ImportError(
                "PyWavelets is required for wavelet excitation. "
                "Install with: pip install PyWavelets"
            )
        
        mean_coeffs = self.state_means[state]
        
        # Select wavelet type based on spectral characteristics
        spectral_centroid = np.mean(np.abs(mean_coeffs))
        spectral_variance = np.var(mean_coeffs)
        
        # Choose wavelet family based on spectral properties
        # High variance -> more complex wavelet (db4)
        # Low variance -> simpler wavelet (haar, db2)
        if spectral_variance > self.WAVELET_VARIANCE_HIGH:
            wavelet_name = 'db4'  # Daubechies 4
        elif spectral_variance > self.WAVELET_VARIANCE_MED:
            wavelet_name = 'db2'  # Daubechies 2
        else:
            wavelet_name = 'haar'  # Simplest
        
        # Determine decomposition level based on duration
        max_level = pywt.dwt_max_level(duration_samples, wavelet_name)
        decomp_level = min(max_level, 4)  # Cap at 4 levels
        
        # Generate wavelet coefficients from spectral state
        # Use LPC coefficients to modulate wavelet structure
        rng = rng if rng is not None else self._rng
        
        # Create wavelet packet with state-dependent characteristics
        # Detail coefficients (high frequency) influenced by LPC magnitude
        detail_scale = float(np.linalg.norm(mean_coeffs))
        
        # Approximation coefficients (low frequency) influenced by spectral centroid
        approx_scale = float(1.0 + spectral_centroid)
        
        # Generate base signal with controlled randomness
        base_length = duration_samples
        base_signal = rng.standard_normal(base_length).astype(np.float32)
        
        # Decompose signal
        coeffs = pywt.wavedec(base_signal, wavelet_name, level=decomp_level)
        
        # Modulate coefficients based on spectral state
        # Approximation coefficients (lowest frequency)
        coeffs[0] = coeffs[0] * approx_scale * 0.5
        
        # Detail coefficients (higher frequencies)
        for i in range(1, len(coeffs)):
            # Scale detail coefficients by LPC-derived values
            mean_idx = i - 1
            if mean_idx < len(mean_coeffs):
                scale_factor = float(np.abs(mean_coeffs[mean_idx]) * detail_scale)
            else:
                scale_factor = detail_scale * 0.5
            coeffs[i] = coeffs[i] * scale_factor * 0.3
        
        # Reconstruct signal from modified coefficients
        excitation = pywt.waverec(coeffs, wavelet_name)
        
        # Ensure correct length
        if len(excitation) > duration_samples:
            excitation = excitation[:duration_samples]
        elif len(excitation) < duration_samples:
            excitation = np.pad(excitation, (0, duration_samples - len(excitation)), mode='constant')
        
        excitation = excitation.astype(np.float32)
        
        # Apply envelope for natural attack/decay
        envelope = np.exp(-np.linspace(0, 3, duration_samples)).astype(np.float32)
        excitation *= envelope
        
        # Normalize
        max_val = np.max(np.abs(excitation))
        if max_val > 0:
            excitation = excitation / max_val * self.EXCITATION_NORMALIZATION_SCALE
        
        return excitation
    
    @staticmethod
    def _stabilize_lpc(lpc_coeffs, max_k=0.995):
        """Clamp reflection coefficients and reconstruct a stable AR polynomial."""
        original = np.asarray(lpc_coeffs)
        if original.ndim != 1:
            raise ValueError("lpc_coeffs must be 1-D")
        if not np.isfinite(max_k) or not 0.0 < max_k < 1.0:
            raise ValueError("max_k must be strictly between 0 and 1")
        if original.size == 0:
            return original.copy()
        if not np.all(np.isfinite(original)):
            return np.zeros_like(original)

        order = original.size
        direct = np.concatenate(([1.0], original.astype(np.float64, copy=False)))
        current = direct.copy()
        reflections = np.zeros(order, dtype=np.float64)
        # Step-down Levinson recursion. Clamp before division so malformed or
        # unstable input cannot create a negative prediction-error denominator.
        for index in range(order, 0, -1):
            reflection = float(np.clip(current[index], -max_k, max_k))
            reflections[index - 1] = reflection
            denominator = max(1.0 - reflection * reflection, 1.0 - max_k * max_k)
            previous = np.zeros(index, dtype=np.float64)
            previous[0] = 1.0
            for coefficient in range(1, index):
                previous[coefficient] = (
                    current[coefficient] - reflection * current[index - coefficient]
                ) / denominator
            if not np.all(np.isfinite(previous)):
                return np.zeros_like(original)
            current = previous

        stable = np.array([1.0], dtype=np.float64)
        for index, reflection in enumerate(reflections):
            updated = np.zeros(index + 2, dtype=np.float64)
            updated[0] = 1.0
            for coefficient in range(1, index + 1):
                updated[coefficient] = (
                    stable[coefficient] + reflection * stable[index + 1 - coefficient]
                )
            updated[index + 1] = reflection
            stable = updated
        result = stable[1:]
        if not np.all(np.isfinite(result)):
            return np.zeros_like(original)
        return result.astype(original.dtype if original.dtype.kind == "f" else np.float64)

    def _lpc_synthesis_filter(self, excitation, lpc_coeffs, zi=None):
        """Filter one excitation block, optionally preserving IIR state across hops."""
        excitation = np.asarray(excitation, dtype=np.float64)
        lpc_coeffs = np.asarray(lpc_coeffs, dtype=np.float64)
        if excitation.ndim != 1 or not np.all(np.isfinite(excitation)):
            raise ValueError("excitation must be a finite 1-D array")
        if lpc_coeffs.ndim != 1 or not np.all(np.isfinite(lpc_coeffs)):
            raise ValueError("lpc_coeffs must be a finite 1-D array")
        stable_coeffs = self._stabilize_lpc(lpc_coeffs)
        filter_coeffs = np.concatenate(([1.0], stable_coeffs.astype(np.float64)))
        if zi is not None:
            zi = np.asarray(zi, dtype=np.float64)
            if zi.shape != (len(filter_coeffs) - 1,) or not np.all(np.isfinite(zi)):
                raise ValueError(f"zi must be finite with shape ({len(filter_coeffs) - 1},)")
            synthesized, new_zi = signal.lfilter([1.0], filter_coeffs, excitation, zi=zi)
        else:
            synthesized = signal.lfilter([1.0], filter_coeffs, excitation)
            new_zi = None
        synthesized = np.nan_to_num(synthesized, nan=0.0, posinf=1e6, neginf=-1e6)
        synthesized = np.clip(synthesized, -1e6, 1e6).astype(np.float32)

        # Only standalone blocks are scaled here. Continuous generation carries zi
        # and receives a single global gain stage after all hops are joined.
        peak = float(np.max(np.abs(synthesized))) if synthesized.size else 0.0
        if peak > 0.0 and zi is None:
            synthesized = synthesized / peak * 0.8
        if zi is not None:
            return synthesized, np.asarray(new_zi, dtype=np.float64)
        return synthesized

    def _apply_gain_floor(
        self,
        audio: np.ndarray,
        *,
        minimum_peak: float = 0.75,
        target_peak: float = 0.9,
    ) -> np.ndarray:
        """Apply a bounded global gain stage; silence stays silent."""
        for name, value in (("minimum_peak", minimum_peak), ("target_peak", target_peak)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if target_peak < minimum_peak:
            raise ValueError("target_peak must be greater than or equal to minimum_peak")
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(arr)))
        if peak <= np.finfo(np.float32).tiny:
            return arr
        if peak <= minimum_peak:
            target_level = float(minimum_peak)
        elif peak > target_peak:
            target_level = float(target_peak)
        else:
            return arr
        return (arr * (target_level / peak)).astype(np.float32, copy=False)

    def _get_training_residual_for_state(self, state, rng=None, n_samples=None):
        """
        Get a representative residual from training data for a given state.

        Args:
            state: HMM state index
            rng: Random number generator (optional)

        Returns:
            residual: Training residual signal for this state
        """
        if not isinstance(state, (int, np.integer)) or not 0 <= int(state) < self.n_states:
            raise ValueError("state index is out of range")
        if rng is None:
            rng = self._rng
        target_length = self.hop_size if n_samples is None else int(n_samples)
        if target_length <= 0:
            raise ValueError("n_samples must be positive")

        if self.residual_signals is None or self.residual_signals.size == 0:
            return rng.standard_normal(target_length).astype(np.float32) * 0.05

        assignments = self._estimate_state_assignments()
        if assignments is None:
            # Fallback: use random training residual
            idx = rng.integers(0, len(self.residual_signals))
            residual = self.residual_signals[idx]
        else:
            # Get residuals for frames assigned to this state
            mask = assignments == state
            if np.any(mask):
                # Randomly select one of the frames assigned to this state
                state_indices = np.where(mask)[0]
                idx = rng.choice(state_indices)
                residual = self.residual_signals[idx]
            else:
                # State has no assignments, use random residual
                idx = rng.integers(0, len(self.residual_signals))
                residual = self.residual_signals[idx]

        # Ensure correct length
        if residual.shape[0] < target_length:
            residual = np.pad(
                residual,
                (0, target_length - residual.shape[0]),
                mode="constant",
            )
        elif residual.shape[0] > target_length:
            residual = residual[:target_length]

        return residual.astype(np.float32, copy=True)

    def render_preview(self, n_frames=32, sample_rate=None, state_sequence=None, rng=None, seed=None):
        """Render a short, state-continuous preview without borrowing EM buffers."""
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError("Model must be trained before rendering a preview")
        if isinstance(n_frames, (bool, np.bool_)) or not isinstance(n_frames, (int, np.integer)) or n_frames <= 0:
            raise ValueError("n_frames must be a positive integer")
        if sample_rate is None:
            output_rate = self._coerce_sample_rate(self.sample_rate or 16000)
        else:
            output_rate = self._coerce_sample_rate(sample_rate)
        analysis_rate = self.sample_rate or output_rate
        rng = rng if rng is not None else np.random.default_rng(seed)
        if state_sequence is None:
            sequence = self.sample_state_sequence(int(n_frames), rng=rng)
        else:
            sequence = self._validate_state_sequence(
                state_sequence, int(n_frames), allow_pad=True
            )

        prototypes = self._state_spectral_prototypes(rng=rng, seed=seed)
        required_samples = int(n_frames) * self.hop_size
        preview = np.zeros(required_samples, dtype=np.float32)
        filter_state = np.zeros(self.lpc_order, dtype=np.float64)
        for idx, state in enumerate(sequence):
            lpc_coeffs, residual_template = prototypes[state]
            excitation = np.asarray(residual_template[: self.hop_size], dtype=np.float32)
            if excitation.size < self.hop_size:
                excitation = np.pad(excitation, (0, self.hop_size - excitation.size))
            frame_audio, filter_state = self._lpc_synthesis_filter(
                excitation, lpc_coeffs, zi=filter_state
            )
            start = idx * self.hop_size
            preview[start : start + self.hop_size] = frame_audio[: self.hop_size]

        # Fade only the outer edges; applying a Hann window to the whole preview
        # used to attenuate every interior frame and create a large level dip.
        fade = min(max(1, int(round(0.005 * analysis_rate))), required_samples // 2)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            preview[:fade] *= ramp
            preview[-fade:] *= ramp[::-1]
        if output_rate != analysis_rate:
            divisor = math.gcd(output_rate, analysis_rate)
            preview = signal.resample_poly(
                preview, output_rate // divisor, analysis_rate // divisor
            ).astype(np.float32, copy=False)
        preview = self._apply_gain_floor(preview, minimum_peak=0.75, target_peak=0.9)
        return preview.astype(np.float32, copy=False), output_rate

    def _validate_state_sequence(self, state_sequence, target_frames, *, allow_pad=False):
        if isinstance(target_frames, (bool, np.bool_)) or not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
            raise ValueError("target_frames must be a positive integer")
        try:
            sequence = list(state_sequence)
        except TypeError as exc:
            raise ValueError("state_sequence must be an iterable of state indices") from exc
        if not sequence:
            raise ValueError("state_sequence must contain at least one state")
        normalized = []
        for state in sequence:
            if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)):
                raise ValueError("state_sequence values must be integer state indices")
            state = int(state)
            if not 0 <= state < self.n_states:
                raise ValueError(f"state_sequence contains out-of-range state {state}")
            normalized.append(state)
        if len(normalized) < target_frames:
            if not allow_pad:
                raise ValueError(
                    f"state_sequence length ({len(normalized)}) < target_frames ({target_frames})"
                )
            normalized.extend([normalized[-1]] * (target_frames - len(normalized)))
        return normalized[:target_frames]

    def sample_state_sequence(
        self,
        target_frames: int,
        temperature: float = 1.0,
        rng=None,
        seed: Optional[int] = None,
    ):
        """Sample a finite Markov trajectory with a strictly positive temperature."""
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError(_ERR_HMM_NOT_INITIALIZED)
        if isinstance(target_frames, (bool, np.bool_)) or not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
            raise ValueError("target_frames must be a positive integer")
        if isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, (int, float, np.number)) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        rng = rng if rng is not None else np.random.default_rng(seed)
        transitions, initial, _, _ = self._validated_model_parameters()

        def tempered(probabilities):
            logits = np.log(np.clip(probabilities, np.finfo(np.float64).tiny, None)) / float(temperature)
            logits -= np.max(logits)
            weights = np.exp(logits)
            return weights / weights.sum()

        state = int(rng.choice(self.n_states, p=tempered(initial)))
        path = [state]
        for _ in range(1, int(target_frames)):
            state = int(rng.choice(self.n_states, p=tempered(transitions[state])))
            path.append(state)
        return path

    def synthesize_audio(
        self,
        target_duration_seconds,
        sample_rate=16000,
        fidelity=0.0,
        excitation_type="Karplus-Strong",
        state_sequence=None,
        method="viterbi",
        temperature=1.0,
        rng=None,
        seed=None,
    ):
        """Generate exactly the requested duration, with rate-correct resampling."""
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError("Must train the model first using extract_features() and iterative_refinement()")
        if isinstance(target_duration_seconds, (bool, np.bool_)) or not isinstance(target_duration_seconds, (int, float, np.number)) or not np.isfinite(target_duration_seconds) or target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be finite and greater than zero")
        output_rate = self._coerce_sample_rate(sample_rate)
        if isinstance(fidelity, (bool, np.bool_)) or not isinstance(fidelity, (int, float, np.number)) or not np.isfinite(fidelity) or not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be a finite number between 0.0 and 1.0")
        if isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, (int, float, np.number)) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        excitation_map = {
            "karplus-strong": "Karplus-Strong",
            "wavelet": "Wavelet",
        }
        if not isinstance(excitation_type, str) or excitation_type.casefold() not in excitation_map:
            raise ValueError("excitation_type must be 'Karplus-Strong' or 'Wavelet'")
        excitation_type = excitation_map[excitation_type.casefold()]
        method_aliases = {"viterbi": "viterbi", "stochastic": "stochastic", "markov": "stochastic", "sample": "stochastic"}
        if not isinstance(method, str) or method.casefold() not in method_aliases:
            raise ValueError("method must be 'viterbi' or 'stochastic'/'markov'")
        method = method_aliases[method.casefold()]

        _, _, state_means, state_covariances = self._validated_model_parameters()

        analysis_rate = self.sample_rate or output_rate
        target_output_samples = int(round(float(target_duration_seconds) * output_rate))
        target_analysis_samples = int(round(float(target_duration_seconds) * analysis_rate))
        if target_output_samples <= 0 or target_analysis_samples <= 0:
            raise ValueError("requested duration is shorter than one output sample")
        target_frames = max(1, int(math.ceil(target_analysis_samples / self.hop_size)))
        rng = rng if rng is not None else np.random.default_rng(seed)

        if state_sequence is not None:
            sequence = self._validate_state_sequence(state_sequence, target_frames)
        elif method == "stochastic":
            sequence = self.sample_state_sequence(
                target_frames, temperature=temperature, rng=rng
            )
        else:
            sequence = self._a_star_search(target_frames)

        audio = np.zeros(target_frames * self.hop_size, dtype=np.float32)
        filter_state = np.zeros(self.lpc_order, dtype=np.float64)
        for idx, state in enumerate(sequence):
            if fidelity >= 1.0:
                excitation = self._get_training_residual_for_state(
                    state, rng=rng, n_samples=self.hop_size
                )
            elif fidelity <= 0.0:
                if excitation_type == "Karplus-Strong":
                    excitation = self._karplus_strong_excitation(state, self.hop_size, rng=rng)
                else:
                    excitation = self._wavelet_excitation(state, self.hop_size, rng=rng)
            else:
                residual = self._get_training_residual_for_state(
                    state, rng=rng, n_samples=self.hop_size
                )
                if excitation_type == "Karplus-Strong":
                    synthetic = self._karplus_strong_excitation(state, self.hop_size, rng=rng)
                else:
                    synthetic = self._wavelet_excitation(state, self.hop_size, rng=rng)
                excitation = fidelity * residual + (1.0 - fidelity) * synthetic

            frame_audio, filter_state = self._lpc_synthesis_filter(
                excitation, state_means[state], zi=filter_state
            )
            start = idx * self.hop_size
            audio[start : start + self.hop_size] = frame_audio[: self.hop_size]

        # Synthesize at the trained analysis rate so frame timing and LPC pole
        # frequencies stay coherent, then convert to the caller's requested rate.
        audio = audio[:target_analysis_samples]
        if output_rate != analysis_rate:
            divisor = math.gcd(output_rate, analysis_rate)
            audio = signal.resample_poly(
                audio, output_rate // divisor, analysis_rate // divisor
            ).astype(np.float32, copy=False)
        if len(audio) < target_output_samples:
            audio = np.pad(audio, (0, target_output_samples - len(audio)))
        else:
            audio = audio[:target_output_samples]
        audio = self._apply_gain_floor(audio, minimum_peak=0.75, target_peak=0.9)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _atomic_write(filepath, writer):
        """Write an artifact beside its destination and atomically replace it."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{filepath.name}.", suffix=filepath.suffix, dir=str(filepath.parent)
        )
        os.close(fd)
        try:
            writer(Path(temp_name))
            os.replace(temp_name, filepath)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _restore_model_parameters(self, transition, initial, means, covariances):
        """Validate, normalize and regularize parameters loaded from an artifact."""
        transition = np.asarray(transition, dtype=np.float64)
        initial = np.asarray(initial, dtype=np.float64)
        means = np.asarray(means, dtype=np.float64)
        covariances = np.asarray(covariances, dtype=np.float64)
        expected_cov_shape = (self.n_states, self.lpc_order, self.lpc_order)
        if transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition_matrix shape does not match model metadata")
        if initial.shape != (self.n_states,):
            raise ValueError("initial_probabilities shape does not match model metadata")
        if means.shape != (self.n_states, self.lpc_order):
            raise ValueError("state_means shape does not match model metadata")
        if covariances.shape != expected_cov_shape:
            raise ValueError("state_covariances shape does not match model metadata")
        if not all(np.all(np.isfinite(array)) for array in (transition, initial, means, covariances)):
            raise ValueError("model artifact contains NaN or infinite parameters")
        if np.any(transition < 0) or np.any(initial < 0):
            raise ValueError("model probabilities must be non-negative")
        if np.any(transition.sum(axis=1) <= 0) or initial.sum() <= 0:
            raise ValueError("model probabilities must have positive mass")
        transition = self._normalize_probabilities(transition, axis=1)
        initial = self._normalize_probabilities(initial)

        repaired_covariances = np.empty_like(covariances)
        for state in range(self.n_states):
            cov = 0.5 * (covariances[state] + covariances[state].T)
            eigenvalues = np.linalg.eigvalsh(cov)
            minimum = float(eigenvalues.min())
            if minimum < self.COVARIANCE_MIN_EIGENVALUE:
                cov += np.eye(self.lpc_order) * (
                    self.COVARIANCE_MIN_EIGENVALUE - minimum
                )
            repaired_covariances[state] = 0.5 * (cov + cov.T)
        self.transition_matrix = transition
        self.initial_probabilities = initial
        self.state_means = means
        self.state_covariances = repaired_covariances

    def _validate_training_artifacts(self):
        """Check optional cached arrays remain aligned after loading or adaptation."""
        count = None
        if self.lpc_coefficients is not None:
            lpc = np.asarray(self.lpc_coefficients)
            if lpc.ndim != 2 or lpc.shape[1] != self.lpc_order or not np.all(np.isfinite(lpc)):
                raise ValueError("lpc_coefficients artifact has an invalid shape or values")
            count = len(lpc)
        if self.training_frames is not None:
            frames = np.asarray(self.training_frames)
            if frames.ndim != 2 or frames.shape[1] != self.frame_size or not np.all(np.isfinite(frames)):
                raise ValueError("training_frames artifact has an invalid shape or values")
            if count is not None and len(frames) != count:
                raise ValueError("training_frames and lpc_coefficients lengths differ")
            count = len(frames) if count is None else count
        if count is None:
            if self.sequence_starts is not None:
                raise ValueError("sequence_starts cannot be loaded without training artifacts")
            return
        for name, array in (
            ("residual_signals", self.residual_signals),
            ("perceptual_features", self.perceptual_features),
            ("spectral_flux", self.spectral_flux),
        ):
            if array is not None and (np.asarray(array).ndim == 0 or len(array) != count or not np.all(np.isfinite(array))):
                raise ValueError(f"{name} artifact has inconsistent length or non-finite values")
        if self.residual_signals is not None and (
            np.asarray(self.residual_signals).ndim != 2
            or np.asarray(self.residual_signals).shape[1] != self.hop_size
        ):
            raise ValueError("residual_signals artifact must have shape (n, hop_size)")
        if self.perceptual_features is not None and np.asarray(self.perceptual_features).ndim != 2:
            raise ValueError("perceptual_features artifact must be 2-D")
        if self.spectral_flux is not None and np.asarray(self.spectral_flux).ndim != 1:
            raise ValueError("spectral_flux artifact must be 1-D")
        if self.frame_metadata:
            for key, values in self.frame_metadata.items():
                values = np.asarray(values)
                if values.ndim != 1 or len(values) != count or not np.all(np.isfinite(values)):
                    raise ValueError(f"frame metadata '{key}' has inconsistent shape or non-finite values")
        starts = self._normalize_sequence_starts(count, self.sequence_starts)
        self._artifacts.sequence_starts = starts

    def export_model(
        self,
        filepath,
        *,
        use_compression=True,
        precision=np.float32,
        pack_covariances=True,
        include_training_artifacts=False,
        include_adaptive_statistics=True,
    ):
        """Atomically export parameters and optional aligned training artifacts."""
        transition, initial, means, covariances = self._validated_model_parameters()
        self._compute_and_cache_cholesky_factors()
        target_dtype = np.dtype(precision)
        if target_dtype.kind != "f":
            raise ValueError("precision must be a floating-point dtype")
        filepath = Path(filepath)
        if filepath.suffix != ".npz":
            filepath = filepath.with_suffix(".npz")

        metadata = {
            "version": 2,
            "n_states": int(self.n_states),
            "lpc_order": int(self.lpc_order),
            "frame_size": int(self.frame_size),
            "hop_size": int(self.hop_size),
            "smoothness_weight": float(self.smoothness_weight),
            "covariance_floor": float(self.covariance_floor),
            "transition_pseudocount": float(self.transition_pseudocount),
            "random_state": int(self.random_state),
            "pack_covariances": bool(pack_covariances),
            "precision": target_dtype.name,
            "includes_training_artifacts": bool(include_training_artifacts),
            "includes_adaptive_statistics": bool(include_adaptive_statistics),
            "sample_rate": int(self.sample_rate) if self.sample_rate is not None else None,
            "adaptive_memory_limit": int(self.adaptive_memory_limit),
            "augmentation_applied": bool(self.augmentation_applied),
        }
        payload = {
            "transition_matrix": transition.astype(target_dtype, copy=False),
            "initial_probabilities": initial.astype(target_dtype, copy=False),
            "state_means": means.astype(target_dtype, copy=False),
        }
        if pack_covariances:
            tri_rows, tri_cols = np.tril_indices(self.lpc_order)
            payload["state_covariances_packed"] = covariances[:, tri_rows, tri_cols].astype(target_dtype, copy=False)
        else:
            payload["state_covariances"] = covariances.astype(target_dtype, copy=False)

        if include_training_artifacts:
            for key, value in (
                ("training_frames", self.training_frames),
                ("lpc_coefficients", self.lpc_coefficients),
                ("residual_signals", self.residual_signals),
                ("perceptual_features", self.perceptual_features),
                ("spectral_flux", self.spectral_flux),
            ):
                if value is not None:
                    payload[key] = np.asarray(value).astype(target_dtype, copy=False)
            if self.sequence_starts is not None:
                payload["sequence_starts"] = np.asarray(self.sequence_starts, dtype=np.int64)
            if self.frame_metadata:
                metadata["frame_metadata_keys"] = list(self.frame_metadata.keys())
                for key, values in self.frame_metadata.items():
                    arr = np.asarray(values)
                    payload[f"frame_metadata__{key}"] = arr.astype(target_dtype, copy=False) if arr.dtype.kind == "f" else arr

        if include_adaptive_statistics and self._adaptive_stats is not None:
            stats = self._adaptive_stats
            payload.update({
                "adaptive_state_frame_counts": stats.state_frame_counts.astype(target_dtype, copy=False),
                "adaptive_transition_counts": stats.transition_counts.astype(target_dtype, copy=False),
                "adaptive_mean_accumulator": stats.mean_accumulator.astype(target_dtype, copy=False),
                "adaptive_second_moment": stats.second_moment_accumulator.astype(target_dtype, copy=False),
                "adaptive_initial_counts": stats.initial_counts.astype(target_dtype, copy=False),
                "adaptive_total_frames": np.array(stats.total_frames, dtype=target_dtype),
                "adaptive_last_adaptation": np.array(stats.last_adaptation or "", dtype=np.str_),
            })
        payload["metadata"] = np.array(json.dumps(metadata), dtype=np.str_)
        saver = np.savez_compressed if use_compression else np.savez
        self._atomic_write(filepath, lambda temporary: saver(temporary, **payload))
        logger.info("Model exported to %s", filepath)

    @classmethod
    def load_model(cls, filepath):
        """Load and validate an SSGS ``.npz`` artifact (including v1 archives)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        try:
            archive = np.load(filepath, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not open model archive: {exc}") from exc
        with archive as data:
            if "metadata" not in data:
                raise ValueError("Model archive is missing metadata")
            try:
                metadata = json.loads(data["metadata"].item())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Model archive metadata is invalid JSON") from exc
            version = int(metadata.get("version", 1))
            if version not in (1, 2):
                raise ValueError(f"Unsupported model archive version: {version}")
            try:
                model = cls(
                    n_states=int(metadata["n_states"]),
                    lpc_order=int(metadata["lpc_order"]),
                    frame_size=int(metadata["frame_size"]),
                    hop_size=int(metadata["hop_size"]),
                    smoothness_weight=float(metadata.get("smoothness_weight", 0.5)),
                    adaptive_memory_limit=int(metadata.get("adaptive_memory_limit", 6000)),
                    random_state=int(metadata.get("random_state", 0)),
                    covariance_floor=float(metadata.get("covariance_floor", 1e-4)),
                    transition_pseudocount=float(metadata.get("transition_pseudocount", 1e-6)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Model archive configuration is invalid: {exc}") from exc

            if metadata.get("pack_covariances", False):
                if "state_covariances_packed" not in data:
                    raise ValueError("Model archive is missing packed covariances")
                packed = np.asarray(data["state_covariances_packed"], dtype=np.float64)
                expected = model.lpc_order * (model.lpc_order + 1) // 2
                if packed.shape != (model.n_states, expected):
                    raise ValueError("Packed covariance tensor has invalid shape")
                rows, cols = np.tril_indices(model.lpc_order)
                covariances = np.zeros((model.n_states, model.lpc_order, model.lpc_order), dtype=np.float64)
                covariances[:, rows, cols] = packed
                covariances[:, cols, rows] = packed
            elif "state_covariances" in data:
                covariances = np.asarray(data["state_covariances"], dtype=np.float64)
            else:
                raise ValueError("Model archive is missing state covariances")
            required = ("transition_matrix", "initial_probabilities", "state_means")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError("Model archive missing required arrays: " + ", ".join(missing))
            model._restore_model_parameters(
                data["transition_matrix"], data["initial_probabilities"],
                data["state_means"], covariances,
            )

            for key in ("training_frames", "lpc_coefficients", "residual_signals", "perceptual_features", "spectral_flux"):
                if key in data:
                    setattr(model, key, data[key].astype(np.float32, copy=False))
            if "sequence_starts" in data:
                model.sequence_starts = data["sequence_starts"]
            elif model.lpc_coefficients is not None or model.training_frames is not None:
                model.sequence_starts = np.array([0], dtype=np.int64)
            meta_keys = metadata.get("frame_metadata_keys", [])
            if meta_keys:
                frame_meta = {}
                for key in meta_keys:
                    tensor_key = f"frame_metadata__{key}"
                    if tensor_key not in data:
                        raise ValueError(f"Model archive is missing frame metadata '{key}'")
                    frame_meta[key] = data[tensor_key].copy()
                model.frame_metadata = frame_meta
            model.sample_rate = metadata.get("sample_rate")
            model.augmentation_applied = bool(metadata.get("augmentation_applied", False))

            if metadata.get("includes_adaptive_statistics") and "adaptive_state_frame_counts" in data:
                required_stats = (
                    "adaptive_transition_counts", "adaptive_mean_accumulator",
                    "adaptive_second_moment", "adaptive_initial_counts", "adaptive_total_frames",
                )
                missing_stats = [key for key in required_stats if key not in data]
                if missing_stats:
                    raise ValueError("Adaptive statistics incomplete: " + ", ".join(missing_stats))
                model._adaptive_stats = _AdaptiveStatistics(
                    state_frame_counts=np.asarray(data["adaptive_state_frame_counts"], dtype=np.float64),
                    transition_counts=np.asarray(data["adaptive_transition_counts"], dtype=np.float64),
                    mean_accumulator=np.asarray(data["adaptive_mean_accumulator"], dtype=np.float64),
                    second_moment_accumulator=np.asarray(data["adaptive_second_moment"], dtype=np.float64),
                    initial_counts=np.asarray(data["adaptive_initial_counts"], dtype=np.float64),
                    total_frames=float(data["adaptive_total_frames"].item()),
                    last_adaptation=str(data["adaptive_last_adaptation"].item() or "") if "adaptive_last_adaptation" in data else None,
                )
            model._validate_training_artifacts()

        model._reset_caches()
        model._compute_and_cache_cholesky_factors()
        model._transition_cost_matrix()
        return model

    def save_checkpoint(
        self,
        filepath,
        *,
        precision=np.float32,
        include_training_artifacts=True,
        include_adaptive_statistics=True,
    ):
        """Atomically save a validated, pickle-free safetensors checkpoint."""
        try:
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for checkpoint saving. Install with: pip install safetensors"
            ) from exc
        transition, initial, means, covariances = self._validated_model_parameters()
        self._compute_and_cache_cholesky_factors()
        target_dtype = np.dtype(precision)
        if target_dtype.kind != "f":
            raise ValueError("precision must be a floating-point dtype")
        filepath = Path(filepath)
        if filepath.suffix != ".safetensors":
            filepath = filepath.with_suffix(".safetensors")

        tensors = {
            "transition_matrix": transition.astype(target_dtype, copy=False),
            "initial_probabilities": initial.astype(target_dtype, copy=False),
            "state_means": means.astype(target_dtype, copy=False),
            "state_covariances": np.asarray(self.state_covariances).astype(target_dtype, copy=False),
        }
        if include_training_artifacts:
            for key, value in (
                ("training_frames", self.training_frames),
                ("lpc_coefficients", self.lpc_coefficients),
                ("residual_signals", self.residual_signals),
                ("perceptual_features", self.perceptual_features),
                ("spectral_flux", self.spectral_flux),
            ):
                if value is not None and np.asarray(value).size:
                    tensors[key] = np.asarray(value).astype(target_dtype, copy=False)
            if self.sequence_starts is not None:
                tensors["sequence_starts"] = np.asarray(self.sequence_starts, dtype=np.int64)
            for key, values in (self.frame_metadata or {}).items():
                arr = np.asarray(values)
                if arr.dtype.kind == "f":
                    arr = arr.astype(target_dtype, copy=False)
                tensors[f"frame_metadata__{key}"] = arr

        metadata = {
            "format": "safetensors",
            "version": "2",
            "n_states": str(self.n_states),
            "lpc_order": str(self.lpc_order),
            "frame_size": str(self.frame_size),
            "hop_size": str(self.hop_size),
            "smoothness_weight": str(self.smoothness_weight),
            "covariance_floor": str(self.covariance_floor),
            "transition_pseudocount": str(self.transition_pseudocount),
            "random_state": str(self.random_state),
            "precision": target_dtype.name,
            "includes_training_artifacts": str(bool(include_training_artifacts)),
            "includes_adaptive_statistics": str(bool(include_adaptive_statistics)),
            "sample_rate": str(self.sample_rate) if self.sample_rate is not None else "None",
            "frame_metadata_keys": json.dumps(
                list(self.frame_metadata.keys())
                if include_training_artifacts and self.frame_metadata
                else []
            ),
            "adaptive_memory_limit": str(self.adaptive_memory_limit),
            "augmentation_applied": str(bool(self.augmentation_applied)),
        }
        if include_adaptive_statistics and self._adaptive_stats is not None:
            stats = self._adaptive_stats
            tensors.update({
                "adaptive_state_frame_counts": np.asarray(stats.state_frame_counts, dtype=target_dtype),
                "adaptive_transition_counts": np.asarray(stats.transition_counts, dtype=target_dtype),
                "adaptive_mean_accumulator": np.asarray(stats.mean_accumulator, dtype=target_dtype),
                "adaptive_second_moment": np.asarray(stats.second_moment_accumulator, dtype=target_dtype),
                "adaptive_initial_counts": np.asarray(stats.initial_counts, dtype=target_dtype),
                "adaptive_total_frames": np.array(stats.total_frames, dtype=target_dtype),
            })
            metadata["adaptive_last_adaptation"] = stats.last_adaptation or ""

        tensors = {key: np.ascontiguousarray(value) for key, value in tensors.items()}
        self._atomic_write(
            filepath,
            lambda temporary: save_file(tensors, temporary, metadata=metadata),
        )
        logger.info("Checkpoint saved to %s", filepath)

    @staticmethod
    def _validate_checkpoint_tensors(tensors, metadata):
        required_keys = (
            "transition_matrix", "initial_probabilities", "state_means", "state_covariances"
        )
        missing = [key for key in required_keys if key not in tensors]
        if missing:
            raise ValueError("Checkpoint missing required tensors: " + ", ".join(missing))
        transition = np.asarray(tensors["transition_matrix"])
        if transition.ndim != 2 or transition.shape[0] != transition.shape[1] or transition.shape[0] == 0:
            raise ValueError("transition_matrix must be a non-empty square 2D array")
        n_states = transition.shape[0]
        declared_states = int(metadata.get("n_states", n_states))
        if n_states != declared_states:
            raise ValueError("transition_matrix shape does not match n_states metadata")
        initial = np.asarray(tensors["initial_probabilities"])
        if initial.shape != (n_states,):
            raise ValueError("initial_probabilities must have shape (n_states,)")
        means = np.asarray(tensors["state_means"])
        if means.ndim != 2 or means.shape[0] != n_states:
            raise ValueError("state_means must be a 2D array with n_states rows")
        lpc_order = int(metadata.get("lpc_order", means.shape[1]))
        if lpc_order <= 0 or means.shape[1] != lpc_order:
            raise ValueError("state_means shape does not match lpc_order metadata")
        covariances = np.asarray(tensors["state_covariances"])
        if covariances.shape != (n_states, lpc_order, lpc_order):
            raise ValueError(
                f"state_covariances must have shape ({n_states}, {lpc_order}, {lpc_order})"
            )
        for name, array in (("transition_matrix", transition), ("initial_probabilities", initial),
                            ("state_means", means), ("state_covariances", covariances)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")
        if np.any(transition < 0) or np.any(initial < 0):
            raise ValueError("checkpoint probability tensors must be non-negative")
        if np.any(transition.sum(axis=1) <= 0) or initial.sum() <= 0:
            raise ValueError("checkpoint probability tensors must have positive mass")

    @classmethod
    def load_checkpoint(cls, filepath):
        """Load a validated safetensors checkpoint without executing serialized code."""
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for checkpoint loading. Install with: pip install safetensors"
            ) from exc
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # Bound and parse the small JSON header before asking safetensors to map arrays.
        try:
            with filepath.open("rb") as handle:
                header_size_bytes = handle.read(8)
                if len(header_size_bytes) != 8:
                    raise ValueError("Invalid safetensors file: unable to read header size")
                header_size = int.from_bytes(header_size_bytes, byteorder="little")
                max_header_size = 1024 * 1024
                if header_size <= 0 or header_size > max_header_size:
                    raise ValueError(f"Invalid safetensors file: header size {header_size} out of bounds")
                if 8 + header_size > filepath.stat().st_size:
                    raise ValueError("Invalid safetensors file: declared header size exceeds file size")
                raw_header = handle.read(header_size)
                if len(raw_header) != header_size:
                    raise ValueError("Invalid safetensors file: incomplete header data")
                header = json.loads(raw_header.decode("utf-8"))
                metadata = header.get("__metadata__", {})
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid safetensors file: metadata must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensors header: {exc}") from exc

        try:
            tensors = load_file(filepath)
        except Exception as exc:
            raise ValueError(f"Could not load safetensors checkpoint: {exc}") from exc
        cls._validate_checkpoint_tensors(tensors, metadata)
        try:
            model = cls(
                n_states=int(metadata.get("n_states", 34)),
                lpc_order=int(metadata.get("lpc_order", 12)),
                frame_size=int(metadata.get("frame_size", 1024)),
                hop_size=int(metadata.get("hop_size", 256)),
                smoothness_weight=float(metadata.get("smoothness_weight", 0.5)),
                adaptive_memory_limit=int(metadata.get("adaptive_memory_limit", 6000)),
                random_state=int(metadata.get("random_state", 0)),
                covariance_floor=float(metadata.get("covariance_floor", 1e-4)),
                transition_pseudocount=float(metadata.get("transition_pseudocount", 1e-6)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint configuration is invalid: {exc}") from exc

        model._restore_model_parameters(
            tensors["transition_matrix"], tensors["initial_probabilities"],
            tensors["state_means"], tensors["state_covariances"],
        )
        for key in ("training_frames", "lpc_coefficients", "residual_signals", "perceptual_features", "spectral_flux"):
            if key in tensors:
                setattr(model, key, tensors[key].astype(np.float32, copy=False))
        if "sequence_starts" in tensors:
            model.sequence_starts = tensors["sequence_starts"]
        elif model.lpc_coefficients is not None or model.training_frames is not None:
            model.sequence_starts = np.array([0], dtype=np.int64)

        try:
            meta_keys = json.loads(metadata.get("frame_metadata_keys", "[]"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("frame_metadata_keys metadata is invalid JSON") from exc
        if not isinstance(meta_keys, list):
            raise ValueError("frame_metadata_keys metadata must be a JSON list")
        frame_meta = {}
        for key in meta_keys:
            tensor_key = f"frame_metadata__{key}"
            if tensor_key not in tensors:
                raise ValueError(f"Checkpoint is missing frame metadata tensor '{key}'")
            frame_meta[key] = tensors[tensor_key].copy()
        model.frame_metadata = frame_meta

        sample_rate = metadata.get("sample_rate", "None")
        model.sample_rate = None if sample_rate in (None, "None", "") else int(sample_rate)
        model.augmentation_applied = metadata.get("augmentation_applied", "False") == "True"

        includes_adaptive = metadata.get("includes_adaptive_statistics", "False") == "True"
        if includes_adaptive and "adaptive_state_frame_counts" in tensors:
            adaptive_keys = (
                "adaptive_transition_counts", "adaptive_mean_accumulator",
                "adaptive_second_moment", "adaptive_initial_counts", "adaptive_total_frames",
            )
            missing_stats = [key for key in adaptive_keys if key not in tensors]
            if missing_stats:
                raise ValueError("Adaptive statistics incomplete: " + ", ".join(missing_stats))
            model._adaptive_stats = _AdaptiveStatistics(
                state_frame_counts=np.asarray(tensors["adaptive_state_frame_counts"], dtype=np.float64),
                transition_counts=np.asarray(tensors["adaptive_transition_counts"], dtype=np.float64),
                mean_accumulator=np.asarray(tensors["adaptive_mean_accumulator"], dtype=np.float64),
                second_moment_accumulator=np.asarray(tensors["adaptive_second_moment"], dtype=np.float64),
                initial_counts=np.asarray(tensors["adaptive_initial_counts"], dtype=np.float64),
                total_frames=float(np.asarray(tensors["adaptive_total_frames"]).item()),
                last_adaptation=metadata.get("adaptive_last_adaptation") or None,
            )
            stats = model._adaptive_stats
            expected = (
                (model.n_states,),
                (model.n_states, model.n_states),
                (model.n_states, model.lpc_order),
                (model.n_states, model.lpc_order, model.lpc_order),
                (model.n_states,),
            )
            for name, array, shape in zip(
                ("state_frame_counts", "transition_counts", "mean_accumulator", "second_moment_accumulator", "initial_counts"),
                (stats.state_frame_counts, stats.transition_counts, stats.mean_accumulator, stats.second_moment_accumulator, stats.initial_counts),
                expected,
            ):
                if array.shape != shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"Adaptive statistic {name} has invalid shape or values")
            if not np.isfinite(stats.total_frames) or stats.total_frames < 0:
                raise ValueError("Adaptive total_frames must be finite and non-negative")

        model._validate_training_artifacts()
        model._reset_caches()
        model._compute_and_cache_cholesky_factors()
        model._transition_cost_matrix()
        logger.info("Checkpoint loaded from %s", filepath)
        return model

    def extract_features_from_files(self, audio_files, sample_rate=16000):
        """Extract multiple audio inputs, preserving each file as an HMM sequence.

        Path inputs are decoded with soundfile and resampled to ``sample_rate``.
        Array inputs are assumed to already use that rate. Mono and samples-by-
        channels arrays are accepted; all features remain in chronological order.
        """
        sample_rate = self._coerce_sample_rate(sample_rate)
        audio_files = list(audio_files)
        if not audio_files:
            raise ValueError("audio_files must contain at least one input")

        all_frames = []
        all_lpc = []
        all_residuals = []
        all_perceptual = []
        all_flux = []
        sequence_starts = []
        total_frames = 0
        mel_filters = self._mel_filter_bank(sample_rate, self.frame_size)
        erb_filters = self._erb_filter_bank(sample_rate, self.frame_size)
        window = np.hanning(self.frame_size).astype(np.float32)

        for idx, audio_input in enumerate(audio_files):
            if isinstance(audio_input, (str, Path)):
                try:
                    import soundfile as sf
                    raw_audio, file_sr = sf.read(audio_input, dtype="float32", always_2d=False)
                except Exception as exc:
                    raise ValueError(f"Failed to load audio file {audio_input}: {exc}") from exc
                audio = self._prepare_audio(raw_audio, int(file_sr), sample_rate)
                description = str(audio_input)
            else:
                audio = self._prepare_audio(audio_input, sample_rate, sample_rate)
                description = f"input {idx + 1}"

            frames = self._frame_signal(audio)
            processed, coeffs, residuals, perceptual, flux = self._process_frame_batch(
                frames, mel_filters, erb_filters, window
            )
            if coeffs.size == 0:
                logger.warning("Skipping silent audio %s", description)
                continue
            sequence_starts.append(total_frames)
            total_frames += len(coeffs)
            all_frames.append(processed)
            all_lpc.append(coeffs)
            all_residuals.append(residuals)
            all_perceptual.append(perceptual)
            all_flux.append(flux)
            logger.info("Processed %s: %d frames", description, len(coeffs))

        if not all_frames:
            raise ValueError("No valid frames extracted from any audio input")

        self.training_frames = np.concatenate(all_frames, axis=0)
        self.lpc_coefficients = np.concatenate(all_lpc, axis=0)
        self.residual_signals = np.concatenate(all_residuals, axis=0)
        self.perceptual_features = np.concatenate(all_perceptual, axis=0)
        self.spectral_flux = np.concatenate(all_flux, axis=0).astype(np.float32, copy=False)
        self.sample_rate = sample_rate
        self.sequence_starts = np.asarray(sequence_starts, dtype=np.int64)

        self._compute_frame_metadata(sample_rate)
        if self.spectral_flux.size:
            metadata = self.frame_metadata
            metadata["spectral_flux"] = self.spectral_flux
            self.frame_metadata = metadata

        self._reset_caches()
        self._compute_and_cache_cholesky_factors()
        logger.info(
            "Extracted %d frames from %d non-silent audio inputs",
            len(self.lpc_coefficients),
            len(sequence_starts),
        )

    def train(
        self,
        audio_signal,
        sample_rate=16000,
        n_em_iterations=10,
        *,
        preview_every: int = 0,
        preview_duration_frames: int = 32,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Complete training pipeline
        
        Args:
            audio_signal: Training audio signal (single array) or list of audio signals/file paths
            sample_rate: Sampling rate
            n_em_iterations: Number of EM iterations
            preview_every: Forward previews every N EM iterations (0 disables)
            preview_duration_frames: Frame count per preview render
            preview_callback: Optional callable receiving preview metadata
        """
        print("Phase 1: Training (HMM Parameter Estimation)")
        print("=" * 50)
        
        # Step 1: Feature Extraction
        print("Step 1: Extracting features (LPC analysis)...")
        
        # Check if we have multiple files/signals
        if isinstance(audio_signal, (list, tuple)):
            print(f"Batch mode: Training on {len(audio_signal)} audio files/signals")
            self.extract_features_from_files(audio_signal, sample_rate)
        else:
            # Single signal
            self.extract_features(audio_signal, sample_rate)
            print(f"Extracted {len(self.lpc_coefficients)} frames of LPC coefficients")
        
        # Step 2: State Initialization
        print("Step 2: Initializing HMM parameters...")
        self.initialize_hmm_parameters()
        
        # Step 3: Iterative Learning
        print("Step 3: Running EM algorithm...")
        self.iterative_refinement(
            n_em_iterations,
            preview_every=preview_every,
            preview_duration_frames=preview_duration_frames,
            preview_callback=preview_callback,
        )
        
        # Step 4: Graph Constraints
        print("Step 4: Applying graph constraints...")
        self.identify_graph_constraints()
        
        print("Training complete!")

    def adapt_to_audio(
        self,
        audio_signal,
        sample_rate=16000,
        *,
        adaptation_rate: float = 0.25,
        stability_bias: float = 0.05,
        memory_blend: float = 0.7,
        memory_limit: Optional[int] = None,
        n_adaptation_iterations: int = 1,
        rng=None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Adapt the model to new audio using exponential-moving-average (EMA)
        statistics (not persistent memory; adaptation_rate controls decay).

        Args:
            audio_signal: New audio samples to adapt to.
            sample_rate: Sampling rate of the input audio.
            adaptation_rate: Blend factor for new statistics (0.0-1.0).
            stability_bias: Regularization strength to avoid degenerate updates (0.0-1.0).
            memory_blend: Fraction of memory reserved for existing training frames.
            memory_limit: Optional override of adaptive_memory_limit.
            n_adaptation_iterations: Additional EM iterations on the new data.
            rng: Optional NumPy random generator.
            seed: Optional random seed if rng is None.

        Returns:
            Dictionary of adaptation metrics.
        """
        if self.transition_matrix is None:
            raise ValueError("Model must be trained before adaptation")
        if isinstance(n_adaptation_iterations, (bool, np.bool_)) or not isinstance(n_adaptation_iterations, (int, np.integer)) or n_adaptation_iterations < 0:
            raise ValueError("n_adaptation_iterations must be a non-negative integer")
        if memory_limit is None:
            memory_limit = self.adaptive_memory_limit
        if isinstance(memory_limit, (bool, np.bool_)) or not isinstance(memory_limit, (int, np.integer)) or memory_limit <= 0:
            raise ValueError("memory_limit must be a positive integer")
        memory_limit = int(memory_limit)
        source_rate = self._coerce_sample_rate(sample_rate)
        analysis_rate = self.sample_rate or source_rate
        self._validate_adaptation_inputs(adaptation_rate, stability_bias, memory_blend, memory_limit)

        (
            frames,
            lpc_coeffs,
            residuals,
            perceptual_vectors,
            flux_values,
            frame_metadata,
        ) = self._extract_features_for_adaptation(audio_signal, sample_rate)

        rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else self._rng)
        self._merge_training_memory(
            frames,
            lpc_coeffs,
            residuals,
            perceptual_vectors,
            flux_values,
            memory_limit=memory_limit,
            memory_blend=memory_blend,
            rng=rng,
        )

        self.sample_rate = analysis_rate
        self._compute_frame_metadata(self.sample_rate)
        if self.spectral_flux is not None and self.spectral_flux.size:
            metadata = self.frame_metadata
            metadata["spectral_flux"] = self.spectral_flux
            self.frame_metadata = metadata

        self._reset_caches()
        self._compute_and_cache_cholesky_factors()

        observations = lpc_coeffs.astype(np.float64, copy=False)
        adaptation_starts = np.array([0], dtype=np.int64)
        gamma, xi, log_likelihood = self._expectation_step(
            observations, sequence_starts=adaptation_starts
        )

        for _ in range(int(n_adaptation_iterations)):
            self._maximization_step(
                observations, gamma, xi, sequence_starts=adaptation_starts
            )
            gamma, xi, log_likelihood = self._expectation_step(
                observations, sequence_starts=adaptation_starts
            )

        self._update_adaptive_statistics(
            observations,
            gamma,
            xi,
            adaptation_rate=adaptation_rate,
            stability_bias=stability_bias,
            sequence_starts=adaptation_starts,
        )
        self._apply_adaptive_statistics(stability_bias=stability_bias)

        occupancy = gamma.mean(axis=0)
        occupancy /= occupancy.sum() + 1e-12
        entropy = float(-np.sum(occupancy * np.log(occupancy + 1e-12)))

        avg_energy = float(np.mean(frame_metadata["energy"])) if "energy" in frame_metadata else 0.0
        avg_rms = float(np.mean(frame_metadata["rms"])) if "rms" in frame_metadata else 0.0
        metrics = {
            "log_likelihood": float(log_likelihood),
            "occupancy_entropy": entropy,
            "adaptation_rate": float(adaptation_rate),
            "memory_limit": float(memory_limit),
            "frames_used": float(observations.shape[0]),
            "avg_energy": avg_energy,
            "avg_rms": avg_rms,
        }
        self._caches.training_history.append(
            {
                "iteration": len(self._caches.training_history) + 1,
                "log_likelihood": metrics["log_likelihood"],
                "occupancy_entropy": metrics["occupancy_entropy"],
                "avg_smoothness_penalty": float("nan"),
                "adaptation_rate": metrics["adaptation_rate"],
                "frames_used": metrics["frames_used"],
            }
        )
        return metrics
    
    def generate(
        self,
        duration_seconds,
        sample_rate=16000,
        fidelity=0.0,
        excitation_type='Karplus-Strong',
        state_sequence=None,
        method='viterbi',
        temperature=1.0,
        rng=None,
        seed=None,
    ):
        """
        Generate new audio

        Args:
            duration_seconds: Duration of generated audio
            sample_rate: Output sample rate
            fidelity: Reconstruction fidelity (0.0-1.0)
                     0.0 = fully synthetic/novel generation
                     1.0 = high fidelity reconstruction of training audio
                     0.5 = balanced mix showing variance
            excitation_type: Type of excitation ('Karplus-Strong' or 'Wavelet')
            state_sequence: Optional precomputed state sequence
            method: Sequence decoding method ('viterbi' or 'stochastic'/'markov')
            temperature: Temperature for stochastic sequence sampling

        Args:
            rng: Optional NumPy random generator for deterministic excitation.
            seed: Optional seed used if rng is not provided.

        Returns:
            Generated audio signal
        """
        print("Phase 2: Inference (Audio Generation)")
        print("=" * 50)

        return self.synthesize_audio(
            duration_seconds,
            sample_rate,
            fidelity=fidelity,
            excitation_type=excitation_type,
            state_sequence=state_sequence,
            method=method,
            temperature=temperature,
            rng=rng,
            seed=seed,
        )


def example_usage():
    """
    Example demonstrating how to use the SSGS system with fidelity control
    """
    # Initialize SSGS with enhanced generative capabilities (34 states)
    ssgs = SpectralStateGuidedSynthesis(
        n_states=34,
        lpc_order=12,
        frame_size=1024,
        hop_size=256
    )

    # Generate a simple training signal (sine wave with harmonics)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create harmonic-rich signal for training
    training_signal = (
        0.5 * np.sin(2 * np.pi * 220 * t) +  # A3 fundamental
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 octave
        0.2 * np.sin(2 * np.pi * 660 * t) +  # E5 fifth
        0.1 * np.sin(2 * np.pi * 880 * t)    # A5 double octave
    )

    # Add some envelope variation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    training_signal *= envelope

    # Add slight noise for realism
    training_signal += 0.01 * np.random.randn(len(training_signal))

    print("Spectral-State Guided Synthesis (SSGS) Example")
    print("=" * 50)

    # Train the model
    ssgs.train(training_signal, sample_rate, n_em_iterations=10)

    # Generate audio with different fidelity levels
    print("\n" + "=" * 50)
    print("Generating audio with different fidelity levels:")
    print("=" * 50)

    # High fidelity (reconstruction - should sound like original)
    print("\n1. High fidelity reconstruction (fidelity=1.0)")
    reconstructed_audio = ssgs.generate(
        duration_seconds=2.0,
        sample_rate=sample_rate,
        fidelity=1.0
    )

    # Medium fidelity (balanced variance)
    print("\n2. Balanced variance (fidelity=0.5)")
    balanced_audio = ssgs.generate(
        duration_seconds=2.0,
        sample_rate=sample_rate,
        fidelity=0.5
    )

    # Low fidelity (fully synthetic/novel)
    print("\n3. Fully synthetic generation (fidelity=0.0)")
    synthetic_audio = ssgs.generate(
        duration_seconds=3.0,
        sample_rate=sample_rate,
        fidelity=0.0
    )

    print("\n" + "=" * 50)
    print("Generation complete!")
    print(f"Reconstructed: {len(reconstructed_audio)} samples ({len(reconstructed_audio)/sample_rate:.2f}s)")
    print(f"Balanced: {len(balanced_audio)} samples ({len(balanced_audio)/sample_rate:.2f}s)")
    print(f"Synthetic: {len(synthetic_audio)} samples ({len(synthetic_audio)/sample_rate:.2f}s)")

    return ssgs, {
        'training': training_signal,
        'reconstructed': reconstructed_audio,
        'balanced': balanced_audio,
        'synthetic': synthetic_audio
    }


if __name__ == "__main__":
    # Run the example
    ssgs, audio_dict = example_usage()
    print("\nTip: Use fidelity parameter to control reconstruction vs. generation:")
    print("  - fidelity=1.0: Direct copy/reconstruction of training audio")
    print("  - fidelity=0.5: Balanced variance (mix of original and novel)")
    print("  - fidelity=0.0: Fully synthetic/novel generation")
