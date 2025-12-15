
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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

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

# Note: Warning filtering has been removed for production safety.
# If specific warnings need suppression, use context managers:
# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', category=SpecificWarning)
#     # your code here


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


class SpectralStateGuidedSynthesis:
    """
    Spectral-State Guided Synthesis (SSGS) Algorithm
    A modular two-stage generative model that decouples high-level musical/phonetic 
    structure from low-level spectral content using HMM and LPC synthesis.
    """
    
    def __init__(
        self,
        n_states=34,
        lpc_order=12,
        frame_size=1024,
        hop_size=256,
        smoothness_weight=0.5,
    ):
        """
        Initialize SSGS parameters
        
        Args:
            n_states: Number of HMM states (default: 34 for enhanced generative capabilities)
            lpc_order: Order of Linear Prediction coefficients
            frame_size: Size of analysis frames
            hop_size: Hop size between frames
        """
        self.n_states = n_states
        self.lpc_order = lpc_order
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.smoothness_weight = smoothness_weight
        
        # Internal state containers keep responsibilities isolated yet accessible.
        self._params = _ModelParameters()
        self._artifacts = _TrainingArtifacts()
        self._caches = _ComputationCaches()
        self._feature_indexes: Dict[Tuple[str, bool], FeatureIndex] = {}
        self._max_preview_snapshots = 6

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

    @property
    def initial_probabilities(self):
        return self._params.initial_probabilities

    @initial_probabilities.setter
    def initial_probabilities(self, probabilities):
        self._params.initial_probabilities = probabilities

    @property
    def state_means(self):
        return self._params.state_means

    @state_means.setter
    def state_means(self, means):
        self._params.state_means = means

    @property
    def state_covariances(self):
        return self._params.state_covariances

    @state_covariances.setter
    def state_covariances(self, covariances):
        self._params.state_covariances = covariances

    @property
    def training_frames(self):
        return self._artifacts.training_frames

    @training_frames.setter
    def training_frames(self, frames):
        self._artifacts.training_frames = frames

    @property
    def lpc_coefficients(self):
        return self._artifacts.lpc_coefficients

    @lpc_coefficients.setter
    def lpc_coefficients(self, coeffs):
        self._artifacts.lpc_coefficients = coeffs

    @property
    def residual_signals(self):
        return self._artifacts.residual_signals

    @residual_signals.setter
    def residual_signals(self, residuals):
        self._artifacts.residual_signals = residuals

    @property
    def perceptual_features(self):
        return self._artifacts.perceptual_features

    @perceptual_features.setter
    def perceptual_features(self, features):
        self._artifacts.perceptual_features = features

    @property
    def spectral_flux(self):
        return self._artifacts.spectral_flux

    @spectral_flux.setter
    def spectral_flux(self, values):
        self._artifacts.spectral_flux = values

    @property
    def frame_metadata(self):
        return self._artifacts.frame_metadata

    @frame_metadata.setter
    def frame_metadata(self, metadata):
        if metadata is None:
            self._artifacts.frame_metadata = {}
        else:
            self._artifacts.frame_metadata = metadata

    @property
    def sample_rate(self):
        return self._artifacts.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value is None:
            self._artifacts.sample_rate = None
        else:
            self._artifacts.sample_rate = int(value)

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

    def _reset_caches(self):
        self._caches.reset()
        self._reset_feature_indexes()

    def _reset_feature_indexes(self):
        """Drop any cached nearest-neighbor structures tied to frame features."""
        self._feature_indexes = {}

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
        """Ensure per-state Cholesky factors and log-determinants are cached."""
        if self.state_covariances is None:
            return

        caches = self._caches
        if caches.chol_factors is not None and len(caches.chol_factors) == self.n_states:
            return

        chol_cache: List[np.ndarray] = []
        log_det_cache = np.empty(self.n_states, dtype=np.float64)

        for state in range(self.n_states):
            cov = self.state_covariances[state]
            try:
                chol = cholesky(cov, lower=True, overwrite_a=False, check_finite=False)
            except LinAlgError:
                cov = cov + np.eye(self.lpc_order) * 1e-4
                self.state_covariances[state] = cov
                chol = cholesky(cov, lower=True, overwrite_a=False, check_finite=False)

            chol_cache.append(chol)
            log_det_cache[state] = 2.0 * np.sum(np.log(np.diag(chol)))

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
            metadata["lpc_norm"] = np.linalg.norm(self.lpc_coefficients, axis=1).astype(
                np.float32, copy=False
            )

        if (
            self.residual_signals is not None
            and len(self.residual_signals) == len(frames)
            and self.residual_signals.shape[1] > 0
        ):
            residual_energy = np.sum(self.residual_signals * self.residual_signals, axis=1)
            residual_rms = np.sqrt(residual_energy / self.residual_signals.shape[1])
            metadata["residual_rms"] = residual_rms.astype(np.float32, copy=False)

        if sample_rate:
            window = np.hanning(frame_len)
            windowed = frames * window
            spectrum = np.abs(np.fft.rfft(windowed, axis=1))
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sample_rate))
            power = spectrum + 1e-12

            centroid = (power * freqs).sum(axis=1) / power.sum(axis=1)
            metadata["spectral_centroid"] = centroid.astype(np.float32, copy=False)

            bandwidth = np.sqrt(
                ((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / power.sum(axis=1)
            )
            metadata["spectral_bandwidth"] = bandwidth.astype(np.float32, copy=False)

            dominant_idx = np.argmax(power, axis=1)
            metadata["dominant_frequency"] = freqs[dominant_idx].astype(np.float32, copy=False)

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
                self.frame_metadata[key][:, None].astype(np.float64, copy=False)
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
            parts: Sequence[np.ndarray] = [np.asarray(vector["lpc"], dtype=np.float64)]
            for key in index.metadata_keys:
                if key not in vector:
                    raise ValueError(f"Vector dictionary missing required key '{key}'")
                parts.append(np.atleast_1d(np.asarray(vector[key], dtype=np.float64)))
            arr = np.concatenate(parts, axis=0)
        else:
            arr = np.asarray(vector, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError("Feature query vector must be 1-D")

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
        if k < 1:
            raise ValueError("k must be >= 1")

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
        if k < 1:
            raise ValueError("k must be >= 1")

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
        if k < 1:
            raise ValueError("k must be >= 1")

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

    def _frame_signal(self, audio_signal):
        """
        Convert audio signal into overlapping frames using stride tricks.
        """
        audio_signal = np.asarray(audio_signal, dtype=np.float32)
        if audio_signal.ndim != 1:
            raise ValueError("audio_signal must be 1-D")

        if len(audio_signal) < self.frame_size:
            pad_width = self.frame_size - len(audio_signal)
            audio_signal = np.pad(audio_signal, (0, pad_width), mode="constant")

        remainder = len(audio_signal) % self.hop_size
        if remainder:
            pad_width = self.hop_size - remainder
            audio_signal = np.pad(audio_signal, (0, pad_width), mode="constant")

        total_frames = 1 + (len(audio_signal) - self.frame_size) // self.hop_size
        shape = (total_frames, self.frame_size)
        strides = (audio_signal.strides[0] * self.hop_size, audio_signal.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio_signal, shape=shape, strides=strides)
        return frames.copy()

    def _autocorrelation(self, frame):
        """Compute autocorrelation sequence needed for LPC."""
        corr = np.correlate(frame, frame, mode="full")
        mid = len(frame) - 1
        return corr[mid : mid + self.lpc_order + 1]

    def _compute_lpc(self, frame):
        """
        Compute LPC coefficients using the autocorrelation / Toeplitz approach.
        """
        autocorr = self._autocorrelation(frame)
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
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if fmax is None:
            fmax = sample_rate / 2.0
        if fmax <= fmin:
            raise ValueError("fmax must be greater than fmin for mel filter bank")

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
        row_sums[row_sums == 0.0] = 1.0
        filters /= row_sums
        return filters

    def _erb_filter_bank(self, sample_rate, n_fft, n_bands=16, fmin=30.0, fmax=None):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if fmax is None:
            fmax = sample_rate / 2.0
        if fmax <= fmin:
            raise ValueError("fmax must be greater than fmin for ERB filter bank")

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
        row_sums[row_sums == 0.0] = 1.0
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

        self._compute_frame_metadata(self.sample_rate)
        if self.spectral_flux is not None and self.spectral_flux.size:
            metadata = self.frame_metadata
            metadata["spectral_flux"] = self.spectral_flux
            self.frame_metadata = metadata

        self._reset_caches()
        self._compute_and_cache_cholesky_factors()
        return self.lpc_coefficients, self.residual_signals
    
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
        if self.sample_rate is None or self.sample_rate <= 0:
            return

        mel_filters = self._mel_filter_bank(self.sample_rate, self.frame_size)
        erb_filters = self._erb_filter_bank(self.sample_rate, self.frame_size)
        window = np.hanning(self.frame_size).astype(np.float32)

        augmented_frames = [self.training_frames]
        augmented_lpc = [self.lpc_coefficients]
        augmented_residuals = [self.residual_signals]

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

    def initialize_hmm_parameters(self):
        """
        Phase 1, Step 2: State Initialization using Clustering
        
        Uses k-means clustering to initialize HMM parameters from LPC coefficients
        """
        if self.lpc_coefficients is None:
            raise ValueError("Must extract features first using extract_features()")

        self._apply_residual_augmentation()

        num_frames = self.lpc_coefficients.shape[0]
        if num_frames < self.n_states:
            warnings.warn(
                "Fewer frames than states; reducing state count for initialization",
                RuntimeWarning,
            )
            self.n_states = num_frames

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

        centroids, labels = vq.kmeans2(normalized_features, self.n_states, minit="++", iter=40)
        if isinstance(labels, tuple):  # Older scipy returns (labels, dist)
            labels = labels[0]

        self.initial_probabilities = np.zeros(self.n_states, dtype=np.float64)
        self.transition_matrix = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        self.state_means = np.zeros((self.n_states, self.lpc_order), dtype=np.float64)
        self.state_covariances = np.array(
            [np.eye(self.lpc_order, dtype=np.float64) * 0.05 for _ in range(self.n_states)]
        )

        # Recover means in original scale
        for state in range(self.n_states):
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

        for idx, state in enumerate(labels):
            if idx == 0:
                self.initial_probabilities[state] += 1.0
            else:
                prev_state = labels[idx - 1]
                self.transition_matrix[prev_state, state] += 1.0

        self.initial_probabilities += 1e-3
        self.initial_probabilities /= self.initial_probabilities.sum()

        self.transition_matrix += 1e-6
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix /= row_sums

        self._reset_caches()
        self._caches.training_history.clear()
        self._compute_and_cache_cholesky_factors()
        self._transition_cost_matrix()
    
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
        combined_max = combined.max()
        if combined_max > 0.0:
            combined /= combined_max
        self._caches.psychoacoustic_weight = combined
        return combined

    def _expectation_step(self, observations):
        """
        E-step: Compute posterior probabilities of state sequences
        
        Args:
            observations: LPC coefficient sequences
            
        Returns:
            gamma: State posterior probabilities
            xi: Transition posterior probabilities
        """
        n_frames = observations.shape[0]
        log_emissions = self._compute_emission_log_probs(observations)

        log_init = np.log(self.initial_probabilities + 1e-12)
        log_trans = np.log(self.transition_matrix + 1e-12)

        alpha = np.empty((n_frames, self.n_states), dtype=np.float64)
        beta = np.empty((n_frames, self.n_states), dtype=np.float64)

        alpha[0] = log_init + log_emissions[0]
        for t in range(1, n_frames):
            alpha[t] = logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
            alpha[t] += log_emissions[t]

        beta[-1] = 0.0
        for t in range(n_frames - 2, -1, -1):
            temp = log_trans + log_emissions[t + 1] + beta[t + 1]
            beta[t] = logsumexp(temp, axis=1)

        log_gamma = alpha + beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        log_xi = (
            alpha[:-1, :, None]
            + log_trans[np.newaxis, :, :]
            + log_emissions[1:, None, :]
            + beta[1:, None, :]
        )
        log_xi -= logsumexp(log_xi.reshape(n_frames - 1, -1), axis=1)[:, None, None]
        xi = np.exp(log_xi)

        log_likelihood = logsumexp(alpha[-1])
        return gamma, xi, float(log_likelihood)
    
    def _compute_emission_log_probs(self, observations):
        """Vectorized Gaussian log-likelihoods for all states."""
        n_frames = observations.shape[0]
        self._compute_and_cache_cholesky_factors()
        caches = self._caches
        log_probs = np.empty((n_frames, self.n_states), dtype=np.float64)
        gaussian_const = self.lpc_order * np.log(2.0 * np.pi)

        obs_t = np.asarray(observations).T
        if obs_t.shape != (self.lpc_order, n_frames):
            raise ValueError("Unexpected observation matrix shape during emission computation")

        if (
            caches.emission_observations is None
            or caches.emission_observations.shape[1] != n_frames
        ):
            caches.emission_observations = np.empty((self.lpc_order, n_frames), dtype=np.float64)

        obs_view = caches.emission_observations
        np.copyto(obs_view, obs_t, casting="safe")

        if (
            caches.emission_workspace is None
            or caches.emission_workspace.shape[1] != n_frames
        ):
            caches.emission_workspace = np.empty((self.lpc_order, n_frames), dtype=np.float64)

        workspace = caches.emission_workspace

        for state in range(self.n_states):
            mean = self.state_means[state]
            chol = caches.chol_factors[state]
            # Avoid creating large temporaries by reusing the shared workspace buffer.
            np.subtract(obs_view, mean[:, None], out=workspace)
            solved = solve_triangular(
                chol,
                workspace,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            quad = np.sum(solved * solved, axis=0)
            log_probs[:, state] = -0.5 * (
                quad + gaussian_const + caches.log_determinants[state]
            )

        return log_probs
    
    def _maximization_step(self, observations, gamma, xi):
        """
        M-step: Update HMM parameters
        
        Args:
            observations: LPC coefficient sequences
            gamma: State posterior probabilities
            xi: Transition posterior probabilities
        """
        n_frames = observations.shape[0]

        self.initial_probabilities = gamma[0] + 1e-6
        self.initial_probabilities /= self.initial_probabilities.sum()

        xi_sum = np.sum(xi, axis=0)
        gamma_sum = np.sum(gamma[:-1], axis=0)[:, None]
        self.transition_matrix = xi_sum / (gamma_sum + 1e-12)
        self.transition_matrix = np.clip(self.transition_matrix, 1e-9, None)
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        for state in range(self.n_states):
            weights = gamma[:, state]
            weight_sum = weights.sum()
            if weight_sum <= 1e-9:
                continue

            mean = np.dot(weights, observations) / weight_sum
            diff = observations - mean
            cov = (weights[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
            cov /= weight_sum
            cov += np.eye(self.lpc_order) * 1e-5
            self.state_means[state] = mean
            self.state_covariances[state] = cov

        self._reset_caches()
    
    def iterative_refinement(
        self,
        n_iterations=10,
        *,
        preview_every: int = 0,
        preview_duration_frames: int = 32,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Phase 1, Step 3: Iterative Learning using EM Algorithm
        
        Args:
            n_iterations: Number of EM iterations
            preview_every: Generate a preview every N iterations (0 disables)
            preview_duration_frames: Frame count for each preview render
            preview_callback: Optional callable receiving preview metadata
        """
        if self.lpc_coefficients is None:
            raise ValueError("Must initialize HMM parameters first")
        
        if preview_every < 0:
            raise ValueError("preview_every must be >= 0")
        if preview_duration_frames <= 0:
            raise ValueError("preview_duration_frames must be > 0")

        print(f"Starting EM training with {n_iterations} iterations...")
        self._caches.training_history.clear()
        
        for iteration in range(n_iterations):
            gamma, xi, log_likelihood = self._expectation_step(self.lpc_coefficients)

            occupancy = gamma.mean(axis=0)
            occupancy = occupancy / (occupancy.sum() + 1e-12)
            entropy = float(-np.sum(occupancy * np.log(occupancy + 1e-12)))

            smooth_matrix = self._spectral_smoothness_matrix()
            smooth_penalty = float(
                np.sum(xi * smooth_matrix[None, :, :]) / (xi.shape[0] + 1e-12)
            )

            history_entry = {
                "iteration": iteration + 1,
                "log_likelihood": float(log_likelihood),
                "occupancy_entropy": entropy,
                "avg_smoothness_penalty": smooth_penalty,
            }
            self._caches.training_history.append(history_entry)

            self._maximization_step(self.lpc_coefficients, gamma, xi)
            print(
                "Iteration {idx}: Log-likelihood = {ll:.2f} | Entropy = {ent:.3f} | "
                "Smoothness = {smooth:.3f}".format(
                    idx=iteration + 1,
                    ll=log_likelihood,
                    ent=entropy,
                    smooth=smooth_penalty,
                )
            )

            preview_snapshot = None
            if preview_every and ((iteration + 1) % preview_every == 0):
                preview_frames = int(preview_duration_frames)
                preview_sr_arg = int(self.sample_rate or 16000)
                preview_audio, preview_sr = self.render_preview(
                    n_frames=preview_frames,
                    sample_rate=preview_sr_arg,
                )
                preview_snapshot = self._record_preview_snapshot(
                    iteration=iteration + 1,
                    frames=preview_frames,
                    audio=preview_audio,
                    sample_rate=preview_sr,
                )
                history_entry["preview_index"] = len(self.preview_snapshots) - 1
                message = (
                    f"  Preview stored ({preview_frames} frames @ {preview_sr} Hz)"
                )
                print(message)
                if preview_callback is not None:
                    try:
                        preview_callback(preview_snapshot)
                    except Exception as exc:  # pragma: no cover - defensive log
                        warnings.warn(
                            f"Preview callback raised an exception: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
    
    def identify_graph_constraints(self):
        """
        Phase 1, Step 4: Graph Component Identification
        
        Identifies structural degeneracies in the HMM transition matrix
        """
        adjacency = (self.transition_matrix > 1e-6).astype(np.float64)
        graph = csr_matrix(adjacency)
        n_components, _ = _strong_components(graph)

        for state in range(self.n_states):
            row = self.transition_matrix[state]
            self_loop = row[state]
            outgoing = row.sum() - self_loop
            incoming = self.transition_matrix[:, state].sum() - self_loop

            if outgoing < 1e-3 or incoming < 1e-3:
                row.fill(1.0 / self.n_states)
                row[state] = min(self_loop, 0.85)
                remainder = 1.0 - row[state]
                if self.n_states > 1:
                    row[:] = remainder / (self.n_states - 1)
                    row[state] = 1.0 - remainder
            row[:] = np.clip(row, 1e-6, None)
            row /= row.sum()

        self._reset_caches()
        self._transition_cost_matrix()
        print(f"Identified {n_components} strongly connected components")
    
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
        if self.transition_matrix is None:
            raise ValueError("Transition matrix is not initialized")

        if self._caches.transition_cost is None:
            transition_cost = -np.log(self.transition_matrix + 1e-12)
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
        Phase 2, Step 5: State sequence decoding using a smoothness-aware DP
        
        Args:
            target_duration_frames: Target number of frames
            max_expanded: Maximum number of nodes to expand
            
        Returns:
            optimal_state_sequence: Most probable state sequence
        """
        return self._decode_state_sequence(target_duration_frames)

    def _decode_state_sequence(self, target_frames):
        """Deterministic dynamic-programming decode with smoothness penalty."""
        transition_cost = self._transition_cost_matrix()
        initial_cost = -np.log(self.initial_probabilities + 1e-12)

        dp = np.empty((target_frames, self.n_states), dtype=np.float64)
        backpointer = np.zeros((target_frames, self.n_states), dtype=np.int32)

        dp[0] = initial_cost
        for t in range(1, target_frames):
            costs = dp[t - 1][:, None] + transition_cost
            backpointer[t] = np.argmin(costs, axis=0)
            dp[t] = np.min(costs, axis=0)

        end_state = int(np.argmin(dp[-1]))
        path = [0] * target_frames
        path[-1] = end_state

        for t in range(target_frames - 1, 0, -1):
            end_state = backpointer[t, end_state]
            path[t - 1] = end_state

        return path

    def _state_spectral_prototypes(self):
        if self._caches.spectral_prototypes is not None:
            return self._caches.spectral_prototypes
        if self.state_means is None:
            raise ValueError("State means are not initialized")

        residuals = (
            self.residual_signals
            if self.residual_signals is not None and self.residual_signals.size
            else None
        )
        assignments = self._estimate_state_assignments()
        rng = np.random.default_rng()
        prototypes: List[Tuple[np.ndarray, np.ndarray]] = []

        for state in range(self.n_states):
            if (
                residuals is not None
                and assignments is not None
                and residuals.shape[0] == assignments.shape[0]
            ):
                mask = assignments == state
                if np.any(mask):
                    residual_template = residuals[mask].mean(axis=0)
                else:
                    residual_template = rng.standard_normal(self.hop_size).astype(np.float32) * 0.05
            else:
                residual_template = rng.standard_normal(self.hop_size).astype(np.float32) * 0.05

            residual_template = residual_template.astype(np.float32, copy=False)
            if residual_template.shape[0] < self.hop_size:
                residual_template = np.pad(
                    residual_template,
                    (0, self.hop_size - residual_template.shape[0]),
                    mode="constant",
                )
            elif residual_template.shape[0] > self.hop_size:
                residual_template = residual_template[: self.hop_size]

            prototypes.append(
                (
                    self.state_means[state].astype(np.float32, copy=False),
                    residual_template,
                )
            )

        self._caches.spectral_prototypes = prototypes
        return prototypes
    
    def _karplus_strong_excitation(self, state, duration_samples):
        """
        Phase 2, Step 6: Excitation Generation using Delay-Line Feedback
        
        Args:
            state: HMM state for parameterization
            duration_samples: Number of samples to generate
            
        Returns:
            excitation_signal: Karplus-Strong excitation signal
        """
        mean_coeffs = self.state_means[state]
        spectral_centroid = np.mean(np.abs(mean_coeffs))

        base_delay = int(20 + 80 * (1 - np.tanh(spectral_centroid)))
        delay_line = max(2, base_delay + np.random.randint(-5, 6))

        excitation = np.zeros(duration_samples, dtype=np.float32)
        noise_burst = np.random.randn(delay_line).astype(np.float32) * 0.5

        damping = 0.98 - 0.1 * spectral_centroid
        damping = float(np.clip(damping, 0.95, 0.99))

        if delay_line >= duration_samples:
            excitation[:duration_samples] = noise_burst[:duration_samples]
        else:
            excitation[:delay_line] = noise_burst
            for i in range(delay_line, duration_samples):
                excitation[i] = damping * 0.5 * (
                    excitation[i - delay_line] + excitation[i - delay_line + 1]
                )

        envelope = np.exp(-np.linspace(0, 5, duration_samples)).astype(np.float32)
        excitation *= envelope
        return excitation
    
    def _lpc_synthesis_filter(self, excitation, lpc_coeffs):
        """
        Phase 2, Step 7: Audio Synthesis using LPC filtering

        Args:
            excitation: Excitation signal
            lpc_coeffs: LPC coefficients

        Returns:
            synthesized_audio: Filtered audio signal
        """
        filter_coeffs = np.concatenate([[1.0], lpc_coeffs])
        try:
            synthesized = signal.lfilter([1.0], filter_coeffs, excitation).astype(np.float32)
        except (ValueError, LinAlgError, RuntimeError) as e:
            logger.warning(f"LPC filter failed ({type(e).__name__}: {e}), falling back to convolution")
            synthesized = np.convolve(excitation, filter_coeffs[::-1], mode="same").astype(np.float32)

        max_val = np.max(np.abs(synthesized))
        if max_val > 0:
            synthesized = synthesized / max_val * 0.8
        return synthesized

    def _apply_gain_floor(
        self,
        audio: np.ndarray,
        *,
        minimum_peak: float = 0.75,
        target_peak: float = 0.9,
    ) -> np.ndarray:
        """
        Normalize audio while ensuring a minimum peak level for audibility.
        """
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return arr

        peak = float(np.max(np.abs(arr)))
        if peak <= 0.0:
            return arr

        floor = float(minimum_peak)
        if floor <= 0.0:
            raise ValueError("minimum_peak must be positive")

        target = float(target_peak)
        if target <= 0.0:
            raise ValueError("target_peak must be positive")
        if target < floor:
            raise ValueError("target_peak must be greater than or equal to minimum_peak")

        if peak <= floor:
            target_level = floor
        elif peak > target:
            target_level = target
        else:
            return arr

        scaled = arr * (target_level / peak)
        return scaled.astype(np.float32, copy=False)

    def _get_training_residual_for_state(self, state, rng=None):
        """
        Get a representative residual from training data for a given state.

        Args:
            state: HMM state index
            rng: Random number generator (optional)

        Returns:
            residual: Training residual signal for this state
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.residual_signals is None or self.residual_signals.size == 0:
            # No residuals available, return noise
            return rng.standard_normal(self.hop_size).astype(np.float32) * 0.05

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
        if residual.shape[0] < self.hop_size:
            residual = np.pad(
                residual,
                (0, self.hop_size - residual.shape[0]),
                mode="constant",
            )
        elif residual.shape[0] > self.hop_size:
            residual = residual[: self.hop_size]

        return residual.astype(np.float32, copy=True)

    def render_preview(self, n_frames=32, sample_rate=None, state_sequence=None):
        """Generate a lightweight audio preview using cached prototypes.

        Returns a tuple ``(audio, sample_rate)`` for quick auditioning.
        """
        if self.transition_matrix is None or self.initial_probabilities is None:
            raise ValueError("Model must be trained before rendering a preview")
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")

        sample_rate = int(sample_rate or (self.sample_rate or 16000))
        prototypes = self._state_spectral_prototypes()

        required_samples = n_frames * self.hop_size
        workspace = self._caches.emission_workspace
        min_cols = max(required_samples // max(self.lpc_order, 1) + 1, 1)
        if workspace is None or workspace.size < required_samples:
            workspace = np.empty((self.lpc_order, min_cols), dtype=np.float64)
            self._caches.emission_workspace = workspace
        elif workspace.shape[1] < min_cols:
            workspace = np.empty((self.lpc_order, min_cols), dtype=np.float64)
            self._caches.emission_workspace = workspace

        preview_buffer = workspace.ravel()[:required_samples]
        preview_buffer.fill(0.0)

        if state_sequence is not None:
            sequence = list(state_sequence[:n_frames])
            if not sequence:
                raise ValueError("state_sequence must contain at least one element")
            if len(sequence) < n_frames:
                sequence.extend([sequence[-1]] * (n_frames - len(sequence)))
        else:
            rng = np.random.default_rng()
            init_probs = self.initial_probabilities / self.initial_probabilities.sum()
            current_state = int(rng.choice(self.n_states, p=init_probs))
            sequence = [current_state]
            for _ in range(1, n_frames):
                probs = self.transition_matrix[current_state]
                probs = probs / probs.sum()
                current_state = int(rng.choice(self.n_states, p=probs))
                sequence.append(current_state)

        for idx, state in enumerate(sequence):
            lpc_coeffs, residual_template = prototypes[state]
            if residual_template.shape[0] < self.hop_size:
                excitation = np.pad(
                    residual_template,
                    (0, self.hop_size - residual_template.shape[0]),
                    mode="constant",
                )
            else:
                excitation = residual_template[: self.hop_size]
            frame_audio = self._lpc_synthesis_filter(excitation, lpc_coeffs)
            start = idx * self.hop_size
            end = start + self.hop_size
            preview_buffer[start:end] += frame_audio[: self.hop_size]

        if required_samples > 1:
            preview_buffer[:required_samples] *= np.hanning(required_samples)

        preview = np.tanh(preview_buffer[:required_samples])
        preview = self._apply_gain_floor(preview, minimum_peak=0.75, target_peak=0.9)
        return preview.astype(np.float32), sample_rate
    
    def synthesize_audio(self, target_duration_seconds, sample_rate=16000, fidelity=0.0):
        """
        Phase 2: Complete Audio Synthesis Pipeline

        Args:
            target_duration_seconds: Target duration in seconds
            sample_rate: Output sample rate
            fidelity: Reconstruction fidelity (0.0-1.0)
                     0.0 = fully synthetic (Karplus-Strong excitation)
                     1.0 = high fidelity reconstruction (training residuals)
                     Values in between blend both approaches

        Returns:
            audio_output: Generated audio signal
        """
        if self.transition_matrix is None:
            raise ValueError("Must train the model first using extract_features() and iterative_refinement()")

        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be between 0.0 and 1.0")

        target_frames = int(target_duration_seconds * sample_rate / self.hop_size)
        print(f"Synthesizing {target_duration_seconds}s of audio ({target_frames} frames)")
        print(f"Fidelity: {fidelity:.2f} (0.0=synthetic, 1.0=reconstruction)")

        # Step 5: Decode optimal state sequence using A* search
        print("Step 5: Decoding state sequence with A* search...")
        state_sequence = self._a_star_search(target_frames)

        audio_output = np.zeros(target_frames * self.hop_size, dtype=np.float32)
        rng = np.random.default_rng()

        print("Step 6-8: Generating excitation, synthesizing audio...")
        for idx, state in enumerate(state_sequence):
            if fidelity >= 1.0:
                # Pure reconstruction mode: use training residuals only
                excitation = self._get_training_residual_for_state(state, rng)
            elif fidelity <= 0.0:
                # Pure synthetic mode: use Karplus-Strong only
                excitation = self._karplus_strong_excitation(state, self.hop_size)
            else:
                # Blend mode: mix both approaches
                residual_excitation = self._get_training_residual_for_state(state, rng)
                synthetic_excitation = self._karplus_strong_excitation(state, self.hop_size)
                # Linear blend based on fidelity parameter
                excitation = (fidelity * residual_excitation +
                            (1.0 - fidelity) * synthetic_excitation)

            frame_audio = self._lpc_synthesis_filter(excitation, self.state_means[state])
            start = idx * self.hop_size
            end = start + self.hop_size
            audio_output[start:end] += frame_audio[: self.hop_size]

        window = np.hanning(len(audio_output))
        audio_output *= window

        audio_output = self._apply_gain_floor(audio_output, minimum_peak=0.75, target_peak=0.9)
        return audio_output.astype(np.float32)
    
    def export_model(
        self,
        filepath,
        *,
        use_compression=True,
        precision=np.float32,
        pack_covariances=True,
        include_training_artifacts=False,
    ):
        """Serialize trained model parameters to disk with optional compression."""
        if self.transition_matrix is None:
            raise ValueError("Model must be trained before exporting")

        target_dtype = np.dtype(precision)
        if target_dtype.kind != "f":
            raise ValueError("precision must be a floating-point dtype")

        filepath = Path(filepath)
        if filepath.suffix != ".npz":
            filepath = filepath.with_suffix(".npz")

        metadata = {
            "version": 1,
            "n_states": int(self.n_states),
            "lpc_order": int(self.lpc_order),
            "frame_size": int(self.frame_size),
            "hop_size": int(self.hop_size),
            "smoothness_weight": float(self.smoothness_weight),
            "pack_covariances": bool(pack_covariances),
            "precision": target_dtype.name,
            "includes_training_artifacts": bool(include_training_artifacts),
            "sample_rate": int(self.sample_rate) if self.sample_rate is not None else None,
        }

        payload = {
            "transition_matrix": self.transition_matrix.astype(target_dtype, copy=False),
            "initial_probabilities": self.initial_probabilities.astype(target_dtype, copy=False),
            "state_means": self.state_means.astype(target_dtype, copy=False),
        }

        if pack_covariances:
            tri_rows, tri_cols = np.tril_indices(self.lpc_order)
            packed = self.state_covariances[:, tri_rows, tri_cols]
            payload["state_covariances_packed"] = packed.astype(target_dtype, copy=False)
        else:
            payload["state_covariances"] = self.state_covariances.astype(target_dtype, copy=False)

        if include_training_artifacts and self.training_frames is not None:
            payload["training_frames"] = self.training_frames.astype(target_dtype, copy=False)
            if self.lpc_coefficients is not None:
                payload["lpc_coefficients"] = self.lpc_coefficients.astype(target_dtype, copy=False)
            if self.residual_signals is not None:
                payload["residual_signals"] = self.residual_signals.astype(target_dtype, copy=False)
            if self.perceptual_features is not None:
                payload["perceptual_features"] = self.perceptual_features.astype(
                    target_dtype,
                    copy=False,
                )
            if self.spectral_flux is not None:
                payload["spectral_flux"] = self.spectral_flux.astype(target_dtype, copy=False)
            if self.frame_metadata:
                metadata["frame_metadata_keys"] = list(self.frame_metadata.keys())
                for key, values in self.frame_metadata.items():
                    arr = np.asarray(values)
                    if arr.dtype.kind == "f":
                        arr = arr.astype(target_dtype, copy=False)
                    payload[f"frame_metadata__{key}"] = arr

        payload["metadata"] = np.array(json.dumps(metadata), dtype=np.str_)

        saver = np.savez_compressed if use_compression else np.savez
        saver(filepath, **payload)

    @classmethod
    def load_model(cls, filepath):
        """Load a serialized model produced by ``export_model``."""
        filepath = Path(filepath)
        with np.load(filepath, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())

            model = cls(
                n_states=metadata["n_states"],
                lpc_order=metadata["lpc_order"],
                frame_size=metadata["frame_size"],
                hop_size=metadata["hop_size"],
                smoothness_weight=metadata.get("smoothness_weight", 0.5),
            )

            transition = data["transition_matrix"].astype(np.float64, copy=False)
            transition = np.clip(transition, 1e-12, None)
            transition /= transition.sum(axis=1, keepdims=True)
            model.transition_matrix = transition

            init_probs = data["initial_probabilities"].astype(np.float64, copy=False)
            init_probs = np.clip(init_probs, 1e-12, None)
            model.initial_probabilities = init_probs / init_probs.sum()
            model.state_means = data["state_means"].astype(np.float64, copy=False)

            if metadata.get("pack_covariances", False):
                tri_rows, tri_cols = np.tril_indices(model.lpc_order)
                packed = data["state_covariances_packed"].astype(np.float64, copy=False)
                full_cov = np.zeros(
                    (model.n_states, model.lpc_order, model.lpc_order),
                    dtype=np.float64,
                )
                full_cov[:, tri_rows, tri_cols] = packed
                full_cov[:, tri_cols, tri_rows] = packed
                model.state_covariances = full_cov
            else:
                model.state_covariances = data["state_covariances"].astype(np.float64, copy=False)

            if "training_frames" in data:
                model.training_frames = data["training_frames"].astype(np.float32, copy=False)
            if "lpc_coefficients" in data:
                model.lpc_coefficients = data["lpc_coefficients"].astype(np.float32, copy=False)
            if "residual_signals" in data:
                model.residual_signals = data["residual_signals"].astype(np.float32, copy=False)
            if "perceptual_features" in data:
                model.perceptual_features = data["perceptual_features"].astype(np.float32, copy=False)
            if "spectral_flux" in data:
                model.spectral_flux = data["spectral_flux"].astype(np.float32, copy=False)

            meta_keys = metadata.get("frame_metadata_keys", [])
            if meta_keys:
                frame_meta: Dict[str, np.ndarray] = {}
                for key in meta_keys:
                    frame_meta[key] = data[f"frame_metadata__{key}"].copy()
                model.frame_metadata = frame_meta
            else:
                model.frame_metadata = {}

            model.sample_rate = metadata.get("sample_rate")

        model._reset_caches()
        model._compute_and_cache_cholesky_factors()
        model._transition_cost_matrix()
        return model

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
            audio_signal: Training audio signal
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
    
    def generate(self, duration_seconds, sample_rate=16000, fidelity=0.0):
        """
        Generate new audio

        Args:
            duration_seconds: Duration of generated audio
            sample_rate: Output sample rate
            fidelity: Reconstruction fidelity (0.0-1.0)
                     0.0 = fully synthetic/novel generation
                     1.0 = high fidelity reconstruction of training audio
                     0.5 = balanced mix showing variance

        Returns:
            Generated audio signal
        """
        print("Phase 2: Inference (Audio Generation)")
        print("=" * 50)

        return self.synthesize_audio(duration_seconds, sample_rate, fidelity=fidelity)


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
