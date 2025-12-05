
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

import numpy as np
import scipy
from scipy import signal
from scipy.cluster import vq
from scipy.fftpack import dct, idct
from scipy.stats import multivariate_normal
import heapq
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SpectralStateGuidedSynthesis:
    """
    Spectral-State Guided Synthesis (SSGS) Algorithm
    A modular two-stage generative model that decouples high-level musical/phonetic 
    structure from low-level spectral content using HMM and LPC synthesis.
    """
    
    def __init__(self, n_states=16, lpc_order=12, frame_size=1024, hop_size=256):
        """
        Initialize SSGS parameters
        
        Args:
            n_states: Number of HMM states
            lpc_order: Order of Linear Prediction coefficients
            frame_size: Size of analysis frames
            hop_size: Hop size between frames
        """
        self.n_states = n_states
        self.lpc_order = lpc_order
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # HMM Parameters
        self.transition_matrix = None
        self.initial_probabilities = None
        self.state_means = None
        self.state_covariances = None
        
        # Training artifacts
        self.training_frames = None
        self.lpc_coefficients = None
        self.residual_signals = None
        
    def _analyze_frame(self, frame):
        """
        Analyze a single frame using Linear Prediction
        
        Args:
            frame: Audio frame samples
            
        Returns:
            lpc_coeffs: LPC coefficients
            residual: Residual error signal
        """
        # Apply window to reduce spectral leakage
        windowed = frame * np.hanning(len(frame))
        
        # Compute LPC coefficients using autocorrelation method
        try:
            lpc_coeffs, _ = scipy.signal.lfilter([1], [1], windowed)
            lpc_coeffs = scipy.signal.lpc(windowed, self.lpc_order)
        except:
            # Fallback to simple coefficients if LPC fails
            lpc_coeffs = np.zeros(self.lpc_order + 1)
            lpc_coeffs[0] = 1.0
            
        # Generate residual by filtering with LPC coefficients
        try:
            residual = scipy.signal.lfilter(lpc_coeffs, [1], windowed)
        except:
            residual = windowed
            
        return lpc_coeffs, residual
    
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
        frames = []
        lpc_coeffs_list = []
        residual_list = []
        
        # Pad signal to ensure complete frames
        pad_length = self.frame_size - (len(audio_signal) % self.frame_size)
        if pad_length != self.frame_size:
            audio_signal = np.pad(audio_signal, (0, pad_length), mode='constant')
        
        # Extract overlapping frames
        for i in range(0, len(audio_signal) - self.frame_size + 1, self.hop_size):
            frame = audio_signal[i:i + self.frame_size]
            
            # Skip frames with very low energy
            if np.sum(frame ** 2) < 1e-10:
                continue
                
            lpc_coeffs, residual = self._analyze_frame(frame)
            
            frames.append(frame)
            lpc_coeffs_list.append(lpc_coeffs[1:])  # Exclude the leading 1
            residual_list.append(residual[:self.hop_size])  # Store only hop size
        
        self.training_frames = np.array(frames)
        self.lpc_coefficients = np.array(lpc_coeffs_list)
        self.residual_signals = np.array(residual_list)
        
        return self.lpc_coefficients, self.residual_signals
    
    def initialize_hmm_parameters(self):
        """
        Phase 1, Step 2: State Initialization using Clustering
        
        Uses k-means clustering to initialize HMM parameters from LPC coefficients
        """
        if self.lpc_coefficients is None:
            raise ValueError("Must extract features first using extract_features()")
        
        # Use k-means clustering on LPC coefficients
        # Normalize coefficients for better clustering
        normalized_coeffs = self.lpc_coefficients / (np.std(self.lpc_coefficients) + 1e-10)
        
        # Perform k-means clustering
        centroids, distortion = vq.kmeans2(normalized_coeffs, self.n_states, minit='++')
        
        # Assign each frame to a cluster
        cluster_indices, _ = vq.vq(normalized_coeffs, centroids)
        
        # Initialize HMM parameters
        self.initial_probabilities = np.zeros(self.n_states)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        self.state_means = centroids
        self.state_covariances = np.array([np.eye(self.lpc_order) * 0.01 for _ in range(self.n_states)])
        
        # Count transitions and initial states
        for i, state in enumerate(cluster_indices):
            if i == 0:
                self.initial_probabilities[state] += 1
            else:
                prev_state = cluster_indices[i-1]
                self.transition_matrix[prev_state, state] += 1
        
        # Normalize probabilities
        self.initial_probabilities = self.initial_probabilities / (np.sum(self.initial_probabilities) + 1e-10)
        
        # Normalize transition matrix with smoothing
        self.transition_matrix = self.transition_matrix + 1e-6  # Add small value for smoothing
        row_sums = np.sum(self.transition_matrix, axis=1)
        self.transition_matrix = self.transition_matrix / (row_sums[:, np.newaxis] + 1e-10)
        
        # Update state covariances based on clustered data
        for state in range(self.n_states):
            state_frames = normalized_coeffs[cluster_indices == state]
            if len(state_frames) > 1:
                self.state_covariances[state] = np.cov(state_frames.T) + 1e-6 * np.eye(self.lpc_order)
    
    def _expectation_step(self, observations):
        """
        E-step: Compute posterior probabilities of state sequences
        
        Args:
            observations: LPC coefficient sequences
            
        Returns:
            gamma: State posterior probabilities
            xi: Transition posterior probabilities
        """
        n_frames = len(observations)
        gamma = np.zeros((n_frames, self.n_states))
        xi = np.zeros((n_frames - 1, self.n_states, self.n_states))
        
        # Forward algorithm
        alpha = np.zeros((n_frames, self.n_states))
        alpha[0] = self.initial_probabilities * self._compute_emission_prob(observations[0])
        
        for t in range(1, n_frames):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * self._compute_emission_prob(observations[t], j)
        
        # Backward algorithm
        beta = np.zeros((n_frames, self.n_states))
        beta[-1] = np.ones(self.n_states)
        
        for t in range(n_frames - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_matrix[i] * self._compute_emission_prob(observations[t+1]) * beta[t+1])
        
        # Compute gamma and xi
        for t in range(n_frames):
            gamma[t] = alpha[t] * beta[t]
            gamma[t] = gamma[t] / (np.sum(gamma[t]) + 1e-10)
            
            if t < n_frames - 1:
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * \
                                      self._compute_emission_prob(observations[t+1], j) * beta[t+1, j]
                
                xi[t] = xi[t] / (np.sum(xi[t]) + 1e-10)
        
        return gamma, xi
    
    def _compute_emission_prob(self, observation, state=None):
        """
        Compute emission probability for observation(s)
        
        Args:
            observation: LPC coefficients
            state: Specific state (if None, compute for all states)
            
        Returns:
            Emission probabilities
        """
        if state is not None:
            try:
                mvn = multivariate_normal(self.state_means[state], self.state_covariances[state])
                return mvn.pdf(observation)
            except:
                return 1e-10
        else:
            probs = np.zeros(self.n_states)
            for s in range(self.n_states):
                try:
                    mvn = multivariate_normal(self.state_means[s], self.state_covariances[s])
                    probs[s] = mvn.pdf(observation)
                except:
                    probs[s] = 1e-10
            return probs + 1e-10
    
    def _maximization_step(self, observations, gamma, xi):
        """
        M-step: Update HMM parameters
        
        Args:
            observations: LPC coefficient sequences
            gamma: State posterior probabilities
            xi: Transition posterior probabilities
        """
        n_frames = len(observations)
        
        # Update initial probabilities
        self.initial_probabilities = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-10)
        
        # Update state means and covariances
        for state in range(self.n_states):
            # Weighted average for means
            weighted_sum = np.sum(gamma[:, state, np.newaxis] * observations, axis=0)
            weight_sum = np.sum(gamma[:, state])
            
            if weight_sum > 1e-10:
                self.state_means[state] = weighted_sum / weight_sum
                
                # Update covariance
                diff = observations - self.state_means[state]
                weighted_cov = np.sum(gamma[:, state, np.newaxis, np.newaxis] * 
                                     diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0)
                self.state_covariances[state] = weighted_cov / weight_sum + 1e-6 * np.eye(self.lpc_order)
    
    def iterative_refinement(self, n_iterations=10):
        """
        Phase 1, Step 3: Iterative Learning using EM Algorithm
        
        Args:
            n_iterations: Number of EM iterations
        """
        if self.lpc_coefficients is None:
            raise ValueError("Must initialize HMM parameters first")
        
        print(f"Starting EM training with {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            # E-step
            gamma, xi = self._expectation_step(self.lpc_coefficients)
            
            # M-step
            self._maximization_step(self.lpc_coefficients, gamma, xi)
            
            # Compute log-likelihood for monitoring
            log_likelihood = 0
            for t in range(len(self.lpc_coefficients)):
                if t == 0:
                    log_likelihood += np.log(np.sum(self.initial_probabilities * 
                                                  self._compute_emission_prob(self.lpc_coefficients[t])) + 1e-10)
                else:
                    likelihood_sum = 0
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            if t > 0:
                                likelihood_sum += gamma[t-1, i] * self.transition_matrix[i, j] * \
                                               self._compute_emission_prob(self.lpc_coefficients[t], j)
                    log_likelihood += np.log(likelihood_sum + 1e-10)
            
            print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.2f}")
    
    def identify_graph_constraints(self):
        """
        Phase 1, Step 4: Graph Component Identification
        
        Identifies structural degeneracies in the HMM transition matrix
        """
        # Find strongly connected components
        visited = [False] * self.n_states
        sccs = []
        
        def dfs(node, stack, on_stack):
            visited[node] = True
            on_stack[node] = True
            stack.append(node)
            
            for neighbor in range(self.n_states):
                if self.transition_matrix[node, neighbor] > 1e-6:
                    if not visited[neighbor]:
                        dfs(neighbor, stack, on_stack)
                    elif on_stack[neighbor]:
                        # Found a cycle
                        idx = stack.index(neighbor)
                        scc = stack[idx:]
                        if scc not in sccs:
                            sccs.append(scc)
            
            on_stack[node] = False
            stack.pop()
        
        # Find SCCs
        for i in range(self.n_states):
            if not visited[i]:
                dfs(i, [], [False] * self.n_states)
        
        # Penalize isolated or highly self-looping states
        for state in range(self.n_states):
            self_loop_prob = self.transition_matrix[state, state]
            outgoing_prob = np.sum(self.transition_matrix[state, :]) - self_loop_prob
            incoming_prob = np.sum(self.transition_matrix[:, state]) - self_loop_prob
            
            # If state is too isolated, reduce its self-loop probability
            if outgoing_prob < 0.1 and incoming_prob < 0.1 and self_loop_prob > 0.8:
                self.transition_matrix[state, state] *= 0.5
                # Redistribute probability to other states
                other_states = [i for i in range(self.n_states) if i != state]
                if other_states:
                    redistributed = self.transition_matrix[state, state] * 0.5 / len(other_states)
                    self.transition_matrix[state, other_states] += redistributed
        
        print(f"Identified {len(sccs)} strongly connected components")
    
    def _spectral_smoothness_cost(self, state1, state2):
        """
        Compute spectral smoothness cost between two states
        
        Args:
            state1: First HMM state
            state2: Second HMM state
            
        Returns:
            Smoothness cost (lower is smoother)
        """
        # Euclidean distance between LPC coefficient means
        distance = np.linalg.norm(self.state_means[state1] - self.state_means[state2])
        
        # Convert to cost (higher distance = higher cost)
        cost = distance / (np.std(self.state_means.flatten()) + 1e-10)
        
        return cost
    
    def _a_star_search(self, target_duration_frames, max_expanded=10000):
        """
        Phase 2, Step 5: State Sequence Decoding using A* Search
        
        Args:
            target_duration_frames: Target number of frames
            max_expanded: Maximum number of nodes to expand
            
        Returns:
            optimal_state_sequence: Most probable state sequence
        """
        class SearchNode:
            def __init__(self, state, frame_idx, path, g_cost, h_cost):
                self.state = state
                self.frame_idx = frame_idx
                self.path = path
                self.g_cost = g_cost  # Cost from start
                self.h_cost = h_cost  # Heuristic cost to goal
                self.f_cost = g_cost + h_cost
            
            def __lt__(self, other):
                return self.f_cost < other.f_cost
        
        # Heuristic: estimated remaining cost based on average transition cost
        avg_transition_cost = np.mean(np.sum(-np.log(self.transition_matrix + 1e-10), axis=1))
        avg_smoothness_cost = np.mean([self._spectral_smoothness_cost(i, j) 
                                      for i in range(self.n_states) for j in range(self.n_states)])
        
        remaining_frames_heuristic = (avg_transition_cost + avg_smoothness_cost) * target_duration_frames
        
        open_set = []
        closed_set = set()
        expanded_count = 0
        
        # Initialize with all possible starting states
        for state in range(self.n_states):
            if self.initial_probabilities[state] > 1e-10:
                g_cost = -np.log(self.initial_probabilities[state])
                h_cost = remaining_frames_heuristic
                node = SearchNode(state, 0, [state], g_cost, h_cost)
                heapq.heappush(open_set, node)
        
        while open_set and expanded_count < max_expanded:
            current = heapq.heappop(open_set)
            
            # Check if we've reached target duration
            if current.frame_idx >= target_duration_frames - 1:
                return current.path
            
            state_key = (current.state, current.frame_idx)
            if state_key in closed_set:
                continue
            
            closed_set.add(state_key)
            expanded_count += 1
            
            # Expand to next states
            for next_state in range(self.n_states):
                transition_prob = self.transition_matrix[current.state, next_state]
                
                if transition_prob > 1e-10:
                    # Compute costs
                    transition_cost = -np.log(transition_prob)
                    smoothness_cost = self._spectral_smoothness_cost(current.state, next_state)
                    
                    g_cost = current.g_cost + transition_cost + smoothness_cost
                    
                    # Heuristic: estimated cost to reach goal from next state
                    remaining_frames = target_duration_frames - (current.frame_idx + 1)
                    h_cost = remaining_frames * (avg_transition_cost + avg_smoothness_cost) * 0.5
                    
                    next_node = SearchNode(
                        next_state,
                        current.frame_idx + 1,
                        current.path + [next_state],
                        g_cost,
                        h_cost
                    )
                    
                    heapq.heappush(open_set, next_node)
        
        # If no complete path found, return the best partial path
        if open_set:
            best = heapq.heappop(open_set)
            return best.path
        else:
            # Fallback to most probable path
            return np.random.choice(self.n_states, target_duration_frames)
    
    def _karplus_strong_excitation(self, state, duration_samples):
        """
        Phase 2, Step 6: Excitation Generation using Delay-Line Feedback
        
        Args:
            state: HMM state for parameterization
            duration_samples: Number of samples to generate
            
        Returns:
            excitation_signal: Karplus-Strong excitation signal
        """
        # State-specific parameters based on LPC coefficients
        # Higher frequency content = shorter delay line
        mean_coeffs = self.state_means[state]
        spectral_centroid = np.mean(np.abs(mean_coeffs))
        
        # Delay line length inversely proportional to spectral centroid
        base_delay = int(20 + 80 * (1 - np.tanh(spectral_centroid)))
        delay_line = base_delay + np.random.randint(-5, 6)  # Add variation
        
        # Initialize with random noise burst
        excitation = np.zeros(duration_samples)
        noise_burst = np.random.randn(delay_line) * 0.5
        
        # Karplus-Strong algorithm with state-specific damping
        damping = 0.98 - 0.1 * spectral_centroid  # Higher centroid = less damping
        damping = np.clip(damping, 0.95, 0.99)
        
        for i in range(duration_samples):
            if i < delay_line:
                excitation[i] = noise_burst[i]
            else:
                # Average and dampen
                excitation[i] = damping * (excitation[i - delay_line] + excitation[i - delay_line + 1]) / 2
        
        # Apply envelope to make it more natural
        envelope = np.exp(-np.linspace(0, 5, duration_samples))
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
        # Create LPC filter coefficients (prepend 1 for the denominator)
        filter_coeffs = np.concatenate([[1], lpc_coeffs])
        
        # Apply filter
        try:
            synthesized = scipy.signal.lfilter([1], filter_coeffs, excitation)
        except:
            # Fallback: simple filtering
            synthesized = np.convolve(excitation, filter_coeffs[::-1], mode='same')
        
        # Apply gentle gain normalization
        max_val = np.max(np.abs(synthesized))
        if max_val > 0:
            synthesized = synthesized / max_val * 0.8
        
        return synthesized
    
    def synthesize_audio(self, target_duration_seconds, sample_rate=16000):
        """
        Phase 2: Complete Audio Synthesis Pipeline
        
        Args:
            target_duration_seconds: Target duration in seconds
            sample_rate: Output sample rate
            
        Returns:
            audio_output: Generated audio signal
        """
        if self.transition_matrix is None:
            raise ValueError("Must train the model first using extract_features() and iterative_refinement()")
        
        target_frames = int(target_duration_seconds * sample_rate / self.hop_size)
        print(f"Synthesizing {target_duration_seconds}s of audio ({target_frames} frames)")
        
        # Step 5: Decode optimal state sequence using A* search
        print("Step 5: Decoding state sequence with A* search...")
        state_sequence = self._a_star_search(target_frames)
        
        # Pad if necessary
        while len(state_sequence) < target_frames:
            state_sequence.append(state_sequence[-1])
        
        audio_output = []
        
        print("Step 6-8: Generating excitation, synthesizing audio...")
        for frame_idx, state in enumerate(state_sequence[:target_frames]):
            # Step 6: Generate Karplus-Strong excitation
            excitation = self._karplus_strong_excitation(state, self.hop_size)
            
            # Step 7: Apply LPC synthesis filter
            lpc_coeffs = self.state_means[state]
            frame_audio = self._lpc_synthesis_filter(excitation, lpc_coeffs)
            
            audio_output.append(frame_audio)
        
        # Step 8: Concatenate and finalize
        audio_output = np.concatenate(audio_output)
        
        # Apply final smoothing to reduce clicks
        window = np.hanning(len(audio_output))
        audio_output = audio_output * window
        
        # Normalize to prevent clipping
        audio_output = audio_output / (np.max(np.abs(audio_output)) + 1e-10) * 0.9
        
        return audio_output
    
    def train(self, audio_signal, sample_rate=16000, n_em_iterations=10):
        """
        Complete training pipeline
        
        Args:
            audio_signal: Training audio signal
            sample_rate: Sampling rate
            n_em_iterations: Number of EM iterations
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
        self.iterative_refinement(n_em_iterations)
        
        # Step 4: Graph Constraints
        print("Step 4: Applying graph constraints...")
        self.identify_graph_constraints()
        
        print("Training complete!")
    
    def generate(self, duration_seconds, sample_rate=16000):
        """
        Generate new audio
        
        Args:
            duration_seconds: Duration of generated audio
            sample_rate: Output sample rate
            
        Returns:
            Generated audio signal
        """
        print("Phase 2: Inference (Audio Generation)")
        print("=" * 50)
        
        return self.synthesize_audio(duration_seconds, sample_rate)


def example_usage():
    """
    Example demonstrating how to use the SSGS system
    """
    # Initialize SSGS
    ssgs = SpectralStateGuidedSynthesis(
        n_states=16,
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
    
    # Generate new audio
    generated_audio = ssgs.generate(duration_seconds=3.0, sample_rate=sample_rate)
    
    print(f"\nGenerated audio with {len(generated_audio)} samples")
    print(f"Duration: {len(generated_audio) / sample_rate:.2f} seconds")
    print(f"Max amplitude: {np.max(np.abs(generated_audio)):.3f}")
    
    return ssgs, generated_audio


if __name__ == "__main__":
    # Run the example
    ssgs, audio = example_usage()
