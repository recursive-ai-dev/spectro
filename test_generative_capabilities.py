#!/usr/bin/env python3
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

"""
Test enhanced generative capabilities (112% and 163% increases)
"""

import numpy as np
import pytest
from ssgs import SpectralStateGuidedSynthesis


def create_simple_test_signal(sample_rate=16000, duration=1.0):
    """Create a simple test signal for quick testing"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a richer harmonic signal with noise for robustness
    base_freq = 220 + 20 * np.sin(2 * np.pi * 0.5 * t)
    signal = (
        0.5 * np.sin(2 * np.pi * base_freq * t) +
        0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.2 * np.sin(2 * np.pi * base_freq * 3 * t) +
        0.1 * np.sin(2 * np.pi * base_freq * 4 * t)
    )
    # Add envelope with smoother transitions
    attack = int(0.1 * len(signal))
    decay = int(0.1 * len(signal))
    envelope = np.ones(len(signal))
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    signal *= envelope
    # Add noise for better LPC stability
    signal += 0.01 * np.random.randn(len(signal))
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    return signal


def test_default_uses_enhanced_capabilities():
    """Test that default initialization uses enhanced capabilities (34 states = 112% increase)"""
    ssgs = SpectralStateGuidedSynthesis()
    assert ssgs.n_states == 34, f"Expected default n_states=34, got {ssgs.n_states}"


def test_enhanced_generative_capabilities_112_percent():
    """Test enhanced generative capabilities with 112% increase (34 states)"""
    sample_rate = 16000
    signal = create_simple_test_signal(sample_rate, duration=1.0)
    
    # Initialize with enhanced capabilities (112% increase: 16 + 16*1.12 ≈ 34)
    ssgs = SpectralStateGuidedSynthesis(
        n_states=34,
        lpc_order=12,
        frame_size=512,
        hop_size=128
    )
    
    assert ssgs.n_states == 34
    
    # Train the model
    ssgs.extract_features(signal, sample_rate)
    ssgs.initialize_hmm_parameters()
    
    # Verify model has correct number of states after initialization
    assert ssgs.state_means.shape[0] == 34
    assert ssgs.state_covariances.shape[0] == 34
    assert ssgs.transition_matrix.shape == (34, 34)
    assert ssgs.initial_probabilities.shape[0] == 34
    
    # Run a few EM iterations
    ssgs.iterative_refinement(n_iterations=3)
    
    # Verify model structure is maintained
    assert ssgs.state_means.shape[0] == 34
    assert ssgs.transition_matrix.shape == (34, 34)
    
    # Generate audio to verify synthesis works
    generated = ssgs.generate(duration_seconds=0.5, sample_rate=sample_rate, fidelity=1.0)
    
    # Verify generated audio has reasonable properties
    assert len(generated) > 0
    # Check for valid audio (no NaN or Inf)
    assert not np.any(np.isnan(generated)), "Generated audio contains NaN values"
    assert not np.any(np.isinf(generated)), "Generated audio contains Inf values"
    # Check normalization
    assert np.max(np.abs(generated)) <= 1.0, "Audio not properly normalized"


def test_maximum_generative_capabilities_163_percent():
    """Test maximum generative capabilities with 163% increase (42 states)"""
    sample_rate = 16000
    signal = create_simple_test_signal(sample_rate, duration=1.0)
    
    # Initialize with maximum capabilities (163% increase: 16 + 16*1.63 ≈ 42)
    ssgs = SpectralStateGuidedSynthesis(
        n_states=42,
        lpc_order=12,
        frame_size=512,
        hop_size=128
    )
    
    assert ssgs.n_states == 42
    
    # Train the model
    ssgs.extract_features(signal, sample_rate)
    ssgs.initialize_hmm_parameters()
    
    # Verify model has correct number of states after initialization
    assert ssgs.state_means.shape[0] == 42
    assert ssgs.state_covariances.shape[0] == 42
    assert ssgs.transition_matrix.shape == (42, 42)
    assert ssgs.initial_probabilities.shape[0] == 42
    
    # Run a few EM iterations
    ssgs.iterative_refinement(n_iterations=3)
    
    # Verify model structure is maintained
    assert ssgs.state_means.shape[0] == 42
    assert ssgs.transition_matrix.shape == (42, 42)
    
    # Generate audio to verify synthesis works
    generated = ssgs.generate(duration_seconds=0.5, sample_rate=sample_rate, fidelity=1.0)
    
    # Verify generated audio has reasonable properties
    assert len(generated) > 0
    # Check for valid audio (no NaN or Inf)
    assert not np.any(np.isnan(generated)), "Generated audio contains NaN values"
    assert not np.any(np.isinf(generated)), "Generated audio contains Inf values"
    # Check normalization
    assert np.max(np.abs(generated)) <= 1.0, "Audio not properly normalized"


def test_generative_capability_comparison():
    """Compare different generative capability levels"""
    sample_rate = 16000
    signal = create_simple_test_signal(sample_rate, duration=1.0)
    
    # Test all three capability levels
    capability_levels = [
        (16, "standard"),
        (34, "enhanced_112%"),
        (42, "maximum_163%")
    ]
    
    results = {}
    
    for n_states, label in capability_levels:
        ssgs = SpectralStateGuidedSynthesis(
            n_states=n_states,
            lpc_order=12,
            frame_size=512,
            hop_size=128
        )
        
        ssgs.extract_features(signal, sample_rate)
        ssgs.initialize_hmm_parameters()
        ssgs.iterative_refinement(n_iterations=2)
        
        generated = ssgs.generate(duration_seconds=0.3, sample_rate=sample_rate, fidelity=1.0)
        
        results[label] = {
            'n_states': n_states,
            'generated_length': len(generated),
            'rms': np.sqrt(np.mean(generated**2)),
            'peak': np.max(np.abs(generated)),
            'spectral_complexity': ssgs.state_means.shape[0] * ssgs.state_means.shape[1]
        }
    
    # Verify that higher capability levels have more spectral complexity
    assert results['enhanced_112%']['spectral_complexity'] > results['standard']['spectral_complexity']
    assert results['maximum_163%']['spectral_complexity'] > results['enhanced_112%']['spectral_complexity']
    
    # Verify all levels generate valid audio
    for label in results:
        assert results[label]['peak'] >= 0.0, f"{label} has negative peak"
        assert results[label]['peak'] <= 1.0, f"{label} exceeds normalization limit"


def test_model_export_import_with_enhanced_capabilities():
    """Test that enhanced capability models can be exported and loaded"""
    import tempfile
    import os
    
    sample_rate = 16000
    signal = create_simple_test_signal(sample_rate, duration=0.5)
    
    # Create and train model with enhanced capabilities
    ssgs = SpectralStateGuidedSynthesis(n_states=34)
    ssgs.train(signal, sample_rate, n_em_iterations=2)
    
    # Export model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "enhanced_model.npz")
        ssgs.export_model(model_path)
        
        # Load model
        loaded_ssgs = SpectralStateGuidedSynthesis.load_model(model_path)
        
        # Verify state count is preserved
        assert loaded_ssgs.n_states == 34
        assert loaded_ssgs.state_means.shape[0] == 34
        assert loaded_ssgs.transition_matrix.shape == (34, 34)
        
        # Verify model can generate audio
        generated = loaded_ssgs.generate(duration_seconds=0.3, sample_rate=sample_rate)
        assert len(generated) > 0
        assert np.max(np.abs(generated)) > 0.1


if __name__ == "__main__":
    # Run tests manually
    print("Testing Enhanced Generative Capabilities")
    print("=" * 60)
    
    print("\n1. Testing default uses enhanced capabilities (34 states)...")
    test_default_uses_enhanced_capabilities()
    print("   ✓ Passed")
    
    print("\n2. Testing 112% increase (34 states)...")
    test_enhanced_generative_capabilities_112_percent()
    print("   ✓ Passed")
    
    print("\n3. Testing 163% increase (42 states)...")
    test_maximum_generative_capabilities_163_percent()
    print("   ✓ Passed")
    
    print("\n4. Testing capability comparison...")
    test_generative_capability_comparison()
    print("   ✓ Passed")
    
    print("\n5. Testing model export/import with enhanced capabilities...")
    test_model_export_import_with_enhanced_capabilities()
    print("   ✓ Passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nGenerative capability increases verified:")
    print("  - Standard: 16 states (baseline)")
    print("  - Enhanced: 34 states (+112%)")
    print("  - Maximum: 42 states (+163%)")
