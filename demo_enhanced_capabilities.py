#!/usr/bin/env python3
"""
Demo: Enhanced Generative Capabilities (112% and 163% increases)
================================================================

This demo shows the impact of increased generative capabilities
through higher HMM state counts.
"""

import numpy as np
from ssgs import SpectralStateGuidedSynthesis

def create_rich_test_signal(sample_rate=16000, duration=2.0):
    """Create a rich harmonic signal for demonstration"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Varying fundamental frequency (vibrato)
    base_freq = 220 + 30 * np.sin(2 * np.pi * 0.5 * t)
    
    # Rich harmonic series with time-varying amplitudes
    signal = np.zeros_like(t)
    for h in range(1, 7):
        freq = base_freq * h
        amplitude = 0.6 / h  # Natural harmonic decay
        amplitude *= (1 + 0.3 * np.sin(2 * np.pi * h * 0.2 * t))  # Modulation
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add formant-like characteristics
    formant1 = 800 + 100 * np.sin(2 * np.pi * 0.3 * t)
    formant2 = 1200 + 150 * np.sin(2 * np.pi * 0.4 * t)
    signal += 0.2 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t/duration)
    signal += 0.15 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t/duration)
    
    # Envelope
    attack_time = 0.15
    decay_time = 0.3
    envelope = np.ones_like(t)
    attack_idx = int(attack_time * sample_rate)
    decay_start_idx = int((duration - decay_time) * sample_rate)
    envelope[:attack_idx] = np.linspace(0, 1, attack_idx)
    envelope[decay_start_idx:] = np.linspace(1, 0, len(envelope) - decay_start_idx)
    signal *= envelope
    
    # Add slight noise for realism
    signal += 0.005 * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    
    return signal


def demonstrate_capability_levels():
    """Demonstrate all three capability levels"""
    print("=" * 70)
    print("SSGS Enhanced Generative Capabilities Demo")
    print("=" * 70)
    
    sample_rate = 16000
    
    # Create training signal
    print("\nCreating training signal...")
    training_signal = create_rich_test_signal(sample_rate, duration=2.0)
    print(f"  Training signal: {len(training_signal)} samples ({len(training_signal)/sample_rate:.1f}s)")
    
    capability_configs = [
        (16, "Standard", "Original baseline"),
        (34, "Enhanced", "112% increase - NEW DEFAULT"),
        (42, "Maximum", "163% increase"),
    ]
    
    for n_states, level, description in capability_configs:
        print("\n" + "=" * 70)
        print(f"{level} Generative Capabilities: {n_states} states")
        print(f"  {description}")
        print("=" * 70)
        
        # Initialize model
        print(f"\nInitializing SSGS with {n_states} states...")
        ssgs = SpectralStateGuidedSynthesis(
            n_states=n_states,
            lpc_order=14,
            frame_size=1024,
            hop_size=256
        )
        
        # Train
        print(f"Training model...")
        ssgs.train(training_signal, sample_rate, n_em_iterations=8)
        
        # Generate with different fidelity levels
        print(f"\nGenerating audio samples:")
        
        for fidelity, fid_label in [(1.0, "reconstruction"), (0.5, "balanced"), (0.0, "synthetic")]:
            generated = ssgs.generate(
                duration_seconds=1.5,
                sample_rate=sample_rate,
                fidelity=fidelity
            )
            
            rms = np.sqrt(np.mean(generated**2))
            peak = np.max(np.abs(generated))
            spectral_energy = np.sum(np.abs(np.fft.rfft(generated))**2)
            
            print(f"  - {fid_label:15s} (fidelity={fidelity:.1f}): "
                  f"RMS={rms:.3f}, Peak={peak:.3f}, "
                  f"Spectral Energy={spectral_energy:.1e}")
        
        # Model complexity metrics
        print(f"\nModel Complexity:")
        print(f"  - HMM States: {ssgs.n_states}")
        print(f"  - LPC Order: {ssgs.lpc_order}")
        print(f"  - Total Parameters: {ssgs.state_means.shape[0] * ssgs.state_means.shape[1]}")
        print(f"  - Transition Matrix Size: {ssgs.transition_matrix.shape}")
        
        # Transition matrix sparsity
        sparsity = 1 - np.count_nonzero(ssgs.transition_matrix > 0.01) / (n_states ** 2)
        transition_entropy = -np.sum(
            ssgs.transition_matrix * np.log(ssgs.transition_matrix + 1e-10)
        )
        print(f"  - Transition Sparsity: {sparsity:.3f}")
        print(f"  - Transition Entropy: {transition_entropy:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print("  • More states = richer spectral representation")
    print("  • Enhanced (34 states) is the new default")
    print("  • Maximum (42 states) provides highest generative capacity")
    print("  • All levels support full fidelity spectrum (0.0 to 1.0)")
    print("\nUsage:")
    print("  # Use new default (enhanced)")
    print("  ssgs = SpectralStateGuidedSynthesis()")
    print()
    print("  # Use maximum capabilities")
    print("  ssgs = SpectralStateGuidedSynthesis(n_states=42)")
    print()
    print("  # Use original baseline")
    print("  ssgs = SpectralStateGuidedSynthesis(n_states=16)")


if __name__ == "__main__":
    demonstrate_capability_levels()
