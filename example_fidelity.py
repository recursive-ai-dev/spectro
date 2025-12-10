#!/usr/bin/env python3
"""
Example demonstrating the fidelity parameter in SSGS
=====================================================

This example shows how to use the fidelity parameter to control
whether the model reconstructs the training audio (high fidelity)
or generates novel variations (low fidelity).

Fidelity Levels:
  - 1.0: High fidelity reconstruction - directly copies the training audio
  - 0.5: Balanced variance - mix of original patterns and novel elements
  - 0.0: Fully synthetic - generates novel audio based on learned patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from ssgs import SpectralStateGuidedSynthesis

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Audio files won't be saved.")
    print("Install with: pip install soundfile")


def load_or_create_training_audio(audio_file=None, sample_rate=16000, duration=3.0):
    """
    Load audio from file or create a synthetic training signal

    Args:
        audio_file: Path to audio file (optional)
        sample_rate: Sample rate if creating synthetic signal
        duration: Duration if creating synthetic signal

    Returns:
        audio_signal, sample_rate
    """
    if audio_file and HAS_SOUNDFILE:
        try:
            audio, sr = sf.read(audio_file)
            if audio.ndim > 1:  # Convert stereo to mono
                audio = audio.mean(axis=1)
            print(f"Loaded audio from {audio_file}")
            print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr} Hz")
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            print("Falling back to synthetic signal...")

    # Create synthetic training signal
    print("Creating synthetic training signal...")
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a musical phrase with variation
    base_freq = 220 + 30 * np.sin(2 * np.pi * 0.3 * t)  # Vibrato

    # Harmonic series
    signal = (
        0.5 * np.sin(2 * np.pi * base_freq * t) +
        0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.2 * np.sin(2 * np.pi * base_freq * 3 * t) +
        0.1 * np.sin(2 * np.pi * base_freq * 4 * t)
    )

    # Add formant-like characteristics
    formant1 = 800 + 100 * np.sin(2 * np.pi * 0.4 * t)
    formant2 = 1200 + 150 * np.sin(2 * np.pi * 0.5 * t)
    signal += 0.15 * np.sin(2 * np.pi * formant1 * t)
    signal += 0.1 * np.sin(2 * np.pi * formant2 * t)

    # Envelope
    attack = 0.1
    decay = 0.2
    envelope = np.ones_like(t)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    signal *= envelope

    # Add slight noise
    signal += 0.005 * np.random.randn(len(signal))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    print(f"  Duration: {duration}s, Sample rate: {sample_rate} Hz")
    return signal, sample_rate


def demonstrate_fidelity_spectrum():
    """
    Demonstrate how different fidelity levels affect the output
    """
    print("=" * 70)
    print("SSGS Fidelity Demonstration")
    print("=" * 70)

    # Step 1: Load or create training audio
    print("\nStep 1: Preparing training audio...")
    # You can replace None with a path to your own audio file
    training_audio, sample_rate = load_or_create_training_audio(
        audio_file=None,  # Set to your audio file path
        sample_rate=16000,
        duration=3.0
    )

    # Step 2: Initialize and train SSGS
    print("\nStep 2: Training SSGS model...")
    ssgs = SpectralStateGuidedSynthesis(
        n_states=20,
        lpc_order=14,
        frame_size=1024,
        hop_size=256,
        smoothness_weight=0.5
    )

    print(f"  Model configuration:")
    print(f"    - States: {ssgs.n_states}")
    print(f"    - LPC order: {ssgs.lpc_order}")
    print(f"    - Frame size: {ssgs.frame_size}")
    print(f"    - Hop size: {ssgs.hop_size}")

    ssgs.train(training_audio, sample_rate, n_em_iterations=15)

    # Step 3: Generate audio with different fidelity levels
    print("\n" + "=" * 70)
    print("Step 3: Generating audio at different fidelity levels...")
    print("=" * 70)

    fidelity_levels = [
        (1.0, "high_fidelity", "High Fidelity (Reconstruction)"),
        (0.8, "high_variance", "High with Variance"),
        (0.5, "balanced", "Balanced Mix"),
        (0.2, "low_variance", "Low with Some Original"),
        (0.0, "fully_synthetic", "Fully Synthetic (Novel)")
    ]

    outputs = {}

    for fidelity, label, description in fidelity_levels:
        print(f"\n{description} (fidelity={fidelity:.1f})")
        audio = ssgs.generate(
            duration_seconds=2.5,
            sample_rate=sample_rate,
            fidelity=fidelity
        )
        outputs[label] = audio
        print(f"  ✓ Generated {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")

    # Step 4: Save audio files
    if HAS_SOUNDFILE:
        print("\n" + "=" * 70)
        print("Step 4: Saving audio files...")
        print("=" * 70)

        # Save training audio
        sf.write('training_audio.wav', training_audio, sample_rate)
        print(f"  ✓ Saved training_audio.wav")

        # Save generated audio
        for label, audio in outputs.items():
            filename = f'generated_{label}.wav'
            sf.write(filename, audio, sample_rate)
            print(f"  ✓ Saved {filename}")

        print("\nAll audio files saved! Compare them to hear the difference.")

    # Step 5: Visualize results
    print("\n" + "=" * 70)
    print("Step 5: Creating visualization...")
    print("=" * 70)

    fig, axes = plt.subplots(len(outputs) + 1, 2, figsize=(15, 3 * (len(outputs) + 1)))
    fig.suptitle('SSGS Fidelity Spectrum: Reconstruction vs Generation', fontsize=14)

    # Plot training audio
    t_train = np.linspace(0, len(training_audio)/sample_rate, len(training_audio))
    axes[0, 0].plot(t_train, training_audio, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Training Audio - Time Domain')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training spectrum
    spectrum = np.abs(np.fft.rfft(training_audio))
    freqs = np.fft.rfftfreq(len(training_audio), 1/sample_rate)
    axes[0, 1].semilogy(freqs[:3000], spectrum[:3000], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title('Training Audio - Frequency Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 3000])

    # Plot generated audio for each fidelity level
    for idx, (fidelity, label, description) in enumerate(fidelity_levels, start=1):
        audio = outputs[label]
        t_gen = np.linspace(0, len(audio)/sample_rate, len(audio))

        # Time domain
        axes[idx, 0].plot(t_gen, audio, 'r-', alpha=0.7, linewidth=0.5)
        axes[idx, 0].set_title(f'{description} - Time Domain')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Frequency domain
        spectrum_gen = np.abs(np.fft.rfft(audio))
        freqs_gen = np.fft.rfftfreq(len(audio), 1/sample_rate)
        axes[idx, 1].semilogy(freqs_gen[:3000], spectrum_gen[:3000], 'r-', alpha=0.7, linewidth=0.5)
        axes[idx, 1].set_title(f'{description} - Frequency Spectrum')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim([0, 3000])

    plt.tight_layout()
    plt.savefig('fidelity_spectrum_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization: fidelity_spectrum_analysis.png")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nFidelity Parameter Guide:")
    print("  fidelity=1.0 → Reconstruction mode")
    print("               Uses actual training residuals")
    print("               Output closely matches training audio")
    print("               Great for: Compression, style transfer base")
    print()
    print("  fidelity=0.5 → Balanced variance mode")
    print("               Blends training residuals with synthetic excitation")
    print("               Output shows variation while keeping structure")
    print("               Great for: Creative variations, improvisation")
    print()
    print("  fidelity=0.0 → Generative mode")
    print("               Uses only synthetic Karplus-Strong excitation")
    print("               Output is novel but inspired by training")
    print("               Great for: Music generation, sound design")
    print()
    print("All done! Check the generated audio files and visualization.")

    return ssgs, training_audio, outputs


if __name__ == "__main__":
    ssgs, training, generated_outputs = demonstrate_fidelity_spectrum()

    print("\n" + "=" * 70)
    print("Quick Start Examples")
    print("=" * 70)
    print("""
# Example 1: Copy a song with high fidelity
ssgs.generate(duration_seconds=10.0, fidelity=1.0)

# Example 2: Generate variations of a song
ssgs.generate(duration_seconds=10.0, fidelity=0.5)

# Example 3: Create novel audio inspired by training
ssgs.generate(duration_seconds=10.0, fidelity=0.0)

# Example 4: Load your own audio and train
import soundfile as sf
audio, sr = sf.read('your_song.wav')
ssgs = SpectralStateGuidedSynthesis(n_states=20, lpc_order=14)
ssgs.train(audio, sr, n_em_iterations=15)
reconstructed = ssgs.generate(duration_seconds=5.0, fidelity=1.0)
sf.write('reconstructed_song.wav', reconstructed, sr)
    """)
