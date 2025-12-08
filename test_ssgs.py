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
import matplotlib.pyplot as plt
import soundfile as sf
from ssgs import SpectralStateGuidedSynthesis
import warnings
warnings.filterwarnings('ignore')

def create_test_signal(sample_rate=16000, duration=2.0):
    """
    Create a complex test signal with multiple frequency components
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency that changes over time (pitch contour)
    base_freq = 220 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Vibrato
    
    # Generate harmonic series with time-varying amplitudes
    signal = np.zeros_like(t)
    
    # Fundamental
    signal += 0.6 * np.sin(2 * np.pi * base_freq * t)
    
    # Harmonics with varying strengths
    for h in range(2, 6):
        freq = base_freq * h
        amplitude = 0.3 / h  # Natural harmonic decay
        amplitude *= (1 + 0.3 * np.sin(2 * np.pi * h * 0.2 * t))  # Modulation
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add formant-like characteristics (vocal-like resonances)
    formant1 = 800 + 100 * np.sin(2 * np.pi * 0.3 * t)
    formant2 = 1200 + 150 * np.sin(2 * np.pi * 0.4 * t)
    
    signal += 0.2 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t/duration)
    signal += 0.15 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t/duration)
    
    # Apply overall envelope (attack, sustain, decay)
    attack_time = 0.1
    decay_time = 0.3
    
    envelope = np.ones_like(t)
    attack_idx = int(attack_time * sample_rate)
    decay_start_idx = int((duration - decay_time) * sample_rate)
    
    envelope[:attack_idx] = np.linspace(0, 1, attack_idx)
    envelope[decay_start_idx:] = np.linspace(1, 0, len(envelope) - decay_start_idx)
    
    signal *= envelope
    
    # Add small amount of noise for realism
    signal += 0.005 * np.random.randn(len(signal))
    
    return signal

def analyze_and_plot_results(ssgs, original_signal, generated_signal, sample_rate=16000):
    """
    Analyze and plot the results of SSGS synthesis
    """
    # Time arrays
    t_orig = np.linspace(0, len(original_signal)/sample_rate, len(original_signal))
    t_gen = np.linspace(0, len(generated_signal)/sample_rate, len(generated_signal))
    
    # Create comprehensive plots
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Spectral-State Guided Synthesis (SSGS) Analysis', fontsize=16)
    
    # 1. Time domain waveforms
    axes[0, 0].plot(t_orig, original_signal, 'b-', alpha=0.7, label='Original')
    axes[0, 0].set_title('Original Signal - Time Domain')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(t_gen, generated_signal, 'r-', alpha=0.7, label='Generated')
    axes[0, 1].set_title('Generated Signal - Time Domain')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 2. Frequency spectra (FFT)
    def compute_spectrum(signal, sample_rate):
        n_fft = len(signal)
        freqs = np.fft.fftfreq(n_fft, 1/sample_rate)
        spectrum = np.fft.fft(signal)
        magnitude = np.abs(spectrum)
        return freqs[:n_fft//2], magnitude[:n_fft//2]
    
    freq_orig, mag_orig = compute_spectrum(original_signal, sample_rate)
    freq_gen, mag_gen = compute_spectrum(generated_signal, sample_rate)
    
    axes[1, 0].semilogy(freq_orig[:2000], mag_orig[:2000], 'b-', alpha=0.7)
    axes[1, 0].set_title('Original Signal - Frequency Spectrum')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 2000])
    
    axes[1, 1].semilogy(freq_gen[:2000], mag_gen[:2000], 'r-', alpha=0.7)
    axes[1, 1].set_title('Generated Signal - Frequency Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 2000])
    
    # 3. Spectrograms
    def compute_spectrogram(signal, sample_rate):
        power, freqs, bins, _ = plt.specgram(signal, Fs=sample_rate, NFFT=1024, noverlap=512)
        return freqs, bins, 10 * np.log10(power + 1e-10)
    
    f_orig, t_spec_orig, S_orig = compute_spectrogram(original_signal, sample_rate)
    f_gen, t_spec_gen, S_gen = compute_spectrogram(generated_signal, sample_rate)
    
    axes[2, 0].imshow(S_orig, aspect='auto', origin='lower', extent=[t_spec_orig[0], t_spec_orig[-1], f_orig[0], f_orig[-1]], cmap='viridis')
    axes[2, 0].set_title('Original Signal - Spectrogram')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Frequency (Hz)')
    axes[2, 0].set_ylim([0, 2000])
    
    axes[2, 1].imshow(S_gen, aspect='auto', origin='lower', extent=[t_spec_gen[0], t_spec_gen[-1], f_gen[0], f_gen[-1]], cmap='viridis')
    axes[2, 1].set_title('Generated Signal - Spectrogram')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Frequency (Hz)')
    axes[2, 1].set_ylim([0, 2000])
    
    # 4. HMM Analysis
    if hasattr(ssgs, 'transition_matrix') and ssgs.transition_matrix is not None:
        # Transition matrix heatmap
        im = axes[3, 0].imshow(ssgs.transition_matrix, cmap='hot', aspect='auto')
        axes[3, 0].set_title('HMM Transition Matrix')
        axes[3, 0].set_xlabel('To State')
        axes[3, 0].set_ylabel('From State')
        plt.colorbar(im, ax=axes[3, 0])
        
        # State mean coefficients visualization
        axes[3, 1].imshow(ssgs.state_means, cmap='coolwarm', aspect='auto')
        axes[3, 1].set_title('State LPC Coefficients (Means)')
        axes[3, 1].set_xlabel('LPC Coefficient Index')
        axes[3, 1].set_ylabel('HMM State')
        plt.colorbar(ax=axes[3, 1])
    
    plt.tight_layout()
    plt.savefig('ssgs_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Basic statistics
    orig_stats = {
        'RMS': np.sqrt(np.mean(original_signal**2)),
        'Peak': np.max(np.abs(original_signal)),
        'Zero crossings': np.sum(np.diff(np.sign(original_signal)) != 0),
        'Spectral centroid': np.sum(freq_orig * mag_orig) / (np.sum(mag_orig) + 1e-10)
    }
    
    gen_stats = {
        'RMS': np.sqrt(np.mean(generated_signal**2)),
        'Peak': np.max(np.abs(generated_signal)),
        'Zero crossings': np.sum(np.diff(np.sign(generated_signal)) != 0),
        'Spectral centroid': np.sum(freq_gen * mag_gen) / (np.sum(mag_gen) + 1e-10)
    }
    
    print(f"{'Metric':<20} {'Original':<15} {'Generated':<15} {'Ratio':<10}")
    print("-" * 60)
    for metric in orig_stats:
        ratio = gen_stats[metric] / (orig_stats[metric] + 1e-10)
        print(f"{metric:<20} {orig_stats[metric]:<15.4f} {gen_stats[metric]:<15.4f} {ratio:<10.4f}")
    
    # Spectral similarity
    correlation = np.corrcoef(mag_orig[:1000], mag_gen[:1000])[0, 1]
    print(f"\nSpectral correlation (0-1000 Hz): {correlation:.4f}")
    
    return orig_stats, gen_stats

def save_audio_files(original_signal, generated_signal, sample_rate=16000):
    """
    Save audio files for listening comparison
    """
    try:
        sf.write('original_signal.wav', original_signal, sample_rate)
        sf.write('generated_signal.wav', generated_signal, sample_rate)
        print(f"\nAudio files saved:")
        print(f"  - original_signal.wav ({len(original_signal)/sample_rate:.2f}s)")
        print(f"  - generated_signal.wav ({len(generated_signal)/sample_rate:.2f}s)")
    except ImportError:
        print("\nWarning: soundfile not installed. Cannot save audio files.")
        print("Install with: pip install soundfile")
    except Exception as e:
        print(f"\nError saving audio files: {e}")

def run_comprehensive_test():
    """
    Run a comprehensive test of the SSGS algorithm
    """
    print("SPECTRAL-STATE GUIDED SYNTHESIS (SSGS) - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Parameters
    sample_rate = 16000
    training_duration = 3.0
    generation_duration = 2.0
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Training duration: {training_duration} seconds")
    print(f"Generation duration: {generation_duration} seconds")
    print()
    
    # Step 1: Create training signal
    print("Step 1: Creating complex training signal...")
    original_signal = create_test_signal(sample_rate, training_duration)
    print(f"  Created signal with {len(original_signal)} samples")
    print(f"  Signal RMS: {np.sqrt(np.mean(original_signal**2)):.4f}")
    
    # Step 2: Initialize and train SSGS
    print("\nStep 2: Initializing SSGS model...")
    ssgs = SpectralStateGuidedSynthesis(
        n_states=20,
        lpc_order=14,
        frame_size=1024,
        hop_size=256
    )
    print("  Model initialized with:")
    print(f"    - States: {ssgs.n_states}")
    print(f"    - LPC order: {ssgs.lpc_order}")
    print(f"    - Frame size: {ssgs.frame_size}")
    print(f"    - Hop size: {ssgs.hop_size}")
    
    print("\nStep 3: Training SSGS model...")
    ssgs.train(original_signal, sample_rate, n_em_iterations=15)
    
    # Step 4: Generate new audio
    print(f"\nStep 4: Generating {generation_duration}s of new audio...")
    generated_signal = ssgs.generate(generation_duration, sample_rate)
    print(f"  Generated signal with {len(generated_signal)} samples")
    print(f"  Signal RMS: {np.sqrt(np.mean(generated_signal**2)):.4f}")
    
    # Step 5: Analysis and visualization
    print("\nStep 5: Analyzing and visualizing results...")
    orig_stats, gen_stats = analyze_and_plot_results(ssgs, original_signal, generated_signal, sample_rate)
    
    # Step 6: Save audio files
    save_audio_files(original_signal, generated_signal, sample_rate)
    
    # Step 7: Performance evaluation
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    # Compute reconstruction quality metrics
    if len(original_signal) > len(generated_signal):
        test_signal = original_signal[:len(generated_signal)]
    else:
        test_signal = generated_signal[:len(original_signal)]
    
    # Signal-to-Noise Ratio (treating one as reference)
    error_power = np.sum((test_signal - generated_signal[:len(test_signal)])**2) + 1e-10
    snr_db = 10 * np.log10(np.sum(test_signal**2) / error_power)
    print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
    
    # HMM model complexity analysis
    if hasattr(ssgs, 'transition_matrix'):
        transition_entropy = -np.sum(ssgs.transition_matrix * np.log(ssgs.transition_matrix + 1e-10))
        print(f"HMM transition entropy: {transition_entropy:.4f}")
        
        # Sparsity measure
        sparsity = 1 - np.count_nonzero(ssgs.transition_matrix > 0.01) / (ssgs.n_states ** 2)
        print(f"Transition matrix sparsity: {sparsity:.4f}")
    
    print("\nSSGS test completed successfully!")
    print("Check 'ssgs_analysis.png' for visual results.")
    print("Check audio files for listening comparison.")
    
    return ssgs, original_signal, generated_signal

if __name__ == "__main__":
    # Run the comprehensive test
    ssgs, orig, gen = run_comprehensive_test()
