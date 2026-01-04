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
Common test utilities for SSGS testing and demonstration.
This module provides reusable test signal generation functions.
"""

import numpy as np


def _resolve_rng(rng=None, seed=None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def create_test_signal(
    sample_rate=16000,
    duration=2.0,
    base_freq_offset=50,
    harmonic_range=(2, 6),
    include_fundamental=True,
    attack_time=0.1,
    decay_time=0.3,
    noise_level=0.005,
    normalize=False,
    rng=None,
    seed=None,
):
    """
    Create a complex test signal with multiple frequency components.
    
    This is a comprehensive signal generator used across test suites.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        base_freq_offset: Vibrato depth in Hz
        harmonic_range: Tuple (start, end) for harmonic series
        include_fundamental: Whether to include the fundamental frequency
        attack_time: Attack time in seconds
        decay_time: Decay time in seconds
        noise_level: Noise amplitude
        normalize: Whether to normalize output to 0.8 max amplitude
        rng: Optional NumPy random generator for deterministic noise
        seed: Optional seed used when rng is not provided
    
    Returns:
        signal: Generated test signal as numpy array
    """
    rng = _resolve_rng(rng, seed)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency that changes over time (pitch contour/vibrato)
    base_freq = 220 + base_freq_offset * np.sin(2 * np.pi * 0.5 * t)
    
    # Generate harmonic series with time-varying amplitudes
    signal = np.zeros_like(t)
    
    # Fundamental (optional)
    if include_fundamental:
        signal += 0.6 * np.sin(2 * np.pi * base_freq * t)
    
    # Harmonics with varying strengths
    for h in range(harmonic_range[0], harmonic_range[1]):
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
    envelope = np.ones_like(t)
    attack_idx = int(attack_time * sample_rate)
    decay_start_idx = int((duration - decay_time) * sample_rate)
    
    envelope[:attack_idx] = np.linspace(0, 1, attack_idx)
    envelope[decay_start_idx:] = np.linspace(1, 0, len(envelope) - decay_start_idx)
    
    signal *= envelope
    
    # Add noise for realism
    signal += noise_level * rng.standard_normal(len(signal))
    
    # Normalize if requested
    if normalize:
        signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    
    return signal


def create_rich_test_signal(sample_rate=16000, duration=2.0, rng=None, seed=None):
    """
    Create a rich harmonic signal for demonstration.
    
    This variant includes the fundamental and more harmonics.
    Used primarily in enhanced capabilities demos.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        signal: Generated rich test signal as numpy array
    """
    return create_test_signal(
        sample_rate=sample_rate,
        duration=duration,
        base_freq_offset=30,
        harmonic_range=(1, 7),
        include_fundamental=False,  # Harmonics start from 1 in range
        attack_time=0.15,
        decay_time=0.3,
        noise_level=0.005,
        normalize=True,
        rng=rng,
        seed=seed,
    )


def create_simple_test_signal(sample_rate=16000, duration=1.0, rng=None, seed=None):
    """
    Create a simple test signal for quick testing.
    
    This is a lighter-weight version optimized for unit tests
    that need to run quickly.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        signal: Generated simple test signal as numpy array
    """
    rng = _resolve_rng(rng, seed)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simpler harmonic signal
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
    signal += 0.01 * rng.standard_normal(len(signal))
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    
    return signal


def create_fidelity_demo_signal(sample_rate=16000, duration=3.0, rng=None, seed=None):
    """
    Create a musical test signal for fidelity demonstrations.
    
    This variant has different vibrato and envelope characteristics
    optimized for showing the fidelity parameter effects.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        signal: Generated test signal as numpy array
    """
    rng = _resolve_rng(rng, seed)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Musical phrase with variation
    base_freq = 220 + 30 * np.sin(2 * np.pi * 0.3 * t)  # Slower vibrato
    
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
    
    # Different envelope
    attack = 0.1
    decay = 0.2
    envelope = np.ones_like(t)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    signal *= envelope
    
    # Add slight noise
    signal += 0.005 * rng.standard_normal(len(signal))
    
    # Normalize with safety guard
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    
    return signal
