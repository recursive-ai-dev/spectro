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
Example: Train SSGS model on audio files and use safetensor checkpoints.

This example demonstrates:
1. Training on multiple audio files from folders
2. Saving periodic checkpoints during training
3. Loading checkpoints to resume or generate audio
4. Uploading/downloading checkpoints (save to local directory)
"""

from pathlib import Path

import numpy as np

from ssgs import SpectralStateGuidedSynthesis


def example_train_and_checkpoint():
    """Example: Train model and save checkpoints."""
    print("=" * 60)
    print("Example: Training with Checkpoints")
    print("=" * 60)
    
    # Setup
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create synthetic training data (in practice, use real audio files)
    print("\n1. Generating synthetic training data...")
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create multiple audio signals with different characteristics
    signals = []
    for freq in [220, 330, 440]:  # Different fundamental frequencies
        signal = (
            0.5 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * freq * 3 * t)
        )
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        signal *= envelope
        signals.append(signal)
    
    print(f"   Created {len(signals)} training signals")
    
    # Train model
    print("\n2. Training model with checkpointing...")
    ssgs = SpectralStateGuidedSynthesis(
        n_states=16,
        lpc_order=12,
        frame_size=1024,
        hop_size=256,
    )
    
    # Train on multiple signals
    ssgs.train(signals, sample_rate, n_em_iterations=10)
    
    # Save checkpoint after training
    checkpoint_path = checkpoint_dir / "example_checkpoint.safetensors"
    print(f"\n3. Saving checkpoint to {checkpoint_path}...")
    ssgs.save_checkpoint(checkpoint_path)
    print("   ✓ Checkpoint saved")
    
    return checkpoint_path


def example_load_and_generate(checkpoint_path):
    """Example: Load checkpoint and generate audio."""
    print("\n" + "=" * 60)
    print("Example: Load Checkpoint and Generate")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from {checkpoint_path}...")
    ssgs = SpectralStateGuidedSynthesis.load_checkpoint(checkpoint_path)
    print("   ✓ Checkpoint loaded")
    print(f"   Model has {ssgs.n_states} states")
    
    # Generate audio with different fidelity levels
    sample_rate = 16000
    
    print("\n2. Generating audio samples...")
    
    # Fully synthetic
    print("   - Generating synthetic audio (fidelity=0.0)...")
    synthetic = ssgs.generate(
        duration_seconds=2.0,
        sample_rate=sample_rate,
        fidelity=0.0
    )
    print(f"     ✓ Generated {len(synthetic)} samples")
    
    # Balanced
    print("   - Generating balanced audio (fidelity=0.5)...")
    balanced = ssgs.generate(
        duration_seconds=2.0,
        sample_rate=sample_rate,
        fidelity=0.5
    )
    print(f"     ✓ Generated {len(balanced)} samples")
    
    # High fidelity
    print("   - Generating high-fidelity audio (fidelity=1.0)...")
    hifi = ssgs.generate(
        duration_seconds=2.0,
        sample_rate=sample_rate,
        fidelity=1.0
    )
    print(f"     ✓ Generated {len(hifi)} samples")
    
    # Save generated audio
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("\n3. Saving generated audio...")
    try:
        import soundfile as sf
        
        sf.write(output_dir / "generated_synthetic.wav", synthetic, sample_rate)
        sf.write(output_dir / "generated_balanced.wav", balanced, sample_rate)
        sf.write(output_dir / "generated_hifi.wav", hifi, sample_rate)
        
        print(f"   ✓ Audio saved to {output_dir}/")
        print("     - generated_synthetic.wav (novel generation)")
        print("     - generated_balanced.wav (mixed)")
        print("     - generated_hifi.wav (reconstruction)")
    except ImportError:
        print("   ! soundfile not available, skipping audio save")
    
    return synthetic, balanced, hifi


def example_checkpoint_upload_download():
    """
    Example: 'Upload' and 'Download' checkpoints (local directory simulation).
    
    In practice, you would integrate with cloud storage (S3, GCS, etc.)
    or a model registry. This example shows the local file operations.
    """
    print("\n" + "=" * 60)
    print("Example: Checkpoint Upload/Download (Local)")
    print("=" * 60)
    
    # Simulate upload: copy checkpoint to 'uploaded' directory
    upload_dir = Path("uploaded_checkpoints")
    upload_dir.mkdir(exist_ok=True)
    
    local_checkpoint = Path("models/example_checkpoint.safetensors")
    
    if not local_checkpoint.exists():
        print("\n   ! Checkpoint not found. Run example_train_and_checkpoint() first.")
        return
    
    print(f"\n1. 'Uploading' checkpoint (copying to {upload_dir})...")
    uploaded_checkpoint = upload_dir / local_checkpoint.name
    
    import shutil
    shutil.copy2(local_checkpoint, uploaded_checkpoint)
    print(f"   ✓ Checkpoint 'uploaded' to {uploaded_checkpoint}")
    print(f"   Size: {uploaded_checkpoint.stat().st_size / 1024:.1f} KB")
    
    # Simulate download: load from 'uploaded' directory
    print(f"\n2. 'Downloading' checkpoint (loading from {upload_dir})...")
    ssgs = SpectralStateGuidedSynthesis.load_checkpoint(uploaded_checkpoint)
    print("   ✓ Checkpoint 'downloaded' and loaded")
    print(f"   Model ready with {ssgs.n_states} states")
    
    # Verify it works
    print("\n3. Verifying downloaded checkpoint...")
    test_audio = ssgs.generate(duration_seconds=1.0, sample_rate=16000)
    print(f"   ✓ Generated {len(test_audio)} samples from downloaded checkpoint")
    
    print("\nNote: In production, integrate with:")
    print("  - Cloud storage (S3, GCS, Azure Blob)")
    print("  - Model registries (MLflow, Weights & Biases)")
    print("  - Version control for models (DVC, Git LFS)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" SSGS Checkpoint Examples")
    print("=" * 70)
    
    # Example 1: Train and save
    checkpoint_path = example_train_and_checkpoint()
    
    # Example 2: Load and generate
    example_load_and_generate(checkpoint_path)
    
    # Example 3: Upload/Download simulation
    example_checkpoint_upload_download()
    
    print("\n" + "=" * 70)
    print(" Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train on real audio: python train_on_folder.py --folders training_001")
    print("  2. Load checkpoint: ssgs = SpectralStateGuidedSynthesis.load_checkpoint('models/checkpoint_final.safetensors')")
    print("  3. Generate audio: audio = ssgs.generate(duration_seconds=5.0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
