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
Training script for SSGS model on audio files from training folders.

This script:
1. Loads all audio files from specified training folders
2. Trains SSGS model on the combined audio
3. Saves checkpoints periodically during training
4. Exports final model in both .npz and .safetensors formats

Usage:
    python train_on_folder.py --folders training_001 training_002
    python train_on_folder.py --folders training_001 --checkpoint-interval 5
    python train_on_folder.py --resume models/checkpoint_epoch_5.safetensors
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ssgs import SpectralStateGuidedSynthesis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_audio_files(
    folders: List[str],
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """Find supported audio files case-insensitively and return a stable unique list."""
    if extensions is None:
        extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]
    normalized_extensions = {
        (extension if str(extension).startswith(".") else f".{extension}").casefold()
        for extension in extensions
    }
    if not normalized_extensions:
        raise ValueError("extensions must contain at least one file suffix")

    discovered = set()
    for folder_name in folders:
        folder = Path(folder_name)
        if not folder.exists():
            logger.warning("Folder not found: %s", folder)
            continue
        if not folder.is_dir():
            logger.warning("Not a directory: %s", folder)
            continue
        candidates = folder.rglob("*") if recursive else folder.glob("*")
        matches = [path for path in candidates if path.is_file() and path.suffix.casefold() in normalized_extensions]
        discovered.update(matches)
        logger.info("Found %d supported audio files in %s", len(matches), folder)
    return sorted(discovered, key=lambda path: (str(path).casefold(), str(path)))


def train_model(
    audio_files: List[Path],
    sample_rate: int = 16000,
    n_states: int = 34,
    n_em_iterations: int = 20,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 5,
    resume_from: Optional[Path] = None,
    random_state: Optional[int] = 0,
) -> SpectralStateGuidedSynthesis:
    """
    Train SSGS model on audio files with checkpointing.
    
    Args:
        audio_files: List of audio file paths
        sample_rate: Target sample rate for audio
        n_states: Number of HMM states
        n_em_iterations: Number of EM iterations
        checkpoint_dir: Directory to save checkpoints (default: models/)
        checkpoint_interval: Save checkpoint every N iterations
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained SSGS model
    """
    if not audio_files:
        raise ValueError(
            "No audio files provided for training. "
            "Check folder paths, extensions, or enable recursive search."
        )
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states <= 0:
        raise ValueError("n_states must be a positive integer")
    if isinstance(n_em_iterations, bool) or not isinstance(n_em_iterations, int) or n_em_iterations < 0:
        raise ValueError("n_em_iterations must be a non-negative integer")
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int) or checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be a non-negative integer")
    if resume_from is not None and not Path(resume_from).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    if checkpoint_dir is None:
        checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize or load model
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ssgs = SpectralStateGuidedSynthesis.load_checkpoint(resume_from)
        if ssgs.lpc_coefficients is None or len(ssgs.lpc_coefficients) == 0:
            logger.info("Checkpoint does not contain cached training features. Re-extracting from audio files...")
            audio_file_strs = [str(f) for f in audio_files]
            ssgs.extract_features_from_files(audio_file_strs, sample_rate)
        logger.info("Model loaded and ready for EM refinement.")
    else:
        logger.info("Initializing new model")
        ssgs = SpectralStateGuidedSynthesis(
            n_states=n_states,
            lpc_order=12,
            frame_size=1024,
            hop_size=256,
            smoothness_weight=0.5,
            random_state=random_state,
        )
        
        # Feature extraction
        logger.info(f"Processing {len(audio_files)} audio files...")
        logger.info("Phase 1: Feature Extraction")
        logger.info("=" * 60)
        
        # Convert paths to strings for the model
        audio_file_strs = [str(f) for f in audio_files]
        
        # Extract features from all files
        ssgs.extract_features_from_files(audio_file_strs, sample_rate)
        logger.info(f"Total frames extracted: {len(ssgs.lpc_coefficients)}")
        
        # Initialize HMM parameters
        logger.info("\nPhase 2: HMM Initialization")
        logger.info("=" * 60)
        ssgs.initialize_hmm_parameters()
        
        # Save initial checkpoint
        initial_checkpoint = checkpoint_dir / "checkpoint_initial.safetensors"
        ssgs.save_checkpoint(initial_checkpoint)
        logger.info(f"Initial checkpoint saved: {initial_checkpoint}")
    
    # EM Training with checkpointing
    logger.info("\nPhase 3: EM Training with Checkpointing")
    logger.info("=" * 60)
    logger.info(f"Running {n_em_iterations} EM iterations")
    logger.info(f"Checkpoints will be saved every {checkpoint_interval} iterations")
    
    # Custom training loop with checkpointing
    for iteration in range(n_em_iterations):
        # Run single EM iteration via the public API
        log_likelihood, entropy, smooth_penalty = ssgs.run_em_iteration()
        
        logger.info(
            f"Iteration {iteration + 1}/{n_em_iterations}: "
            f"Log-likelihood = {log_likelihood:.2f} | "
            f"Entropy = {entropy:.3f} | "
            f"Smoothness = {smooth_penalty:.3f}"
        )
        
        # Save checkpoint at intervals
        if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{iteration + 1}.safetensors"
            ssgs.save_checkpoint(checkpoint_path)
            logger.info(f"  → Checkpoint saved: {checkpoint_path}")
    
    # Apply graph constraints
    logger.info("\nPhase 4: Graph Constraints")
    logger.info("=" * 60)
    ssgs.identify_graph_constraints()
    
    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.safetensors"
    ssgs.save_checkpoint(final_checkpoint)
    logger.info(f"Final checkpoint saved: {final_checkpoint}")
    
    # Also save in .npz format for compatibility
    final_npz = checkpoint_dir / "model_final.npz"
    ssgs.export_model(final_npz, include_training_artifacts=True)
    logger.info(f"Final model (.npz) saved: {final_npz}")
    
    logger.info("\nTraining complete!")
    return ssgs


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train SSGS model on audio files with safetensor checkpointing"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["training_001", "training_002"],
        help="Folders containing training audio files (default: training_001 training_002)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=34,
        help="Number of HMM states (default: 34)"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=20,
        help="Number of EM iterations (default: 20)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save checkpoints (default: models/)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N iterations (default: 5, 0 to disable)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp3", ".wav", ".flac", ".ogg"],
        help="Audio file extensions to process (default: .mp3 .wav .flac .ogg)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible model initialization (default: 0)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for audio files recursively within folders",
    )
    
    args = parser.parse_args()
    
    # Find audio files
    logger.info("Searching for audio files...")
    audio_files = find_audio_files(
        args.folders,
        args.extensions,
        recursive=args.recursive,
    )
    
    if not audio_files:
        logger.error(
            "No audio files found! Please check folder paths, extensions, "
            "or enable recursive search."
        )
        return 1
    
    logger.info(f"Found {len(audio_files)} audio files")
    for i, file in enumerate(audio_files[:5]):  # Show first 5
        logger.info(f"  {i+1}. {file.name}")
    if len(audio_files) > 5:
        logger.info(f"  ... and {len(audio_files) - 5} more")
    
    # Train model
    try:
        model = train_model(
            audio_files=audio_files,
            sample_rate=args.sample_rate,
            n_states=args.n_states,
            n_em_iterations=args.n_iterations,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume,
            random_state=args.seed,
        )
        
        # Generate a test sample
        logger.info("\nGenerating test sample...")
        test_audio = model.generate(
            duration_seconds=3.0,
            sample_rate=args.sample_rate,
            fidelity=0.0  # Fully synthetic
        )
        
        # Save test audio
        test_output = args.checkpoint_dir / "test_generation.wav"
        try:
            import soundfile as sf
            sf.write(test_output, test_audio, args.sample_rate)
            logger.info(f"Test audio saved: {test_output}")
        except ImportError:
            logger.warning("soundfile not available, skipping test audio save")
        
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS: Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Checkpoints saved in: {args.checkpoint_dir}")
        logger.info(f"To generate audio, use the final checkpoint:")
        logger.info(f"  {args.checkpoint_dir / 'checkpoint_final.safetensors'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
