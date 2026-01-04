# Training SSGS Models with Safetensor Checkpoints

This document describes how to train SSGS models on audio files with persistent checkpointing using the safetensors format.

## Overview

SSGS now supports:
- **Training on multiple audio files** from directories
- **Persistent checkpointing** during training using safetensors format
- **Checkpoint upload/download** (save/load from local directories or cloud storage)
- **Verified safetensor format** ensuring security and reproducibility

## Quick Start

### 1. Train on Audio Files

Train a model on MP3 files in the `training_001` and `training_002` folders:

```bash
python train_on_folder.py --folders training_001 training_002
```

This will:
- Extract features from all audio files
- Train the SSGS model with EM iterations
- Save checkpoints periodically during training
- Export the final model in both `.safetensors` and `.npz` formats

### 2. Use Checkpoints in Python

```python
from ssgs import SpectralStateGuidedSynthesis

# Load a trained checkpoint
model = SpectralStateGuidedSynthesis.load_checkpoint('models/checkpoint_final.safetensors')

# Generate audio
audio = model.generate(
    duration_seconds=5.0,
    sample_rate=16000,
    fidelity=0.0  # 0.0=synthetic, 1.0=reconstruction
)

# Save audio
import soundfile as sf
sf.write('output.wav', audio, 16000)
```

## Training Script Options

The `train_on_folder.py` script supports the following options:

```bash
python train_on_folder.py [OPTIONS]

Options:
  --folders FOLDER [FOLDER ...]
                        Folders containing training audio files
                        (default: training_001 training_002)
  
  --sample-rate RATE    Target sample rate (default: 16000)
  
  --n-states N          Number of HMM states (default: 34)
                        More states = more expressive generation
  
  --n-iterations N      Number of EM iterations (default: 20)
  
  --checkpoint-dir DIR  Directory to save checkpoints (default: models/)
  
  --checkpoint-interval N
                        Save checkpoint every N iterations (default: 5)
                        Set to 0 to disable periodic checkpoints
  
  --resume PATH         Resume training from checkpoint
  
  --extensions EXT [EXT ...]
                        Audio file extensions to process
                        (default: .mp3 .wav .flac .ogg)
```

### Examples

```bash
# Train on custom folders with more iterations
python train_on_folder.py --folders my_audio --n-iterations 30

# Train with frequent checkpoints
python train_on_folder.py --checkpoint-interval 2

# Resume from checkpoint
python train_on_folder.py --resume models/checkpoint_epoch_10.safetensors

# Train with more states for higher quality
python train_on_folder.py --n-states 42
```

## Checkpoint Format

Checkpoints are saved in the **safetensors** format, which provides:

- **Security**: No arbitrary code execution (unlike pickle)
- **Speed**: Fast loading with zero-copy when possible
- **Portability**: Works across platforms and languages
- **Verification**: Built-in checksums ensure data integrity

### Checkpoint Contents

Each checkpoint contains:
- **Model parameters**: Transition matrix, state means, covariances
- **Training artifacts**: LPC coefficients, residual signals, frame metadata
- **Configuration**: Model hyperparameters and settings

### File Sizes

- **Minimal checkpoint** (without training artifacts): ~6 KB (8 states)
- **Full checkpoint** (with training artifacts): ~1.7 MB per 100 frames
- For large datasets (70K+ frames), checkpoints can be ~2 GB

## Checkpoint Operations

### Save Checkpoint

```python
from ssgs import SpectralStateGuidedSynthesis

# Train model
ssgs = SpectralStateGuidedSynthesis(n_states=34)
ssgs.train(audio_signal, sample_rate=16000, n_em_iterations=20)

# Save checkpoint
ssgs.save_checkpoint(
    'models/my_checkpoint.safetensors',
    include_training_artifacts=True  # Include for reconstruction ability
)
```

### Load Checkpoint

```python
# Load from checkpoint
ssgs = SpectralStateGuidedSynthesis.load_checkpoint('models/my_checkpoint.safetensors')

# Model is ready to use
audio = ssgs.generate(duration_seconds=3.0)
```

### Upload/Download Checkpoints

For production use, integrate with cloud storage:

```python
import boto3  # AWS S3 example
from pathlib import Path

# Save checkpoint locally
ssgs.save_checkpoint('checkpoint.safetensors')

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file('checkpoint.safetensors', 'my-bucket', 'models/checkpoint.safetensors')

# Download from S3
s3.download_file('my-bucket', 'models/checkpoint.safetensors', 'downloaded.safetensors')

# Load downloaded checkpoint
ssgs = SpectralStateGuidedSynthesis.load_checkpoint('downloaded.safetensors')
```

## Testing

Run the checkpoint tests to verify functionality:

```bash
# Test checkpoint save/load
python test_checkpoints.py

# Run example demonstrations
python example_checkpoints.py
```

## Compatibility

- **Safetensors format** is platform-independent
- Checkpoints can be loaded in different environments
- Compatible with model registries (MLflow, Weights & Biases)
- Can be versioned with DVC or Git LFS

## Best Practices

1. **Save checkpoints frequently** during long training runs
2. **Include training artifacts** for high-fidelity reconstruction
3. **Version your checkpoints** with meaningful names (e.g., `model_v1.2.safetensors`)
4. **Use cloud storage** for backup and sharing
5. **Test checkpoints** after saving to ensure they load correctly

## Troubleshooting

### Out of Memory

If training on large datasets causes memory issues:
- Reduce `--n-states` (try 16 or 24 instead of 34)
- Process fewer files at once
- Use a machine with more RAM

### Checkpoint Won't Load

If a checkpoint fails to load:
- Verify the file is not corrupted (check file size)
- Ensure safetensors is installed: `pip install safetensors`
- Check for version compatibility

### Generation Quality

For better generation quality:
- Use more training data
- Increase `--n-states` (try 34 or 42)
- Run more EM iterations (`--n-iterations 30`)
- Adjust fidelity parameter in `generate()`

## See Also

- [SSGS Main README](README.md) - General SSGS documentation
- [Enhanced Capabilities](ENHANCED_CAPABILITIES.md) - Advanced features
- [train_on_folder.py](train_on_folder.py) - Training script source
- [example_checkpoints.py](example_checkpoints.py) - Example code
