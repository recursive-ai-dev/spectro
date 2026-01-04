# SSGS Training Workflow Demo

This document demonstrates the complete training workflow with safetensor checkpoints.

## Step 1: Training on Audio Files

```bash
# Train on audio files from training folders
python train_on_folder.py --folders training_001 training_002 \
    --n-iterations 20 \
    --checkpoint-interval 5 \
    --n-states 34
```

**Output:**
- `models/checkpoint_initial.safetensors` - After initialization
- `models/checkpoint_epoch_5.safetensors` - After 5 iterations
- `models/checkpoint_epoch_10.safetensors` - After 10 iterations
- `models/checkpoint_epoch_15.safetensors` - After 15 iterations
- `models/checkpoint_epoch_20.safetensors` - After 20 iterations
- `models/checkpoint_final.safetensors` - Final trained model
- `models/model_final.npz` - Final model in .npz format (for compatibility)

## Step 2: Load and Generate

```python
from ssgs import SpectralStateGuidedSynthesis
import soundfile as sf

# Load the final checkpoint
model = SpectralStateGuidedSynthesis.load_checkpoint(
    'models/checkpoint_final.safetensors'
)

# Generate synthetic audio (novel)
synthetic = model.generate(
    duration_seconds=5.0,
    sample_rate=16000,
    fidelity=0.0,  # Fully synthetic
    excitation_type='Karplus-Strong'
)
sf.write('outputs/synthetic.wav', synthetic, 16000)

# Generate high-fidelity reconstruction
reconstruction = model.generate(
    duration_seconds=5.0,
    sample_rate=16000,
    fidelity=1.0  # High fidelity
)
sf.write('outputs/reconstruction.wav', reconstruction, 16000)

# Generate balanced mix
balanced = model.generate(
    duration_seconds=5.0,
    sample_rate=16000,
    fidelity=0.5  # 50/50 mix
)
sf.write('outputs/balanced.wav', balanced, 16000)
```

## Step 3: Resume Training

```python
# Resume from a checkpoint to continue training
python train_on_folder.py \
    --resume models/checkpoint_epoch_10.safetensors \
    --n-iterations 10  # Train 10 more iterations
```

## Step 4: Upload/Download (Cloud Integration)

```python
# Example: Upload to AWS S3
import boto3
from pathlib import Path

s3 = boto3.client('s3')

# Upload checkpoint
checkpoint_path = 'models/checkpoint_final.safetensors'
s3.upload_file(
    checkpoint_path,
    'my-model-bucket',
    'ssgs/production/v1.0/checkpoint_final.safetensors'
)

# Download checkpoint
s3.download_file(
    'my-model-bucket',
    'ssgs/production/v1.0/checkpoint_final.safetensors',
    'downloaded_checkpoint.safetensors'
)

# Load and use downloaded checkpoint
from ssgs import SpectralStateGuidedSynthesis
model = SpectralStateGuidedSynthesis.load_checkpoint(
    'downloaded_checkpoint.safetensors'
)
audio = model.generate(duration_seconds=3.0)
```

## Step 5: Verify Checkpoints

```python
# Test that checkpoints work correctly
python test_checkpoints.py
```

**Expected output:**
```
============================================================
SSGS Checkpoint Tests
============================================================
✓ All checkpoint tests passed!
============================================================
```

## Production Workflow

1. **Training**: Train on large audio dataset
2. **Checkpointing**: Save checkpoints every N iterations
3. **Validation**: Generate test audio after each checkpoint
4. **Upload**: Upload best checkpoint to cloud storage
5. **Deployment**: Download checkpoint in production environment
6. **Inference**: Load checkpoint and generate audio

## File Formats

### Safetensors (.safetensors)
- **Recommended format** for new projects
- Secure (no code execution)
- Fast loading
- Cross-platform
- Verifiable checksums

### NumPy (.npz)
- **Legacy format** for compatibility
- Compressed by default
- Python-specific
- Still supported

Both formats contain the same model data and can be used interchangeably.

## Checkpoint Size Reference

| Configuration | Frames | States | Size (with artifacts) | Size (minimal) |
|--------------|--------|--------|----------------------|----------------|
| Small | 100 | 8 | ~2 MB | ~6 KB |
| Medium | 1,000 | 16 | ~20 MB | ~12 KB |
| Large | 10,000 | 34 | ~200 MB | ~30 KB |
| Very Large | 70,000+ | 34 | ~2 GB | ~30 KB |

**Note:** Minimal checkpoints only store model parameters, not training artifacts.
