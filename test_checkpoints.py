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
Tests for safetensor checkpoint save/load functionality.
"""

import tempfile
from pathlib import Path

import numpy as np

from ssgs import SpectralStateGuidedSynthesis
from test_utils import create_test_signal


def test_checkpoint_save_load():
    """Test that we can save and load checkpoints in safetensors format."""
    print("=" * 60)
    print("Test: Checkpoint Save/Load")
    print("=" * 60)
    
    # Create and train a simple model
    print("\n1. Creating and training model...")
    sample_rate = 16000
    duration = 1.0
    signal = create_test_signal(duration, sample_rate)
    
    ssgs = SpectralStateGuidedSynthesis(
        n_states=8,  # Small for fast testing
        lpc_order=12,
        frame_size=512,
        hop_size=128,
    )
    
    ssgs.train(signal, sample_rate, n_em_iterations=3)
    print("   ✓ Model trained")
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.safetensors"
        
        print(f"\n2. Saving checkpoint to: {checkpoint_path}")
        ssgs.save_checkpoint(checkpoint_path)
        print(f"   ✓ Checkpoint saved ({checkpoint_path.stat().st_size} bytes)")
        
        # Load checkpoint
        print("\n3. Loading checkpoint...")
        loaded_ssgs = SpectralStateGuidedSynthesis.load_checkpoint(checkpoint_path)
        print("   ✓ Checkpoint loaded")
        
        # Verify parameters match
        print("\n4. Verifying loaded parameters...")
        assert loaded_ssgs.n_states == ssgs.n_states, "n_states mismatch"
        assert loaded_ssgs.lpc_order == ssgs.lpc_order, "lpc_order mismatch"
        assert loaded_ssgs.frame_size == ssgs.frame_size, "frame_size mismatch"
        assert loaded_ssgs.hop_size == ssgs.hop_size, "hop_size mismatch"
        
        # Check array equality
        np.testing.assert_allclose(
            loaded_ssgs.transition_matrix,
            ssgs.transition_matrix,
            rtol=1e-5,
            err_msg="Transition matrix mismatch"
        )
        print("   ✓ transition_matrix matches")
        
        np.testing.assert_allclose(
            loaded_ssgs.initial_probabilities,
            ssgs.initial_probabilities,
            rtol=1e-5,
            err_msg="Initial probabilities mismatch"
        )
        print("   ✓ initial_probabilities match")
        
        np.testing.assert_allclose(
            loaded_ssgs.state_means,
            ssgs.state_means,
            rtol=1e-5,
            err_msg="State means mismatch"
        )
        print("   ✓ state_means match")
        
        # Note: Slight differences in covariances are expected due to regularization
        # applied after loading to ensure positive definiteness
        np.testing.assert_allclose(
            loaded_ssgs.state_covariances,
            ssgs.state_covariances,
            rtol=0.01,  # 1% tolerance to account for regularization
            atol=1e-4,  # Absolute tolerance for small values
            err_msg="State covariances mismatch (beyond expected regularization)"
        )
        print("   ✓ state_covariances match")
        
        # Check training artifacts if present
        if ssgs.lpc_coefficients is not None:
            np.testing.assert_allclose(
                loaded_ssgs.lpc_coefficients,
                ssgs.lpc_coefficients,
                rtol=1e-5,
                err_msg="LPC coefficients mismatch"
            )
            print("   ✓ lpc_coefficients match")
        
        print("\n5. Testing generation from loaded checkpoint...")
        gen_audio = loaded_ssgs.generate(duration_seconds=0.5, sample_rate=sample_rate)
        assert len(gen_audio) > 0, "Generated audio is empty"
        print(f"   ✓ Generated {len(gen_audio)} samples")
    
    print("\n" + "=" * 60)
    print("✓ All checkpoint tests passed!")
    print("=" * 60)


def test_checkpoint_without_training_artifacts():
    """Test checkpoint save/load without training artifacts."""
    print("\n" + "=" * 60)
    print("Test: Checkpoint Without Training Artifacts")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 0.5
    signal = create_test_signal(duration, sample_rate)
    
    print("\n1. Training model...")
    ssgs = SpectralStateGuidedSynthesis(n_states=8, lpc_order=12)
    ssgs.train(signal, sample_rate, n_em_iterations=2)
    print("   ✓ Model trained")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "minimal_checkpoint.safetensors"
        
        print("\n2. Saving checkpoint without training artifacts...")
        ssgs.save_checkpoint(checkpoint_path, include_training_artifacts=False)
        size_without = checkpoint_path.stat().st_size
        print(f"   ✓ Saved ({size_without} bytes)")
        
        print("\n3. Loading minimal checkpoint...")
        loaded = SpectralStateGuidedSynthesis.load_checkpoint(checkpoint_path)
        print("   ✓ Loaded successfully")
        
        print("\n4. Verifying core parameters preserved...")
        np.testing.assert_allclose(loaded.transition_matrix, ssgs.transition_matrix, rtol=1e-5)
        np.testing.assert_allclose(loaded.state_means, ssgs.state_means, rtol=1e-5)
        print("   ✓ Core parameters match")
        
        print("\n5. Generating audio from minimal checkpoint...")
        audio = loaded.generate(duration_seconds=0.5, sample_rate=sample_rate)
        assert len(audio) > 0
        print(f"   ✓ Generated {len(audio)} samples")
    
    print("\n" + "=" * 60)
    print("✓ Minimal checkpoint test passed!")
    print("=" * 60)


def test_checkpoint_file_format():
    """Verify that checkpoint files are valid safetensors format."""
    print("\n" + "=" * 60)
    print("Test: Safetensors File Format Validation")
    print("=" * 60)
    
    sample_rate = 16000
    signal = create_test_signal(0.5, sample_rate)
    
    print("\n1. Creating model...")
    ssgs = SpectralStateGuidedSynthesis(n_states=8, lpc_order=12)
    ssgs.train(signal, sample_rate, n_em_iterations=2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "format_test.safetensors"
        
        print("\n2. Saving checkpoint...")
        ssgs.save_checkpoint(checkpoint_path)
        
        print("\n3. Verifying file format...")
        # Check file exists and has content
        assert checkpoint_path.exists(), "Checkpoint file not created"
        file_size = checkpoint_path.stat().st_size
        assert file_size > 0, "Checkpoint file is empty"
        print(f"   ✓ File exists ({file_size} bytes)")
        
        # Verify safetensors header structure
        with open(checkpoint_path, 'rb') as f:
            # First 8 bytes are header size
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, byteorder='little')
            assert header_size > 0, "Invalid header size"
            print(f"   ✓ Valid safetensors header (size: {header_size} bytes)")
            
            # Read header
            header_bytes = f.read(header_size)
            import json
            header = json.loads(header_bytes.decode('utf-8'))
            
            # Check metadata
            assert '__metadata__' in header, "Missing metadata"
            metadata = header['__metadata__']
            assert metadata.get('format') == 'safetensors', "Wrong format"
            print("   ✓ Valid metadata")
            
            # Check tensors are listed
            tensor_keys = [k for k in header.keys() if k != '__metadata__']
            assert len(tensor_keys) > 0, "No tensors in checkpoint"
            print(f"   ✓ Contains {len(tensor_keys)} tensors")
            
            # Verify required tensors
            required = ['transition_matrix', 'initial_probabilities', 'state_means', 'state_covariances']
            for tensor_name in required:
                assert tensor_name in tensor_keys, f"Missing required tensor: {tensor_name}"
            print(f"   ✓ All required tensors present")
    
    print("\n" + "=" * 60)
    print("✓ File format validation passed!")
    print("=" * 60)


def run_all_tests():
    """Run all checkpoint tests."""
    print("\n" + "=" * 60)
    print("SSGS Checkpoint Tests")
    print("=" * 60)
    
    try:
        test_checkpoint_save_load()
        test_checkpoint_without_training_artifacts()
        test_checkpoint_file_format()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
