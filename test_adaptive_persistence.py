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

from ssgs import SpectralStateGuidedSynthesis
from test_utils import create_simple_test_signal, create_test_signal


def assert_not_allclose(a, b, tol=1e-4):
    if np.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError("Expected arrays to differ after adaptation")


def run_adaptation_persistence_test():
    np.random.seed(0)

    sample_rate = 12000
    training_signal = create_simple_test_signal(sample_rate=sample_rate, duration=1.5)
    adaptation_signal = create_test_signal(
        sample_rate=sample_rate,
        duration=1.0,
        base_freq_offset=80,
        harmonic_range=(2, 5),
        noise_level=0.003,
        normalize=True,
    )

    ssgs = SpectralStateGuidedSynthesis(
        n_states=6,
        lpc_order=8,
        frame_size=512,
        hop_size=128,
        adaptive_memory_limit=800,
    )

    ssgs.train(training_signal, sample_rate=sample_rate, n_em_iterations=2)
    transition_before = ssgs.transition_matrix.copy()
    memory_before = ssgs.training_frames.shape[0]

    metrics = ssgs.adapt_to_audio(
        adaptation_signal,
        sample_rate=sample_rate,
        adaptation_rate=0.35,
        stability_bias=0.05,
        memory_blend=0.6,
        memory_limit=600,
        n_adaptation_iterations=1,
    )

    assert metrics["frames_used"] > 0
    assert_not_allclose(transition_before, ssgs.transition_matrix)
    assert ssgs.training_frames.shape[0] <= 600
    assert ssgs.training_frames.shape[0] != memory_before

    export_path = "adaptive_model_test.npz"
    ssgs.export_model(export_path, include_training_artifacts=False, include_adaptive_statistics=True)
    reloaded = SpectralStateGuidedSynthesis.load_model(export_path)
    if reloaded._adaptive_stats is None:
        raise AssertionError("Adaptive statistics not restored from export")
    if reloaded._adaptive_stats.state_frame_counts.shape[0] != ssgs.n_states:
        raise AssertionError("Adaptive statistics shape mismatch after reload")

    # Exercise adaptation rate boundaries (-1 to 12)
    for rate in range(-1, 13):
        if 0 <= rate <= 1:
            ssgs.adapt_to_audio(
                adaptation_signal,
                sample_rate=sample_rate,
                adaptation_rate=float(rate),
                stability_bias=0.05,
                memory_blend=0.6,
                memory_limit=600,
                n_adaptation_iterations=0,
            )
        else:
            try:
                ssgs.adapt_to_audio(
                    adaptation_signal,
                    sample_rate=sample_rate,
                    adaptation_rate=float(rate),
                    stability_bias=0.05,
                    memory_blend=0.6,
                    memory_limit=600,
                    n_adaptation_iterations=0,
                )
            except ValueError:
                continue
            raise AssertionError(f"Expected ValueError for adaptation_rate={rate}")


if __name__ == "__main__":
    run_adaptation_persistence_test()
    print("Adaptive persistence tests passed.")
