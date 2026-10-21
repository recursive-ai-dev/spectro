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


def _generate_ar_process(coeffs, n_samples, noise_std=0.05, seed=17):
    rng = np.random.default_rng(seed)
    order = len(coeffs)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    signal = np.zeros(n_samples, dtype=np.float64)
    for idx in range(order, n_samples):
        signal[idx] = np.dot(coeffs, signal[idx - order : idx][::-1]) + noise[idx]
    return signal.astype(np.float32, copy=False)


def test_ar4_recovery_with_single_state():
    ar_coeffs = np.array([0.65, -0.3, 0.2, -0.1], dtype=np.float64)
    audio = _generate_ar_process(ar_coeffs, n_samples=24000, noise_std=0.03, seed=91)

    model = SpectralStateGuidedSynthesis(
        n_states=1,
        lpc_order=4,
        frame_size=512,
        hop_size=256,
    )
    model.train(audio, sample_rate=16000, n_em_iterations=6)

    estimated = model.state_means[0]
    expected = -ar_coeffs
    assert np.allclose(estimated, expected, atol=0.05), (
        f"Expected LPC mean near {expected}, got {estimated}"
    )
