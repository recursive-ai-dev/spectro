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

"""SSGS — Spectral-State Guided Synthesis.

Hybrid generative audio model combining linear predictive coding (LPC),
hidden Markov models (HMMs), spectral clustering, global graph search,
and a physically-inspired Karplus-Strong excitation engine.

Module layout:
    ssgs.model   — full model implementation (SpectralStateGuidedSynthesis)
    ssgs.core    — dependency-light numeric foundation (pure NumPy)
    ssgs.native  — ctypes wrapper for the compiled C kernels (ssgs_native.c)
"""

from .model import (
    FeatureIndex,
    FrameQueryResult,
    SpectralStateGuidedSynthesis,
    __version__,
)

__all__ = [
    "FeatureIndex",
    "FrameQueryResult",
    "SpectralStateGuidedSynthesis",
    "__version__",
]
