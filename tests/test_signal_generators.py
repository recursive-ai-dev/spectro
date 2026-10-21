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

"""Verify that all signal generators in ssgs.signals produce valid signals."""

import numpy as np

from ssgs.signals import (
    create_fidelity_demo_signal,
    create_rich_test_signal,
    create_simple_test_signal,
    create_test_signal,
)


def test_signal_generators():
    sr = 16000
    for gen_fn in [create_test_signal, create_rich_test_signal, create_simple_test_signal, create_fidelity_demo_signal]:
        sig = gen_fn(sample_rate=sr, duration=0.2, seed=42)
        assert len(sig) == int(sr * 0.2)
        assert not np.isnan(sig).any()
        assert not np.isinf(sig).any()
        assert np.max(np.abs(sig)) > 0.0
