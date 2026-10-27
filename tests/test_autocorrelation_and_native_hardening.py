#!/usr/bin/env python3
"""Regression tests from the adversarial / alternative-method evaluation.

Covers (see docs/ADVERSARIAL_EVALUATION.md):
  * ``_autocorrelation`` -- lag-limited replacement for the full
    ``np.correlate(mode="full")``: equivalence with the old method on
    adversarial frames, in both the direct (order <= 64) and FFT branches.
  * ``ssgs.native`` -- import must survive a missing .so, and every wrapper
    must validate before crossing the ctypes boundary (previously shape
    mismatches returned garbage from out-of-bounds reads, non-PSD
    covariances returned NaNs, and all -inf forward-backward returned NaNs).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ssgs.model import SpectralStateGuidedSynthesis

SRC = Path(__file__).resolve().parents[1] / "src"


def _full_correlate_reference(frame, order):
    corr = np.correlate(frame, frame, mode="full")
    mid = len(frame) - 1
    return corr[mid:mid + order + 1]


def _model(order=12, frame_size=512):
    return SpectralStateGuidedSynthesis(
        n_states=4, lpc_order=order, frame_size=frame_size, hop_size=128)


@pytest.mark.parametrize("order", [1, 4, 12, 32, 64, 65, 128, 200])
@pytest.mark.parametrize("kind", ["noise", "zero", "constant", "impulse", "sine", "alternating"])
def test_autocorrelation_matches_full_correlate(order, kind):
    rng = np.random.default_rng(order)
    n = 1024
    frame = {
        "noise": rng.standard_normal(n),
        "zero": np.zeros(n),
        "constant": np.full(n, 3.5),
        "impulse": np.eye(1, n, 0).ravel(),
        "sine": np.sin(2 * np.pi * 5 * np.arange(n) / n),
        "alternating": (-1.0) ** np.arange(n),
    }[kind]
    got = _model(order=order, frame_size=n)._autocorrelation(frame)
    ref = _full_correlate_reference(frame, order)
    assert got.shape == (order + 1,)
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-8 * max(1.0, ref[0]))


def test_autocorrelation_lpc_end_to_end_unchanged():
    rng = np.random.default_rng(7)
    m = _model()
    for _ in range(10):
        frame = rng.standard_normal(512) * np.hanning(512)
        autocorr_old = _full_correlate_reference(frame, m.lpc_order)
        from ssgs.core import lpc_coefficients
        assert np.allclose(lpc_coefficients(m._autocorrelation(frame), m.lpc_order),
                           lpc_coefficients(autocorr_old, m.lpc_order), atol=1e-10)
        assert np.allclose(m._compute_lpc(frame),
                           lpc_coefficients(autocorr_old, m.lpc_order), atol=1e-5)


def test_autocorrelation_rejects_adversarial_frames():
    m = _model()
    with pytest.raises(ValueError):
        m._autocorrelation(np.full(512, np.nan))
    with pytest.raises(ValueError):
        m._autocorrelation(np.zeros((2, 512)))
    with pytest.raises(ValueError):
        m._autocorrelation(np.zeros(8))  # not longer than lpc_order


# ---------------------------------------------------------------- native ---
native = pytest.importorskip("ssgs.native")

pytestmark_native = pytest.mark.skipif(native._lib is None, reason="native .so not built")


@pytestmark_native
def test_native_gaussian_non_psd_matches_core_oracle():
    from ssgs import core
    means, obs = np.zeros((1, 2)), np.zeros((2, 2))
    bad_cov = np.array([[[-1.0, 0.0], [0.0, -1.0]]])
    got = native.gaussian_log_probs(means, bad_cov, obs)
    assert np.all(np.isfinite(got))
    assert np.allclose(got, core.gaussian_log_probs(means, bad_cov, obs), atol=1e-6)


@pytestmark_native
def test_native_forward_backward_all_neg_inf_raises():
    with pytest.raises(FloatingPointError):
        native.forward_backward(np.full((4, 2), -np.inf),
                                np.log([0.5, 0.5]), np.log(np.full((2, 2), 0.5)))


@pytestmark_native
@pytest.mark.parametrize("call", [
    lambda: native.gaussian_log_probs(np.zeros((2, 3)), np.zeros((1, 2, 2)), np.zeros((2, 3))),
    lambda: native.gaussian_log_probs(np.zeros((1, 2)), np.zeros((1, 2, 2)), np.zeros((2, 3))),
    lambda: native.decode_sequence(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), 0.5, 0),
    lambda: native.decode_sequence(np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), 0.5, 4),
    lambda: native.decode_sequence(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), -1.0, 4),
    lambda: native.lpc_levinson(np.zeros(5), 12),
    lambda: native.lpc_levinson(np.full(13, np.nan), 12),
    lambda: native.forward_backward(np.zeros((0, 2)), np.zeros(2), np.zeros((2, 2))),
    lambda: native.forward_backward(np.full((3, 2), np.nan), np.zeros(2), np.zeros((2, 2))),
])
def test_native_wrappers_validate_before_ctypes(call):
    with pytest.raises((ValueError, FloatingPointError)):
        call()


def test_native_import_survives_missing_shared_library(tmp_path):
    """Fresh-clone behaviour: no .so -> _lib is None, model falls back to core."""
    pkg = tmp_path / "ssgs"
    shutil.copytree(SRC / "ssgs", pkg, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import ssgs.native as n\n"
        "assert n._lib is None, n._lib\n"
        "import ssgs; m = ssgs.SpectralStateGuidedSynthesis(n_states=2)\n"  # core fallback works
        "try:\n n.lpc_levinson([1.0] + [0.0] * 12, 12)\n"
        "except RuntimeError as e: print('ok:', e)\n"
        "else: raise SystemExit('expected RuntimeError')\n"
    ) % tmp_path
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok:" in proc.stdout
