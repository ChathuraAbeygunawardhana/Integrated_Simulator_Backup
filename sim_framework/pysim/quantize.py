"""Quantisation helpers: float32 <-> fixed-point Qm.n, plus per-tensor
calibration that picks frac_bits to maximise dynamic range without saturation.

The systolic array consumes integer data (parameter DATA_WIDTH); the host
quantises float weights/activations into Qm.n before pushing them into the
AXI memory model.
"""
from __future__ import annotations

import numpy as np


def float_to_fixed(arr: np.ndarray, total_bits: int = 32, frac_bits: int = 16) -> np.ndarray:
    """Round to nearest, saturate to signed [INT_MIN, INT_MAX]."""
    scale = 1 << frac_bits
    int_max = (1 << (total_bits - 1)) - 1
    int_min = -(1 << (total_bits - 1))
    scaled = np.rint(arr.astype(np.float64) * scale)
    clipped = np.clip(scaled, int_min, int_max).astype(_int_dtype(total_bits))
    return clipped


def fixed_to_float(arr: np.ndarray, frac_bits: int = 16) -> np.ndarray:
    scale = float(1 << frac_bits)
    return arr.astype(np.float64) / scale


def calibrate_per_tensor(arr: np.ndarray, total_bits: int = 32, headroom_bits: int = 1) -> int:
    """Pick the largest frac_bits such that no element overflows.

    headroom_bits: extra integer bits to leave above the max abs value
    (prevents overflow during MAC accumulation in higher-precision accum).
    """
    a = np.abs(arr).max()
    if a == 0 or not np.isfinite(a):
        return total_bits - 1 - headroom_bits
    # Need int_bits >= ceil(log2(a)) + 1 (for sign).
    int_bits = max(1, int(np.ceil(np.log2(a))) + 1 + headroom_bits)
    frac_bits = max(0, total_bits - int_bits)
    return frac_bits


def _int_dtype(total_bits: int) -> np.dtype:
    if total_bits <= 8:   return np.int8
    if total_bits <= 16:  return np.int16
    if total_bits <= 32:  return np.int32
    return np.int64
