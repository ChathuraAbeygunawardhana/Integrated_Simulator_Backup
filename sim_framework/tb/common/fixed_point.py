"""Q m.n fixed-point converter used by the cocotb testbenches."""
from __future__ import annotations

import numpy as np


class FixedPointConverter:
    def __init__(self, total_bits: int = 32, frac_bits: int = 16):
        self.total_bits = total_bits
        self.frac_bits  = frac_bits
        self.scale      = 1 << frac_bits
        self.int_max    = (1 << (total_bits - 1)) - 1
        self.int_min    = -(1 << (total_bits - 1))

    def to_fixed(self, x):
        a = np.asarray(x, dtype=np.float64)
        scaled = np.rint(a * self.scale)
        clipped = np.clip(scaled, self.int_min, self.int_max).astype(np.int64)
        return clipped

    def to_float(self, x):
        a = np.asarray(x, dtype=np.int64)
        return a.astype(np.float64) / self.scale
