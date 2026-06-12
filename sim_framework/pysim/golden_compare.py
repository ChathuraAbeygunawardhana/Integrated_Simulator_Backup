"""Compare simulator output against a TF golden reference dumped by tf_dump.py.

Reads the simulator's output region (from the cocotb AXI memory model or
from a dumped sim_output.npy file) and compares element-wise against
expected.npy with a user-supplied tolerance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .quantize import fixed_to_float


@dataclass
class CompareResult:
    passed: bool
    n_total: int
    n_within_tol: int
    max_abs_err: float
    mean_abs_err: float
    fraction_within: float

    def summary(self) -> str:
        return (f"{self.n_within_tol}/{self.n_total} within tol  "
                f"(max|err|={self.max_abs_err:.4g}, "
                f"mean|err|={self.mean_abs_err:.4g})")


def compare_arrays(actual: np.ndarray, expected: np.ndarray,
                   tol: float = 0.02) -> CompareResult:
    if actual.shape != expected.shape:
        # Try to reshape if total sizes match
        if actual.size == expected.size:
            actual = actual.reshape(expected.shape)
        else:
            return CompareResult(False, expected.size, 0,
                                 float("inf"), float("inf"), 0.0)
    err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    n_within = int(np.sum(err <= tol))
    return CompareResult(
        passed=(n_within == expected.size),
        n_total=expected.size,
        n_within_tol=n_within,
        max_abs_err=float(err.max()),
        mean_abs_err=float(err.mean()),
        fraction_within=n_within / expected.size,
    )


def compare_layer(layer_dir: str | Path, sim_output: np.ndarray,
                  tol: Optional[float] = None) -> CompareResult:
    """Compare a simulator's float-domain output against the TF expected
    from a layer dump directory."""
    layer_path = Path(layer_dir)
    expected = np.load(layer_path / "expected.npy")
    if tol is None:
        tol = 0.02
    return compare_arrays(sim_output, expected, tol)


def compare_from_hex(layer_dir: str | Path, sim_hex_path: str | Path,
                     tol: float = 0.02) -> CompareResult:
    """Read the simulator's fixed-point hex output, dequantize, compare."""
    layer_path = Path(layer_dir)
    quant = json.loads((layer_path / "quant.json").read_text())
    fb = int(quant["expected_frac_bits"])
    expected = np.load(layer_path / "expected.npy")
    # Parse hex as signed N-bit values
    width = int(quant.get("fp_total_bits", 32))
    sign_mask = 1 << (width - 1)
    full_mask = (1 << width) - 1
    vals = []
    with open(sim_hex_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16) & full_mask
            if v & sign_mask:
                v -= 1 << width
            vals.append(v)
    actual_fixed = np.array(vals, dtype=np.int64).reshape(expected.shape)
    actual_float = fixed_to_float(actual_fixed, frac_bits=fb)
    return compare_arrays(actual_float, expected, tol)
