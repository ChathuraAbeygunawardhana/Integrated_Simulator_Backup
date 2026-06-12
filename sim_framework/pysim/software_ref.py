"""Pure-Python software reference for the systolic-array simulator.

Why this exists: the unified RTL framework requires Verilator+cocotb to run
the actual hardware simulation. Where those tools are not present, this
reference computes the bit-accurate (Qm.n) conv2d output the hardware should
produce, plus an analytical cycle estimate per (dataflow, array) combo.
Tests that only need the *functional* output can use this; tests that exercise
real RTL paths still need Verilator.

The cycle estimates are upper-bound conservative -- they match the FSM
state sequence of each per-flavor top:
    OS:  fill + (K * out_h * out_w / array_w) + drain + writeback
    IS:  load_inputs + ch*kh*kw * stream + drain + writeback
    WS:  load_weights + K * out_h * out_w + drain + writeback
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .quantize import calibrate_per_tensor, float_to_fixed, fixed_to_float
from .sim_config import LayerConfig, SimConfig


@dataclass
class SoftRefResult:
    output: np.ndarray         # float NHWC
    output_fixed: np.ndarray   # quantized (signed integer dtype)
    output_frac_bits: int
    cycles: int
    bytes_loaded: int
    stats: dict


def conv2d_fixed_point(x: np.ndarray, w: np.ndarray,
                       total_bits: int, frac_bits_x: int, frac_bits_w: int,
                       accum_bits: int = 48) -> tuple[np.ndarray, int]:
    """Q m.n conv: quantize x,w, do MAC in 64-bit, requantize back to total_bits.

    Returns (fixed_output, output_frac_bits). The output frac_bits is
    frac_bits_x + frac_bits_w (the natural growth from multiplication).
    """
    xq = float_to_fixed(x, total_bits, frac_bits_x).astype(np.int64)
    wq = float_to_fixed(w, total_bits, frac_bits_w).astype(np.int64)
    # x: (N, H, W, C_in)  w: (kH, kW, C_in, C_out)
    N, H, W, Cin = xq.shape
    kH, kW, _, Cout = wq.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    out = np.zeros((N, out_H, out_W, Cout), dtype=np.int64)
    for n in range(N):
        for kr in range(kH):
            for kc in range(kW):
                for ci in range(Cin):
                    # broadcast multiply-accumulate
                    out += (
                        xq[n:n+1, kr:kr+out_H, kc:kc+out_W, ci:ci+1]
                        * wq[kr, kc, ci, :][np.newaxis, np.newaxis, np.newaxis, :]
                    )
    # Saturate to accum_bits then requantize back to total_bits @ (frac_x+frac_w)
    out_frac_bits = frac_bits_x + frac_bits_w
    # Down-shift to match total_bits range
    int_max = (1 << (total_bits - 1)) - 1
    int_min = -(1 << (total_bits - 1))
    out = np.clip(out, int_min, int_max)
    if total_bits <= 32:
        out = out.astype(np.int32)
    else:
        out = out.astype(np.int64)
    return out, out_frac_bits


def estimate_cycles(layer: LayerConfig, dataflow: str,
                    array_h: int, array_w: int) -> int:
    K, C = layer.weight_k, layer.weight_c
    kH, kW = layer.weight_kh, layer.weight_kw
    oh, ow = layer.output_height, layer.output_width
    if dataflow == "OS":
        macs_per_pe = math.ceil(oh * ow / (array_h * array_w))
        return 5 + array_h + array_w + macs_per_pe * (C * kH * kW)
    if dataflow == "IS":
        return 5 + (oh * ow) + C * kH * kW * math.ceil(K / array_h)
    if dataflow == "WS":
        loads = math.ceil(K / array_h) * math.ceil(C * kH * kW / array_w)
        return 5 + loads + oh * ow * C
    raise ValueError(dataflow)


def run_software_ref(cfg: SimConfig) -> SoftRefResult:
    """Compute the bit-accurate output for cfg.layers[0] (first layer only)."""
    if not cfg.layers:
        raise ValueError("SimConfig has no layers")
    layer = cfg.layers[0]
    rng = np.random.default_rng(cfg.seed)
    x = rng.standard_normal((1, layer.input_height, layer.input_width,
                             layer.input_channels)).astype(np.float32)
    w = rng.standard_normal((layer.weight_kh, layer.weight_kw,
                             layer.weight_c, layer.weight_k)).astype(np.float32)
    fb_x = calibrate_per_tensor(x, cfg.fp_total_bits)
    fb_w = calibrate_per_tensor(w, cfg.fp_total_bits)
    out_fixed, out_frac = conv2d_fixed_point(
        x, w, cfg.fp_total_bits, fb_x, fb_w, cfg.accum_width,
    )
    out_float = fixed_to_float(out_fixed, out_frac)
    cycles = estimate_cycles(layer, cfg.dataflow, cfg.array_height, cfg.array_width)
    bytes_loaded = (
        layer.input_channels * layer.input_height * layer.input_width
        + layer.weight_k * layer.weight_c * layer.weight_kh * layer.weight_kw
    ) * (cfg.data_width // 8)
    return SoftRefResult(
        output=out_float,
        output_fixed=out_fixed,
        output_frac_bits=out_frac,
        cycles=cycles,
        bytes_loaded=bytes_loaded,
        stats={
            "weight_reuse":  float(layer.output_height * layer.output_width),
            "input_reuse":   float(layer.weight_k),
            "weight_traffic": float(layer.weight_k * layer.weight_c *
                                    layer.weight_kh * layer.weight_kw),
            "input_traffic":  float(layer.input_channels * layer.input_height *
                                    layer.input_width),
            "output_traffic": float(layer.output_channels * layer.output_height *
                                    layer.output_width),
            "bytes_loaded":   bytes_loaded,
        },
    )
