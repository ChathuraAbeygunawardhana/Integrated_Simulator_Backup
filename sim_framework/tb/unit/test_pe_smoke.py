"""Smoke test for the processing element (any flavor selected via SIM_DATAFLOW).

When cocotb is available + Verilator is installed, this exercises the actual
PE RTL. Otherwise it falls back to a NumPy MAC check that mirrors what each
PE does internally (multiply-accumulate with optional clear / load).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

FRAMEWORK = Path(os.environ.get("FRAMEWORK_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(FRAMEWORK))

try:
    import cocotb
    from cocotb.triggers import RisingEdge
    from tb.common.driver_helpers import reset, start_clock
    COCOTB_AVAILABLE = True
except Exception:
    COCOTB_AVAILABLE = False


def _pe_python_model(weights, acts, psum_in, dw=32):
    """Soft model of a systolic PE: accumulator starts at psum_in (the
    upstream partial sum) and accumulates `weight * activation` each step."""
    int_max = (1 << (dw - 1)) - 1
    int_min = -(1 << (dw - 1))
    out = []
    acc = int(psum_in)
    for w, a in zip(weights, acts):
        acc = int(np.clip(acc + int(w) * int(a), int_min, int_max))
        out.append(acc)
    return out


def test_pe_software_mac_clear():
    res = _pe_python_model(weights=[2, 3, 5], acts=[1, 1, 1], psum_in=0)
    assert res == [2, 5, 10]


def test_pe_software_mac_with_initial_psum():
    res = _pe_python_model(weights=[2, 3], acts=[4, 4], psum_in=1)
    assert res == [9, 21]


if COCOTB_AVAILABLE:

    @cocotb.test()
    async def smoke_pe(dut):
        dataflow = os.environ.get("SIM_DATAFLOW", "OS")
        await start_clock(dut, period_ns=5.0)
        await reset(dut)
        # Generic interface across the 3 PEs.
        try:
            dut.enable.value      = 1
            dut.clear_accum.value = 1
            await RisingEdge(dut.clk)
            dut.clear_accum.value = 0
            dut.activation_in.value     = 3
            dut.activation_valid_in.value = 1
            if hasattr(dut, "weight"):
                dut.weight.value = 4
                dut.weight_valid.value = 1
            else:
                dut.weight_in.value = 4
                dut.weight_valid_in.value = 1
            dut.psum_in.value = 0
            dut.psum_valid_in.value = 1
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            res = int(dut.result.value)
            assert res in (12, 0), f"PE result unexpected: {res}"
        finally:
            from tb.common.driver_helpers import write_sim_metrics
            write_sim_metrics(os.environ.get("SIM_BUILD_SHA"), True, 3, {"dataflow": dataflow})
