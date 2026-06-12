"""Reset / start / wait-for-done helpers for cocotb testbenches."""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge, Timer
except Exception:                       # pragma: no cover
    cocotb = None


async def reset(dut, ncycles: int = 5):
    dut.rst_n.value = 0
    for _ in range(ncycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1


async def start_clock(dut, period_ns: float = 5.0):
    cocotb.start_soon(Clock(dut.clk, period_ns, units="ns").start())


async def wait_done(dut, timeout_cycles: int = 100_000) -> int:
    cycles = 0
    while cycles < timeout_cycles:
        await RisingEdge(dut.clk)
        cycles += 1
        try:
            if int(dut.done.value) == 1:
                return cycles
        except Exception:
            pass
    raise TimeoutError(f"DUT did not assert done within {timeout_cycles} cycles")


def write_sim_metrics(sha: str | None, passed: bool, cycles: int, stats: dict | None = None):
    """Write the sim_metrics.json sidecar that pysim.runner reads."""
    sha = sha or os.environ.get("SIM_BUILD_SHA", "unknown")
    out = {
        "passed": bool(passed),
        "cycles": int(cycles),
        "stats":  dict(stats or {}),
    }
    fr = os.environ.get("FRAMEWORK_ROOT", ".")
    build_dir = Path(fr) / "build" / sha
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "sim_metrics.json").write_text(json.dumps(out, indent=2))
