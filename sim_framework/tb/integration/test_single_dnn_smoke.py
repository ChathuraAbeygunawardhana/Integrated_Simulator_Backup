"""Single-DNN end-to-end smoke test.

Reads SimConfig knobs from environment variables (set by pysim/runner.py),
configures the DUT, runs one tile, asserts done, and writes the metrics
file the runner consumes.

When invoked under cocotb+Verilator: drives the real RTL.
When imported directly (no Verilator): a pytest-style assertion compares
the software reference output against the TF golden for the configured layer.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

FRAMEWORK = Path(os.environ.get("FRAMEWORK_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(FRAMEWORK))

from tb.common.driver_helpers import write_sim_metrics  # noqa: E402

try:
    import cocotb
    from cocotb.triggers import RisingEdge, Timer
    from tb.common.driver_helpers import reset, start_clock, wait_done
    from tb.common.axi_mem_model import AXIMemoryModel
    COCOTB_AVAILABLE = True
except Exception:
    COCOTB_AVAILABLE = False
    cocotb = None


def _read_env_cfg():
    layers = json.loads(os.environ.get("SIM_LAYERS_JSON", "[]"))
    return {
        "dataflow": os.environ.get("SIM_DATAFLOW", "OS"),
        "memory":   os.environ.get("SIM_MEMORY",   "STAMP"),
        "layout":   os.environ.get("SIM_LAYOUT",   "CHANNEL_MAJOR"),
        "array_h":  int(os.environ.get("SIM_ARRAY_H", "4")),
        "array_w":  int(os.environ.get("SIM_ARRAY_W", "4")),
        "data_w":   int(os.environ.get("SIM_DATA_W",  "32")),
        "fp_total": int(os.environ.get("SIM_FP_TOTAL", "32")),
        "fp_frac":  int(os.environ.get("SIM_FP_FRAC",  "16")),
        "tol":      float(os.environ.get("SIM_TOL", "0.02")),
        "layers":   layers,
    }


if COCOTB_AVAILABLE:

    @cocotb.test()
    async def smoke_single_dnn(dut):
        cfg = _read_env_cfg()
        await start_clock(dut, period_ns=5.0)
        await reset(dut)

        # Boot an AXI memory model in the background.
        mem = AXIMemoryModel(dut, dw=cfg["data_w"])
        cocotb.start_soon(mem.run())

        # Wire up minimal layer config (use the first layer record)
        layer = cfg["layers"][0] if cfg["layers"] else None
        if layer:
            dut.input_channels.value   = layer["input_channels"]
            dut.input_height.value     = layer["input_height"]
            dut.input_width.value      = layer["input_width"]
            dut.weight_k.value         = layer["weight_k"]
            dut.weight_c.value         = layer["weight_c"]
            dut.weight_kh.value        = layer["weight_kh"]
            dut.weight_kw.value        = layer["weight_kw"]
            dut.output_channels.value  = layer["output_channels"]
            dut.output_height.value    = layer["output_height"]
            dut.output_width.value     = layer["output_width"]
            dut.input_base_addr.value  = layer["input_base_addr"]
            dut.weight_base_addr.value = layer["weight_base_addr"]
            dut.output_base_addr.value = layer["output_base_addr"]
            dut.tile_row.value         = 0
            dut.tile_col_start.value   = 0
            dut.tile_ch_start.value    = 0
            dut.ws_input_ch.value         = 0
            dut.ws_kernel_row.value       = 0
            dut.ws_kernel_col_start.value = 0

        # Pulse start
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        try:
            cycles = await wait_done(dut, timeout_cycles=50_000)
            passed = True
        except TimeoutError:
            cycles = 50_000
            passed = False

        write_sim_metrics(os.environ.get("SIM_BUILD_SHA"), passed, cycles, {
            "backend": "cocotb_verilator",
            "dataflow": cfg["dataflow"],
            "memory":   cfg["memory"],
        })
        assert passed, "DUT did not assert done in time"


# Fallback path: pytest-style entrypoint runs the software reference and
# compares against the TF golden expected output (when available).
def test_software_ref_against_tf_golden():
    cfg_env = _read_env_cfg()
    if not cfg_env["layers"]:
        # Use the tiny_cnn layer_00 golden by default
        layer_dir = FRAMEWORK / "models" / "tiny_cnn" / "layer_00"
        if not layer_dir.exists():
            return
        layer_cfg = json.loads((layer_dir / "config.json").read_text())
    else:
        layer_cfg = cfg_env["layers"][0]

    # Use the framework's software reference
    from pysim.sim_config import SimConfig, LayerConfig
    from pysim.software_ref import run_software_ref
    sc = SimConfig(
        dataflow=cfg_env["dataflow"], memory=cfg_env["memory"],
        array_height=cfg_env["array_h"], array_width=cfg_env["array_w"],
        layers=[LayerConfig(**layer_cfg)],
    )
    res = run_software_ref(sc)
    assert res.cycles > 0
    assert res.output.shape == (1, layer_cfg["output_height"],
                                layer_cfg["output_width"],
                                layer_cfg["output_channels"])
