"""Multi-DNN end-to-end smoke test (both basic and DNN-aware schedulers).

Mirrors test_single_dnn_smoke but selects multi_dnn_top as TOPLEVEL and
exercises the dispatch FSM with N tasks.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FRAMEWORK = Path(os.environ.get("FRAMEWORK_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(FRAMEWORK))

from tb.common.driver_helpers import write_sim_metrics  # noqa: E402

try:
    import cocotb
    from cocotb.triggers import RisingEdge
    from tb.common.driver_helpers import reset, start_clock
    from tb.common.axi_mem_model import AXIMemoryModel
    COCOTB_AVAILABLE = True
except Exception:
    COCOTB_AVAILABLE = False


if COCOTB_AVAILABLE:

    @cocotb.test()
    async def smoke_multi_dnn(dut):
        sched = int(os.environ.get("SIM_SCHEDULER", "0"))
        await start_clock(dut, period_ns=5.0)
        await reset(dut)
        mem = AXIMemoryModel(dut, dw=int(os.environ.get("SIM_DATA_W", "32")))
        cocotb.start_soon(mem.run())

        dut.scheduler_select.value = sched

        # Issue 3 basic tasks
        n_tasks = 3
        completed = 0
        for tid in range(n_tasks):
            dut.task_valid.value = 1
            dut.task_id.value    = tid
            dut.burst_time.value = 20
            dut.task_priority.value = 1
            dut.deadline.value = 100
            await RisingEdge(dut.clk)
        dut.task_valid.value = 0

        for _ in range(20_000):
            await RisingEdge(dut.clk)
            try:
                if int(dut.task_complete_pulse.value):
                    completed += 1
                    if completed == n_tasks:
                        break
            except AttributeError:
                pass

        passed = completed == n_tasks
        write_sim_metrics(os.environ.get("SIM_BUILD_SHA"), passed, 0, {
            "backend": "cocotb_verilator",
            "scheduler": sched,
            "tasks_completed": completed,
            "tasks_expected": n_tasks,
        })
        assert passed, f"Only {completed}/{n_tasks} tasks completed"


def test_multi_dnn_sched_id_validation():
    """Pure-python sanity check on scheduler-name <-> id mapping."""
    from pysim.sim_config import SimConfig
    sc = SimConfig(top="multi_dnn_top", scheduler="FIFO")
    assert sc.scheduler_select() == 0
    sc = SimConfig(top="multi_dnn_top", scheduler="AIMT")
    assert sc.scheduler_select() == 11
    sc = SimConfig(top="multi_dnn_top", scheduler="BATCHDNN_PP")
    assert sc.scheduler_select() == 13
