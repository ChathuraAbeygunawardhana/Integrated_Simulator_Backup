"""Cycle-budget timing tests.

For each (dataflow, array) configuration, compute the analytical lower
bound for an end-to-end conv2d execution and assert that the simulator
(or software reference, when Verilator is missing) does not exceed
3x the bound. This catches pipeline regressions that don't change
functional correctness.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

FRAMEWORK = Path(os.environ.get("FRAMEWORK_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(FRAMEWORK))

import pytest

from pysim import runner
from pysim.sim_config import LayerConfig, SimConfig
from pysim.software_ref import estimate_cycles


def _budget(layer: LayerConfig, dataflow: str, array_h: int, array_w: int) -> int:
    # Analytical lower bound: total MACs / array_size + fill+drain overhead
    macs = (layer.weight_k * layer.weight_c * layer.weight_kh * layer.weight_kw
            * layer.output_height * layer.output_width)
    lower_bound = math.ceil(macs / (array_h * array_w))
    overhead = (array_h + array_w) * 2  # fill + drain
    return max(64, (lower_bound + overhead) * 3)


@pytest.mark.parametrize("dataflow", ["OS", "IS", "WS"])
@pytest.mark.parametrize("memory",   ["STAMP", "PAGED"])
def test_cycle_budget(dataflow, memory):
    layer = LayerConfig(
        input_channels=4, input_height=6, input_width=6,
        weight_k=4, weight_c=4, weight_kh=3, weight_kw=3,
        output_channels=4, output_height=4, output_width=4,
    )
    cfg = SimConfig(dataflow=dataflow, memory=memory,
                    array_height=4, array_width=4, layers=[layer])
    budget = _budget(layer, dataflow, cfg.array_height, cfg.array_width)
    result = runner.run(cfg)
    assert result.passed, f"sim did not pass: {result.stats}"
    assert result.cycles <= budget, (
        f"cycle-budget regression: {result.cycles} > {budget} "
        f"({dataflow}/{memory})"
    )
