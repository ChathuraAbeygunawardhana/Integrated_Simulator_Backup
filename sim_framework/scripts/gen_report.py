#!/usr/bin/env python3
"""Generate the final framework report (docs/final_report.md) from
verification matrix output, timing reports, and a sample optimizer run."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT))

REPORT = FRAMEWORK_ROOT / "docs" / "final_report.md"


def _vivado_summary(rpt_path: Path) -> str:
    if not rpt_path.exists():
        return "_(Vivado synthesis was not run.)_"
    txt = rpt_path.read_text()
    # Grab the first 60 lines of the timing-summary; full file is in timing/reports/.
    head = "\n".join(txt.splitlines()[:60])
    return f"```\n{head}\n```"


def _python_test_summary() -> str:
    csv = FRAMEWORK_ROOT / "build" / "matrix_results.csv"
    if not csv.exists():
        return "_(No matrix results yet. Run `scripts/iterate_until_pass.py`.)_"
    rows = csv.read_text().splitlines()
    return f"```\n{chr(10).join(rows[:25])}\n```\n\nFull CSV at `build/matrix_results.csv`."


def main():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    body = f"""# Unified Systolic-Array Simulator -- Framework Report

Generated: {now}

## 1. Verification matrix

{_python_test_summary()}

## 2. Vivado timing closure

### single_dnn_top
{_vivado_summary(FRAMEWORK_ROOT / 'timing' / 'reports' / 'single_dnn_timing_summary.rpt')}

### multi_dnn_top
{_vivado_summary(FRAMEWORK_ROOT / 'timing' / 'reports' / 'multi_dnn_timing_summary.rpt')}

## 3. Loop optimizer cost-model integration

`pysim.cost_model.SimulatorCostModel` plugs into
`optimizer.src.blocking_optimizer.BlockingOptimizer` via the new `analyzer`
constructor argument and the new `SearchMode.SIMULATOR_GUIDED` mode:

```python
from blocking_optimizer import BlockingOptimizer, SearchMode
from pysim.cost_model  import SimulatorCostModel
from pysim             import SimConfig

base = SimConfig(dataflow='OS', memory='STAMP', array_height=4, array_width=4)
sim  = SimulatorCostModel(K=32, C=32, H=14, W=14, KH=3, KW=3, base_cfg=base)
opt  = BlockingOptimizer(K=32, C=32, H=14, W=14, KH=3, KW=3, analyzer=sim)
best, stats = opt.search(mode=SearchMode.SIMULATOR_GUIDED)
```

Two-stage strategy: the analytical `MemoryAccessAnalyzer` pre-filters all
720 (loop-order, tile-size) combinations down to the top-N=20; the
simulator analyzer re-scores the survivors with real cycle counts.

## 4. Tensorflow golden flow

`models/dump_tf_model.py` dumps a pretrained Keras model into per-layer
.npy + .hex artifacts. `pysim/golden_compare.py` compares simulator output
against the dumped expected.npy with a configurable tolerance.

The smoke model `models/tiny_cnn` is committed; for larger regression flows:

```bash
python models/dump_tf_model.py --model mobilenet_v2 --out models/mobilenet_v2
```

## 5. Known limitations

* Verilator and Vivado are not installed in the default test environment;
  when missing, the framework runner falls back to the bit-accurate
  software reference (`pysim.software_ref`) and timing-closure phase is
  skipped. Both fall-backs are clearly labelled in the matrix CSV.
* Single-outstanding AXI in the v1 backend serialises concurrent memory
  requests from multi-DNN schedulers (documented v1 limitation).
* `extended_memory_controller.sv` (stamp + prefetch) is deferred to v2.
"""
    REPORT.write_text(body)
    print(f"[gen_report] wrote {REPORT}")


if __name__ == "__main__":
    main()
