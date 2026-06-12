# Getting Started

This guide walks through installing the framework, running the smoke tests,
generating the TF golden reference, sweeping the full verification matrix,
and (optionally) closing timing in Vivado.

## 1. Install

The framework targets Ubuntu 22.04 / 24.04. `install_deps.sh` handles both
the apt-side packages (Verilator, gtkwave) and the Python venv (cocotb,
TensorFlow, numpy, pyyaml, matplotlib, pandas, tqdm).

```bash
./scripts/install_deps.sh
source .venv/bin/activate
./scripts/check_env.sh
```

Expected output: every required tool reports OK; optional tools (Vivado,
yosys) warn if missing. Vivado is only needed for the post-synthesis
timing-closure step.

### Verilator note

Cocotb 2.x requires Verilator >= 5.0. Ubuntu 22.04 ships Verilator 4.038;
`install_deps.sh` prints the build-from-source recipe in that case.
Ubuntu 24.04 ships Verilator 5.x already.

### Without RTL tools

The framework runner (`pysim.runner.run`) auto-detects Verilator. If it
is missing, it transparently falls back to `pysim.software_ref` -- a
bit-accurate Python implementation of the systolic computation. The
fall-back enables the full pytest suite and matrix sweep to run end-to-end
without any HDL toolchain. Tests that exercise real RTL paths (cocotb
tests, Vivado synth) skip gracefully and are flagged in the report.

## 2. Dump a TF golden reference

```bash
python models/dump_tf_model.py --model tiny_cnn --out models/tiny_cnn
```

This generates `models/tiny_cnn/layer_NN/{config.json,input.npy,weights.npy,
expected.npy,input.hex,weights.hex,expected.hex,quant.json}` per layer.

For a larger regression model:

```bash
python models/dump_tf_model.py --model mobilenet_v2 --out models/mobilenet_v2
```

## 3. Run the smoke test

```bash
python -m pytest tb/integration/test_single_dnn_smoke.py -v
python -m pytest tb/integration/test_tf_golden_conv2d.py -v
```

A green run of these two confirms:
* the framework's Python driver can build a SimConfig and call the runner
* the runner picks the right backend (Verilator or software reference)
* the golden comparator parses the dumped TF artifacts correctly

## 4. Sweep the full verification matrix

```bash
python scripts/iterate_until_pass.py
```

This expands `configs/ci_matrix.yaml` into 12 (default) configurations,
runs each, archives any failures under `build/failures/<sha>/`, and
iterates up to `--max-iter` times. Output:

* `docs/verification_matrix.md` -- one row per config, pass/fail/cycles
* `build/matrix_results.csv`    -- machine-readable

## 5. Timing closure (Vivado, optional)

```bash
vivado -mode batch -source timing/vivado/synth_single_dnn.tcl -tclargs $(pwd)
vivado -mode batch -source timing/vivado/synth_multi_dnn.tcl  -tclargs $(pwd)
```

Output: `timing/reports/single_dnn_timing_summary.rpt` and
`timing/reports/single_dnn_utilization.rpt` (likewise for multi_dnn).

The script asserts WNS >= 0 at 200 MHz (5 ns period defined in
`timing/sdc/single_dnn.xdc`). Relax the period if synthesis fails.

## 6. Generate the final report

```bash
python scripts/gen_report.py
```

Writes `docs/final_report.md` consolidating verification matrix + Vivado
timing summaries + a sample optimizer integration snippet.

## 7. Loop-optimizer with hardware-in-the-loop cost model

```python
from blocking_optimizer import BlockingOptimizer, SearchMode
from pysim.cost_model  import SimulatorCostModel
from pysim             import SimConfig

base = SimConfig(dataflow='OS', memory='STAMP', array_height=4, array_width=4)
sim  = SimulatorCostModel(K=32, C=32, H=14, W=14, KH=3, KW=3, base_cfg=base)
opt  = BlockingOptimizer(K=32, C=32, H=14, W=14, KH=3, KW=3, analyzer=sim)
best, all_stats = opt.search(mode=SearchMode.SIMULATOR_GUIDED)
print(best)
```

The two-stage search filters with the analytical analyzer first (cheap,
720 candidates), then re-scores the top-N=20 survivors with the
simulator-backed analyzer.

## 8. Troubleshooting

* `verilator: command not found` -- install via apt or build from source.
  The runner still works via the software-ref fallback.
* `ImportError: tensorflow` -- only needed for the TF golden flow; install
  with `pip install tensorflow-cpu==2.18.0`.
* `KeyError: SIM_LAYERS_JSON` -- you're invoking a cocotb test directly
  without going through `pysim.runner.run`. The runner sets every required
  env var.
* See `docs/triage.md` for failure-pattern triage recipes.
