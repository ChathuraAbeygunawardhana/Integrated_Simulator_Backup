# Unified Systolic-Array Simulator Framework

A consolidated simulator framework for a systolic-array hardware accelerator.
Two flavors -- **single-DNN** and **multi-DNN** -- share one parameterised
RTL core and a Python driver layer that exposes every control knob: dataflow
(OS / IS / WS), on-chip memory backend (stamp-based vs page-table), array
geometry, data widths, scheduler choice, fixed-point format.

The loop optimizer (under `optimizer/`) is kept separate but can be
configured to use the single-DNN simulator as its hardware-in-the-loop
cost model.

## Repo layout

```
sim_framework/
├── rtl/             Unified SystemVerilog RTL (PE, fetchers, dataflows,
│                    memory backends, schedulers, single_dnn_top, multi_dnn_top)
├── pysim/           Python driver layer (SimConfig, runner, cost_model,
│                    tf_dump, golden_compare, software_ref)
├── tb/              cocotb test bench (unit, integration, timing)
├── models/          TF golden-reference flow (dump_tf_model.py + tiny_cnn)
├── optimizer/       Loop / blocking optimizer (analytical + simulator-guided)
├── timing/          Vivado synthesis + report_timing scripts
├── scripts/         install_deps.sh, check_env.sh, run_all_tests.sh,
│                    iterate_until_pass.py, gen_report.py
├── configs/         YAML presets + ci_matrix.yaml
└── docs/            Generated and hand-written documentation
```

## Quick start

```bash
./scripts/install_deps.sh      # apt + pip; creates .venv
source .venv/bin/activate
./scripts/check_env.sh         # verifies verilator/cocotb/tf/vivado

# dump a small pretrained TF model into golden vectors
python models/dump_tf_model.py --model tiny_cnn --out models/tiny_cnn

# run the full test suite (pytest + matrix + Vivado-if-present)
./scripts/run_all_tests.sh

# generate the final report
python scripts/gen_report.py
```

See [`docs/getting_started.md`](docs/getting_started.md) for the long form
and [`docs/knob_reference.md`](docs/knob_reference.md) for the complete
control-knob catalogue.

## Tests at a glance

| Suite | Path | What |
|---|---|---|
| Unit (PE, fixed-point) | `tb/unit/`           | Per-flavor PE MAC, quantizer round-trip |
| Integration (E2E)      | `tb/integration/`    | single_dnn_top, multi_dnn_top, TF golden |
| Timing (cycle-budget)  | `tb/timing/`         | Latency upper-bound per (dataflow, memory) |
| Matrix sweep           | `scripts/iterate_until_pass.py` | Sweep config matrix, report pass/fail |

Without Verilator installed, the framework runner falls back to a bit-accurate
Python software reference (`pysim/software_ref.py`); all tests still execute
and produce a populated `docs/verification_matrix.md`.

## What this framework integrates

| Source tree | Status |
|---|---|
| `Archive/files_os/` -- Output Stationary | Lifted into `rtl/array/systolic_array_os.sv` (verbatim, imports unified pkg) |
| `Archive/files_IS/` -- Input Stationary  | Lifted into `rtl/array/systolic_array_is.sv` |
| `Archive/files_ws/` -- Weight Stationary | Lifted into `rtl/array/systolic_array_ws.sv` |
| `static_hash_and_tagless_memory/stamp_based_memory_controller.sv` | Wrapped by `rtl/memory/stamp_memory_backend.sv` |
| `static_hash_and_tagless_memory/paged_table.v` | Wrapped by `rtl/memory/paged_memory_backend.sv` |
| `Multi_DNN_schedulers/{task,advanced_task}_scheduler.sv` | Unified under `rtl/schedulers/multi_scheduler_wrapper.sv` (existing) |
| `Multi_DNN_schedulers/{aimt,batchdnn,batchdnn_pp}_scheduler.sv` | Unified under new `rtl/schedulers/dnn_scheduler_wrapper.sv` |
| `systolic_optimizer_codebase/` | Lifted into `optimizer/` with injectable analyzer |
| three duplicate `*_pkg.sv` files | Replaced by canonical `rtl/pkg/sim_framework_pkg.sv` (with thin re-export aliases for legacy imports) |
| `stamp_memory_pkg.sv` empty `package systolic_array_pkg` stub | Removed (collision blocker) |
| `extended_memory_controller.sv` | Deferred to v2 (per user decision) |
