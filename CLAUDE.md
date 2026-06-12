# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout — what is active vs. reference

The repo started as four independent stacks (a systolic-array dataflow study,
an on-chip memory study, a multi-DNN scheduler study, and a Python loop
optimizer). All four are now consolidated into **`sim_framework/`** — that
is the active tree. The four legacy directories remain untouched as
reference only:

```
Archive/                          # legacy: 3 dataflow flavors (OS/IS/WS)
static_hash_and_tagless_memory/   # legacy: stamp ctrl + page_table
Multi_DNN_schedulers/             # legacy: 5 schedulers + wrapper
systolic_optimizer_codebase/      # legacy: Python loop optimizer
sim_framework/                    # <-- active unified framework
```

Do not edit the legacy directories. If you need to verify behaviour of an
original module, read it as reference, then change the copy under
`sim_framework/rtl/`.

## Toolchain availability

This environment **does not have Verilator or Vivado** (no sudo, no apt
install). The framework was designed to cope:

- `pysim/runner.py` auto-detects Verilator; if missing it transparently
  falls back to `pysim/software_ref.py` (a bit-accurate NumPy
  implementation of the systolic computation). The full pytest suite and
  matrix sweep run green in this mode.
- Vivado synthesis (Phase I of the build plan) is wrapped in
  `scripts/run_all_tests.sh` and the top `Makefile` with a "skip if
  vivado not found" guard.
- Python deps live in `sim_framework/.venv/` (cocotb, tensorflow 2.18,
  pyyaml, pytest, numpy). The venv is gitignored; recreate with
  `./scripts/install_deps.sh`.

If you ever do gain access to Verilator/Vivado, the cocotb tests in
`tb/integration/test_*.py` and the TCL scripts in `timing/vivado/` are
ready to run unmodified.

## Common commands

All commands assume `cd sim_framework && source .venv/bin/activate` first.

```bash
# Bring up the environment from scratch
./scripts/install_deps.sh        # apt + pip; creates .venv
./scripts/check_env.sh           # verifies verilator/cocotb/tf/vivado

# Generate the TF golden reference (only needed once or after model changes)
python models/dump_tf_model.py --model tiny_cnn --out models/tiny_cnn

# Test
PYTHONPATH=. python -m pytest tb/ -v                                # full
PYTHONPATH=. python -m pytest tb/unit/test_pe_smoke.py -v           # one file
PYTHONPATH=. python -m pytest tb/timing/test_cycle_budget.py -v -k OS  # filter
make smoke                                                          # via Makefile

# Sweep the configuration matrix and write the verification report
PYTHONPATH=. python scripts/iterate_until_pass.py                   # 12 configs
PYTHONPATH=. python scripts/iterate_until_pass.py --max-iter 5      # retry budget

# Lint RTL (requires verilator)
make -C rtl -f Makefile.lint lint

# Vivado timing closure (requires vivado)
vivado -mode batch -source timing/vivado/synth_single_dnn.tcl -tclargs $(pwd)

# Generate the framework's final report
python scripts/gen_report.py
```

The `Makefile` has aggregate targets: `install`, `check_env`, `lint`,
`smoke`, `full_test`, `matrix`, `timing_closure`, `report`, `all`, `clean`.

## Architecture — the load-bearing pieces

### Control knobs flow through `SimConfig`

`pysim/sim_config.py` defines a `SimConfig` dataclass that captures every
runtime knob: `top` (`single_dnn_top` | `multi_dnn_top`), `dataflow`
(`OS`/`IS`/`WS`), `memory` (`STAMP`/`PAGED`), array geometry, data widths,
quantisation, scheduler choice (one of 14 names), per-layer workload. The
framework's two-way mapping is on this object: `verilator_params()` emits
`-GNAME=value` overrides for `verilator`, `cocotb_env()` emits env vars
that the cocotb tests read. **Always plumb new knobs through `SimConfig`**
rather than parameterising lower layers directly.

### Package-collision fix (do not re-break it)

The three legacy per-flavor `*_pkg.sv` files defined byte-identical
`mem_layout_t` enums; `stamp_memory_pkg.sv` also contained an empty stub
of `package systolic_array_pkg` that collided with the OS package the
moment two stacks were compiled together. The fix:

- `rtl/pkg/sim_framework_pkg.sv` — canonical types + new `dataflow_e`,
  `mem_backend_e`, `sched_output_t`
- The same file contains thin re-export aliases for
  `systolic_array_pkg`, `input_stationary_pkg`, `weight_stationary_pkg`
  so unmodified legacy `import` lines still resolve
- `rtl/pkg/stamp_memory_pkg.sv` — the empty stub is removed and the
  stamp-side struct is renamed `stamp_mem_layout_t` to avoid clashing
  with the dataflow enum

If you add types/enums shared across flavors, put them in
`sim_framework_pkg`. If you add a new legacy-package field, mirror it in
both files (the typedefs must stay in sync — `rtl/Makefile.lint` catches
divergences).

### Wrapper instantiation, not merged modules

The three per-flavor systolic-array tops have *different ports* (WS has a
2D weight port and a 1D output port; OS/IS are the inverse). The
canonical pattern across the framework is:

- `systolic_array_wrap.sv` exposes the **union** of all per-flavor ports
  and uses `generate (DATAFLOW)` to instantiate one body and tie off the
  unused-flavor ports
- `mem_backend_wrap.sv` does the same for `MEMORY` (stamp vs paged)
- `unified_scheduler_wrapper.sv` does the same for `scheduler_select`

Do not try to merge per-flavor modules into one `case (DATAFLOW)` block —
the port-shape heterogeneity makes that hostile.

### Split-FSM in `single_dnn_top`

`single_dnn_top.sv` has an outer FSM (`S_IDLE → S_MEM → S_COMPUTE →
S_FINAL`) that exposes **two separate strobes** — `phase_mem_done` and
`phase_compute_done` — so that AI-MT and BatchDNN schedulers can issue
memory tasks (MT) and compute tasks (CT) against the same DNN. The
dispatch FSM in `multi_dnn_top.sv` routes those strobes back to the
scheduler's `mem_access_done` / `compute_done` inputs. Basic
(non-DNN-aware) schedulers only see `phase_compute_done` as
`task_complete`.

### Two-wrapper scheduler design

Schedulers come in two families with **incompatible** interfaces:

- Basic + advanced (11 algorithms, FIFO/SJF/...) use a FIFO task queue —
  wrapped by `multi_scheduler_wrapper.sv` (existing legacy)
- DNN-aware (AI-MT / BatchDNN / BatchDNN++) use a scheduling-table load
  + MT/CT split outputs — wrapped by new `dnn_scheduler_wrapper.sv`

`unified_scheduler_wrapper.sv` muxes both behind a 5-bit
`scheduler_select` and a tagged-union `sched_output_t` (defined in
`sim_framework_pkg`). When adding a scheduler, decide which family it
belongs to and update only that wrapper.

### Optimizer integration via injectable analyzer

`optimizer/src/blocking_optimizer.py::BlockingOptimizer.__init__` accepts
an `analyzer=` keyword. Default is the analytical `MemoryAccessAnalyzer`
(closed-form DRAM traffic). To run the optimizer with the simulator as
its cost model:

```python
from blocking_optimizer  import BlockingOptimizer, SearchMode
from pysim.cost_model    import SimulatorCostModel
from pysim               import SimConfig

base = SimConfig(dataflow='OS', memory='STAMP', array_height=4, array_width=4)
sim  = SimulatorCostModel(K=32, C=32, H=14, W=14, KH=3, KW=3, base_cfg=base)
opt  = BlockingOptimizer(K=32, C=32, H=14, W=14, KH=3, KW=3, analyzer=sim)
best, _ = opt.search(mode=SearchMode.SIMULATOR_GUIDED)
```

`SearchMode.SIMULATOR_GUIDED` does a two-stage search — analytical
pre-filter (top-N=20) followed by simulator scoring on the survivors —
because pure-simulator search is too slow over the 720-permutation × tile
grid. `SimulatorCostModel.traffic_continuous()` deliberately raises
`NotImplementedError` so `SearchMode.GRADIENT` is disabled when this
analyzer is in use.

The simulator cost-model caches results in
`pysim/cache/<RTL_REVISION>_<sha>.json`. The `RTL_REVISION` env var lets
you invalidate the cache when RTL changes (default is `"default"`).

### Runner fallback — what `runner.run()` actually does

`pysim/runner.py::run(cfg)`:

1. Hashes the SimConfig, looks for `build/<sha>/sim_metrics.json` — if
   present, returns the cached result.
2. If `verilator` is on PATH: renders `tb/Makefile.cocotb`, spawns
   `make`, parses `sim_metrics.json` (written by the cocotb test via
   `tb/common/driver_helpers.write_sim_metrics`).
3. If not: imports `pysim/software_ref.py`, runs the bit-accurate NumPy
   conv2d, returns a `RunResult` with `stats["backend"] = "software_ref"`.

The cocotb tests in `tb/` are written to work in *both* modes: each test
file has a `@cocotb.test()` coroutine **and** standalone pytest-style
functions that exercise the same logic via the software reference. Keep
this pattern for new tests.

## Adding a test

1. Decide the layer: `tb/unit/` (single-module), `tb/integration/` (full
   `*_top` E2E), or `tb/timing/` (cycle-budget).
2. Inside the file:
   - Top of file: `try: import cocotb ... except: COCOTB_AVAILABLE = False`
   - Cocotb coroutine guarded by `if COCOTB_AVAILABLE:`
   - One or more pytest functions that work without cocotb (use
     `pysim.software_ref` or `pysim.golden_compare`)
3. Tests read SimConfig knobs from env vars set by `pysim/runner.py`
   (see `tb/integration/test_single_dnn_smoke.py::_read_env_cfg` for the
   pattern).
4. To add the test to the matrix: extend `configs/ci_matrix.yaml`.

## Documentation pointers

The reports under `sim_framework/docs/` are the canonical reference:

- `getting_started.md` — install + smoke + full run + Vivado + optimizer
- `architecture.md` — RTL hierarchy diagram + split-FSM state machine
- `knob_reference.md` — every `SimConfig` field with units / RTL mapping
- `triage.md` — failure-pattern recipes (timeout, mismatch, X-prop, ...)
- `final_report.md` — auto-regenerated by `scripts/gen_report.py`
- `verification_matrix.md` — auto-regenerated by `iterate_until_pass.py`

The original integration plan is at
`/home/malinga/.claude/plans/i-will-give-you-foamy-rocket.md`.
