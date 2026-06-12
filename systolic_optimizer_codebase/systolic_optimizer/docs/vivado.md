# Vivado Integration Guide

## Overview

The hardware flow supports two simulators:
- **CocoTB + Verilator** — fast, open-source, recommended for development
- **Vivado xsim** — for pre-synthesis validation and FPGA deployment

## Prerequisites

- Vivado 2024.1+ (free WebPACK edition works for Xilinx 7-series)
- Python 3.9+, `subprocess` in stdlib
- SystemVerilog source files for the systolic array RTL

## Quick Start

```bash
# Verify installation
vivado -version && xvlog -version && xelab -version && xsim -version

# Build project (first time)
make -C hardware -f Makefile.vivado project

# Batch simulation
make -C hardware -f Makefile.vivado sim

# GUI simulation
make -C hardware -f Makefile.vivado sim-gui
```

## Python API

```python
from hardware.vivado_setup import VivadoSimulator

sim = VivadoSimulator(project_name="systolic_array_sim")
success = sim.run_full_flow(
    verilog_sources=["systolic_spad_top.sv", "scratchpad_ctrl.sv", ...],
    testbench_files=["systolic_spad_tb.sv"],
    gui=False
)
```

## Supported FPGA Targets

Change `FPGA_PART` in `hardware/Makefile.vivado`:

| Board family | Part number |
|---|---|
| Zynq-7000 (default) | `xc7z020clg400-1` |
| Artix-7 | `xc7a200tsbg484-1` |
| Zynq UltraScale+ | `xczu9eg-ffvb1156-2-e` |

## Troubleshooting

**`vivado: command not found`**
```bash
# Linux
source /tools/Xilinx/Vivado/2024.1/settings64.sh
```

**`LoopOrder not found`** — use `LoopOrder6D` (the `LoopOrder` enum from earlier iterations has been removed).

**Simulation hangs** — add `XSIM_FLAGS="-maxdelta 1000"` on the make command line.

## CI/CD (GitHub Actions)

```yaml
jobs:
  simulate:
    runs-on: ubuntu-latest
    container:
      image: xilinx/vivado:2024.1
    steps:
      - uses: actions/checkout@v3
      - run: make -C hardware -f Makefile.vivado sim
```
