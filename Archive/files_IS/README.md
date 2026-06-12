# Input Stationary Systolic Array – Complete Implementation

## Overview

This folder contains a fully parameterized implementation of an
**Input Stationary Systolic Array** in both Python (reference model) and
Verilog/SystemVerilog (hardware RTL).

It is the third dataflow module in the project alongside:
- `files_os/` – Output Stationary
- `files_ws/` – Weight Stationary
- `files_IS/`  – **Input Stationary (this folder)**

---

## Input Stationary Dataflow

### Core Idea

> Keep **input activations** stationary inside each PE.  Stream **weights**
> horizontally across each row.  Accumulate **partial sums** vertically
> through each column.

```
  Weights stream →  →  →  →
  ┌──────┬──────┬──────┬──────┐
  │ I₀₀  │ I₀₁  │ I₀₂  │ I₀₃  │  ↓ psum  (out-ch 0)
  ├──────┼──────┼──────┼──────┤
  │ I₁₀  │ I₁₁  │ I₁₂  │ I₁₃  │  ↓ psum  (out-ch 1)
  ├──────┼──────┼──────┼──────┤
  │ I₂₀  │ I₂₁  │ I₂₂  │ I₂₃  │  ↓ psum  (out-ch 2)
  ├──────┼──────┼──────┼──────┤
  │ I₃₀  │ I₃₁  │ I₃₂  │ I₃₃  │  ↓ psum  (out-ch 3)
  └──────┴──────┴──────┴──────┘

  I[r][c] = stationary input for col c (shared across rows r)
  Weights for output-channel r stream through row r
  Final psum exits bottom of each column → writeback
```

### Data Movement

| Data | Movement | Reuse |
|------|----------|-------|
| **Inputs**   | Stationary (loaded once per tile) | Max – never move after load |
| **Weights**  | Stream horizontally (left → right) | Medium – reused across tile columns |
| **Psums**    | Flow vertically (top → bottom) | — |

---

## When to Use Input Stationary

| Use IS when… | Example |
|---|---|
| Input feature maps are large and accessed repeatedly | VGG early layers |
| Input bandwidth is the bottleneck | Bandwidth-limited FPGA designs |
| Many output channels share the same inputs | Grouped convolutions |
| Comparing all three dataflows for a research study | — |

### Comparison with OS and WS

| Aspect | Output Stationary (OS) | Weight Stationary (WS) | **Input Stationary (IS)** |
|--------|------------------------|------------------------|--------------------------|
| Stationary data | Partial sums (outputs) | Weights | **Inputs** |
| Streaming data | Inputs (vertical) | Weights (horizontal) | **Weights (horizontal)** |
| Accumulated data | In each PE | Diagonal flow | **Vertical column flow** |
| Max reuse | Output reuse | Weight reuse | **Input reuse** |
| Memory BW | Low (input reused) | Low (weight loaded once) | **Low (input loaded once)** |
| Best for | Deep CNNs, many channels | Large kernels, weight-bound | **Large inputs, input-bound** |

---

## File Structure

```
files_IS/
├── input_stationary_pkg.sv      # Package: types, enums
├── address_generator_is.sv      # Address calculation (layout-aware)
├── processing_element_is.sv     # Single PE: MAC with stationary input
├── input_fetcher_is.sv          # Input address generator (column-wise)
├── weight_fetcher_is.sv         # Weight streamer (row-wise)
├── output_fetcher_is.sv         # Output writeback addresses (2D tile)
├── input_stationary_top.sv      # Top-level integration + FSM
├── input_stationary_tb.sv       # Testbench (all 3 layouts)
├── systolic_fetchers_is.py      # Python reference model
├── interactive_demo_is.py       # Interactive CLI demo
├── filelist.f                   # Source file list for simulators
├── Makefile                     # Build automation
└── README.md                    # This file
```

---

## Quick Start

### Python

```bash
cd files_IS

# Run the full demonstration
python3 systolic_fetchers_is.py

# Run the interactive CLI
python3 interactive_demo_is.py
```

### Verilog RTL (Icarus Verilog)

```bash
cd files_IS

make sim           # Icarus Verilog (default)
make wave          # Open GTKWave
```

### Other simulators

```bash
make sim SIM=verilator   # Lint check
make sim SIM=questa      # Questa / ModelSim
make sim SIM=vcs         # Synopsys VCS
```

---

## Module Descriptions

### `input_stationary_pkg.sv`
Package with `mem_layout_t` enum (CHANNEL_MAJOR, ROW_MAJOR, COLUMN_MAJOR)
and common struct types – identical encoding to OS and WS packages.

### `address_generator_is.sv`
Converts (idx_0, idx_1, idx_2) + layout → linear address.
Imports `input_stationary_pkg` so the IS filelist compiles standalone.

### `processing_element_is.sv`
Single MAC unit with:
- `input_reg` – stationary, latched once via `load_input` strobe
- `weight_in` / `weight_out` – passes weight rightward
- `psum_in`   / `psum_out`   – passes partial sum downward
- `accumulator` – accumulated result

### `input_fetcher_is.sv`
Generates one address stream per **column**.  Each column address maps to the
input activation at spatial position `(output_row + kh, output_col_start + j + kw)`
for the current input channel.  Also drives the `load_input` strobe to the PEs.

### `weight_fetcher_is.sv`
Generates one weight address per **row** for a given `(input_ch, kernel_row,
kernel_col)` combination.  The caller's FSM advances these counters after each
`fetch_done` pulse until all `(kh, kw, c)` triples are exhausted.

### `output_fetcher_is.sv`
Generates a 2D grid of output addresses `[ARRAY_HEIGHT][ARRAY_WIDTH]` for the
accumulated PE results, matching the tile position `(channel_start, row_start,
col_start)`.

### `input_stationary_top.sv`
Top-level integration:
1. **LOAD_INPUTS** – input fetcher fills each PE column with stationary values
2. **STREAM_WEIGHTS** – weight fetcher drives one weight per row; FSM iterates
   over all `(kh, kw, c)` combinations
3. **WRITEBACK** – output fetcher generates writeback addresses; PE results
   are placed on `output_data`

### `input_stationary_tb.sv`
Testbench covering all three memory layouts with a 4×4 array and a
3-channel, 8×8 input / 16-filter, 3×3 kernel configuration.

---

## Parameters

```systemverilog
parameter int ARRAY_HEIGHT = 4;   // Number of rows (= output channels per tile)
parameter int ARRAY_WIDTH  = 4;   // Number of cols (= output spatial cols per tile)
parameter int DATA_WIDTH   = 32;  // Data path width
parameter int ACCUM_WIDTH  = 48;  // Accumulator width
parameter int ADDR_WIDTH   = 32;  // Address bus width
```

---

## Memory Layouts

All three layouts are supported identically to the OS and WS implementations.

| Layout | Address formula |
|--------|----------------|
| CHANNEL_MAJOR (C,H,W) | `base + c*(H*W) + h*W + w` |
| ROW_MAJOR (H,W,C)     | `base + h*(W*C) + w*C + c` |
| COLUMN_MAJOR (W,H,C)  | `base + w*(H*C) + h*C + c` |

---

## Known Limitations

- No padding support (valid convolution only)
- Stride = 1 assumed
- Simplified memory model in testbench (1-cycle latency)
- Fixed-point arithmetic only

---

## References

1. **Eyeriss**: Chen et al., ISCA 2016 – dataflow taxonomy
2. **Systolic Arrays**: Kung & Leiserson, 1979
3. **Input Stationary**: Used in research accelerators where input bandwidth
   dominates (e.g. early CNN feature extraction layers)

---

**Version**: 1.0
**Dataflow**: Input Stationary
**Status**: Tested reference model; RTL structurally consistent with OS and WS modules
