# Systolic Array Optimizer

A production-ready Python framework for mapping TensorFlow/Keras models onto output-stationary systolic array hardware accelerators. Covers the full stack: loop-order analysis, multi-level blocking, inter-layer data-reuse scheduling, hardware simulation (cocotb / Vivado xsim), and golden-model verification.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)
4. [Core Modules](#core-modules)
5. [Hardware Flow (FPGA / Simulation)](#hardware-flow)
6. [Examples](#examples)
7. [Configuration Reference](#configuration-reference)
8. [Requirements](#requirements)

---

## Architecture Overview

```
TensorFlow Model (pretrained)
         │
         ▼
┌─────────────────────────┐
│   Metadata Table        │  metadata_table.py
│  • Layer topology       │  Extracts shapes, dependency graph,
│  • Tensor lifetimes     │  and blocking factors per layer.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Loop Optimizer        │  loop_optimizer.py
│  • 720 permutations     │  Scores all LoopOrder6D variants for
│  • 6 named presets      │  register reuse, memory traffic.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Blocking Optimizer    │  blocking_optimizer.py
│  • Inter-tile ordering  │  Determines L1/L2 tile sizes and the
│  • Tile-size search     │  best (tile, order) pair.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Inter-Layer Reuse     │  inter_layer_reuse.py
│  • On-chip retention    │  Schedules output reuse, weight reuse,
│  • Fusion detection     │  and vertical/horizontal fusion chains.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   TF Systolic Wrapper   │  tf_systolic_wrapper.py
│  • Register map         │  Drives cocotb / xsim testbenches,
│  • Simulation hooks     │  generates JSON configs per layer.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Hardware Verifier     │  hardware_verifier.py
│  • Fixed-point vectors  │  Compares hardware output against
│  • Tolerance checking   │  TensorFlow golden model.
└─────────────────────────┘
```

---

## Repository Layout

```
systolic_optimizer/
├── src/                        # Core library
│   ├── __init__.py
│   ├── loop_optimizer.py       # LoopOrder6D enum + LoopOptimizer
│   ├── blocking_optimizer.py   # BlockingOptimizer (tile-first approach)
│   ├── metadata_table.py       # LayerMetadata + MetadataTable
│   ├── inter_layer_reuse.py    # Inter-layer reuse scheduler
│   ├── tf_systolic_wrapper.py  # TF ↔ systolic array interface
│   ├── enhanced_wrapper.py     # Extended wrapper with extra hooks
│   ├── analyze_model.py        # CLI model analysis & perf estimation
│   ├── hardware_verification.py # Golden-model comparison primitives
│   ├── hardware_verifier.py    # Full verification flow class
│   └── tf_hardware_verification.py  # TF model → hardware test vectors
│
├── examples/                   # Runnable end-to-end demos
│   ├── complete_system_demo.py # Full pipeline (recommended entry point)
│   ├── blocking_demo.py        # Blocking-first optimization demo
│   ├── loop_order_examples.py  # Loop-order enumeration examples
│   └── run_verification.py     # MNIST model verification demo
│
├── tests/
│   └── test_systolic_cocotb.py # CocoTB hardware testbench
│
├── hardware/                   # FPGA / simulation build files
│   ├── Makefile.cocotb         # CocoTB + Verilator flow
│   ├── Makefile.vivado         # Vivado xsim flow
│   └── vivado_setup.py         # Python Vivado API
│
├── configs/                    # Example JSON configs (generated outputs)
│   ├── network_config.example.json
│   └── optimized_loop_orders.example.json
│
├── docs/
│   └── vivado.md               # Full Vivado integration guide
│
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Minimum:** Python 3.9+, TensorFlow 2.12+, NumPy 1.23+.  
> CocoTB and Vivado are only needed for hardware simulation.

### 2 — Analyse a model

```python
import sys, os
sys.path.insert(0, "src")

import tensorflow as tf
from analyze_model import ModelAnalyzer

model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3))

analyzer = ModelAnalyzer(array_height=8, array_width=8, clock_freq_mhz=200)
report = analyzer.analyze(model)
print(report)
```

### 3 — Optimize loop orders

```python
from loop_optimizer import LoopOrder6D, LoopOptimizer, LoopTileConfig

tile = LoopTileConfig(tile_k=8, tile_c=8, tile_hin=8, tile_win=8)
opt  = LoopOptimizer(array_height=8, array_width=8)

results = {}
for order in LoopOrder6D:
    s = opt.analyze_memory_access_pattern(
        order, k=64, c=32, kh=3, kw=3, h_in=14, w_in=14, tile_config=tile
    )
    results[order.name] = s["weight_reuse"] + s["input_reuse"] + s["output_reuse"]

best = max(results, key=results.get)
print(f"Best loop order: {best}  (combined reuse: {results[best]:.2f})")
```

### 4 — Run the complete pipeline

```bash
python examples/complete_system_demo.py
```

This builds a small CNN, runs the full optimizer stack, and prints a per-layer execution schedule.

### 5 — Verify against hardware

```bash
# Software verification (no HDL required)
python examples/run_verification.py

# CocoTB hardware-in-the-loop
make -C hardware -f Makefile.cocotb test

# Vivado xsim
make -C hardware -f Makefile.vivado sim
```

---

## Core Modules

### `loop_optimizer.py`

Enumerates all 6! = 720 permutations of the six outer convolution loops and scores them for memory reuse.

```python
from loop_optimizer import LoopOrder6D, LoopOptimizer, LoopTileConfig

# Six named presets for common access patterns
LoopOrder6D.K_C_KH_KW_HIN_WIN   # Balanced baseline
LoopOrder6D.C_K_KH_KW_HIN_WIN   # Channel-first (input reuse)
LoopOrder6D.HIN_WIN_K_C_KH_KW   # Spatial-first (locality)
LoopOrder6D.KH_KW_C_K_HIN_WIN   # Kernel-spatial first
LoopOrder6D.C_KH_KW_K_HIN_WIN   # Input reuse emphasis
LoopOrder6D.K_HIN_WIN_C_KH_KW   # Output-channel + spatial grouped

tile = LoopTileConfig(tile_k=8, tile_c=8, tile_hin=8, tile_win=8)
opt  = LoopOptimizer(array_height=8, array_width=8)
stats = opt.analyze_memory_access_pattern(
    LoopOrder6D.K_C_KH_KW_HIN_WIN,
    k=64, c=32, kh=3, kw=3, h_in=14, w_in=14, tile_config=tile
)
# stats keys: loop_order, weight_reuse, input_reuse, output_reuse,
#             weight_accesses, input_accesses, output_accesses
```

### `blocking_optimizer.py`

Applies tiling *before* reordering: determines tile sizes, then searches the best inter-tile loop order.

```python
from blocking_optimizer import BlockingOptimizer, TileLoopOrder

opt = BlockingOptimizer(K=128, C=64, H=56, W=56, KH=3, KW=3,
                        array_height=8, array_width=8, scratchpad_kb=32)
best_cfg, all_stats = opt.search_optimal_config()
# best_cfg is a LoopNestConfig: str(best_cfg) → "LoopNest(blocking=4K×16C, order=K_C_SPATIAL)"
```

### `metadata_table.py`

Manages layer topology, tensor shapes, lifetimes, and execution ordering.

```python
from metadata_table import (
    MetadataTable, LayerMetadata, LayerType,
    TensorMetadata, MemoryLocation
)

t_in  = TensorMetadata(name="x",    shape=(1,14,14,32), size_bytes=14*14*32*4,
                       memory_location=MemoryLocation.DRAM)
t_out = TensorMetadata(name="conv1",shape=(1,14,14,64), size_bytes=14*14*64*4,
                       memory_location=MemoryLocation.DRAM)
layer = LayerMetadata(
    layer_id=0, layer_name="conv1", layer_type=LayerType.CONV2D,
    input_tensors=["x"], output_tensor="conv1",
    input_shape=(1,14,14,32), output_shape=(1,14,14,64),
    kernel_shape=(3,3,32,64),
)
table = MetadataTable()
table.add_tensor(t_in); table.add_tensor(t_out); table.add_layer(layer)
table.print_summary()
```

### `inter_layer_reuse.py`

Schedules on-chip data retention and detects fusion opportunities.

```python
from inter_layer_reuse import ReuseStrategy
# ReuseStrategy: NO_REUSE, OUTPUT_REUSE, WEIGHT_REUSE,
#                VERTICAL_FUSION, HORIZONTAL_FUSION
```

### `tf_systolic_wrapper.py`

The main integration point: wraps a Keras model, runs per-layer optimization, and drives simulation.

```python
from tf_systolic_wrapper import SystolicArrayWrapper
from loop_optimizer import LoopOrder6D

wrapper = SystolicArrayWrapper(
    array_height=8, array_width=8,
    simulator="verilator",          # or "xsim"
    default_loop_order=LoopOrder6D.K_C_KH_KW_HIN_WIN
)
wrapper.optimize_all_layers(model)
success, metrics = wrapper.verify_layer(model, "conv1", test_input)
```

---

## Hardware Flow

### CocoTB / Verilator (recommended for development)

```bash
make -C hardware -f Makefile.cocotb test
```

Requires: `cocotb >= 1.8`, `verilator`.

### Vivado xsim (FPGA deployment)

```bash
# First run — compile + elaborate
make -C hardware -f Makefile.vivado project

# Simulate
make -C hardware -f Makefile.vivado sim

# With waveform GUI
make -C hardware -f Makefile.vivado sim-gui
```

Requires: Vivado 2024.1+. See [`docs/vivado.md`](docs/vivado.md) for full setup, supported FPGA targets, and CI/CD integration.

---

## Examples

| Script | What it shows |
|---|---|
| `examples/complete_system_demo.py` | Full pipeline from Keras model to execution schedule |
| `examples/blocking_demo.py` | Blocking-first optimizer with tile-size search |
| `examples/loop_order_examples.py` | Loop-order enumeration and scoring |
| `examples/run_verification.py` | MNIST model → hardware verification flow |

Run any example from the repo root:

```bash
python examples/complete_system_demo.py
```

---

## Configuration Reference

Generated JSON configs are written per-layer. See [`configs/network_config.example.json`](configs/network_config.example.json) for the schema:

```jsonc
{
  "network_name": "SimpleCNN",
  "array_config": { "height": 4, "width": 4, "clock_mhz": 100 },
  "layers": [
    {
      "id": 0, "name": "conv1", "type": "Conv2D",
      "input_shape": [28, 28, 1], "output_shape": [28, 28, 8],
      "kernel": [3, 3],
      "loop_order": "K_C_KH_KW_HIN_WIN",
      "memory": {
        "input_base":  268435456,
        "weight_base": 268500992,
        "output_base": 268566528
      }
    }
  ]
}
```

---

## Requirements

```
tensorflow>=2.12.0
numpy>=1.23.0
cocotb>=1.8.0          # hardware simulation only
cocotb-bus>=0.2.1      # hardware simulation only
matplotlib>=3.7.0      # optional — visualisation
pandas>=2.0.0          # optional — reporting
tqdm>=4.65.0           # optional — progress bars
pytest>=7.3.0          # development
```

Install all:

```bash
pip install -r requirements.txt
```
