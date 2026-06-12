# Weight Stationary Systolic Array - Complete Implementation

## Overview

This is a fully parameterized implementation of a **Weight Stationary Systolic Array** with configurable memory layouts, provided in both **Python** (reference model) and **Verilog/SystemVerilog** (hardware RTL).

### Key Difference from Output Stationary

**Weight Stationary Dataflow:**
- ✅ Weights remain **stationary** in each PE (loaded once, never move)
- ✅ Inputs are **broadcast horizontally** across each row
- ✅ Partial sums **flow diagonally** through the array
- ✅ Each row produces outputs for different output channels
- ✅ Best for: High weight reuse scenarios, reducing weight memory bandwidth

**Comparison:**

| Aspect | Weight Stationary | Output Stationary |
|--------|------------------|-------------------|
| **Weights** | Stationary in PEs | Stream horizontally |
| **Inputs** | Broadcast per row | Stream vertically |
| **Outputs** | Flow diagonally | Accumulate in place |
| **Best For** | Weight-heavy models | Deep accumulation |
| **Weight Reuse** | Maximum | Medium |
| **Input Bandwidth** | Higher | Lower |

## Architecture

### Weight Stationary Dataflow Visualization

```
Input (broadcast horizontally) →  →  →  →
┌────┬────┬────┬────┐
│W₀₀ │W₀₁ │W₀₂ │W₀₃ │ → Outputs for Channel 0
├────┼────┼────┼────┤
│W₁₀ │W₁₁ │W₁₂ │W₁₃ │ → Outputs for Channel 1
├────┼────┼────┼────┤
│W₂₀ │W₂₁ │W₂₂ │W₂₃ │ → Outputs for Channel 2
├────┼────┼────┼────┤
│W₃₀ │W₃₁ │W₃₂ │W₃₃ │ → Outputs for Channel 3
└────┴────┴────┴────┘
 Partial sums flow diagonally ↘

W = Stationary Weight (one per PE)
Each row processes a different output channel
```

### Data Movement Patterns

1. **Weight Loading**: Each PE receives ONE weight value and keeps it
2. **Input Broadcast**: Same input values sent to all PEs in a row
3. **Partial Sum Flow**: Results propagate diagonally (upper-left → lower-right)
4. **Output Collection**: Final results emerge from rightmost column

## Package Contents

### Python Implementation
- `systolic_fetchers_ws.py` (19KB) - Complete reference implementation
- `interactive_demo_ws.py` (12KB) - Interactive demonstration tool

### Verilog RTL Implementation
- `weight_stationary_pkg.sv` - Package definitions
- `address_generator.sv` - Address calculation module
- `input_fetcher_ws.sv` - Input broadcast address generation
- `weight_fetcher_ws.sv` - Weight preload addresses
- `output_fetcher_ws.sv` - Output writeback addresses
- `processing_element_ws.sv` - PE with weight stationary logic
- `weight_stationary_top.sv` - Complete integrated system
- `weight_stationary_tb.sv` - Testbench
- `Makefile` - Build automation
- `filelist.f` - Source file list

## Memory Layouts

All three layouts are supported:

### 1. Channel Major (C, H, W)
```
address = base + c*(H*W) + h*W + w
```
**Best for**: Depthwise separable convolutions, channel-parallel ops

### 2. Row Major (H, W, C)
```
address = base + h*(W*C) + w*C + c
```
**Best for**: Standard convolutions, most CNN layers

### 3. Column Major (W, H, C)
```
address = base + w*(H*C) + h*C + c
```
**Best for**: Transposed operations, specialized access patterns

## Quick Start

### Python

```bash
cd weight_stationary

# Run interactive demo
python3 interactive_demo_ws.py

# Run basic demo
python3 systolic_fetchers_ws.py
```

### Verilog RTL

```bash
cd weight_stationary/rtl

# Simulate with Icarus Verilog
make sim

# View waveforms
make wave
```

## API Reference

### Python

#### SystolicArrayController
```python
from systolic_fetchers_ws import SystolicArrayController, MemoryLayout

controller = SystolicArrayController(
    config=SystolicArrayConfig(array_height=4, array_width=4),
    layout=MemoryLayout.CHANNEL_MAJOR
)

controller.set_memory_regions(
    input_base=0x10000000,
    weight_base=0x20000000,
    output_base=0x30000000
)

result = controller.compute_conv_tile(
    input_shape=(3, 8, 8),
    weight_shape=(16, 3, 3, 3),
    output_shape=(16, 6, 6),
    output_row_start=0,
    output_col=0,
    output_channel_start=0,
    input_channel=0,
    kernel_row=0,
    kernel_col_start=0
)
```

**Returns:**
```python
{
    'weight_addresses': [[addr, addr, ...], ...],  # 2D array (H×W)
    'input_addresses': [[addr, ...], ...],          # Per row
    'output_addresses': [(addr, (k,h,w)), ...],     # Per row
    'layout': 'channel_major',
    'dataflow': 'weight_stationary'
}
```

### Verilog

#### Module: weight_stationary_top

**Parameters:**
```systemverilog
parameter int ARRAY_HEIGHT = 4;      // Number of rows
parameter int ARRAY_WIDTH  = 4;      // Number of columns
parameter int DATA_WIDTH   = 32;     // Data path width
parameter int ACCUM_WIDTH  = 48;     // Accumulator width
parameter int ADDR_WIDTH   = 32;     // Address width
```

**Key Ports:**
```systemverilog
// Configuration
input  mem_layout_t  mem_layout;
input  logic [15:0]  input_channels, input_height, input_width;
input  logic [15:0]  weight_k, weight_c, weight_kh, weight_kw;
input  logic [15:0]  output_channels, output_height, output_width;

// Control
input  logic         start;
input  logic [15:0]  tile_row_start, tile_col, tile_ch_start;
input  logic [15:0]  input_ch, kernel_row, kernel_col_start;
output logic         done, busy;

// Memory interfaces (inputs, weights, outputs)
// See module for full interface
```

## Usage Examples

### Example 1: Standard 3×3 Convolution

```python
# Python
controller = SystolicArrayController(
    SystolicArrayConfig(array_height=4, array_width=4),
    MemoryLayout.ROW_MAJOR
)

result = controller.compute_conv_tile(
    input_shape=(64, 56, 56),     # 64 channels, 56×56 spatial
    weight_shape=(128, 64, 3, 3), # 128 output, 64 input, 3×3 kernel
    output_shape=(128, 54, 54),   # 128 channels, 54×54 output
    output_row_start=0,
    output_col=0,
    output_channel_start=0,
    input_channel=0,
    kernel_row=0,
    kernel_col_start=0
)
```

```systemverilog
// Verilog
mem_layout = ROW_MAJOR;
input_channels = 64;
input_height = 56;
input_width = 56;
weight_k = 128;
weight_c = 64;
weight_kh = 3;
weight_kw = 3;
output_channels = 128;
output_height = 54;
output_width = 54;

tile_row_start = 0;
tile_col = 0;
tile_ch_start = 0;
input_ch = 0;
kernel_row = 0;
kernel_col_start = 0;

start = 1'b1;
```

### Example 2: Depthwise Separable Convolution

```python
# Python - Depthwise phase
controller = SystolicArrayController(
    SystolicArrayConfig(array_height=4, array_width=4),
    MemoryLayout.CHANNEL_MAJOR  # Better for depthwise
)

result = controller.compute_conv_tile(
    input_shape=(128, 28, 28),
    weight_shape=(128, 1, 3, 3),  # Depthwise: 1 input channel per filter
    output_shape=(128, 26, 26),
    ...
)
```

## Design Considerations

### When to Use Weight Stationary

**Advantages:**
- ✅ Minimal weight data movement (loaded once)
- ✅ Excellent for models with large kernels
- ✅ Good for weight-bound applications
- ✅ Simplifies weight memory system

**Disadvantages:**
- ❌ Higher input data movement (broadcast to all PEs in row)
- ❌ More complex partial sum routing (diagonal flow)
- ❌ May not be optimal for 1×1 convolutions

### Performance Metrics

**Throughput:**
```
MACs/cycle = ARRAY_HEIGHT × ARRAY_WIDTH
```

**Memory Bandwidth Required:**
- **Inputs**: `ARRAY_HEIGHT × DATA_WIDTH × Clock` (higher than output stationary)
- **Weights**: One-time load, then stationary
- **Outputs**: `ARRAY_HEIGHT × DATA_WIDTH` (per column completed)

**Weight Reuse:**
```
Reuse Factor = (Output_H × Output_W) / (KH × KW)
```
Weight stationary maximizes this reuse.

## Differences from Output Stationary

| Feature | Weight Stationary | Output Stationary |
|---------|------------------|-------------------|
| **Weight Movement** | None (stationary) | Stream horizontally |
| **Input Movement** | Broadcast horizontally | Stream vertically |
| **Partial Sum Path** | Diagonal flow | In-place accumulation |
| **Output Location** | End of each row | In each PE |
| **Weight Reuse** | Maximum | Medium |
| **Input Bandwidth** | Higher | Lower |
| **Best Application** | Large kernels, weight reuse | Deep networks, many channels |
| **Complexity** | Medium | Lower |

## File Organization

```
weight_stationary/
├── systolic_fetchers_ws.py      # Python implementation
├── interactive_demo_ws.py       # Interactive demo
├── README.md                    # This file
└── rtl/                         # Verilog implementation
    ├── weight_stationary_pkg.sv
    ├── address_generator.sv
    ├── input_fetcher_ws.sv
    ├── weight_fetcher_ws.sv
    ├── output_fetcher_ws.sv
    ├── processing_element_ws.sv
    ├── weight_stationary_top.sv
    ├── weight_stationary_tb.sv
    ├── Makefile
    └── filelist.f
```

## Simulation

### Run Tests

```bash
cd rtl
make sim               # Icarus Verilog
make sim SIM=verilator # Verilator
make sim SIM=questa    # Questa/ModelSim
make wave              # View waveforms
```

### Expected Output

```
Weight Stationary Systolic Array TB
========================================

Testing CHANNEL_MAJOR
  Weight addr[0][0]: 0x20000000
  Input addr[0]: 0x10000000
  Output addr[0]: 0x30000000

Testing ROW_MAJOR
  ...

Testing COLUMN_MAJOR
  ...

All tests complete!
```

## Verification

The implementation has been verified with:
- ✅ All three memory layouts
- ✅ Address generation correctness
- ✅ Data flow patterns
- ✅ PE operation
- ✅ Partial sum accumulation

## Extending the Design

### Adding Features

**Support for Stride:**
- Modify `input_fetcher_ws.sv` to skip positions based on stride
- Adjust address calculation

**Multiple Accumulation Modes:**
- Add mode selection to PE
- Support different activation functions

**Sparsity Support:**
- Add zero-skip logic to PEs
- Optimize for sparse weights/activations

## Performance Optimization

### For Higher Throughput
1. Increase array size (8×8, 16×16)
2. Pipeline MAC operations in PE
3. Use wider data paths
4. Optimize memory interface

### For Lower Power
1. Clock gate inactive PEs
2. Use low-power memories for weights
3. Implement activation sparsity
4. Dynamic voltage/frequency scaling

### For Smaller Area
1. Time-multiplex smaller array
2. Share address generators
3. Reduce precision (INT8, INT16)
4. Optimize for specific kernel sizes

## Common Applications

- **CNNs with large kernels**: VGG-style networks
- **Depthwise separable convolutions**: MobileNets, EfficientNets
- **Feature extraction**: First layers of networks
- **Traditional image filtering**: Sobel, Gaussian blur, etc.

## References

1. **Weight Stationary Architecture**: Similar to designs in Google TPU v1
2. **Systolic Arrays**: Kung & Leiserson, 1979
3. **DNN Accelerators**: Various academic and commercial implementations

## License

Provided for educational and research purposes.

---

**Version**: 1.0  
**Date**: 2026-01-31  
**Dataflow**: Weight Stationary  
**Status**: Tested and verified
