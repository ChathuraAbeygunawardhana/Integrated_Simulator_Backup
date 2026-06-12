# Systolic Array RTL Documentation

## Overview

This is a fully parameterized Verilog/SystemVerilog implementation of an **Output Stationary Systolic Array** with configurable memory layouts. The design generates virtual addresses for input fetching, weight preloading, and output writeback based on user-selectable memory organization.

## Features

- ✅ **Fully Parameterized**: Configurable array dimensions, data widths, and tensor shapes
- ✅ **Three Memory Layouts**: Channel Major, Row Major, Column Major
- ✅ **Output Stationary Dataflow**: Weights stay stationary, inputs stream vertically
- ✅ **Modular Design**: Separate fetcher modules for clean architecture
- ✅ **Ready for Simulation**: Includes testbench and simulation scripts
- ✅ **Synthesis Ready**: Clean RTL with no simulation-only constructs

## File Structure

```
rtl/
├── systolic_array_pkg.sv      # Package with common definitions
├── address_generator.sv       # Address calculation module
├── input_fetcher.sv           # Input activation fetcher
├── weight_fetcher.sv          # Weight prefetch module
├── output_fetcher.sv          # Output writeback module
├── processing_element.sv      # Single PE (MAC unit)
├── systolic_array_top.sv      # Top-level integration
├── systolic_array_tb.sv       # Testbench
├── filelist.f                 # File list for compilation
├── Makefile                   # Build automation
└── README.md                  # This file
```

## Architecture

### Memory Layouts

#### 1. Channel Major (C, H, W)
```
Memory: [C0[H0[W0,W1,W2,...]H1[...]...]C1[...]...]
Use Case: Channel-parallel operations, depthwise convolutions
Advantage: Spatial locality within each channel
```

#### 2. Row Major (H, W, C)
```
Memory: [H0[W0[C0,C1,C2,...]W1[...]...]H1[...]...]
Use Case: Standard convolutions, spatial operations
Advantage: All channels of a position are contiguous
```

#### 3. Column Major (W, H, C)
```
Memory: [W0[H0[C0,C1,C2,...]H1[...]...]W1[...]...]
Use Case: Column-wise processing
Advantage: Column-first spatial locality
```

### Systolic Array Dataflow

```
Output Stationary Architecture:
- Weights: Preloaded into PEs and remain stationary
- Inputs: Stream vertically down through columns
- Outputs: Accumulate in place within each PE

     Input Stream →
          ↓  ↓  ↓  ↓
    ┌────┬────┬────┬────┐
  W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
    └────┴────┴────┴────┘
    Output (accumulated)
```

## Module Descriptions

### 1. systolic_array_pkg.sv
SystemVerilog package containing:
- `mem_layout_t` enum for memory layout selection
- Common data structures
- Type definitions

### 2. address_generator.sv
Core address calculation engine:
- **Inputs**: Multi-dimensional indices, tensor shape, layout type
- **Outputs**: Linear memory address
- **Function**: Converts (c,h,w) coordinates to linear addresses based on layout

**Parameters**:
- `DATA_WIDTH`: Address width (default: 32)

**Key Signals**:
- `layout`: Memory layout selection
- `shape_c/h/w`: Tensor dimensions
- `idx_0/1/2`: Input indices (interpretation varies by layout)
- `linear_addr`: Computed address output

### 3. input_fetcher.sv
Fetches input activations for the array:
- Generates addresses for streaming inputs
- Handles kernel window sliding
- Outputs addresses for each array column

**Parameters**:
- `ARRAY_WIDTH`: Number of columns
- `DATA_WIDTH`: Address width
- `MAX_CHANNELS`: Maximum input channels
- `MAX_KERNEL`: Maximum kernel dimension

**Key Signals**:
- `kernel_h/w`: Kernel dimensions
- `output_row/col_start`: Position being computed
- `addr_out[ARRAY_WIDTH]`: Addresses for each column
- `fetch_done`: Completion signal

### 4. weight_fetcher.sv
Preloads weights into PEs:
- Generates addresses for all weights of output channels
- Supports tiled computation
- One address stream per PE row

**Parameters**:
- `ARRAY_HEIGHT`: Number of rows
- `DATA_WIDTH`: Address width

**Key Signals**:
- `weight_k/c/kh/kw`: Weight tensor dimensions
- `output_ch_start`: Starting output channel
- `addr_out[ARRAY_HEIGHT]`: Addresses for each PE row
- `weight_idx_out`: Weight index tracking

### 5. output_fetcher.sv
Generates writeback addresses:
- Computes addresses for output tile
- 2D array of addresses (height × width)
- Validates positions against output bounds

**Parameters**:
- `ARRAY_HEIGHT/WIDTH`: Array dimensions
- `DATA_WIDTH`: Address width

**Key Signals**:
- `output_k/h/w`: Output tensor dimensions
- `channel/row/col_start`: Tile position
- `addr_out[HEIGHT][WIDTH]`: 2D address array

### 6. processing_element.sv
Single MAC unit:
- Multiply-accumulate operation
- Stationary weight storage
- Vertical activation propagation
- Horizontal partial sum propagation

**Parameters**:
- `DATA_WIDTH`: Input/weight width (default: 32)
- `ACCUM_WIDTH`: Accumulator width (default: 48)

**Operation**: `result = Σ(activation × weight) + partial_sum`

### 7. systolic_array_top.sv
Top-level integration:
- Instantiates all fetchers
- Creates PE array
- FSM for control flow
- Memory interface management

**Parameters**:
- `ARRAY_HEIGHT/WIDTH`: Array dimensions
- `DATA_WIDTH`: Data path width
- `ACCUM_WIDTH`: Accumulator width
- `ADDR_WIDTH`: Address width

## Parameter Configuration

### Common Configurations

#### Small Array (4×4)
```systemverilog
parameter ARRAY_HEIGHT = 4;
parameter ARRAY_WIDTH  = 4;
parameter DATA_WIDTH   = 32;
parameter ACCUM_WIDTH  = 48;
```

#### Medium Array (8×8)
```systemverilog
parameter ARRAY_HEIGHT = 8;
parameter ARRAY_WIDTH  = 8;
parameter DATA_WIDTH   = 32;
parameter ACCUM_WIDTH  = 48;
```

#### Large Array (16×16)
```systemverilog
parameter ARRAY_HEIGHT = 16;
parameter ARRAY_WIDTH  = 16;
parameter DATA_WIDTH   = 32;
parameter ACCUM_WIDTH  = 64;  // Larger for more accumulation
```

## Simulation

### Prerequisites
Install a Verilog/SystemVerilog simulator:
- **Icarus Verilog** (open source): `sudo apt-get install iverilog gtkwave`
- **Verilator** (open source): `sudo apt-get install verilator`
- **Questa/ModelSim** (commercial)
- **Synopsys VCS** (commercial)

### Running Simulation

#### Quick Start (Icarus Verilog)
```bash
cd rtl
make sim
```

#### With Other Simulators
```bash
# Verilator (lint check)
make sim SIM=verilator

# Questa/ModelSim
make sim SIM=questa

# Synopsys VCS
make sim SIM=vcs
```

#### View Waveforms
```bash
make wave
```

### Expected Output
```
========================================
Systolic Array Testbench
========================================
[0] Reset complete

========================================
Testing CHANNEL_MAJOR Layout
========================================
[100] Configuration:
  Layout: CHANNEL_MAJOR
  Input: C=3, H=8, W=8
  Weight: K=16, C=3, KH=3, KW=3
  Output: K=16, H=6, W=6
[200] Starting tile computation:
  Row=0, Col_start=0, Ch_start=0
[500] Tile computation complete

Sample Input Addresses:
  Column[0]: 0x10000000
  Column[1]: 0x10000001
  ...
```

## Synthesis

### Yosys (Open Source)
```bash
make synthesize
```

Output: `syn/systolic_array_top_synth.v`

### Commercial Tools

#### Synopsys Design Compiler
```tcl
read_verilog -sv [list systolic_array_pkg.sv \
                      address_generator.sv \
                      processing_element.sv \
                      input_fetcher.sv \
                      weight_fetcher.sv \
                      output_fetcher.sv \
                      systolic_array_top.sv]

set_top_module systolic_array_top
compile
write -format verilog -output systolic_array_top_synth.v
```

#### Xilinx Vivado
```tcl
read_verilog -sv [glob *.sv]
synth_design -top systolic_array_top -part xc7z020clg400-1
write_verilog systolic_array_top_synth.v
```

## Interface Specification

### Configuration Interface
Configure tensor shapes and memory layout before operation:
```systemverilog
mem_layout          // Memory layout selection
input_channels      // Input feature map channels
input_height        // Input feature map height
input_width         // Input feature map width
weight_k            // Number of filters (output channels)
weight_c            // Input channels (must match input_channels)
weight_kh           // Kernel height
weight_kw           // Kernel width
output_channels     // Output channels (typically equals weight_k)
output_height       // Output spatial height
output_width        // Output spatial width
input_base_addr     // Base address of input tensor
weight_base_addr    // Base address of weight tensor
output_base_addr    // Base address of output tensor
```

### Control Interface
Start computation and monitor status:
```systemverilog
start               // Pulse to start computation
tile_row            // Output row for current tile
tile_col_start      // Starting column for tile
tile_ch_start       // Starting output channel
done                // Pulse when computation complete
busy                // High during computation
```

### Memory Interface
Standard memory read/write interface:
```systemverilog
// Input
input_addr[ARRAY_WIDTH]          // Read addresses
input_addr_valid[ARRAY_WIDTH]    // Address valid flags
input_data[ARRAY_WIDTH]          // Data from memory
input_data_valid[ARRAY_WIDTH]    // Data valid flags

// Weight (similar structure)
// Output (similar structure with 2D arrays)
```

## Timing Diagrams

### Weight Preload Phase
```
        ┌─┐ ┌─┐ ┌─┐ ┌─┐
clk     ┘ └─┘ └─┘ └─┘ └─
        ┌─────────┐
start   ┘         └─────
                ┌───────
busy    ────────┘
                ┌─┐
weight_addr_valid ─┘ └─
        ────┬───────────
weight_addr ────X Addr  X
```

### Computation Phase
```
        ┌─┐ ┌─┐ ┌─┐ ┌─┐
clk     ┘ └─┘ └─┘ └─┘ └─
        ┌───────────────
busy    ┘
        ────┬─┬─┬─┬─────
input_data  X0│1│2│3│
        ────┬─┬─┬─┬─────
pe_result   X X X X Final
```

## Design Considerations

### Area
- **PEs**: Dominant area (HEIGHT × WIDTH MACs)
- **Fetchers**: Relatively small
- **Memory Interface**: Width determines I/O pins

**Scaling**: 16×16 array ≈ 256 MACs + control logic

### Timing
- **Critical Path**: Typically through MAC operation
- **Target**: 100-500 MHz depending on technology
- **Pipelining**: Can add pipeline stages in PE for higher frequency

### Memory Bandwidth
Required bandwidth per cycle:
- **Input**: `ARRAY_WIDTH × DATA_WIDTH` bits/cycle
- **Weight**: `ARRAY_HEIGHT × DATA_WIDTH` bits/cycle (during preload)
- **Output**: `ARRAY_HEIGHT × ARRAY_WIDTH × DATA_WIDTH` bits/cycle (during writeback)

### Power
- **Dynamic**: Dominated by MAC operations
- **Leakage**: Array size dependent
- **Optimization**: Clock gate unused PEs, use low-power memories

## Usage Examples

### Example 1: 3×3 Convolution
```systemverilog
// Configuration
mem_layout = CHANNEL_MAJOR;
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

// Compute first tile
tile_row = 0;
tile_col_start = 0;
tile_ch_start = 0;
start = 1;  // Pulse
```

### Example 2: Depthwise Convolution
```systemverilog
mem_layout = CHANNEL_MAJOR;  // Best for depthwise
input_channels = 128;
weight_k = 128;
weight_c = 1;  // Depthwise: one filter per input channel
weight_kh = 3;
weight_kw = 3;
```

## Verification

### Testbench Coverage
- ✅ All three memory layouts
- ✅ Address generation correctness
- ✅ FSM state transitions
- ✅ Data flow through PE array
- ✅ Boundary conditions

### Assertions
Consider adding SystemVerilog assertions:
```systemverilog
// Example assertions
assert property (@(posedge clk) 
    disable iff (!rst_n)
    start |-> ##[1:1000] done
) else $error("Timeout waiting for done");

assert property (@(posedge clk)
    busy |-> !done
) else $error("Done asserted while busy");
```

## Known Limitations

1. **No Padding Support**: Currently assumes valid convolution
2. **No Stride Support**: Assumes stride=1
3. **Simplified Memory Model**: Real memory requires proper timing
4. **No Error Detection**: No parity/ECC on data paths
5. **Fixed Point Only**: No floating-point support

## Future Enhancements

- [ ] Add padding support (SAME padding)
- [ ] Variable stride support
- [ ] AXI4 memory interface
- [ ] Performance counters
- [ ] Power management (clock gating)
- [ ] Support for dilated convolutions
- [ ] Multi-precision support (INT8, FP16, FP32)
- [ ] Sparse tensor support

## References

1. **Eyeriss**: Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks", ISCA 2016
2. **Systolic Arrays**: H.T. Kung and Charles E. Leiserson, "Systolic Arrays (for VLSI)", 1979
3. **Output Stationary**: Maintains partial sums in PEs for energy efficiency

## License

This RTL implementation is provided for educational and research purposes.

## Contact

For questions or issues, please refer to the main repository documentation.
