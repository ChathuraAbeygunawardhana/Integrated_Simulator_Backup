# Systolic Array Quick Reference Guide

## Fast Start

### 1. Run Simulation (5 minutes)
```bash
cd rtl
make sim          # Uses Icarus Verilog
make wave         # View waveforms
```

### 2. Change Array Size
Edit `systolic_array_top.sv` or override at instantiation:
```systemverilog
systolic_array_top #(
    .ARRAY_HEIGHT(8),   // Change from 4 to 8
    .ARRAY_WIDTH(8)     // Change from 4 to 8
) dut (...);
```

### 3. Change Memory Layout
Set `mem_layout` input:
```systemverilog
mem_layout = CHANNEL_MAJOR;  // or ROW_MAJOR or COLUMN_MAJOR
```

## Key Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `ARRAY_HEIGHT` | 4 | Number of PE rows | 4-32 |
| `ARRAY_WIDTH` | 4 | Number of PE columns | 4-32 |
| `DATA_WIDTH` | 32 | Bit width of data | 8-32 |
| `ACCUM_WIDTH` | 48 | Accumulator width | 32-64 |
| `ADDR_WIDTH` | 32 | Address bus width | 16-64 |

## Memory Layout Selection Guide

| Layout | Best For | Memory Pattern | Cache Friendly For |
|--------|----------|----------------|-------------------|
| **Channel Major** | Depthwise convolution | `C → H → W` | Channel operations |
| **Row Major** | Standard convolution | `H → W → C` | Spatial row access |
| **Column Major** | Transposed ops | `W → H → C` | Column-wise access |

## Address Calculation Formula

### Channel Major (C, H, W)
```
address = base + c * (H * W) + h * W + w
```

### Row Major (H, W, C)
```
address = base + h * (W * C) + w * C + c
```

### Column Major (W, H, C)
```
address = base + w * (H * C) + h * C + c
```

## Typical Configuration Examples

### MobileNet Depthwise Layer
```systemverilog
mem_layout = CHANNEL_MAJOR;
input_channels = 128;
weight_k = 128;
weight_c = 1;        // Depthwise
weight_kh = 3;
weight_kw = 3;
```

### ResNet Bottleneck
```systemverilog
mem_layout = ROW_MAJOR;
input_channels = 256;
weight_k = 64;
weight_c = 256;
weight_kh = 1;
weight_kw = 1;       // 1x1 convolution
```

### VGG Layer
```systemverilog
mem_layout = ROW_MAJOR;
input_channels = 128;
weight_k = 256;
weight_c = 128;
weight_kh = 3;
weight_kw = 3;
```

## Control Sequence

### Standard Operation Flow
```
1. Configure tensor shapes and base addresses
2. Set memory layout
3. Set tile position (row, col_start, ch_start)
4. Pulse start signal
5. Wait for done signal
6. Read output data
7. Repeat for next tile
```

### Verilog Code Example
```systemverilog
// Step 1-3: Configuration
mem_layout = CHANNEL_MAJOR;
input_channels = 64;
// ... configure all parameters ...
tile_row = 0;
tile_col_start = 0;
tile_ch_start = 0;

// Step 4: Start
@(posedge clk);
start <= 1'b1;
@(posedge clk);
start <= 1'b0;

// Step 5: Wait
@(posedge done);

// Step 6: Read outputs (available on output_data ports)
```

## Performance Metrics

### Throughput (Operations per Cycle)
```
MACs/cycle = ARRAY_HEIGHT × ARRAY_WIDTH
```
Example: 4×4 array = 16 MACs/cycle

### Peak Performance
```
GOPS = (MACs/cycle) × (Clock Frequency in GHz) × 2
```
Example: 16 MACs @ 200MHz = 6.4 GOPS

### Memory Bandwidth Required
```
Input BW = ARRAY_WIDTH × DATA_WIDTH × Clock_Freq (bits/sec)
Weight BW = ARRAY_HEIGHT × DATA_WIDTH × Clock_Freq (during preload)
Output BW = ARRAY_HEIGHT × ARRAY_WIDTH × DATA_WIDTH (during writeback)
```

## Resource Utilization (Approximate)

### FPGA (Xilinx Ultrascale+)
For 4×4 array, 32-bit data:
- **LUTs**: ~8,000 - 12,000
- **FFs**: ~6,000 - 10,000
- **DSPs**: 16 (one per PE)
- **BRAM**: Minimal (mainly for buffering)

### ASIC (28nm)
For 4×4 array, 32-bit data:
- **Area**: ~0.5 - 1.0 mm²
- **Power**: ~50-200 mW @ 200MHz
- **Frequency**: 200-500 MHz (depends on pipeline depth)

## Common Issues & Solutions

### Issue: Timing violation in MAC path
**Solution**: Add pipeline stage in `processing_element.sv`
```systemverilog
// Add register after multiply
always_ff @(posedge clk) begin
    product <= activation_in * weight_reg;
end
always_ff @(posedge clk) begin
    accumulator <= accumulator + product;
end
```

### Issue: High power consumption
**Solution**: Enable clock gating
```systemverilog
// Add in synthesis constraints
set_clock_gating_style -minimum_bitwidth 4
```

### Issue: Large area
**Solution**: 
- Reduce array size
- Share resources across time
- Use smaller data widths

### Issue: Memory bandwidth bottleneck
**Solution**:
- Use wider memory interface
- Add local buffers/caches
- Pipeline memory accesses

## File Organization Checklist

✅ Package first (`systolic_array_pkg.sv`)  
✅ Base modules (`address_generator.sv`, `processing_element.sv`)  
✅ Fetchers (`input_fetcher.sv`, `weight_fetcher.sv`, `output_fetcher.sv`)  
✅ Top level (`systolic_array_top.sv`)  
✅ Testbench last (`systolic_array_tb.sv`)

## Simulation Quick Commands

```bash
# Icarus Verilog
iverilog -g2012 -o sim.vvp -f filelist.f
vvp sim.vvp
gtkwave waveform.vcd

# Verilator (lint)
verilator --lint-only --timing -Wall -f filelist.f

# Questa
vlog -sv -f filelist.f
vsim -c -do "run -all; quit" systolic_array_tb

# VCS
vcs -sverilog -f filelist.f
./simv
```

## Synthesis Quick Commands

```bash
# Yosys
yosys -p "read_verilog -sv *.sv; synth -top systolic_array_top; write_verilog out.v"

# Design Compiler
dc_shell -f synthesis_script.tcl

# Vivado
vivado -mode batch -source synthesis_script.tcl
```

## Debugging Tips

### 1. Check Address Generation
Add this to testbench:
```systemverilog
always @(posedge input_addr_valid[0]) begin
    $display("Input Addr[0]: 0x%08h", input_addr[0]);
end
```

### 2. Monitor PE State
```systemverilog
initial begin
    $monitor("Time=%0t PE[0][0] accum=%0d", $time, 
             dut.gen_pe_rows[0].gen_pe_cols[0].pe_inst.accumulator);
end
```

### 3. Dump All Signals
```systemverilog
initial begin
    $dumpfile("debug.vcd");
    $dumpvars(0, systolic_array_tb);
end
```

## Optimization Strategies

### For Speed
1. Pipeline MAC operation (add FF stages)
2. Use fast adder architectures (Kogge-Stone)
3. Optimize critical paths in address generation
4. Register all outputs

### For Area
1. Time-multiplex smaller array
2. Share address generators
3. Use smaller accumulator width if possible
4. Remove unused features

### For Power
1. Clock gate unused PEs
2. Use lower voltage for non-critical paths
3. Reduce switching activity in accumulators
4. Use multi-Vt cells in synthesis

## License & Support

This RTL is provided as-is for educational purposes.
Refer to main documentation for full details.

---
Last Updated: 2026-01-28
