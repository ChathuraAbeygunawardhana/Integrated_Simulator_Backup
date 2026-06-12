# Output Stationary Systolic Array Address Generator

A comprehensive implementation of virtual address generation for output-stationary systolic arrays with configurable memory layouts.

## Overview

This system generates virtual memory addresses for:
- **Input Activation Fetching**: Streaming inputs to the systolic array
- **Weight Prefetching**: Preloading stationary weights into Processing Elements (PEs)
- **Output Writeback**: Writing computed results back to memory

### Supported Memory Layouts

1. **Channel Major (C, H, W)**: Channels vary slowest
   - Layout: `[C0[H0[W0,W1,...]H1[...]]C1[...]...]`
   - Best for: Channel-parallel operations, depthwise convolutions
   - Spatial positions of one channel are contiguous in memory

2. **Row Major (H, W, C)**: Rows vary slowest
   - Layout: `[H0[W0[C0,C1,...]W1[...]]H1[...]...]`
   - Best for: Spatial operations, pointwise convolutions
   - All channels of one spatial position are contiguous

3. **Column Major (W, H, C)**: Columns vary slowest
   - Layout: `[W0[H0[C0,C1,...]H1[...]]W1[...]...]`
   - Best for: Column-wise access patterns
   - Similar to row major but prioritizes column locality

## Architecture

### Output Stationary Dataflow

In an output stationary systolic array:
- **Weights**: Preloaded and remain stationary in PEs
- **Inputs**: Stream vertically down through the array
- **Outputs**: Accumulate in place within each PE
- **Array Dimensions**: Height × Width (e.g., 4×4, 8×8, 16×16)

```
     Input Stream (vertical)
          ↓  ↓  ↓  ↓
    ┌────┬────┬────┬────┐
→ W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
→ W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
→ W │ PE │ PE │ PE │ PE │
    ├────┼────┼────┼────┤
→ W │ PE │ PE │ PE │ PE │
    └────┴────┴────┴────┘
         Outputs (in PE)

W = Weights (stationary)
PE = Processing Element (MAC unit)
```

## Files

### Core Implementation
- **`systolic_fetchers.py`**: Main implementation with all fetcher classes
  - `AddressGenerator`: Base address calculation logic
  - `InputFetcher`: Input activation address generation
  - `WeightFetcher`: Weight prefetch address generation
  - `OutputFetcher`: Output writeback address generation
  - `SystolicArrayController`: High-level orchestration

### Interactive Tools
- **`interactive_demo.py`**: User-friendly menu-driven demo
  - Select memory layout interactively
  - View detailed address patterns
  - Compare all layouts side-by-side

### Testing
- **`test_fetchers.py`**: Comprehensive test suite
  - Unit tests for all components
  - Layout comparison tests
  - Boundary condition validation

## Installation & Usage

### Prerequisites
```bash
python3 -m pip install numpy
```

### Quick Start

#### 1. Run Interactive Demo
```bash
python3 interactive_demo.py
```

This launches a menu where you can:
- Select different memory layouts
- View address generation details
- Compare layouts side-by-side

#### 2. Run Tests
```bash
python3 test_fetchers.py
```

Validates all functionality with comprehensive unit tests.

#### 3. Run Basic Demo
```bash
python3 systolic_fetchers.py
```

Shows address generation for all three layouts.

### Programmatic Usage

```python
from systolic_fetchers import (
    SystolicArrayController, SystolicArrayConfig, MemoryLayout
)

# Configure systolic array
config = SystolicArrayConfig(array_height=4, array_width=4)

# Choose memory layout
layout = MemoryLayout.CHANNEL_MAJOR  # or ROW_MAJOR or COLUMN_MAJOR

# Create controller
controller = SystolicArrayController(config, layout)

# Set base addresses for memory regions
controller.set_memory_regions(
    input_base=0x10000000,
    weight_base=0x20000000,
    output_base=0x30000000
)

# Generate addresses for convolution tile
result = controller.compute_conv_tile(
    input_shape=(3, 8, 8),      # 3 channels, 8×8 spatial
    weight_shape=(16, 3, 3, 3), # 16 filters, 3 input ch, 3×3 kernel
    output_shape=(16, 6, 6),    # 16 channels, 6×6 output
    output_tile_row=0,
    output_tile_col=0,
    output_channel_start=0
)

# Access generated addresses
weight_addrs = result['weight_addresses']  # Per PE row
input_addrs = result['input_addresses']    # Per array column
output_addrs = result['output_addresses']  # Writeback addresses
```

## API Reference

### SystolicArrayConfig
```python
SystolicArrayConfig(array_height: int, array_width: int)
```
Defines systolic array dimensions.

### MemoryLayout (Enum)
- `CHANNEL_MAJOR`: C, H, W ordering
- `ROW_MAJOR`: H, W, C ordering  
- `COLUMN_MAJOR`: W, H, C ordering

### InputFetcher

#### Methods
```python
set_base_address(addr: int)
```
Set base virtual address for input tensor.

```python
fetch_addresses(
    input_shape: Tuple[int, int, int],  # (C, H, W)
    output_row: int,
    output_col: int,
    kernel_h: int,
    kernel_w: int
) -> List[Tuple[int, Tuple[int, int, int]]]
```
Generate addresses for inputs needed at one output position.
Returns list of (address, (c, h, w)) tuples.

```python
fetch_for_systolic_cycle(
    input_shape: Tuple[int, int, int],
    output_row: int,
    output_col_start: int,
    kernel_h: int,
    kernel_w: int
) -> List[List[int]]
```
Fetch addresses for entire array width (one cycle).
Returns list of address lists, one per column.

### WeightFetcher

#### Methods
```python
set_base_address(addr: int)
```
Set base virtual address for weight tensor.

```python
fetch_addresses(
    weight_shape: Tuple[int, int, int, int],  # (K, C, KH, KW)
    output_channel: int
) -> List[Tuple[int, Tuple[int, int, int, int]]]
```
Generate addresses for weights of one output channel.
Returns list of (address, (k, c, kh, kw)) tuples.

```python
preload_weights_for_array(
    weight_shape: Tuple[int, int, int, int],
    output_channel_start: int
) -> List[List[int]]
```
Preload weights for all PEs in the array.
Returns list of address lists, one per PE row.

### OutputFetcher

#### Methods
```python
set_base_address(addr: int)
```
Set base virtual address for output tensor.

```python
get_output_address(
    output_shape: Tuple[int, int, int],  # (K, H_out, W_out)
    output_channel: int,
    output_row: int,
    output_col: int
) -> int
```
Get address for single output activation.

```python
get_output_tile_addresses(
    output_shape: Tuple[int, int, int],
    channel_start: int,
    row_start: int,
    col_start: int
) -> np.ndarray
```
Get addresses for entire output tile (array_height × array_width).
Returns 2D array of addresses.

```python
writeback_addresses(
    output_shape: Tuple[int, int, int],
    channel_start: int,
    row_start: int,
    col_start: int
) -> List[Tuple[int, Tuple[int, int, int]]]
```
Generate all writeback addresses for completed outputs.
Returns list of (address, (k, h, w)) tuples.

### SystolicArrayController

#### Methods
```python
set_memory_regions(input_base: int, weight_base: int, output_base: int)
```
Configure base addresses for all memory regions.

```python
compute_conv_tile(
    input_shape: Tuple[int, int, int],
    weight_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int],
    output_tile_row: int,
    output_tile_col: int,
    output_channel_start: int
) -> Dict
```
Generate all addresses for computing one output tile.

Returns dictionary:
```python
{
    'weight_addresses': List[List[int]],  # Per PE row
    'input_addresses': List[List[int]],   # Per array column
    'output_addresses': List[Tuple[int, Tuple[int, int, int]]],
    'layout': str  # Layout name
}
```

## Memory Layout Examples

### Example Tensor: Shape (2, 3, 4) - 2 channels, 3×4 spatial

#### Channel Major Layout (C, H, W)
```
Address | Channel | Row | Col | Element
--------|---------|-----|-----|--------
0       | 0       | 0   | 0   | C0H0W0
1       | 0       | 0   | 1   | C0H0W1
2       | 0       | 0   | 2   | C0H0W2
3       | 0       | 0   | 3   | C0H0W3
4       | 0       | 1   | 0   | C0H1W0
...
12      | 1       | 0   | 0   | C1H0W0
```

#### Row Major Layout (H, W, C)
```
Address | Row | Col | Channel | Element
--------|-----|-----|---------|--------
0       | 0   | 0   | 0       | H0W0C0
1       | 0   | 0   | 1       | H0W0C1
2       | 0   | 1   | 0       | H0W1C0
3       | 0   | 1   | 1       | H0W1C1
...
8       | 1   | 0   | 0       | H1W0C0
```

#### Column Major Layout (W, H, C)
```
Address | Col | Row | Channel | Element
--------|-----|-----|---------|--------
0       | 0   | 0   | 0       | W0H0C0
1       | 0   | 0   | 1       | W0H0C1
2       | 0   | 1   | 0       | W0H1C0
3       | 0   | 1   | 1       | W0H1C1
...
6       | 1   | 0   | 0       | W1H0C0
```

## Use Cases by Layout

### Channel Major
**Best for:**
- Depthwise convolutions (process channels independently)
- Channel-wise normalization
- Hardware with channel-parallel datapaths

**Example:** Mobile neural network accelerators

### Row Major
**Best for:**
- Standard convolutions with spatial locality
- Image processing pipelines
- Row-wise access patterns

**Example:** CPU/GPU implementations, general-purpose accelerators

### Column Major
**Best for:**
- Transposed operations
- Column-wise processing
- Certain matrix multiplication patterns

**Example:** Specialized architectures with column-first access

## Performance Considerations

### Cache Efficiency
- **Channel Major**: Exploits channel locality, good for depthwise ops
- **Row Major**: Exploits spatial locality, good for standard convolutions
- **Column Major**: Special cases where column access dominates

### Memory Bandwidth
Different layouts affect:
- Burst access patterns
- Cache line utilization
- Prefetcher effectiveness
- DRAM page hit rates

### Hardware Mapping
Choose layout based on:
1. Memory hierarchy (cache sizes, line sizes)
2. Access pattern of compute kernel
3. DMA transfer granularity
4. Accelerator datapath organization

## Example: 4×4 Array Computing 3×3 Convolution

```
Input: 3 channels, 8×8 spatial
Weights: 16 filters, 3 input channels, 3×3 kernel
Output: 16 channels, 6×6 spatial

Systolic Array: 4 rows × 4 columns

Mapping:
- Array rows → Output channels (4 channels per tile)
- Array columns → Output spatial positions (4 positions per tile)
- Time steps → Accumulate across input channels and kernel

Timeline:
1. Preload weights for channels 0-3 (4 PEs in column 0)
2. Stream inputs for positions (0,0) through (0,3)
3. Accumulate partial sums
4. Writeback outputs when complete
5. Repeat for next tile
```

## Testing

The test suite covers:
- ✓ Address calculation correctness
- ✓ All three memory layouts
- ✓ Boundary conditions
- ✓ Layout comparison
- ✓ Integration testing
- ✓ Memory footprint validation

Run tests:
```bash
python3 test_fetchers.py
```

Expected output:
```
======================================================================
SYSTOLIC ARRAY FETCHER TEST SUITE
======================================================================
test_channel_major_layout ... ok
test_row_major_layout ... ok
...
Tests run: 24
Successes: 24
Failures: 0
Errors: 0
```

## Extending the System

### Adding New Layouts
```python
class MemoryLayout(Enum):
    CHANNEL_MAJOR = "channel_major"
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    CUSTOM_LAYOUT = "custom_layout"  # Add new layout

# Implement in AddressGenerator.get_linear_address()
```

### Custom Systolic Architectures
```python
# Different dataflow (e.g., weight stationary)
class WeightStationaryController(SystolicArrayController):
    def compute_conv_tile(self, ...):
        # Different fetching pattern
        pass
```

## References

- Output Stationary Dataflow: Eyeriss (Chen et al., ISCA 2016)
- Systolic Arrays: H.T. Kung & Charles Leiserson (1979)
- Memory Layouts: NCHW, NHWC in modern DNN frameworks

## License

This implementation is provided for educational and research purposes.

## Author

Claude (Anthropic) - Systolic Array Address Generation System
