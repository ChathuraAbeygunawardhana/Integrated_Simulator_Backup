# Control Knob Reference

Every field of `pysim.sim_config.SimConfig` maps to an RTL parameter or
testbench environment variable. This page is the canonical catalogue.

## Top-level selectors

| Field | Type / values | RTL mapping | Default |
|---|---|---|---|
| `top`        | `single_dnn_top` / `multi_dnn_top` | module name | `single_dnn_top` |
| `dataflow`   | `OS` / `IS` / `WS`                 | `DATAFLOW` param (`dataflow_e` enum) | `OS` |
| `memory`     | `STAMP` / `PAGED`                  | `MEMORY` param (`mem_backend_e` enum) | `STAMP` |
| `scheduler`  | one of 14 names; `None` for single_dnn | `scheduler_select` runtime input (5 bits) | `None` |

Scheduler names map to `scheduler_select` values:

| name | id | family |
|---|---|---|
| FIFO        | 0  | basic |
| LIFO        | 1  | basic |
| SJF         | 2  | basic |
| RR          | 3  | basic |
| PRI         | 4  | basic |
| EDF         | 5  | basic |
| LRU         | 6  | basic |
| SRTF        | 7  | advanced |
| HRRN        | 8  | advanced |
| MLQ         | 9  | advanced |
| MLFQ        | 10 | advanced |
| AIMT        | 11 | DNN-aware |
| BATCHDNN    | 12 | DNN-aware |
| BATCHDNN_PP | 13 | DNN-aware |

## Array geometry

| Field | Default | RTL param |
|---|---|---|
| `array_height` | 4  | `ARRAY_HEIGHT` |
| `array_width`  | 4  | `ARRAY_WIDTH`  |
| `data_width`   | 32 | `DATA_WIDTH`   |
| `accum_width`  | 48 | `ACCUM_WIDTH`  |
| `addr_width`   | 32 | `ADDR_WIDTH`   |

## On-chip memory sizing

| Field | Default | RTL param | Notes |
|---|---|---|---|
| `spad_depth`      | 4096 | `SPAD_DEPTH`     | scratchpad words |
| `axi_data_width`  | 64   | `AXI_DATA_WIDTH` | downstream AXI beat width |
| `metadata_depth`  | 256  | `METADATA_DEPTH` | stamp only |
| `num_pages`       | 256  | `NUM_PAGES`      | paged only |
| `page_size_bits`  | 12   | `PAGE_SIZE_BITS` | paged only (4 KiB pages) |
| `n_mem_ports`     | 4    | `N_MEM_PORTS`    | scratchpad multi-port width |

## Memory layout

`mem_layout` (`CHANNEL_MAJOR`, `ROW_MAJOR`, `COLUMN_MAJOR`) sets the
`mem_layout_t` enum input to the systolic-array wrapper. It controls the
address-generator's indexing formula (see `address_generator_*.sv`).

## Layer workload

`layers: List[LayerConfig]` -- each entry carries:

| Field | Meaning |
|---|---|
| `name` | human-readable label |
| `input_channels`, `input_height`, `input_width` | input tensor shape (NHWC, N=1) |
| `weight_k`, `weight_c`, `weight_kh`, `weight_kw` | weight tensor (K_out, C_in, kH, kW) |
| `output_channels`, `output_height`, `output_width` | output tensor shape |
| `input_base_addr`, `weight_base_addr`, `output_base_addr` | DRAM byte offsets |
| `tile_row`, `tile_col_start`, `tile_ch_start` | tile coordinates fed to OS/IS |
| `ws_input_ch`, `ws_kernel_row`, `ws_kernel_col_start` | WS-specific extras |

## Quantisation / tolerance

| Field | Default | Meaning |
|---|---|---|
| `fp_total_bits` | 32 | total bits of the Qm.n format |
| `fp_frac_bits`  | 16 | fractional bits (default Q16.16) |
| `tolerance`     | 0.02 | max abs error allowed by golden_compare |

The TF dump script (`tf_dump.py`) calibrates `frac_bits` per tensor to
maximise dynamic range. Default is Q16.16 when the array's `DATA_WIDTH=32`.

## Multi-DNN sizing

| Field | Default | RTL param |
|---|---|---|
| `max_tasks`  | 16 | `MAX_TASKS`  |
| `max_layers` | 32 | `MAX_LAYERS` |
| `max_dnns`   | 4  | `MAX_DNNS`   |

## Iteration / timing

| Field | Default | Meaning |
|---|---|---|
| `cycle_budget` | None | per-test cycle upper bound (None = use analytical 3x) |
| `seed`         | 0    | RNG seed for software_ref and tf_dump random inputs |

## Mapping to environment variables (cocotb)

`SimConfig.cocotb_env()` emits:

```
SIM_DATAFLOW, SIM_MEMORY, SIM_LAYOUT, SIM_ARRAY_H, SIM_ARRAY_W,
SIM_DATA_W, SIM_FP_TOTAL, SIM_FP_FRAC, SIM_TOL, SIM_SCHEDULER,
SIM_LAYERS_JSON
```

The cocotb tests read these and configure the DUT accordingly.
