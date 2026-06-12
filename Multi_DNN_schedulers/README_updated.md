# Hardware Task Scheduler System

A comprehensive, parameterized hardware task scheduler implementation supporting
multiple scheduling algorithms — from simple OS-level policies through to
state-of-the-art DNN multi-accelerator schedulers.

---

## Scheduling Algorithms Implemented

### Basic Schedulers (`task_scheduler.sv`)

| Type | `SCHEDULER_TYPE` | Description |
|------|-----------------|-------------|
| FIFO | 0 | First-In First-Out |
| LIFO | 1 | Last-In First-Out (stack) |
| SJF  | 2 | Shortest Job First (non-preemptive) |
| Round Robin | 3 | Fixed time-quantum rotation |
| Priority | 4 | Highest priority value first |
| EDF  | 5 | Earliest Deadline First |
| LRU  | 6 | Least Recently Used |

### Advanced Schedulers (`advanced_task_scheduler.sv`)

| Type | `SCHEDULER_TYPE` | Description |
|------|-----------------|-------------|
| SRTF | 0 | Shortest Remaining Time First (preemptive) |
| HRRN | 1 | Highest Response Ratio Next |
| MLQ  | 2 | Multi-Level Queue (static) |
| MLFQ | 3 | Multi-Level Feedback Queue (dynamic) |

### DNN Accelerator Schedulers *(new)*

| Module | Based on | Key contribution |
|--------|----------|-----------------|
| `aimt_scheduler.sv` | AI-MT (Baek et al., ISCA 2020) | Memory/compute overlap with layer-granularity scheduling |
| `batchdnn_scheduler.sv` | BATCH-DNN (Ranawaka & Stenstrom, EuroPar 2025) | Adaptive dynamic sub-batch splitting and merging |
| `batchdnn_pp_scheduler.sv` | BATCH-DNN++ (2025) | Distance-based throttling + bottleneck expedition + batch slicing |

---

## DNN Scheduler Architecture

### AI-MT (`aimt_scheduler.sv`)

The AI-MT scheduler overlaps memory access with computation across multiple DNN
workloads.  It is the baseline upon which BATCH-DNN and BATCH-DNN++ are built.

**Key signals:**

| Port | Direction | Description |
|------|-----------|-------------|
| `st_write_en` | in | Write one row of the scheduling table |
| `st_layer_idx` | in | Row index (layer number, global across all DNNs) |
| `st_dnn_id` | in | Which DNN workload this layer belongs to |
| `st_prev_layer` | in | Predecessor layer index; `8'hFF` = no predecessor (root) |
| `st_mem_cycles` | in | Cycles to load weights from off-chip memory |
| `st_compute_cycles` | in | Cycles to compute this layer |
| `st_{weight,ifmap,ofmap}_fp` | in | Memory footprints in bytes |
| `mem_access_done` | in | Pulse: current memory-access task finished |
| `compute_done` | in | Pulse: current compute task finished |
| `mt_valid` / `mt_layer_id` | out | Next layer dispatched for memory access |
| `ct_valid` / `ct_layer_id` | out | Next layer dispatched for compute |
| `available_memory` | out | Free on-chip memory (bytes) |
| `stall_detected` | out | High when no task can be scheduled |
| `total_stall_cycles` | out | Cumulative stall counter |

**Scheduling table** is loaded once at compile-time before simulation starts (or
at startup in hardware).  The compiler generates one row per layer containing
cycle counts and memory footprints (Figure 1 / Section 2.1 of the paper).

**Balance condition** (Section 3.4.1, Figure 7):
- Schedule MT only if `mt.mem_cycles ≤ cycles_to_fill_remaining` AND
  (`available_compute_cycles > 1000` OR `mt.mem_cycles ≤ available_compute_cycles`)

---

### BATCH-DNN (`batchdnn_scheduler.sv`)

Extends AI-MT with **adaptive cascaded sub-batch splitting and merging**
(Section 3, Figure 7 Blocks C & D).

**Additional ports over AI-MT:**

| Port | Direction | Description |
|------|-----------|-------------|
| `st_initial_batch` | in | Requested batch size for this DNN |
| `ct_batch_size` | out | Actual batch size dispatched |
| `total_split_events` | out | Number of sub-batch splits performed |
| `total_merge_events` | out | Number of sub-batch merges performed |

**Sub-Batch Splitting** (Box 17–24 of Figure 7):
1. Compute maximum feasible batch: `N_feas = min(avail_mem / ofmap_fp, prev_batch)`
2. If `N_feas < current_batch`: push `(layer, paused_size)` onto the LIFO stack and
   continue with `N_feas` inferences.
3. Update AI-MT shared counters proportionally.

**Sub-Batch Merging** (Box 16 of Figure 7):
1. Peek at top of stack.
2. Walk from the resumed layer to the current layer; check all intermediate layers fit.
3. If yes: pop stack, add batch sizes, continue merged.

**Sub-batching stack** (Figure 6B):  Each DNN has a dedicated hardware LIFO
stack of depth `STACK_DEPTH` (default 16).  Stack entries store `{layer_idx,
batch_size}`.

---

### BATCH-DNN++ (`batchdnn_pp_scheduler.sv`)

Replaces the AI-MT baseline blocks (A & B) with a new scheduler while reusing
the BATCH-DNN sub-batching blocks (C & D).  Three new techniques are added
(Section 4, Figure 9):

#### 4.1.1 Distance-Based Memory Access Throttling
Prevents memory tasks from running too far ahead of the current compute task,
which would exhaust on-chip memory.

- Threshold: `MAX_LAYER_DISTANCE = 5` (empirically determined, Section 4.2.1).
- The MT scheduler iterates the entire MT candidate queue and selects the
  **lowest-indexed layer** (earliest in DAG order) that satisfies both:
  - `mem_req ≤ available_memory`
  - `|mt_layer − ct_current_layer| ≤ MAX_LAYER_DISTANCE`
- Throttled tasks are re-queued; the counter `total_throttle_events` is
  incremented.

#### 4.1.2 Bottleneck Layer Expedition
When a compute task is blocked because its predecessors are not yet done, the
scheduler recursively walks back the dependency chain to find the earliest
unscheduled layer (the "bottleneck") and expedites it.

- If the bottleneck still has a pending memory access, it is injected into the
  MT candidate queue (priority scheduling of that task).
- If memory access is done but compute is not, it is pushed directly to the
  Scheduled CT Queue.
- Counter: `total_expedition_events`.

#### 4.1.3 Batch Slicing
Allows arbitrarily large requested batch sizes (`N_x`) that exceed the hardware
capacity (`N_max`).

```
full_iterations = floor(N_x / N_max)
remainder       = N_x mod N_max
```

Weights are reloaded once per slice iteration.  The `slice_remaining` register
per DNN tracks how many inferences are still pending.

**Additional ports over BATCH-DNN:**

| Port | Direction | Description |
|------|-----------|-------------|
| `st_requested_batch` | in | Arbitrary batch size `N_x` |
| `total_throttle_events` | out | Distance-based throttling counter |
| `total_expedition_events` | out | Bottleneck expedition counter |
| `total_slices_executed` | out | Batch slicing iterations |

---

## File Structure

```
.
├── task_scheduler.sv              # Basic schedulers (FIFO/LIFO/SJF/RR/Priority/EDF/LRU)
├── advanced_task_scheduler.sv     # Advanced schedulers (SRTF/HRRN/MLQ/MLFQ)
├── multi_scheduler_wrapper.sv     # Runtime-selectable wrapper for basic schedulers
├── aimt_scheduler.sv              # AI-MT DNN scheduler  ← NEW
├── batchdnn_scheduler.sv          # BATCH-DNN scheduler  ← NEW
├── batchdnn_pp_scheduler.sv       # BATCH-DNN++ scheduler ← NEW
├── task_scheduler_tb.sv           # Testbench: basic + advanced schedulers
├── dnn_schedulers_tb.sv           # Testbench: AI-MT / BATCH-DNN / BATCH-DNN++ ← NEW
├── Makefile                       # Updated build rules
├── README.md                      # This file
└── EXAMPLES.md                    # Usage examples
```

---

## Running the Simulation

### Icarus Verilog

```bash
# Basic + advanced schedulers (original)
make sim_basic

# DNN schedulers (AI-MT, BATCH-DNN, BATCH-DNN++)
make sim_dnn

# Both
make sim_all
```

### Verilator

```bash
make verilate_basic
make verilate_dnn
```

### ModelSim / QuestaSim

```bash
make modelsim_basic
make modelsim_dnn
```

---

## Instantiation Examples

### AI-MT Scheduler

```systemverilog
aimt_scheduler #(
    .MAX_LAYERS      (32),
    .MAX_DNNS        (4),
    .ONCHIP_MEM_SIZE (44 * 1024 * 1024),   // 44 MB
    .COMPUTE_BAL_THRESH (1000)
) u_aimt (
    .clk             (clk),
    .rst_n           (rst_n),
    // -- table load (once at startup) --
    .st_write_en     (table_wr),
    .st_layer_idx    (layer_idx),
    .st_dnn_id       (dnn_id),
    .st_prev_layer   (prev_layer),
    .st_mem_cycles   (mem_cyc),
    .st_compute_cycles(comp_cyc),
    .st_weight_fp    (w_fp),
    .st_ifmap_fp     (i_fp),
    .st_ofmap_fp     (o_fp),
    .st_total_layers (n_layers),
    // -- runtime --
    .mem_access_done (mem_done_pulse),
    .compute_done    (compute_done_pulse),
    // -- outputs --
    .mt_valid        (mt_go),
    .mt_layer_id     (mt_layer),
    .ct_valid        (ct_go),
    .ct_layer_id     (ct_layer),
    .available_memory(free_mem)
);
```

### BATCH-DNN++ Scheduler

```systemverilog
batchdnn_pp_scheduler #(
    .MAX_LAYERS          (32),
    .MAX_DNNS            (4),
    .BATCH_WIDTH         (8),
    .ONCHIP_MEM_SIZE     (44 * 1024 * 1024),
    .MAX_LAYER_DISTANCE  (5),     // throttling threshold (Section 4.2.1)
    .STACK_DEPTH         (16)
) u_bpp (
    .clk                 (clk),
    .rst_n               (rst_n),
    .st_write_en         (table_wr),
    .st_layer_idx        (layer_idx),
    .st_dnn_id           (dnn_id),
    .st_prev_layer       (prev_layer),
    .st_mem_cycles       (mem_cyc),
    .st_compute_cycles   (comp_cyc),
    .st_weight_fp        (w_fp),
    .st_ifmap_fp         (i_fp),
    .st_ofmap_fp         (o_fp),
    .st_requested_batch  (batch_nx),  // can be larger than hardware max
    .st_total_layers     (n_layers),
    .mem_access_done     (mem_done_pulse),
    .compute_done        (compute_done_pulse),
    .mt_valid            (mt_go),
    .mt_layer_id         (mt_layer),
    .ct_valid            (ct_go),
    .ct_layer_id         (ct_layer),
    .ct_batch_size       (actual_batch),
    .available_memory    (free_mem),
    .total_throttle_events   (n_throttled),
    .total_expedition_events (n_expedited),
    .total_slices_executed   (n_slices)
);
```

---

## Design Notes

### Scheduling Table Population

The scheduling table must be written **before** `rst_n` deasserts or immediately
after reset.  In hardware this would be done by the host CPU's DNN compiler.
Each layer needs:

| Field | Description |
|-------|-------------|
| `layer_idx` | Global sequential layer index (0, 1, 2, …) |
| `dnn_id` | Which DNN workload (0 to MAX_DNNS−1) |
| `prev_layer` | Predecessor index; `8'hFF` for root layers |
| `mem_cycles` | Off-chip weight-load latency in cycles |
| `compute_cycles` | Per-unit-batch computation latency in cycles |
| `weight_fp` | Weight buffer size (bytes) |
| `ifmap_fp` | Input feature map size per inference (bytes) |
| `ofmap_fp` | Output feature map size per inference (bytes) |
| `initial_batch` / `requested_batch` | Batch size `N` (BATCH-DNN) or `N_x` (BATCH-DNN++) |

### Memory Model

On-chip memory is modelled as a flat counter (`available_memory`).  Memory is
allocated when an MT is scheduled and freed when compute completes:
- **Allocated:** `weight_fp + (ifmap_fp + ofmap_fp) × batch_size`
- **Freed after compute:** `ifmap_fp × batch_size`  (OFMAP becomes next IFMAP)
- **Weight freed:** only after all sub-batches of that layer complete (stack
  empty check).

### Synthesis Considerations

- No floating-point arithmetic is used; all division is integer.
- `find_bottleneck()` is a bounded loop (≤ `MAX_LAYERS` iterations) and can be
  pipelined across multiple clock cycles if timing is critical.
- The sub-batching stack is register-based; `STACK_DEPTH × MAX_DNNS` entries
  are required.  At 16 depth × 4 DNNs × (8+8) bits = 128 bytes — negligible.
- Hardware overhead vs. AI-MT baseline: < 1% of on-chip memory (Section 6.5).

---

## Known Limitations

1. The testbench uses shared `mem_access_done`/`compute_done` pulses for all
   three DUTs simultaneously.  In a real system each DUT would have its own
   control signals.
2. The `find_bottleneck()` function is implemented as a combinational loop
   inside `always_ff`; tool support varies.  For synthesis, unroll or convert
   to a multi-cycle FSM.
3. HRRN in `advanced_task_scheduler.sv` uses `real` arithmetic (simulation
   only).  Replace with fixed-point for synthesis.
4. Deadlock detection/recovery (Section 4.2 of BATCH-DNN++) is not yet
   implemented.

---

## Performance Results (from paper)

| Metric | BATCH-DNN vs AI-MT | BATCH-DNN++ vs AI-MT |
|--------|-------------------|----------------------|
| Throughput improvement | +6% | **+70%** (mean) |
| Speedup | ~1× | **3.4×** (mean) |
| Compute utilisation | ~1× | **2.4×** |
| Energy reduction | — | **41%** |

---

## References

1. Baek et al., "A Multi-Neural Network Acceleration Architecture," ISCA 2020.
2. Ranawaka & Stenstrom, "BATCH-DNN: Adaptive and Dynamic Batching for
   Multi-DNN Accelerators," EuroPar 2025.
3. Ranawaka & Stenstrom, "BATCH-DNN++: Improving Resource Utilization in
   Multi-DNN Accelerators with Adaptive and Dynamic Batching and
   Dependency-Aware Scheduling," ACM TACO 2025.
