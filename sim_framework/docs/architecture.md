# Architecture

```
                  +-------------------+
                  |  pysim.runner     |
                  |  + SimConfig      |
                  +---------+---------+
                            | env vars + -Gparams
                            v
+--------------+    +---------------------+    +-----------------+
|  cocotb TBs  |--->|  Verilator binary   |--->|  RTL elaboration|
+--------------+    +---------------------+    +-----------------+
        ^                                            |
        | fall back (no verilator)                   v
        |                                  +-------------------+
        |                                  | single_dnn_top    |
        +-----[ pysim.software_ref ]------>|   or              |
                                           | multi_dnn_top     |
                                           +-------------------+
```

## RTL hierarchy

```
multi_dnn_top
+-- unified_scheduler_wrapper
|   +-- multi_scheduler_wrapper                  # 11 basic + advanced
|   |   +-- task_scheduler           (x7)
|   |   +-- advanced_task_scheduler  (x4)
|   +-- dnn_scheduler_wrapper                    # 3 DNN-aware
|       +-- aimt_scheduler
|       +-- batchdnn_scheduler
|       +-- batchdnn_pp_scheduler
+-- single_dnn_top                               # the inner accelerator
    +-- systolic_array_wrap                      # union-port wrapper
    |   +-- systolic_array_os   (DATAFLOW == DF_OS)
    |   +-- systolic_array_is   (DATAFLOW == DF_IS)
    |   +-- systolic_array_ws   (DATAFLOW == DF_WS)
    +-- mem_backend_wrap                         # MEMORY param picks one
        +-- stamp_memory_backend                 # delta-op-driven scratchpad
        |   +-- stamp_based_memory_controller
        |   +-- axi_read_adapter
        |   +-- scratchpad_ram
        +-- paged_memory_backend                 # MMU + scratchpad
            +-- page_table (xN_PORTS)
            +-- scratchpad_ram
```

## Single-DNN split-FSM (Phase C)

The outer FSM in `single_dnn_top` exposes two separate "done" strobes so
that DNN-aware schedulers (AI-MT / BatchDNN / BatchDNN++) can issue
memory and compute tasks against the same DNN instance:

```
            +-- S_IDLE
            |    | start
            v    v
            +-- S_MEM ---phase_mem_done---> phase_mem_done pulse
            |    | (be_phase_done || MEMORY == MEM_PAGED)
            v    v
            +-- S_COMPUTE ---array_done---> phase_compute_done pulse
            |    |
            v    v
            +-- S_FINAL ---> done pulse, back to IDLE
```

For basic (non-DNN-aware) schedulers, only `phase_compute_done` is
meaningful and the dispatch FSM in `multi_dnn_top` routes that signal as
`task_complete`.

## Memory backend selection

Both backends present the same N-port data plane to the wrapper:

```
    rd_en[N], rd_addr[N], rd_data[N], rd_valid[N]
    wr_en[N], wr_addr[N], wr_data[N]
```

Backend-specific side bands stay at `single_dnn_top`'s port list and are
muxed by `MEMORY`:

| Side band              | STAMP | PAGED |
|---|---|---|
| metadata_wr_*          | used  | tied off |
| phase_start_in / phase_id_in / num_delta_ops_in / phase_base_addr_in | used  | unused (phase auto-completes) |
| pt_write_en / pt_write_vpn / pt_write_ppn / pt_write_valid | unused | used |
| AXI DRAM master        | used  | unused |

## Loop-optimizer integration

```
+-- BlockingOptimizer
|   +-- analyzer: MemoryAccessAnalyzer         (analytical default)
|   |              or SimulatorCostModel       (HW-in-the-loop)
|   +-- search(mode=SIMULATOR_GUIDED)
|       1) analytical pre-filter (top-N=20)
|       2) re-score survivors with simulator
+-- pysim.cost_model.SimulatorCostModel
    +-- analyze(LoopNestConfig) -> dict
        +-- pysim.runner.run(cfg)
            +-- Verilator+cocotb (if installed) OR software_ref fallback
            +-- cache result under pysim/cache/<RTL_REVISION>_<sha>.json
```

## File ownership reminder

The four legacy directories (`Archive/`, `static_hash_and_tagless_memory/`,
`Multi_DNN_schedulers/`, `systolic_optimizer_codebase/`) remain untouched
as reference. All consolidation work lives under `sim_framework/`.
