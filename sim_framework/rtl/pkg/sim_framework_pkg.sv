//=============================================================================
// Package: sim_framework_pkg
// Description: Canonical type & enum definitions for the unified systolic-
//              array simulator framework. Supersedes the three duplicate
//              packages (systolic_array_pkg, input_stationary_pkg,
//              weight_stationary_pkg) from the legacy per-flavor source trees.
//
// COMPILE FIRST. The three legacy package files in this tree are now thin
// re-export aliases that pull from this package, so any flavor top module
// that still says `import systolic_array_pkg::*` will pick up the unified
// definitions transparently.
//=============================================================================

package sim_framework_pkg;

    //-------------------------------------------------------------------------
    // Memory layout encoding (carried over from the three legacy packages).
    //-------------------------------------------------------------------------
    typedef enum logic [1:0] {
        CHANNEL_MAJOR = 2'b00,  // C, H, W
        ROW_MAJOR     = 2'b01,  // H, W, C
        COLUMN_MAJOR  = 2'b10   // W, H, C
    } mem_layout_t;

    //-------------------------------------------------------------------------
    // Address request / data packet structures (unchanged from legacy pkgs).
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic [31:0] addr;
        logic        valid;
        logic        ready;
    } addr_req_t;

    typedef struct packed {
        logic [31:0] data;
        logic [31:0] addr;
        logic        valid;
    } data_packet_t;

    //-------------------------------------------------------------------------
    // Dataflow selector. Used as a parameter on systolic_array_wrap and
    // single_dnn_top to pick which per-flavor systolic array body to elaborate.
    //-------------------------------------------------------------------------
    typedef enum logic [1:0] {
        DF_OS = 2'b00,
        DF_IS = 2'b01,
        DF_WS = 2'b10
    } dataflow_e;

    //-------------------------------------------------------------------------
    // Memory backend selector. Used as a parameter on mem_backend_wrap and
    // single_dnn_top.
    //-------------------------------------------------------------------------
    typedef enum logic [0:0] {
        MEM_STAMP = 1'b0,
        MEM_PAGED = 1'b1
    } mem_backend_e;

    //-------------------------------------------------------------------------
    // Scheduler family. The unified_scheduler_wrapper exposes a 5-bit
    // scheduler_select that the multi_dnn_top dispatch FSM decodes into
    // one of these families:
    //   0..6   : basic task schedulers   (FIFO/LIFO/SJF/RR/Priority/EDF/LRU)
    //   7..10  : advanced task scheds    (SRTF/HRRN/MLQ/MLFQ)
    //   11..13 : DNN-aware schedulers    (AI-MT/BATCH-DNN/BATCH-DNN++)
    //-------------------------------------------------------------------------
    localparam logic [4:0] SCHED_FIFO        = 5'd0;
    localparam logic [4:0] SCHED_LIFO        = 5'd1;
    localparam logic [4:0] SCHED_SJF         = 5'd2;
    localparam logic [4:0] SCHED_RR          = 5'd3;
    localparam logic [4:0] SCHED_PRI         = 5'd4;
    localparam logic [4:0] SCHED_EDF         = 5'd5;
    localparam logic [4:0] SCHED_LRU         = 5'd6;
    localparam logic [4:0] SCHED_SRTF        = 5'd7;
    localparam logic [4:0] SCHED_HRRN        = 5'd8;
    localparam logic [4:0] SCHED_MLQ         = 5'd9;
    localparam logic [4:0] SCHED_MLFQ        = 5'd10;
    localparam logic [4:0] SCHED_AIMT        = 5'd11;
    localparam logic [4:0] SCHED_BATCHDNN    = 5'd12;
    localparam logic [4:0] SCHED_BATCHDNNPP  = 5'd13;

    //-------------------------------------------------------------------------
    // Tagged-union output of the unified_scheduler_wrapper. The dispatch FSM
    // in multi_dnn_top inspects `is_dnn_aware` to decide whether to drive the
    // single_dnn_top with a memory-only phase or a compute phase.
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic        valid;
        logic        is_dnn_aware;     // 0 = basic queue task, 1 = AI-MT family
        logic [7:0]  task_or_layer_id;
        logic [3:0]  dnn_id;           // valid only if is_dnn_aware
        logic        is_compute_task;  // 0 = memory task, 1 = compute task
        logic [15:0] burst_or_cycles;
    } sched_output_t;

endpackage : sim_framework_pkg


//=============================================================================
// Legacy package re-exports.
//
// The three legacy packages (systolic_array_pkg, input_stationary_pkg,
// weight_stationary_pkg) defined byte-identical type sets. To avoid touching
// every fetcher/PE/array source file (~12 files), we re-declare each as a
// thin re-export of the canonical package's types.
//
// Down-side: a SystemVerilog package cannot re-export an `import` directly,
// so we re-declare the typedefs/enums by hand here. They are guaranteed to
// stay in sync because they share the same numerical encoding; if the
// canonical package changes, this block must change too (caught by the
// `verilator --lint-only` step in Phase B).
//
// The empty `package systolic_array_pkg` stub at the bottom of the legacy
// `stamp_memory_pkg.sv` is removed (see rtl/pkg/stamp_memory_pkg.sv).
//=============================================================================

package systolic_array_pkg;
    typedef enum logic [1:0] {
        CHANNEL_MAJOR = 2'b00,
        ROW_MAJOR     = 2'b01,
        COLUMN_MAJOR  = 2'b10
    } mem_layout_t;

    typedef struct packed { logic [31:0] addr; logic valid; logic ready; } addr_req_t;
    typedef struct packed { logic [31:0] data; logic [31:0] addr; logic valid; } data_packet_t;
endpackage : systolic_array_pkg

package input_stationary_pkg;
    typedef enum logic [1:0] {
        CHANNEL_MAJOR = 2'b00,
        ROW_MAJOR     = 2'b01,
        COLUMN_MAJOR  = 2'b10
    } mem_layout_t;

    typedef struct packed { logic [31:0] addr; logic valid; logic ready; } addr_req_t;
    typedef struct packed { logic [31:0] data; logic [31:0] addr; logic valid; } data_packet_t;
endpackage : input_stationary_pkg

package weight_stationary_pkg;
    typedef enum logic [1:0] {
        CHANNEL_MAJOR = 2'b00,
        ROW_MAJOR     = 2'b01,
        COLUMN_MAJOR  = 2'b10
    } mem_layout_t;

    typedef struct packed { logic [31:0] addr; logic valid; logic ready; } addr_req_t;
    typedef struct packed { logic [31:0] data; logic [31:0] addr; logic valid; } data_packet_t;
endpackage : weight_stationary_pkg
