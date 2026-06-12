//=============================================================================
// Package: stamp_memory_pkg
// Description: Type definitions for the stamp-based on-chip memory controller.
//
// Cleaned for the unified simulator framework:
//   * the empty `package systolic_array_pkg` stub at the bottom of the legacy
//     file (lines 54-63 in the original) has been removed -- the real
//     systolic_array_pkg now lives in sim_framework_pkg.sv as a re-export.
//   * the `mem_layout_t` packed-struct that lived here under the same name as
//     the dataflow `mem_layout_t` enum (sim_framework_pkg) has been renamed
//     to `stamp_mem_layout_t` to remove the name clash.
//=============================================================================

package stamp_memory_pkg;

    //-------------------------------------------------------------------------
    // Memory statistics collected by the stamp controller at runtime.
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic [31:0] total_loads;
        logic [31:0] total_moves;
        logic [31:0] total_keeps;
        logic [31:0] bytes_loaded;
        logic [31:0] bytes_moved;
        logic [31:0] cycles_elapsed;
    } memory_stats_t;

    //-------------------------------------------------------------------------
    // Stamp-side memory layout: base addresses of each data region in DRAM.
    //
    // Renamed from `mem_layout_t` -> `stamp_mem_layout_t` to avoid clashing
    // with the dataflow-side mem_layout_t enum in sim_framework_pkg.
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic [31:0] input_base_addr;
        logic [31:0] weight_base_addr;
        logic [31:0] output_base_addr;
        logic [31:0] reserved;
    } stamp_mem_layout_t;

    //-------------------------------------------------------------------------
    // Neural-network operation configuration that the stamp tile-table carries.
    //-------------------------------------------------------------------------
    typedef struct packed {
        logic [7:0] activation_type;  // 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh
        logic [7:0] pool_type;        // 0=None, 1=MaxPool, 2=AvgPool
        logic [7:0] norm_type;        // 0=None, 1=BatchNorm, 2=LayerNorm
        logic [7:0] reserved;
    } nn_operation_config_t;

    //-------------------------------------------------------------------------
    // Delta-operation type codes (mirrors the Python TileType enum in
    // stamp_compiler.py).
    //-------------------------------------------------------------------------
    localparam logic [7:0] OP_KEEP = 8'd0;  // tile stays at same address
    localparam logic [7:0] OP_MOVE = 8'd1;  // tile shifts within SRAM
    localparam logic [7:0] OP_LOAD = 8'd2;  // tile fetched from DRAM

endpackage : stamp_memory_pkg
