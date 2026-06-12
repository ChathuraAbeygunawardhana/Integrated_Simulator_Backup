//=============================================================================
// Package: stamp_memory_pkg
// Description: Type definitions for stamp-based memory management system.
//
// This package MUST be compiled first before any other .sv file in this
// project. In Vivado: add this file first in the Sources panel, or list
// it first in your fileset.
//=============================================================================

package stamp_memory_pkg;

    //=========================================================================
    // Memory Statistics - collected by the stamp controller at runtime
    //=========================================================================
    typedef struct packed {
        logic [31:0] total_loads;      // Number of LOAD operations executed
        logic [31:0] total_moves;      // Number of MOVE operations executed
        logic [31:0] total_keeps;      // Number of KEEP operations executed
        logic [31:0] bytes_loaded;     // Total bytes fetched from off-chip DRAM
        logic [31:0] bytes_moved;      // Total bytes moved within on-chip SRAM
        logic [31:0] cycles_elapsed;   // Total clock cycles since last reset
    } memory_stats_t;

    //=========================================================================
    // Memory Layout - base addresses of each data region in off-chip memory
    //=========================================================================
    typedef struct packed {
        logic [31:0] input_base_addr;   // Start address of input feature map
        logic [31:0] weight_base_addr;  // Start address of weight tensor
        logic [31:0] output_base_addr;  // Start address of output feature map
        logic [31:0] reserved;          // Reserved for future use
    } mem_layout_t;

    //=========================================================================
    // Neural Network Operation Configuration
    //=========================================================================
    typedef struct packed {
        logic [7:0]  activation_type;   // 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh
        logic [7:0]  pool_type;         // 0=None, 1=MaxPool, 2=AvgPool
        logic [7:0]  norm_type;         // 0=None, 1=BatchNorm, 2=LayerNorm
        logic [7:0]  reserved;          // Reserved
    } nn_operation_config_t;

    //=========================================================================
    // Delta Operation Types (mirrors Python TileType enum)
    //=========================================================================
    localparam logic [7:0] OP_KEEP = 8'd0;  // Tile stays at same address - free
    localparam logic [7:0] OP_MOVE = 8'd1;  // Tile shifts within SRAM    - cheap
    localparam logic [7:0] OP_LOAD = 8'd2;  // Tile fetched from DRAM     - expensive

endpackage : stamp_memory_pkg


//=============================================================================
// Minimal systolic_array_pkg - required by stamp_based_memory_controller.sv
// which has "import systolic_array_pkg::*" but uses no types from it.
// Keep this empty stub so the import resolves without error.
//=============================================================================
package systolic_array_pkg;
    // Intentionally empty - the stamp controller imports this for forward
    // compatibility with the full systolic array package but does not use
    // any types from it directly.
endpackage : systolic_array_pkg