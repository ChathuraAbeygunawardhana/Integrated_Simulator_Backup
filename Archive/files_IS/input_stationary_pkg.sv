//=============================================================================
// Package: input_stationary_pkg
// Description: Common definitions for input stationary systolic array
//=============================================================================

package input_stationary_pkg;

    // Memory layout encoding
    typedef enum logic [1:0] {
        CHANNEL_MAJOR = 2'b00,  // C, H, W
        ROW_MAJOR     = 2'b01,  // H, W, C
        COLUMN_MAJOR  = 2'b10   // W, H, C
    } mem_layout_t;

    // Address request structure
    typedef struct packed {
        logic [31:0] addr;
        logic        valid;
        logic        ready;
    } addr_req_t;

    // Data structure for input/weight/output
    typedef struct packed {
        logic [31:0] data;
        logic [31:0] addr;
        logic        valid;
    } data_packet_t;

endpackage : input_stationary_pkg
