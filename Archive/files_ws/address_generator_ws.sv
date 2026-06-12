//=============================================================================
// Module: address_generator_ws
// Description: Converts 3D tensor index to linear memory address
//              Uses weight_stationary_pkg (NOT systolic_array_pkg).
//
// WHY THIS FILE EXISTS
// ---------------------
// The OS modules (input_fetcher_ws etc.) originally referenced address_generator
// which imports systolic_array_pkg::mem_layout_t.  The WS fetchers import
// weight_stationary_pkg::mem_layout_t.  Even though both enums have the same
// encoding, they are DIFFERENT TYPES in SystemVerilog, causing a type mismatch
// at the port connection.  This module is identical in logic to address_generator
// but imports weight_stationary_pkg so the types match throughout WS hierarchy.
//
// LAYOUT FORMULAS (same as OS address_generator)
//   CHANNEL_MAJOR [C,H,W]:  offset = idx_0 * (shape_h * shape_w)
//                                   + idx_1 * shape_w
//                                   + idx_2
//   ROW_MAJOR     [H,W,C]:  offset = idx_0 * (shape_w * shape_c)
//                                   + idx_1 * shape_c
//                                   + idx_2
//   COLUMN_MAJOR  [W,H,C]:  offset = idx_0 * (shape_h * shape_c)
//                                   + idx_1 * shape_c
//                                   + idx_2
//=============================================================================

module address_generator_ws
    import weight_stationary_pkg::*;
#(
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  mem_layout_t            layout,
    input  logic [15:0]            shape_c,
    input  logic [15:0]            shape_h,
    input  logic [15:0]            shape_w,
    input  logic [DATA_WIDTH-1:0]  base_addr,
    input  logic [15:0]            idx_0,
    input  logic [15:0]            idx_1,
    input  logic [15:0]            idx_2,
    input  logic                   idx_valid,

    output logic [DATA_WIDTH-1:0]  linear_addr,
    output logic                   addr_valid
);

    logic [DATA_WIDTH-1:0] offset;

    always_comb begin
        case (layout)
            CHANNEL_MAJOR:
                offset = idx_0 * (32'(shape_h) * shape_w)
                       + idx_1 *  32'(shape_w)
                       + idx_2;
            ROW_MAJOR:
                offset = idx_0 * (32'(shape_w) * shape_c)
                       + idx_1 *  32'(shape_c)
                       + idx_2;
            COLUMN_MAJOR:
                offset = idx_0 * (32'(shape_h) * shape_c)
                       + idx_1 *  32'(shape_c)
                       + idx_2;
            default:
                offset = '0;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            linear_addr <= '0;
            addr_valid  <= 1'b0;
        end else begin
            linear_addr <= base_addr + offset;
            addr_valid  <= idx_valid;
        end
    end

endmodule : address_generator_ws
