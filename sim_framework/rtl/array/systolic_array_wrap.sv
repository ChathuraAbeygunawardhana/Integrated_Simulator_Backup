//=============================================================================
// Module: systolic_array_wrap
// Description: Uniform-port wrapper around the three per-flavor systolic
//              array tops (OS / IS / WS). The DATAFLOW parameter selects
//              which body is instantiated; unused ports are tied off.
//
// Why a wrapper (not a single merged module): the three flavor tops have
// non-uniform port shapes -- in particular WS has a 2D per-PE weight port
// and a 1D output port, while OS/IS have a 1D per-row weight port and a
// 2D output port. Trying to express this in one module via case-generate
// inside the body produces an awkward super-port-list that's hard to wire.
// A thin wrapper around the existing per-flavor tops keeps the existing,
// already-verified modules intact and centralises only the port-shape
// adaptation in this file.
//
// Port shape (union of all three flavors). For each flavor, only the
// matching subset is driven; the rest is tied to 0 by the generate-else
// branches.
//=============================================================================

module systolic_array_wrap
    import sim_framework_pkg::*;
#(
    parameter dataflow_e DATAFLOW    = DF_OS,
    parameter int        ARRAY_HEIGHT = 4,
    parameter int        ARRAY_WIDTH  = 4,
    parameter int        DATA_WIDTH   = 32,
    parameter int        ACCUM_WIDTH  = 48,
    parameter int        ADDR_WIDTH   = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,

    //--- configuration (shared by all flavors)
    input  mem_layout_t           mem_layout,
    input  logic [15:0]           input_channels,
    input  logic [15:0]           input_height,
    input  logic [15:0]           input_width,
    input  logic [15:0]           weight_k,
    input  logic [15:0]           weight_c,
    input  logic [15:0]           weight_kh,
    input  logic [15:0]           weight_kw,
    input  logic [15:0]           output_channels,
    input  logic [15:0]           output_height,
    input  logic [15:0]           output_width,
    input  logic [ADDR_WIDTH-1:0] input_base_addr,
    input  logic [ADDR_WIDTH-1:0] weight_base_addr,
    input  logic [ADDR_WIDTH-1:0] output_base_addr,

    //--- control
    input  logic                  start,
    input  logic [15:0]           tile_row,         // OS/IS: row | WS: tile_row_start
    input  logic [15:0]           tile_col_start,   // OS/IS: col start | WS: tile_col
    input  logic [15:0]           tile_ch_start,    // shared
    input  logic [15:0]           ws_input_ch,      // WS only
    input  logic [15:0]           ws_kernel_row,    // WS only
    input  logic [15:0]           ws_kernel_col_start, // WS only
    output logic                  done,
    output logic                  busy,

    //--- memory ports (UNION)
    // OS/IS: input is per-column (W); WS: input is per-row (H).
    // We expose two named ports; only one is driven per flavor.
    output logic [ADDR_WIDTH-1:0]  input_addr_w  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] input_addr_valid_w,
    input  logic [DATA_WIDTH-1:0]  input_data_w  [ARRAY_WIDTH],
    input  logic [ARRAY_WIDTH-1:0] input_data_valid_w,

    output logic [ADDR_WIDTH-1:0]   input_addr_h  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] input_addr_valid_h,
    input  logic [DATA_WIDTH-1:0]   input_data_h  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] input_data_valid_h,

    // OS/IS weight: 1D per-row. WS weight: 2D per-PE.
    output logic [ADDR_WIDTH-1:0]   weight_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] weight_addr_valid_1d,
    input  logic [DATA_WIDTH-1:0]   weight_data_1d  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] weight_data_valid_1d,

    output logic [ADDR_WIDTH-1:0]                    weight_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_addr_valid_2d,
    input  logic [DATA_WIDTH-1:0]                    weight_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    input  logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_data_valid_2d,

    // OS/IS output: 2D per-PE. WS output: 1D per-row.
    output logic [ADDR_WIDTH-1:0]                    output_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid_2d,
    output logic [DATA_WIDTH-1:0]                    output_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid_2d,

    output logic [ADDR_WIDTH-1:0]   output_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] output_addr_valid_1d,
    output logic [DATA_WIDTH-1:0]   output_data_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] output_data_valid_1d
);

    generate
        if (DATAFLOW == DF_OS) begin : g_os
            systolic_array_top #(
                .ARRAY_HEIGHT (ARRAY_HEIGHT),
                .ARRAY_WIDTH  (ARRAY_WIDTH),
                .DATA_WIDTH   (DATA_WIDTH),
                .ACCUM_WIDTH  (ACCUM_WIDTH),
                .ADDR_WIDTH   (ADDR_WIDTH)
            ) u_os (
                .clk               (clk),
                .rst_n             (rst_n),
                .mem_layout        (mem_layout),
                .input_channels    (input_channels),
                .input_height      (input_height),
                .input_width       (input_width),
                .weight_k          (weight_k),
                .weight_c          (weight_c),
                .weight_kh         (weight_kh),
                .weight_kw         (weight_kw),
                .output_channels   (output_channels),
                .output_height     (output_height),
                .output_width      (output_width),
                .input_base_addr   (input_base_addr),
                .weight_base_addr  (weight_base_addr),
                .output_base_addr  (output_base_addr),
                .start             (start),
                .tile_row          (tile_row),
                .tile_col_start    (tile_col_start),
                .tile_ch_start     (tile_ch_start),
                .done              (done),
                .busy              (busy),
                .input_addr        (input_addr_w),
                .input_addr_valid  (input_addr_valid_w),
                .input_data        (input_data_w),
                .input_data_valid  (input_data_valid_w),
                .weight_addr       (weight_addr_1d),
                .weight_addr_valid (weight_addr_valid_1d),
                .weight_data       (weight_data_1d),
                .weight_data_valid (weight_data_valid_1d),
                .output_addr       (output_addr_2d),
                .output_addr_valid (output_addr_valid_2d),
                .output_data       (output_data_2d),
                .output_data_valid (output_data_valid_2d)
            );

            // Tie off WS-only ports
            assign input_addr_h        = '{default: '0};
            assign input_addr_valid_h  = '0;
            assign weight_addr_2d      = '{default: '0};
            assign weight_addr_valid_2d= '0;
            assign output_addr_1d      = '{default: '0};
            assign output_addr_valid_1d= '0;
            assign output_data_1d      = '{default: '0};
            assign output_data_valid_1d= '0;
        end
        else if (DATAFLOW == DF_IS) begin : g_is
            input_stationary_top #(
                .ARRAY_HEIGHT (ARRAY_HEIGHT),
                .ARRAY_WIDTH  (ARRAY_WIDTH),
                .DATA_WIDTH   (DATA_WIDTH),
                .ACCUM_WIDTH  (ACCUM_WIDTH),
                .ADDR_WIDTH   (ADDR_WIDTH)
            ) u_is (
                .clk               (clk),
                .rst_n             (rst_n),
                .mem_layout        (mem_layout),
                .input_channels    (input_channels),
                .input_height      (input_height),
                .input_width       (input_width),
                .weight_k          (weight_k),
                .weight_c          (weight_c),
                .weight_kh         (weight_kh),
                .weight_kw         (weight_kw),
                .output_channels   (output_channels),
                .output_height     (output_height),
                .output_width      (output_width),
                .input_base_addr   (input_base_addr),
                .weight_base_addr  (weight_base_addr),
                .output_base_addr  (output_base_addr),
                .start             (start),
                .tile_row          (tile_row),
                .tile_col_start    (tile_col_start),
                .tile_ch_start     (tile_ch_start),
                .done              (done),
                .busy              (busy),
                .input_addr        (input_addr_w),
                .input_addr_valid  (input_addr_valid_w),
                .input_data        (input_data_w),
                .input_data_valid  (input_data_valid_w),
                .weight_addr       (weight_addr_1d),
                .weight_addr_valid (weight_addr_valid_1d),
                .weight_data       (weight_data_1d),
                .weight_data_valid (weight_data_valid_1d),
                .output_addr       (output_addr_2d),
                .output_addr_valid (output_addr_valid_2d),
                .output_data       (output_data_2d),
                .output_data_valid (output_data_valid_2d)
            );

            assign input_addr_h        = '{default: '0};
            assign input_addr_valid_h  = '0;
            assign weight_addr_2d      = '{default: '0};
            assign weight_addr_valid_2d= '0;
            assign output_addr_1d      = '{default: '0};
            assign output_addr_valid_1d= '0;
            assign output_data_1d      = '{default: '0};
            assign output_data_valid_1d= '0;
        end
        else begin : g_ws
            weight_stationary_top #(
                .ARRAY_HEIGHT (ARRAY_HEIGHT),
                .ARRAY_WIDTH  (ARRAY_WIDTH),
                .DATA_WIDTH   (DATA_WIDTH),
                .ACCUM_WIDTH  (ACCUM_WIDTH),
                .ADDR_WIDTH   (ADDR_WIDTH)
            ) u_ws (
                .clk               (clk),
                .rst_n             (rst_n),
                .mem_layout        (mem_layout),
                .input_channels    (input_channels),
                .input_height      (input_height),
                .input_width       (input_width),
                .weight_k          (weight_k),
                .weight_c          (weight_c),
                .weight_kh         (weight_kh),
                .weight_kw         (weight_kw),
                .output_channels   (output_channels),
                .output_height     (output_height),
                .output_width      (output_width),
                .input_base_addr   (input_base_addr),
                .weight_base_addr  (weight_base_addr),
                .output_base_addr  (output_base_addr),
                .start             (start),
                .tile_row_start    (tile_row),
                .tile_col          (tile_col_start),
                .tile_ch_start     (tile_ch_start),
                .input_ch          (ws_input_ch),
                .kernel_row        (ws_kernel_row),
                .kernel_col_start  (ws_kernel_col_start),
                .done              (done),
                .busy              (busy),
                .input_addr        (input_addr_h),
                .input_addr_valid  (input_addr_valid_h),
                .input_data        (input_data_h),
                .input_data_valid  (input_data_valid_h),
                .weight_addr       (weight_addr_2d),
                .weight_addr_valid (weight_addr_valid_2d),
                .weight_data       (weight_data_2d),
                .weight_data_valid (weight_data_valid_2d),
                .output_addr       (output_addr_1d),
                .output_addr_valid (output_addr_valid_1d),
                .output_data       (output_data_1d),
                .output_data_valid (output_data_valid_1d)
            );

            assign input_addr_w        = '{default: '0};
            assign input_addr_valid_w  = '0;
            assign weight_addr_1d      = '{default: '0};
            assign weight_addr_valid_1d= '0;
            assign output_addr_2d      = '{default: '0};
            assign output_addr_valid_2d= '0;
            assign output_data_2d      = '{default: '0};
            assign output_data_valid_2d= '0;
        end
    endgenerate

endmodule : systolic_array_wrap
