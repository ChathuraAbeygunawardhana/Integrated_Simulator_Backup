//=============================================================================
// Module: weight_fetcher_is
// Description: Streams weights for input stationary systolic array.
//
// Input Stationary Dataflow:
//   Weights stream horizontally across each row.  Each row corresponds to
//   one kernel position (kh, kw) across all input channels, and the weight
//   for each output channel is driven into the leftmost PE of that row.
//   The weight propagates right through the row while the stationary inputs
//   in each column PE perform the MAC.
//
//   ARRAY_HEIGHT rows process ARRAY_HEIGHT distinct output channels in
//   parallel; ARRAY_WIDTH columns are fed the same weight stream staggered
//   by one cycle to match the skewed pipeline.
//
// Signals produced:
//   addr_out[ARRAY_HEIGHT]  – one weight address per row (for the output
//                             channel mapped to that row)
//   addr_valid[ARRAY_HEIGHT]
//=============================================================================

module weight_fetcher_is
    import input_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // Configuration
    input  mem_layout_t             layout,
    input  logic [15:0]             weight_k,          // Number of output channels (filters)
    input  logic [15:0]             weight_c,          // Input channels
    input  logic [15:0]             weight_kh,         // Kernel height
    input  logic [15:0]             weight_kw,         // Kernel width
    input  logic [DATA_WIDTH-1:0]   base_addr,

    // Control
    input  logic [15:0]             output_ch_start,   // First output channel of tile
    input  logic [15:0]             input_ch,          // Current input channel
    input  logic [15:0]             kernel_row,        // Current kernel row (kh)
    input  logic [15:0]             kernel_col,        // Current kernel column (kw)
    input  logic                    fetch_enable,
    output logic                    fetch_done,

    // Address outputs – one per array row (output channel)
    output logic [DATA_WIDTH-1:0]   addr_out   [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] addr_valid
);

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        FETCH,
        DONE
    } state_t;

    state_t state, next_state;

    // -------------------------------------------------------------------------
    // Weight address calculation – one combinational block per row
    // -------------------------------------------------------------------------
    genvar row;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_row_weights

            logic [15:0]           out_ch;
            logic [31:0]           weight_offset;
            logic [DATA_WIDTH-1:0] weight_addr;
            logic                  position_valid;

            always_comb begin
                out_ch = output_ch_start + row;

                // Bounds check
                position_valid = (out_ch   < weight_k)  &&
                                 (input_ch < weight_c)  &&
                                 (kernel_row < weight_kh) &&
                                 (kernel_col < weight_kw);

                // Weight tensor layout
                case (layout)
                    CHANNEL_MAJOR: begin
                        // Storage order: K, C, KH, KW
                        weight_offset = out_ch   * (weight_c * weight_kh * weight_kw) +
                                        input_ch * (weight_kh * weight_kw) +
                                        kernel_row * weight_kw +
                                        kernel_col;
                    end
                    ROW_MAJOR: begin
                        // Storage order: K, KH, KW, C
                        weight_offset = out_ch    * (weight_kh * weight_kw * weight_c) +
                                        kernel_row * (weight_kw * weight_c) +
                                        kernel_col * weight_c +
                                        input_ch;
                    end
                    COLUMN_MAJOR: begin
                        // Storage order: K, KW, KH, C
                        weight_offset = out_ch    * (weight_kw * weight_kh * weight_c) +
                                        kernel_col * (weight_kh * weight_c) +
                                        kernel_row * weight_c +
                                        input_ch;
                    end
                    default: begin
                        weight_offset = '0;
                    end
                endcase

                weight_addr = base_addr + weight_offset;
            end

            // Register outputs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[row]   <= '0;
                    addr_valid[row] <= 1'b0;
                end else begin
                    if (state == FETCH && position_valid) begin
                        addr_out[row]   <= weight_addr;
                        addr_valid[row] <= 1'b1;
                    end else begin
                        addr_valid[row] <= 1'b0;
                    end
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // FSM – state register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:  if (fetch_enable) next_state = FETCH;
            FETCH:                   next_state = DONE;
            DONE:                    next_state = IDLE;
            default:                 next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Fetch done
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : weight_fetcher_is
