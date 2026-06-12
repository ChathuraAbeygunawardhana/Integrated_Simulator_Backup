//=============================================================================
// Module: output_fetcher_ws  (FIXED)
// Description: Generates output addresses for weight stationary systolic array.
//
// WS OUTPUT CONCEPT
// ------------------
// In WS, each array ROW produces one output element per compute pass.
// Row r writes to output position:
//   channel = channel_start + r
//   row     = row_start     + r   (each row maps to a different spatial row)
//   col     = col               (same column for all rows in one tile)
//
// addr_out is [ARRAY_HEIGHT] (1 address per row, not 2D like weight fetcher).
//
// BUG FIXED (BUG-6)
// ------------------
// Original: captured gen_addr into addr_out when fetch_enable=1.
// address_generator_ws has 1-cycle pipeline latency:
//   T0 IDLE:      fetch_enable=1 → next_state=GENERATE
//   T1 GENERATE:  idx_valid driven → generator pipelines; addr_out captured HERE (WRONG)
//                 gen_addr still holds STALE value from before T1
//   T2 DONE:      gen_valid=1, gen_addr is correct → should capture HERE
//
// FIX: Capture addr_out in DONE state (when gen_valid fires), NOT in GENERATE.
// This is the same fix applied to output_fetcher_is.
//=============================================================================

module output_fetcher_ws
    import weight_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            layout,
    input  logic [15:0]            output_k,
    input  logic [15:0]            output_h,
    input  logic [15:0]            output_w,
    input  logic [DATA_WIDTH-1:0]  base_addr,

    // Tile position
    input  logic [15:0]            channel_start,
    input  logic [15:0]            row_start,
    input  logic [15:0]            col,

    // Control
    input  logic                   fetch_enable,
    output logic                   fetch_done,

    // Outputs
    output logic [DATA_WIDTH-1:0]  addr_out   [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] addr_valid,
    output logic [15:0]             out_ch  [ARRAY_HEIGHT],
    output logic [15:0]             out_row [ARRAY_HEIGHT],
    output logic [15:0]             out_col [ARRAY_HEIGHT]
);

    //=========================================================================
    // FSM: IDLE → GENERATE → DONE → IDLE
    //=========================================================================
    typedef enum logic [1:0] { IDLE, GENERATE, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        case (state)
            IDLE:     next_state = fetch_enable ? GENERATE : IDLE;
            GENERATE: next_state = DONE;
            DONE:     next_state = IDLE;
            default:  next_state = IDLE;
        endcase
    end

    //=========================================================================
    // Per-row address generation
    //=========================================================================
    genvar row;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_rows

            logic [15:0]           idx_0_row, idx_1_row, idx_2_row;
            logic                  idx_valid_row;
            logic [DATA_WIDTH-1:0] gen_addr_row;
            logic                  gen_valid_row;
            logic [15:0]           out_ch_comb, out_row_comb, out_col_comb;
            logic                  position_valid;

            always_comb begin
                out_ch_comb  = channel_start + row[15:0];
                out_row_comb = row_start     + row[15:0];
                out_col_comb = col;

                position_valid = (out_ch_comb  < output_k) &&
                                 (out_row_comb < output_h) &&
                                 (out_col_comb < output_w);

                // Drive idx_valid in GENERATE state only (not in IDLE)
                idx_valid_row = (state == GENERATE) && position_valid;

                case (layout)
                    CHANNEL_MAJOR: begin
                        idx_0_row = out_ch_comb;   // k
                        idx_1_row = out_row_comb;  // h
                        idx_2_row = out_col_comb;  // w
                    end
                    ROW_MAJOR: begin
                        idx_0_row = out_row_comb;  // h
                        idx_1_row = out_col_comb;  // w
                        idx_2_row = out_ch_comb;   // k
                    end
                    COLUMN_MAJOR: begin
                        idx_0_row = out_col_comb;  // w
                        idx_1_row = out_row_comb;  // h
                        idx_2_row = out_ch_comb;   // k
                    end
                    default: begin
                        idx_0_row = '0; idx_1_row = '0; idx_2_row = '0;
                    end
                endcase
            end

            address_generator_ws #(.DATA_WIDTH(DATA_WIDTH)) addr_gen_inst (
                .clk        (clk),
                .rst_n      (rst_n),
                .layout     (layout),
                .shape_c    (output_k),
                .shape_h    (output_h),
                .shape_w    (output_w),
                .base_addr  (base_addr),
                .idx_0      (idx_0_row),
                .idx_1      (idx_1_row),
                .idx_2      (idx_2_row),
                .idx_valid  (idx_valid_row),
                .linear_addr(gen_addr_row),
                .addr_valid (gen_valid_row)
            );

            // FIX BUG-6: Capture addr_out in DONE state (gen_valid fires here)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[row]   <= '0;
                    addr_valid[row] <= 1'b0;
                    out_ch[row]     <= '0;
                    out_row[row]    <= '0;
                    out_col[row]    <= '0;
                end else if (state == DONE && gen_valid_row) begin
                    addr_out[row]   <= gen_addr_row;
                    addr_valid[row] <= 1'b1;
                    out_ch[row]     <= out_ch_comb;
                    out_row[row]    <= out_row_comb;
                    out_col[row]    <= out_col_comb;
                end else begin
                    addr_valid[row] <= 1'b0;
                end
            end
        end
    endgenerate

    //=========================================================================
    // fetch_done - 1-cycle pulse in DONE state
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : output_fetcher_ws