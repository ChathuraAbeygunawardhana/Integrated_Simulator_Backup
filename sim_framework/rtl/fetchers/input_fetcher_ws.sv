//=============================================================================
// Module: input_fetcher_ws  (FIXED)
// Description: Fetches input activations for weight stationary systolic array.
//
// WS INPUT CONCEPT
// -----------------
// In WS dataflow, input activations are BROADCAST HORIZONTALLY across each row.
// Every PE in row r receives the SAME input activation simultaneously.
// The input for row r at kernel position (kh, kw), channel c is:
//   input[ output_row_start + r + kh ][ output_col + kw ][ c ]
//
// For each (kh, kw) iteration, ALL ARRAY_HEIGHT rows are served SIMULTANEOUSLY
// (one address per row), each in the same clock cycle.
//
// Total FETCH cycles = kernel_h × kernel_w
// (input_channel is passed externally and doesn't loop here)
//
// BUGS IN ORIGINAL
// -----------------
// BUG-4  Multi-state FSM (FETCH_INIT / FETCH_KERNEL_H / FETCH_KERNEL_W):
//   - kh_cnt was incremented in FETCH_KERNEL_H BEFORE kw iteration completes,
//     so kernel row 0 was SKIPPED (kh started at 1, not 0).
//   - 2 cycles wasted per kernel position (H state + W state separately).
//   - FETCH_KERNEL_H never asserted idx_valid, so no addresses were generated
//     during that state - only in FETCH_KERNEL_W, which iterated kw linearly.
//
// BUG-5  Used address_generator (systolic_array_pkg) instead of
//         address_generator_ws (weight_stationary_pkg), causing type mismatch.
//
// FIX
// ----
// Single FETCH state: counters advance kh (outer) → kw (inner).
// All ARRAY_HEIGHT address generators run in parallel each FETCH cycle.
// addr_valid[r] and all addr_out[r] asserted simultaneously each cycle.
// fetch_done pulses in DONE state (1 cycle after last FETCH).
//=============================================================================

module input_fetcher_ws
    import weight_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            layout,
    input  logic [15:0]            input_c,
    input  logic [15:0]            input_h,
    input  logic [15:0]            input_w,
    input  logic [15:0]            kernel_h,
    input  logic [15:0]            kernel_w,
    input  logic [DATA_WIDTH-1:0]  base_addr,

    // Tile position
    input  logic [15:0]            output_row_start, // First output row of tile
    input  logic [15:0]            output_col,       // Output column
    input  logic [15:0]            input_channel,    // Current input channel (from top)

    // Control
    input  logic                   fetch_enable,
    output logic                   fetch_done,

    // Address outputs: one per row, all valid simultaneously each FETCH cycle
    output logic [DATA_WIDTH-1:0]  addr_out   [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] addr_valid
);

    //=========================================================================
    // FSM
    //=========================================================================
    typedef enum logic [1:0] { IDLE, FETCH, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    //=========================================================================
    // Counters: kh (outer) → kw (inner)
    //=========================================================================
    logic [15:0] kh_cnt, kw_cnt;

    wire last_kw  = (kw_cnt == kernel_w - 1);
    wire last_kh  = (kh_cnt == kernel_h - 1);
    wire last_all = last_kw && last_kh;

    always_comb begin
        case (state)
            IDLE:  next_state = fetch_enable ? FETCH : IDLE;
            FETCH: next_state = last_all     ? DONE  : FETCH;
            DONE:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kh_cnt <= '0;  kw_cnt <= '0;
        end else if (state == IDLE) begin
            kh_cnt <= '0;  kw_cnt <= '0;
        end else if (state == FETCH) begin
            if (last_kw) begin
                kw_cnt <= '0;
                if (!last_kh) kh_cnt <= kh_cnt + 1;
            end else begin
                kw_cnt <= kw_cnt + 1;
            end
        end
    end

    //=========================================================================
    // Per-row address generators (all run in parallel)
    // Row r: input position = (output_row_start + r + kh_cnt, output_col + kw_cnt, ch)
    //=========================================================================
    genvar r;
    generate
        for (r = 0; r < ARRAY_HEIGHT; r++) begin : gen_row_addrs

            logic [15:0]           idx_0_r, idx_1_r, idx_2_r;
            logic                  idx_valid_r;
            logic [DATA_WIDTH-1:0] gen_addr_r;
            logic                  gen_valid_r;
            logic [15:0]           in_h_pos, in_w_pos;

            always_comb begin
                in_h_pos    = output_row_start + r[15:0] + kh_cnt;
                in_w_pos    = output_col + kw_cnt;
                idx_valid_r = (state == FETCH);

                case (layout)
                    CHANNEL_MAJOR: begin
                        idx_0_r = input_channel;  // c
                        idx_1_r = in_h_pos;       // h
                        idx_2_r = in_w_pos;       // w
                    end
                    ROW_MAJOR: begin
                        idx_0_r = in_h_pos;       // h
                        idx_1_r = in_w_pos;       // w
                        idx_2_r = input_channel;  // c
                    end
                    COLUMN_MAJOR: begin
                        idx_0_r = in_w_pos;       // w
                        idx_1_r = in_h_pos;       // h
                        idx_2_r = input_channel;  // c
                    end
                    default: begin
                        idx_0_r = '0; idx_1_r = '0; idx_2_r = '0;
                    end
                endcase
            end

            address_generator_ws #(.DATA_WIDTH(DATA_WIDTH)) addr_gen_r (
                .clk        (clk),
                .rst_n      (rst_n),
                .layout     (layout),
                .shape_c    (input_c),
                .shape_h    (input_h),
                .shape_w    (input_w),
                .base_addr  (base_addr),
                .idx_0      (idx_0_r),
                .idx_1      (idx_1_r),
                .idx_2      (idx_2_r),
                .idx_valid  (idx_valid_r),
                .linear_addr(gen_addr_r),
                .addr_valid (gen_valid_r)
            );

            // Register: all ARRAY_HEIGHT rows update simultaneously
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[r]   <= '0;
                    addr_valid[r] <= 1'b0;
                end else begin
                    if (gen_valid_r) begin
                        addr_out[r]   <= gen_addr_r;
                        addr_valid[r] <= 1'b1;
                    end else begin
                        addr_valid[r] <= 1'b0;
                    end
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

endmodule : input_fetcher_ws