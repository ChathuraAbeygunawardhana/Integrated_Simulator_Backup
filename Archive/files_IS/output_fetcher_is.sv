//=============================================================================
// Module: output_fetcher_is  (FIXED)
// Description: Manages output addresses for input stationary systolic array.
//
// IS DATAFLOW MAPPING
// --------------------
// In IS: each PE[row][col] accumulates the result for:
//   output channel = channel_start + row
//   output row     = row_start          (same for ALL rows in IS tile)
//   output column  = col_start  + col
//
// After all weights have streamed, one 2-D grid of addresses is produced,
// one per PE position (ARRAY_HEIGHT × ARRAY_WIDTH).
//
// BUG IN ORIGINAL (both fixed)
// ------------------------------
// BUG-8 / BUG-9: The original code wrote addr_out and addr_valid from
//   gen_addr (output of address_generator_is) on the cycle fetch_enable=1.
//   But address_generator_is is a REGISTERED pipeline:
//     idx_valid goes high in state GENERATE
//     gen_addr/gen_valid appear ONE posedge LATER (in state DONE)
//   So writing addr_out when fetch_enable fires captured STALE data from
//   the previous request (whatever gen_addr happened to hold).
//
// FIX: Follow the same 3-cycle pattern used by OS output_fetcher:
//   T0 (IDLE):     fetch_enable sampled  → next=GENERATE
//   T1 (GENERATE): idx_valid_sig=1       → address_generator pipelines indices
//   T2 (DONE):     gen_valid=1           → addr_out captured HERE (not T0/T1)
//                  addr_valid=1, fetch_done=1
//   T3 (IDLE):     addr_valid=0, fetch_done=0  (addr_out holds last value)
//
// addr_out is captured in the DONE state where gen_valid is guaranteed 1.
// addr_valid and fetch_done both pulse HIGH for exactly 1 cycle (in DONE).
//=============================================================================

module output_fetcher_is
    import input_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            layout,
    input  logic [15:0]            output_k,       // Output channels
    input  logic [15:0]            output_h,       // Output height
    input  logic [15:0]            output_w,       // Output width
    input  logic [DATA_WIDTH-1:0]  base_addr,

    // Tile position
    input  logic [15:0]            channel_start,  // First output channel (row 0)
    input  logic [15:0]            row_start,      // Output row (same for all rows in IS)
    input  logic [15:0]            col_start,      // First output column (col 0)

    // Control
    input  logic                   fetch_enable,
    output logic                   fetch_done,

    // 2D address outputs (one per PE)
    output logic [DATA_WIDTH-1:0]  addr_out  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid,

    // Debug / position tracking
    output logic [15:0]            out_ch  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [15:0]            out_row [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [15:0]            out_col [ARRAY_HEIGHT][ARRAY_WIDTH]
);

    //=========================================================================
    // FSM
    //=========================================================================
    typedef enum logic [1:0] {
        IDLE,
        GENERATE,
        DONE
    } state_t;

    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:     if (fetch_enable) next_state = GENERATE;
            GENERATE:                   next_state = DONE;
            DONE:                       next_state = IDLE;
            default:                    next_state = IDLE;
        endcase
    end

    //=========================================================================
    // Per-PE address generation
    //=========================================================================
    logic [15:0] idx_0 [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0] idx_1 [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0] idx_2 [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] idx_valid_sig;
    logic [DATA_WIDTH-1:0] gen_addr [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] gen_valid;

    genvar row, col;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_rows
            for (col = 0; col < ARRAY_WIDTH; col++) begin : gen_cols

                logic [15:0] output_channel;
                logic [15:0] output_row_pos;
                logic [15:0] output_col_pos;
                logic        position_valid;

                // Coordinate mapping
                always_comb begin
                    output_channel = channel_start + row;
                    output_row_pos = row_start;        // IS: same row for whole tile
                    output_col_pos = col_start + col;

                    position_valid = (output_channel < output_k) &&
                                     (output_row_pos  < output_h) &&
                                     (output_col_pos  < output_w);

                    // Index mapping by layout
                    case (layout)
                        CHANNEL_MAJOR: begin
                            idx_0[row][col] = output_channel;
                            idx_1[row][col] = output_row_pos;
                            idx_2[row][col] = output_col_pos;
                        end
                        ROW_MAJOR: begin
                            idx_0[row][col] = output_row_pos;
                            idx_1[row][col] = output_col_pos;
                            idx_2[row][col] = output_channel;
                        end
                        COLUMN_MAJOR: begin
                            idx_0[row][col] = output_col_pos;
                            idx_1[row][col] = output_row_pos;
                            idx_2[row][col] = output_channel;
                        end
                        default: begin
                            idx_0[row][col] = '0;
                            idx_1[row][col] = '0;
                            idx_2[row][col] = '0;
                        end
                    endcase

                    // Drive address generator only in GENERATE state AND in-bounds
                    idx_valid_sig[row][col] = (state == GENERATE) && position_valid;
                end

                // Address generator (1-cycle registered pipeline)
                address_generator_is #(
                    .DATA_WIDTH(DATA_WIDTH)
                ) addr_gen_inst (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .layout     (layout),
                    .shape_c    (output_k),
                    .shape_h    (output_h),
                    .shape_w    (output_w),
                    .base_addr  (base_addr),
                    .idx_0      (idx_0[row][col]),
                    .idx_1      (idx_1[row][col]),
                    .idx_2      (idx_2[row][col]),
                    .idx_valid  (idx_valid_sig[row][col]),
                    .linear_addr(gen_addr[row][col]),
                    .addr_valid (gen_valid[row][col])
                );

                // FIX: Capture addr_out in DONE state (gen_addr is ready here,
                // having been registered by address_generator_is in GENERATE cycle).
                // addr_valid pulses for exactly 1 cycle in DONE.
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        addr_out[row][col]   <= '0;
                        addr_valid[row][col] <= 1'b0;
                        out_ch[row][col]     <= '0;
                        out_row[row][col]    <= '0;
                        out_col[row][col]    <= '0;
                    end else if (state == DONE && gen_valid[row][col]) begin
                        // gen_valid=1 here because GENERATE drove idx_valid last cycle
                        addr_out[row][col]   <= gen_addr[row][col];
                        addr_valid[row][col] <= 1'b1;
                        out_ch[row][col]     <= output_channel;
                        out_row[row][col]    <= output_row_pos;
                        out_col[row][col]    <= output_col_pos;
                    end else begin
                        addr_valid[row][col] <= 1'b0;
                        // addr_out holds last value (not cleared)
                    end
                end

            end
        end
    endgenerate

    //=========================================================================
    // fetch_done - registered 1-cycle pulse in DONE state
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : output_fetcher_is