//=============================================================================
// Module: input_stationary_top  (FIXED)
// Description: Top-level input stationary systolic array.
//
// IS DATAFLOW (3 phases per tile)
// --------------------------------
// Phase 1 - LOAD_INPUTS:
//   input_fetcher_is generates ARRAY_WIDTH addresses simultaneously for each
//   (ch, kh, kw) combination.  PEs latch their stationary activation value
//   via load_input strobe + data valid.
//
// Phase 2 - STREAM_WEIGHTS (loops over all ch, kh, kw):
//   For each (ch, kh, kw): weight_fetcher_is produces one weight address per
//   row (output channel).  That weight enters the leftmost PE of each row and
//   propagates right; each PE multiplies its stationary input by the passing
//   weight and accumulates into a partial sum that flows downward.
//
// Phase 3 - WRITEBACK:
//   output_fetcher_is generates the grid of output addresses; PE results are
//   written to memory.
//
// BUGS FIXED
// -----------
// BUG-10 [psum boundary]:
//   Original:  pe_psum_valid[0][c] = pe_weight_valid[0][0]
//   This tied column c's top psum_valid to ROW 0's leftmost weight valid,
//   which is wrong for columns c>0 because the weight pipeline is staggered.
//   Fix: pe_psum_valid[0][c] = pe_weight_valid[0][c]
//   Each column's top psum_valid tracks THAT COLUMN's weight valid signal,
//   which is the weight after propagating through c pipeline stages.
//   This ensures psum accumulation starts column-by-column as the weight
//   skew propagates rightward through the array (matching IS pipeline timing).
//
// BUG-CONTROL [weight_fetch_enable double-trigger]:
//   Original control block asserted weight_fetch_enable in BOTH FETCH_WEIGHTS
//   and COMPUTE states. COMPUTE immediately transitions to FETCH_WEIGHTS, so
//   the weight fetcher saw enable for 2 consecutive cycles.
//   Fix: assert weight_fetch_enable only in FETCH_WEIGHTS state.
//   COMPUTE is now just a 1-cycle pass-through (transition state).
//=============================================================================

module input_stationary_top
    import input_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int ACCUM_WIDTH  = 48,
    parameter int ADDR_WIDTH   = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            mem_layout,
    input  logic [15:0]            input_channels,
    input  logic [15:0]            input_height,
    input  logic [15:0]            input_width,
    input  logic [15:0]            weight_k,
    input  logic [15:0]            weight_c,
    input  logic [15:0]            weight_kh,
    input  logic [15:0]            weight_kw,
    input  logic [15:0]            output_channels,
    input  logic [15:0]            output_height,
    input  logic [15:0]            output_width,

    // Base addresses
    input  logic [ADDR_WIDTH-1:0]  input_base_addr,
    input  logic [ADDR_WIDTH-1:0]  weight_base_addr,
    input  logic [ADDR_WIDTH-1:0]  output_base_addr,

    // Control
    input  logic                   start,
    input  logic [15:0]            tile_row,
    input  logic [15:0]            tile_col_start,
    input  logic [15:0]            tile_ch_start,
    output logic                   done,
    output logic                   busy,

    // Memory interface - Input (one address per column)
    output logic [ADDR_WIDTH-1:0]  input_addr  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] input_addr_valid,
    input  logic [DATA_WIDTH-1:0]  input_data  [ARRAY_WIDTH],
    input  logic [ARRAY_WIDTH-1:0] input_data_valid,

    // Memory interface - Weight (one address per row)
    output logic [ADDR_WIDTH-1:0]  weight_addr  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] weight_addr_valid,
    input  logic [DATA_WIDTH-1:0]  weight_data  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] weight_data_valid,

    // Memory interface - Output (2D array, one address per PE)
    output logic [ADDR_WIDTH-1:0]  output_addr  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid,
    output logic [DATA_WIDTH-1:0]  output_data  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid
);

    //=========================================================================
    // Internal signals
    //=========================================================================
    logic input_fetch_enable,  input_fetch_done;
    logic weight_fetch_enable, weight_fetch_done;
    logic output_fetch_enable, output_fetch_done;

    logic [ARRAY_WIDTH-1:0] load_input_strobe;

    // Weight loop counters (track which (ch, kh, kw) is being fetched)
    logic [15:0] kh_cnt, kw_cnt, ch_cnt;

    // PE interconnect
    // Weights stream horizontally: pe_weight[row][col=0] = from memory
    logic [DATA_WIDTH-1:0]  pe_weight [ARRAY_HEIGHT][ARRAY_WIDTH+1];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH:0] pe_weight_valid;

    // Partial sums flow vertically downward
    logic [ACCUM_WIDTH-1:0] pe_psum [ARRAY_HEIGHT+1][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT:0][ARRAY_WIDTH-1:0] pe_psum_valid;

    // PE control
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_enable;
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_clear;

    // PE results
    logic [ACCUM_WIDTH-1:0] pe_result [ARRAY_HEIGHT][ARRAY_WIDTH];

    //=========================================================================
    // FSM
    //=========================================================================
    typedef enum logic [3:0] {
        IDLE,
        LOAD_INPUTS,
        WAIT_INPUTS,
        FETCH_WEIGHTS,
        WAIT_WEIGHTS,
        COMPUTE,
        WRITEBACK,
        DONE_STATE
    } state_t;

    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    //=========================================================================
    // FSM next-state logic
    //=========================================================================
    always_comb begin
        next_state = state;
        case (state)
            IDLE:
                if (start) next_state = LOAD_INPUTS;

            LOAD_INPUTS:
                next_state = WAIT_INPUTS;

            WAIT_INPUTS:
                if (input_fetch_done) next_state = FETCH_WEIGHTS;

            FETCH_WEIGHTS:
                next_state = WAIT_WEIGHTS;

            WAIT_WEIGHTS:
                if (weight_fetch_done) begin
                    // Check if this was the LAST (ch, kh, kw) iteration
                    if (ch_cnt  == weight_c  - 1 &&
                        kw_cnt  == weight_kw - 1 &&
                        kh_cnt  == weight_kh - 1)
                        next_state = WRITEBACK;
                    else
                        next_state = COMPUTE;
                end

            COMPUTE:
                // 1-cycle transition back to fetch next weight
                next_state = FETCH_WEIGHTS;

            WRITEBACK:
                if (output_fetch_done) next_state = DONE_STATE;

            DONE_STATE:
                next_state = IDLE;

            default:
                next_state = IDLE;
        endcase
    end

    //=========================================================================
    // Weight loop counters  (ch innermost → kw → kh outermost)
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch_cnt <= '0;
            kw_cnt <= '0;
            kh_cnt <= '0;
        end else if (state == IDLE) begin
            ch_cnt <= '0;
            kw_cnt <= '0;
            kh_cnt <= '0;
        end else if (state == WAIT_WEIGHTS && weight_fetch_done) begin
            // Advance ch → kw → kh
            if (ch_cnt < weight_c - 1) begin
                ch_cnt <= ch_cnt + 1;
            end else begin
                ch_cnt <= '0;
                if (kw_cnt < weight_kw - 1) begin
                    kw_cnt <= kw_cnt + 1;
                end else begin
                    kw_cnt <= '0;
                    if (kh_cnt < weight_kh - 1)
                        kh_cnt <= kh_cnt + 1;
                    else
                        kh_cnt <= '0;
                end
            end
        end
    end

    //=========================================================================
    // Fetcher Instances
    //=========================================================================

    // Input Fetcher (fixed version)
    input_fetcher_is #(
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (ADDR_WIDTH)
    ) input_fetch_inst (
        .clk              (clk),
        .rst_n            (rst_n),
        .layout           (mem_layout),
        .input_c          (input_channels),
        .input_h          (input_height),
        .input_w          (input_width),
        .kernel_h         (weight_kh),
        .kernel_w         (weight_kw),
        .base_addr        (input_base_addr),
        .output_row       (tile_row),
        .output_col_start (tile_col_start),
        .fetch_enable     (input_fetch_enable),
        .fetch_done       (input_fetch_done),
        .load_input       (load_input_strobe),
        .addr_out         (input_addr),
        .addr_valid       (input_addr_valid)
    );

    // Weight Fetcher
    weight_fetcher_is #(
        .ARRAY_HEIGHT (ARRAY_HEIGHT),
        .DATA_WIDTH   (ADDR_WIDTH)
    ) weight_fetch_inst (
        .clk            (clk),
        .rst_n          (rst_n),
        .layout         (mem_layout),
        .weight_k       (weight_k),
        .weight_c       (weight_c),
        .weight_kh      (weight_kh),
        .weight_kw      (weight_kw),
        .base_addr      (weight_base_addr),
        .output_ch_start(tile_ch_start),
        .input_ch       (ch_cnt),
        .kernel_row     (kh_cnt),
        .kernel_col     (kw_cnt),
        .fetch_enable   (weight_fetch_enable),
        .fetch_done     (weight_fetch_done),
        .addr_out       (weight_addr),
        .addr_valid     (weight_addr_valid)
    );

    // Output Fetcher (fixed version)
    output_fetcher_is #(
        .ARRAY_HEIGHT (ARRAY_HEIGHT),
        .ARRAY_WIDTH  (ARRAY_WIDTH),
        .DATA_WIDTH   (ADDR_WIDTH)
    ) output_fetch_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .layout       (mem_layout),
        .output_k     (output_channels),
        .output_h     (output_height),
        .output_w     (output_width),
        .base_addr    (output_base_addr),
        .channel_start(tile_ch_start),
        .row_start    (tile_row),
        .col_start    (tile_col_start),
        .fetch_enable (output_fetch_enable),
        .fetch_done   (output_fetch_done),
        .addr_out     (output_addr),
        .addr_valid   (output_addr_valid),
        .out_ch       (),
        .out_row      (),
        .out_col      ()
    );

    //=========================================================================
    // PE Array boundaries
    //=========================================================================

    // Left boundary: weights enter from memory (one per row)
    genvar r;
    generate
        for (r = 0; r < ARRAY_HEIGHT; r++) begin : gen_left_weight
            assign pe_weight[r][0]       = weight_data[r];
            assign pe_weight_valid[r][0] = weight_data_valid[r];
        end
    endgenerate

    // Top boundary: psum from above = 0
    // psum_valid[0][c] = weight_valid of THAT COLUMN (FIX for BUG-10)
    // As weight skew propagates rightward, each column starts accumulating
    // when its own weight arrives (not when column 0's weight arrives).
    genvar c;
    generate
        for (c = 0; c < ARRAY_WIDTH; c++) begin : gen_top_psum
            assign pe_psum[0][c]       = '0;
            assign pe_psum_valid[0][c] = pe_weight_valid[0][c];
        end
    endgenerate

    //=========================================================================
    // PE grid
    //=========================================================================
    genvar ri, ci;
    generate
        for (ri = 0; ri < ARRAY_HEIGHT; ri++) begin : gen_pe_rows
            for (ci = 0; ci < ARRAY_WIDTH; ci++) begin : gen_pe_cols

                processing_element_is #(
                    .DATA_WIDTH  (DATA_WIDTH),
                    .ACCUM_WIDTH (ACCUM_WIDTH)
                ) pe_inst (
                    .clk                (clk),
                    .rst_n              (rst_n),
                    .enable             (pe_enable[ri][ci]),
                    .clear_accum        (pe_clear[ri][ci]),
                    .load_input         (load_input_strobe[ci]),
                    // Stationary input: all rows in column ci share same data
                    .activation_in      (input_data[ci]),
                    .activation_valid_in(input_data_valid[ci]),
                    // Streaming weight (propagates rightward)
                    .weight_in          (pe_weight[ri][ci]),
                    .weight_valid_in    (pe_weight_valid[ri][ci]),
                    // Partial sum from above (flows downward)
                    .psum_in            (pe_psum[ri][ci]),
                    .psum_valid_in      (pe_psum_valid[ri][ci]),
                    // Outputs
                    .weight_out         (pe_weight[ri][ci+1]),
                    .weight_valid_out   (pe_weight_valid[ri][ci+1]),
                    .psum_out           (pe_psum[ri+1][ci]),
                    .psum_valid_out     (pe_psum_valid[ri+1][ci]),
                    .result             (pe_result[ri][ci])
                );

            end
        end
    endgenerate

    //=========================================================================
    // Control signals
    //=========================================================================
    always_comb begin
        // Defaults
        input_fetch_enable  = 1'b0;
        weight_fetch_enable = 1'b0;
        output_fetch_enable = 1'b0;
        busy                = 1'b1;
        done                = 1'b0;

        for (int r2 = 0; r2 < ARRAY_HEIGHT; r2++) begin
            for (int c2 = 0; c2 < ARRAY_WIDTH; c2++) begin
                pe_enable[r2][c2] = 1'b0;
                pe_clear[r2][c2]  = 1'b0;
            end
        end

        case (state)
            IDLE: begin
                busy = 1'b0;
            end

            LOAD_INPUTS: begin
                input_fetch_enable = 1'b1;
                // Clear all accumulators before fresh tile computation
                for (int r2 = 0; r2 < ARRAY_HEIGHT; r2++)
                    for (int c2 = 0; c2 < ARRAY_WIDTH; c2++)
                        pe_clear[r2][c2] = 1'b1;
            end

            WAIT_INPUTS: begin
                // PEs latch inputs via load_input_strobe (driven by fetcher)
            end

            FETCH_WEIGHTS: begin
                // FIX: assert weight_fetch_enable ONLY here, not in COMPUTE
                weight_fetch_enable = 1'b1;
            end

            WAIT_WEIGHTS: begin
                // Enable all PEs during weight streaming
                for (int r2 = 0; r2 < ARRAY_HEIGHT; r2++)
                    for (int c2 = 0; c2 < ARRAY_WIDTH; c2++)
                        pe_enable[r2][c2] = 1'b1;
            end

            COMPUTE: begin
                // 1-cycle gap between weight fetch iterations
                // (weight_fetch_enable NOT asserted here - avoids double-trigger)
            end

            WRITEBACK: begin
                output_fetch_enable = 1'b1;
            end

            DONE_STATE: begin
                done = 1'b1;
                busy = 1'b0;
            end

            default: ;
        endcase
    end

    //=========================================================================
    // Output writeback data connections
    //=========================================================================
    always_comb begin
        for (int r2 = 0; r2 < ARRAY_HEIGHT; r2++) begin
            for (int c2 = 0; c2 < ARRAY_WIDTH; c2++) begin
                output_data[r2][c2]       = pe_result[r2][c2][DATA_WIDTH-1:0];
                output_data_valid[r2][c2] = output_addr_valid[r2][c2];
            end
        end
    end

endmodule : input_stationary_top