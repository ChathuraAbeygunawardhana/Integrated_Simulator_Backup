//=============================================================================
// Module: input_fetcher_is  (FIXED)
// Description: Fetches input activations for input stationary systolic array.
//
// INPUT STATIONARY CONCEPT
// -------------------------
// In IS dataflow, each PE column (j) holds ONE input activation permanently.
// That activation is:  input[ output_row + kh ][ output_col_start + j + kw ][ ch ]
//
// For each (ch, kh, kw) iteration, ALL ARRAY_WIDTH columns are loaded
// SIMULTANEOUSLY in a single clock cycle. Each column j gets the address
// for its own spatial position.  This is fundamentally different from the
// OS input_fetcher which serves columns one at a time with col_offset cycling.
//
// BUGS IN ORIGINAL (all fixed here)
// -----------------------------------
// BUG-1  Multi-state FSM (FETCH_KERNEL / FETCH_CHANNELS) had no kh/kw counter
//         reset when transitioning back from FETCH_CHANNELS → FETCH_KERNEL.
//         Counters held their max values and iterated incorrectly.
//
// BUG-2  col_offset iterated as the INNERMOST loop inside FETCH_KERNEL,
//         producing addresses ONE column at a time over ARRAY_WIDTH cycles
//         per (kh,kw) position.  IS requires ALL columns simultaneously.
//
// BUG-3  Output demux wrote to addr_out[i] only when col_offset==i, so only
//         one column was valid per cycle - again wrong for IS.
//
// FIX
// ----
// Replaced the multi-state loop with a single FETCH state (same pattern as
// the fixed OS input_fetcher).  Counters iterate:
//   ch_cnt (outermost) → kh_cnt → kw_cnt (innermost)
//
// Each cycle in FETCH: ALL ARRAY_WIDTH address generators are driven
// simultaneously with:
//   col j:  input_h = output_row    + kh_cnt
//            input_w = output_col_start + j   + kw_cnt
//            channel = ch_cnt
//
// addr_out[j] and addr_valid[j] are ALL registered in the same cycle.
// load_input[j] pulses HIGH together with addr_valid[j].
//
// Total FETCH cycles = input_c × kernel_h × kernel_w  (NOT × ARRAY_WIDTH).
//=============================================================================

module input_fetcher_is
    import input_stationary_pkg::*;
#(
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            layout,
    input  logic [15:0]            input_c,           // Input channels
    input  logic [15:0]            input_h,           // Input height
    input  logic [15:0]            input_w,           // Input width
    input  logic [15:0]            kernel_h,          // Kernel height
    input  logic [15:0]            kernel_w,          // Kernel width
    input  logic [DATA_WIDTH-1:0]  base_addr,

    // Control
    input  logic [15:0]            output_row,        // Output row being computed
    input  logic [15:0]            output_col_start,  // First output column of tile
    input  logic                   fetch_enable,
    output logic                   fetch_done,

    // load_input strobe - one per column, pulses with addr_valid
    output logic [ARRAY_WIDTH-1:0] load_input,

    // Address outputs - ALL ARRAY_WIDTH valid SIMULTANEOUSLY each FETCH cycle
    output logic [DATA_WIDTH-1:0]  addr_out  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] addr_valid
);

    //=========================================================================
    // FSM - three states only (same clean pattern as fixed OS input_fetcher)
    //=========================================================================
    typedef enum logic [1:0] { IDLE, FETCH, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    //=========================================================================
    // Counters  (outermost → innermost: ch → kh → kw)
    //=========================================================================
    logic [15:0] ch_cnt, kh_cnt, kw_cnt;

    wire last_kw  = (kw_cnt == kernel_w - 1);
    wire last_kh  = (kh_cnt == kernel_h - 1);
    wire last_ch  = (ch_cnt == input_c  - 1);
    wire last_all = last_kw && last_kh && last_ch;

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
            ch_cnt <= '0;
            kh_cnt <= '0;
            kw_cnt <= '0;
        end else if (state == IDLE) begin
            ch_cnt <= '0;
            kh_cnt <= '0;
            kw_cnt <= '0;
        end else if (state == FETCH) begin
            // Advance innermost (kw) first, then kh, then ch (outermost)
            if (last_kw) begin
                kw_cnt <= '0;
                if (last_kh) begin
                    kh_cnt <= '0;
                    if (!last_ch) ch_cnt <= ch_cnt + 1;
                end else begin
                    kh_cnt <= kh_cnt + 1;
                end
            end else begin
                kw_cnt <= kw_cnt + 1;
            end
        end
    end

    //=========================================================================
    // Address generators - one instance per column  (j = 0..ARRAY_WIDTH-1)
    // Each column j computes address for spatial position (col_start + j + kw_cnt).
    // All run in PARALLEL - all ARRAY_WIDTH addresses are valid simultaneously.
    //=========================================================================
    genvar j;
    generate
        for (j = 0; j < ARRAY_WIDTH; j++) begin : gen_col_addrs

            logic [15:0]           idx_0_j, idx_1_j, idx_2_j;
            logic                  idx_valid_j;
            logic [DATA_WIDTH-1:0] gen_addr_j;
            logic                  gen_valid_j;
            logic [15:0]           input_h_pos_j, input_w_pos_j;

            // Spatial position for column j
            always_comb begin
                input_h_pos_j = output_row       + kh_cnt;
                input_w_pos_j = output_col_start + j[15:0] + kw_cnt;
            end

            // Index mapping by memory layout
            always_comb begin
                idx_valid_j = (state == FETCH);
                case (layout)
                    CHANNEL_MAJOR: begin
                        idx_0_j = ch_cnt;           // c
                        idx_1_j = input_h_pos_j;    // h
                        idx_2_j = input_w_pos_j;    // w
                    end
                    ROW_MAJOR: begin
                        idx_0_j = input_h_pos_j;    // h
                        idx_1_j = input_w_pos_j;    // w
                        idx_2_j = ch_cnt;            // c
                    end
                    COLUMN_MAJOR: begin
                        idx_0_j = input_w_pos_j;    // w
                        idx_1_j = input_h_pos_j;    // h
                        idx_2_j = ch_cnt;            // c
                    end
                    default: begin
                        idx_0_j = '0;
                        idx_1_j = '0;
                        idx_2_j = '0;
                    end
                endcase
            end

            address_generator_is #(
                .DATA_WIDTH(DATA_WIDTH)
            ) addr_gen_j (
                .clk        (clk),
                .rst_n      (rst_n),
                .layout     (layout),
                .shape_c    (input_c),
                .shape_h    (input_h),
                .shape_w    (input_w),
                .base_addr  (base_addr),
                .idx_0      (idx_0_j),
                .idx_1      (idx_1_j),
                .idx_2      (idx_2_j),
                .idx_valid  (idx_valid_j),
                .linear_addr(gen_addr_j),
                .addr_valid (gen_valid_j)
            );

            // Register outputs - all columns update in the SAME cycle
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[j]   <= '0;
                    addr_valid[j] <= 1'b0;
                    load_input[j] <= 1'b0;
                end else begin
                    if (gen_valid_j) begin
                        addr_out[j]   <= gen_addr_j;
                        addr_valid[j] <= 1'b1;
                        load_input[j] <= 1'b1;   // Strobe PE column j to latch input
                    end else begin
                        addr_valid[j] <= 1'b0;
                        load_input[j] <= 1'b0;
                    end
                end
            end

        end
    endgenerate

    //=========================================================================
    // fetch_done - registered pulse in DONE state (1 cycle)
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : input_fetcher_is