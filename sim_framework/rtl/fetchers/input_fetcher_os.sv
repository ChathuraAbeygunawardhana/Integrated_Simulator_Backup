//=============================================================================
// Module: input_fetcher  (FIXED)
// Description: Fetches input activations for output stationary systolic array
//
// FIX: The original FETCH_KERNEL → FETCH_COL → FETCH_CHANNELS multi-state
//      loop suffered from a counter/FSM synchronization problem:
//        - FETCH_KERNEL loops on itself while kh/kw are not at their maxima.
//        - But the counters are updated based on the *current* state register,
//          so on the cycle FETCH_KERNEL transitions to FETCH_COL the counters
//          show the last (max) kh/kw values but have not been reset yet.
//        - When FETCH_COL decides to go back to FETCH_KERNEL (for next col),
//          FETCH_INIT is skipped so kh/kw are never reset → first kw_cnt
//          increment immediately wraps to a wrong value and the kernel scan
//          finishes in 1 cycle instead of kh×kw cycles for every column
//          after the first.  For larger channel counts this corrupts the
//          col_capture logic causing some output columns to never receive a
//          valid address, leaving the testbench wait(done) hanging.
//
//      Solution: Collapse all three loops into one FETCH state that advances
//      col_offset (innermost) → kw → kh → ch (outermost) per clock cycle
//      and generates the address_generator indices each cycle.
//      fetch_done pulses for exactly 1 cycle in DONE.
//=============================================================================

module input_fetcher
    import systolic_array_pkg::*;
#(
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // Configuration
    input  mem_layout_t           layout,
    input  logic [15:0]           input_c,           // Input channels
    input  logic [15:0]           input_h,           // Input height
    input  logic [15:0]           input_w,           // Input width
    input  logic [15:0]           kernel_h,          // Kernel height
    input  logic [15:0]           kernel_w,          // Kernel width
    input  logic [DATA_WIDTH-1:0] base_addr,

    // Control
    input  logic [15:0]           output_row,        // Output row being computed
    input  logic [15:0]           output_col_start,  // Starting output column
    input  logic                  fetch_enable,
    output logic                  fetch_done,

    // Address outputs (one per array column)
    output logic [DATA_WIDTH-1:0] addr_out   [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] addr_valid
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
    // Counters
    // Loop order (outermost→innermost): ch → kh → kw → col_offset
    //=========================================================================
    logic [15:0] ch_cnt, kh_cnt, kw_cnt, col_cnt;

    wire last_col = (col_cnt  == ARRAY_WIDTH - 1);
    wire last_kw  = (kw_cnt   == kernel_w - 1);
    wire last_kh  = (kh_cnt   == kernel_h - 1);
    wire last_ch  = (ch_cnt   == input_c  - 1);
    wire last_all = last_col && last_kw && last_kh && last_ch;

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
            col_cnt <= '0;
            kw_cnt  <= '0;
            kh_cnt  <= '0;
            ch_cnt  <= '0;
        end else if (state == IDLE) begin
            col_cnt <= '0;
            kw_cnt  <= '0;
            kh_cnt  <= '0;
            ch_cnt  <= '0;
        end else if (state == FETCH) begin
            // Innermost: col_cnt
            if (last_col) begin
                col_cnt <= '0;
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
            end else begin
                col_cnt <= col_cnt + 1;
            end
        end
    end

    //=========================================================================
    // Address generator instance
    //=========================================================================
    logic [15:0]           idx_0, idx_1, idx_2;
    logic                  idx_valid;
    logic [DATA_WIDTH-1:0] gen_addr;
    logic                  gen_valid;

    // Compute current input position
    logic [15:0] input_h_pos, input_w_pos;
    always_comb begin
        input_h_pos = output_row       + kh_cnt;
        input_w_pos = output_col_start + col_cnt + kw_cnt;
    end

    always_comb begin
        idx_valid = (state == FETCH);

        case (layout)
            CHANNEL_MAJOR: begin
                idx_0 = ch_cnt;       // c
                idx_1 = input_h_pos;  // h
                idx_2 = input_w_pos;  // w
            end
            ROW_MAJOR: begin
                idx_0 = input_h_pos;  // h
                idx_1 = input_w_pos;  // w
                idx_2 = ch_cnt;       // c
            end
            COLUMN_MAJOR: begin
                idx_0 = input_w_pos;  // w
                idx_1 = input_h_pos;  // h
                idx_2 = ch_cnt;       // c
            end
            default: begin
                idx_0 = '0;
                idx_1 = '0;
                idx_2 = '0;
            end
        endcase
    end

    address_generator #(
        .DATA_WIDTH(DATA_WIDTH)
    ) addr_gen_inst (
        .clk(clk),
        .rst_n(rst_n),
        .layout(layout),
        .shape_c(input_c),
        .shape_h(input_h),
        .shape_w(input_w),
        .base_addr(base_addr),
        .idx_0(idx_0),
        .idx_1(idx_1),
        .idx_2(idx_2),
        .idx_valid(idx_valid),
        .linear_addr(gen_addr),
        .addr_valid(gen_valid)
    );

    //=========================================================================
    // Latch col_cnt one cycle early (address_generator has 1-cycle latency)
    // so the column index that matches gen_valid is col_capture not col_cnt.
    //=========================================================================
    logic [15:0] col_capture;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col_capture <= '0;
        else if (state == FETCH)
            col_capture <= col_cnt;   // capture current col before it increments
    end

    //=========================================================================
    // Per-column address capture
    //=========================================================================
    genvar i;
    generate
        for (i = 0; i < ARRAY_WIDTH; i++) begin : gen_col_addrs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[i]   <= '0;
                    addr_valid[i] <= 1'b0;
                end else begin
                    if (gen_valid && col_capture == 16'(i)) begin
                        addr_out[i]   <= gen_addr;
                        addr_valid[i] <= 1'b1;
                    end else begin
                        addr_valid[i] <= 1'b0;
                    end
                end
            end
        end
    endgenerate

    //=========================================================================
    // fetch_done: pulse HIGH for exactly 1 cycle in DONE
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : input_fetcher