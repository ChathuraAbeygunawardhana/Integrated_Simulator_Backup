//=============================================================================
// Module: output_fetcher_is_tb  (FIXED v2 - XSim compatible)
//
// ROOT CAUSE OF COMPILE ERROR  (same as input_fetcher_is_tb)
// ----------------------------
// Vivado XSim rejects "automatic" variable declarations inside initial
// blocks and begin...end procedural scopes.  Legal only on task/function
// declarations and inside automatic tasks/functions.
//
// FIX: All local variables moved to MODULE scope with unique names.
// For-loop body variables ("automatic logic exp = ...") replaced with
// module-scope temporaries assigned just before use.
//
// TIMING NOTE (for address validity)
// ------------------------------------
//   T0 IDLE:      fetch_enable sampled
//   T1 GENERATE:  idx_valid driven → address_generator_is pipelines
//   T2 DONE:      gen_valid=1 → addr_out captured, addr_valid=1, fetch_done=1
//   T3 IDLE:      addr_valid=0, fetch_done=0, addr_out holds last value
//
// The snapshot always_ff captures addr_out and addr_valid ON the fetch_done
// posedge (T2).  The main sequence reads snap_* values, which remain stable.
//=============================================================================

`timescale 1ns/1ps

module output_fetcher_is_tb;

    import input_stationary_pkg::*;

    localparam int ARRAY_HEIGHT = 4;
    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;
    mem_layout_t           layout;
    logic [15:0]           output_k, output_h, output_w;
    logic [DATA_WIDTH-1:0] base_addr;
    logic [15:0]           channel_start, row_start, col_start;
    logic                  fetch_enable, fetch_done;

    logic [DATA_WIDTH-1:0]                    addr_out  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid;
    logic [15:0]                              out_ch    [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              out_row   [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              out_col   [ARRAY_HEIGHT][ARRAY_WIDTH];

    //=========================================================================
    // DUT
    //=========================================================================
    output_fetcher_is #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .layout       (layout),
        .output_k     (output_k),
        .output_h     (output_h),
        .output_w     (output_w),
        .base_addr    (base_addr),
        .channel_start(channel_start),
        .row_start    (row_start),
        .col_start    (col_start),
        .fetch_enable (fetch_enable),
        .fetch_done   (fetch_done),
        .addr_out     (addr_out),
        .addr_valid   (addr_valid),
        .out_ch       (out_ch),
        .out_row      (out_row),
        .out_col      (out_col)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Module-scope variables (XSim compatible - no "automatic" in initial)
    //=========================================================================
    int pass_count;
    int fail_count;

    // Snapshot registers: latched on fetch_done posedge
    logic [DATA_WIDTH-1:0]                    snap_addr_out  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] snap_addr_valid;
    logic [15:0]                              snap_out_ch  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              snap_out_row [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              snap_out_col [ARRAY_HEIGHT][ARRAY_WIDTH];

    // Temp variables for address checks inside for loops
    logic [15:0] tmp_k, tmp_h, tmp_w;
    logic [31:0] tmp_exp;
    logic [15:0] tmp_ch, tmp_vexp_ch;
    logic [15:0] tmp_widx;
    bit          tmp_vexp_bit;
    int          tmp_retrig_fail;

    //=========================================================================
    // Snapshot always_ff: capture addr_out the cycle fetch_done fires (T2)
    //=========================================================================
    always_ff @(posedge clk) begin
        if (fetch_done) begin
            for (int r=0; r<ARRAY_HEIGHT; r++) begin
                for (int c=0; c<ARRAY_WIDTH; c++) begin
                    snap_addr_out [r][c] <= addr_out [r][c];
                    snap_out_ch   [r][c] <= out_ch   [r][c];
                    snap_out_row  [r][c] <= out_row  [r][c];
                    snap_out_col  [r][c] <= out_col  [r][c];
                end
            end
            snap_addr_valid <= addr_valid;
        end
    end

    //=========================================================================
    // Tasks
    //=========================================================================
    task automatic reset_dut();
        rst_n         = 0;
        fetch_enable  = 0;
        layout        = CHANNEL_MAJOR;
        output_k      = 8; output_h = 6; output_w = 6;
        base_addr     = 32'h3000_0000;
        channel_start = 0; row_start = 0; col_start = 0;
        repeat(5) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // Trigger fetch, wait for fetch_done, wait 2 more cycles for snapshot NBA
    task automatic trigger_and_settle(input int max_cyc);
        int cnt;
        cnt = 0;
        @(negedge clk); fetch_enable = 1;
        @(posedge clk); @(negedge clk); fetch_enable = 0;
        while (!fetch_done && cnt < max_cyc) begin
            @(posedge clk); cnt++;
        end
        repeat(2) @(posedge clk);  // NBA settle for snapshot
    endtask

    // Address formula helpers
    function automatic logic [31:0] cm_addr(
        input logic [31:0] ba, input logic [15:0] K, H, W, k, h, w);
        return ba + k*(H*W) + h*W + w;
    endfunction

    function automatic logic [31:0] rm_addr(
        input logic [31:0] ba, input logic [15:0] K, H, W, k, h, w);
        return ba + h*(W*K) + w*K + k;
    endfunction

    function automatic logic [31:0] colm_addr(
        input logic [31:0] ba, input logic [15:0] K, H, W, k, h, w);
        return ba + w*(H*K) + h*K + k;
    endfunction

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        pass_count = 0; fail_count = 0;

        $display("========================================");
        $display("output_fetcher_is testbench (FIXED v2)");
        $display("========================================");
        $display("Results read from snapshot latched on fetch_done posedge.");
        $display("");
        reset_dut();

        //----------------------------------------------------------------------
        // TEST 1: fetch_done asserts
        //----------------------------------------------------------------------
        $display("--- Test 1: fetch_done - CHANNEL_MAJOR ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        channel_start=0; row_start=0; col_start=0;
        trigger_and_settle(200);
        $display("  PASS  fetch_done observed"); pass_count++;

        //----------------------------------------------------------------------
        // TEST 2: CHANNEL_MAJOR address values
        // output_k=8, H=6, W=6, channel_start=0, row_start=2, col_start=1
        // [r][c]: k=r, h=2, w=1+c
        // CM offset = k*H*W + h*W + w = r*36 + 12 + 1+c = 36r+13+c
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 2: CHANNEL_MAJOR address values ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        base_addr=32'h3000_0000; channel_start=0; row_start=2; col_start=1;
        trigger_and_settle(200);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_k = channel_start + r;
                tmp_h = row_start;
                tmp_w = col_start + c;
                tmp_exp = cm_addr(base_addr, output_k, output_h, output_w,
                                  tmp_k, tmp_h, tmp_w);
                if (snap_addr_valid[r][c] && snap_addr_out[r][c] === tmp_exp) begin
                    $display("  PASS  CM [%0d][%0d] addr=0x%08h", r, c, snap_addr_out[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  CM [%0d][%0d] got=0x%08h exp=0x%08h valid=%0b",
                             r, c, snap_addr_out[r][c], tmp_exp, snap_addr_valid[r][c]);
                    fail_count++;
                end
            end
        end

        //----------------------------------------------------------------------
        // TEST 3: ROW_MAJOR address values
        // channel_start=0, row_start=3, col_start=0
        // [r][c]: k=r, h=3, w=c
        // RM offset = h*W*K + w*K + k = 3*6*8 + c*8 + r = 144+8c+r
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 3: ROW_MAJOR address values ---");
        layout=ROW_MAJOR; channel_start=0; row_start=3; col_start=0;
        trigger_and_settle(200);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_k = channel_start + r;
                tmp_h = row_start;
                tmp_w = col_start + c;
                if (tmp_k >= output_k || tmp_w >= output_w) continue;
                tmp_exp = rm_addr(base_addr, output_k, output_h, output_w,
                                  tmp_k, tmp_h, tmp_w);
                if (snap_addr_valid[r][c] && snap_addr_out[r][c] === tmp_exp) begin
                    $display("  PASS  RM [%0d][%0d] addr=0x%08h", r, c, snap_addr_out[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  RM [%0d][%0d] got=0x%08h exp=0x%08h valid=%0b",
                             r, c, snap_addr_out[r][c], tmp_exp, snap_addr_valid[r][c]);
                    fail_count++;
                end
            end
        end

        //----------------------------------------------------------------------
        // TEST 4: COLUMN_MAJOR address values
        // channel_start=0, row_start=0, col_start=2
        // [r][c]: k=r, h=0, w=2+c
        // ColM offset = w*H*K + h*K + k = (2+c)*6*8 + 0 + r = 48(2+c)+r
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 4: COLUMN_MAJOR address values ---");
        layout=COLUMN_MAJOR; channel_start=0; row_start=0; col_start=2;
        trigger_and_settle(200);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_k = channel_start + r;
                tmp_h = row_start;
                tmp_w = col_start + c;
                if (tmp_k >= output_k || tmp_w >= output_w) continue;
                tmp_exp = colm_addr(base_addr, output_k, output_h, output_w,
                                    tmp_k, tmp_h, tmp_w);
                if (snap_addr_valid[r][c] && snap_addr_out[r][c] === tmp_exp) begin
                    $display("  PASS  ColM [%0d][%0d] addr=0x%08h", r, c, snap_addr_out[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  ColM [%0d][%0d] got=0x%08h exp=0x%08h valid=%0b",
                             r, c, snap_addr_out[r][c], tmp_exp, snap_addr_valid[r][c]);
                    fail_count++;
                end
            end
        end

        //----------------------------------------------------------------------
        // TEST 5: OOB masking - channel (ch_start=2, output_k=4)
        // rows 0,1: ch 2,3 valid; rows 2,3: ch 4,5 OOB
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 5: OOB masking - channel ---");
        layout=CHANNEL_MAJOR; output_k=4; output_h=6; output_w=6;
        channel_start=2; row_start=0; col_start=0;
        trigger_and_settle(200);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_ch       = channel_start + r;
            tmp_vexp_bit = (tmp_ch < output_k);
            if (tmp_vexp_bit && snap_addr_valid[r][0]) begin
                $display("  PASS  row[%0d] ch=%0d in-bounds addr_valid", r, tmp_ch);
                pass_count++;
            end else if (!tmp_vexp_bit && !snap_addr_valid[r][0]) begin
                $display("  PASS  row[%0d] ch=%0d OOB masked", r, tmp_ch);
                pass_count++;
            end else begin
                $display("  FAIL  row[%0d] ch=%0d addr_valid=%0b exp=%0b",
                         r, tmp_ch, snap_addr_valid[r][0], tmp_vexp_bit);
                fail_count++;
            end
        end

        //----------------------------------------------------------------------
        // TEST 6: OOB masking - column (col_start=4, output_w=6)
        // cols 0,1: w 4,5 valid; cols 2,3: w 6,7 OOB
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 6: OOB masking - column ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        channel_start=0; row_start=0; col_start=4;
        trigger_and_settle(200);

        for (int c=0; c<ARRAY_WIDTH; c++) begin
            tmp_widx     = col_start + c;
            tmp_vexp_bit = (tmp_widx < output_w);
            if (tmp_vexp_bit && snap_addr_valid[0][c]) begin
                $display("  PASS  col[%0d] w=%0d in-bounds addr_valid", c, tmp_widx);
                pass_count++;
            end else if (!tmp_vexp_bit && !snap_addr_valid[0][c]) begin
                $display("  PASS  col[%0d] w=%0d OOB masked", c, tmp_widx);
                pass_count++;
            end else begin
                $display("  FAIL  col[%0d] w=%0d addr_valid=%0b exp=%0b",
                         c, tmp_widx, snap_addr_valid[0][c], tmp_vexp_bit);
                fail_count++;
            end
        end

        //----------------------------------------------------------------------
        // TEST 7: Position outputs (out_ch, out_row, out_col)
        // channel_start=1, row_start=2, col_start=0
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 7: Position outputs (out_ch/out_row/out_col) ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        channel_start=1; row_start=2; col_start=0;
        trigger_and_settle(200);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_k = channel_start + r;  // exp_ch
                tmp_h = row_start;           // exp_row
                tmp_w = col_start + c;       // exp_col
                if (tmp_k >= output_k || tmp_w >= output_w) continue;
                if (snap_out_ch[r][c]  === tmp_k &&
                    snap_out_row[r][c] === tmp_h &&
                    snap_out_col[r][c] === tmp_w) begin
                    $display("  PASS  pos[%0d][%0d] ch=%0d row=%0d col=%0d",
                             r, c, snap_out_ch[r][c], snap_out_row[r][c], snap_out_col[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  pos[%0d][%0d] got(ch=%0d,row=%0d,col=%0d) exp(ch=%0d,row=%0d,col=%0d)",
                             r, c,
                             snap_out_ch[r][c], snap_out_row[r][c], snap_out_col[r][c],
                             tmp_k, tmp_h, tmp_w);
                    fail_count++;
                end
            end
        end

        //----------------------------------------------------------------------
        // TEST 8: Re-trigger
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 8: Re-trigger ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        channel_start=0; row_start=0; col_start=0;
        trigger_and_settle(200);
        trigger_and_settle(200);

        tmp_retrig_fail = 0;
        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_k   = channel_start + r;
                tmp_h   = row_start;
                tmp_w   = col_start + c;
                tmp_exp = cm_addr(base_addr, output_k, output_h, output_w,
                                  tmp_k, tmp_h, tmp_w);
                if (!snap_addr_valid[r][c] || snap_addr_out[r][c] !== tmp_exp)
                    tmp_retrig_fail++;
            end
        end
        if (tmp_retrig_fail == 0) begin
            $display("  PASS  Re-trigger: addresses correct"); pass_count++;
        end else begin
            $display("  FAIL  Re-trigger: %0d mismatches", tmp_retrig_fail); fail_count++;
        end

        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #100; $finish;
    end

    initial begin #1_000_000; $display("TIMEOUT"); $finish; end
    initial begin
        $dumpfile("output_fetcher_is.vcd");
        $dumpvars(0, output_fetcher_is_tb);
    end

endmodule : output_fetcher_is_tb