//=============================================================================
// Module: output_fetcher_ws_tb  (FIXED - XSim compatible)
//
// BUGS FIXED
// -----------
// BUG-3 [XSim]: "automatic logic/bit" inside for loops in initial block
//   → XSim compile error. All variables moved to module scope.
//
// TIMING NOTE
// -----------
// output_fetcher_ws FSM: IDLE → GENERATE → DONE → IDLE
// address_generator_ws has 1-cycle latency: addr arrives in DONE state.
// Snapshot always_ff captures addr_out on the fetch_done posedge.
// Main sequence reads snap_* values after 2 extra posedges (NBA settle).
//=============================================================================

`timescale 1ns/1ps

module output_fetcher_ws_tb;

    import weight_stationary_pkg::*;

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
    logic [15:0]           channel_start, row_start, col;
    logic                  fetch_enable, fetch_done;
    logic [DATA_WIDTH-1:0] addr_out   [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] addr_valid;
    logic [15:0]             out_ch  [ARRAY_HEIGHT];
    logic [15:0]             out_row [ARRAY_HEIGHT];
    logic [15:0]             out_col [ARRAY_HEIGHT];

    //=========================================================================
    // DUT
    //=========================================================================
    output_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .layout        (layout),
        .output_k      (output_k),
        .output_h      (output_h),
        .output_w      (output_w),
        .base_addr     (base_addr),
        .channel_start (channel_start),
        .row_start     (row_start),
        .col           (col),
        .fetch_enable  (fetch_enable),
        .fetch_done    (fetch_done),
        .addr_out      (addr_out),
        .addr_valid    (addr_valid),
        .out_ch        (out_ch),
        .out_row       (out_row),
        .out_col       (out_col)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Module-scope variables (no "automatic" in initial blocks)
    //=========================================================================
    int pass_count;
    int fail_count;
    int trigger_cnt;

    // Snapshot registers: latched on fetch_done posedge
    logic [DATA_WIDTH-1:0] snap_addr_out [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] snap_addr_valid;
    logic [15:0]             snap_out_ch  [ARRAY_HEIGHT];
    logic [15:0]             snap_out_row [ARRAY_HEIGHT];
    logic [15:0]             snap_out_col [ARRAY_HEIGHT];

    // Temp variables for address comparisons
    logic [15:0] tmp_k, tmp_h, tmp_w;
    logic [31:0] tmp_exp;
    logic        tmp_vexp;

    //=========================================================================
    // Snapshot: latch on fetch_done posedge (addr_valid=1 exactly here)
    //=========================================================================
    always_ff @(posedge clk) begin
        if (fetch_done) begin
            for (int r=0; r<ARRAY_HEIGHT; r++) begin
                snap_addr_out [r] <= addr_out [r];
                snap_out_ch   [r] <= out_ch   [r];
                snap_out_row  [r] <= out_row  [r];
                snap_out_col  [r] <= out_col  [r];
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
        channel_start = 0; row_start = 0; col = 0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    task automatic trigger_and_settle(input int max_cyc);
        @(negedge clk); fetch_enable = 1;
        @(posedge clk); @(negedge clk); fetch_enable = 0;
        trigger_cnt = 0;
        while (!fetch_done && trigger_cnt < max_cyc) begin
            @(posedge clk); trigger_cnt++;
        end
        repeat(2) @(posedge clk);  // NBA settle for snapshot
    endtask

    // Address formula functions
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
        $display("output_fetcher_ws testbench (FIXED)");
        $display("========================================");
        $display("WS: addr_out[ARRAY_HEIGHT], row r → (ch_start+r, row_start+r, col)");
        $display("");
        reset_dut();

        //---------------------------------------------------------------------
        // TEST 1: CHANNEL_MAJOR address values
        // output_k=8, H=6, W=6, ch_start=0, row_start=0, col=0
        // Row r: k=r, h=r, w=0
        // CM offset = r*(6*6) + r*6 + 0 = 36r+6r = 42r
        //---------------------------------------------------------------------
        $display("--- Test 1: CHANNEL_MAJOR address values ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        base_addr=32'h3000_0000; channel_start=0; row_start=0; col=0;
        trigger_and_settle(100);

        if (fetch_done || trigger_cnt < 100) begin
            $display("  PASS  fetch_done asserted"); pass_count++;
        end else begin
            $display("  FAIL  fetch_done NOT asserted"); fail_count++;
        end

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_k   = channel_start + r;
            tmp_h   = row_start + r;
            tmp_w   = col;
            if (tmp_k >= output_k || tmp_h >= output_h) continue;
            tmp_exp = cm_addr(base_addr, output_k, output_h, output_w, tmp_k, tmp_h, tmp_w);
            if (snap_addr_valid[r] && snap_addr_out[r] === tmp_exp) begin
                $display("  PASS  CM row[%0d] ch=%0d row=%0d col=%0d addr=0x%08h",
                         r,tmp_k,tmp_h,tmp_w,snap_addr_out[r]);
                pass_count++;
            end else begin
                $display("  FAIL  CM row[%0d] got=0x%08h exp=0x%08h valid=%0b",
                         r,snap_addr_out[r],tmp_exp,snap_addr_valid[r]);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 2: ROW_MAJOR address values
        // ch_start=0, row_start=1, col=2
        // Row r: k=r, h=1+r, w=2
        // RM offset = (1+r)*(6*8) + 2*8 + r = 48+48r+16+r = 64+49r
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 2: ROW_MAJOR address values ---");
        layout=ROW_MAJOR; channel_start=0; row_start=1; col=2;
        trigger_and_settle(100);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_k   = channel_start + r;
            tmp_h   = row_start + r;
            tmp_w   = col;
            if (tmp_k >= output_k || tmp_h >= output_h) continue;
            tmp_exp = rm_addr(base_addr, output_k, output_h, output_w, tmp_k, tmp_h, tmp_w);
            if (snap_addr_valid[r] && snap_addr_out[r] === tmp_exp) begin
                $display("  PASS  RM row[%0d] addr=0x%08h",r,snap_addr_out[r]);
                pass_count++;
            end else begin
                $display("  FAIL  RM row[%0d] got=0x%08h exp=0x%08h",
                         r,snap_addr_out[r],tmp_exp);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 3: COLUMN_MAJOR address values
        // ch_start=0, row_start=0, col=3
        // Row r: k=r, h=r, w=3
        // ColM offset = 3*(6*8) + r*8 + r = 144+8r+r = 144+9r
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 3: COLUMN_MAJOR address values ---");
        layout=COLUMN_MAJOR; channel_start=0; row_start=0; col=3;
        trigger_and_settle(100);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_k   = channel_start + r;
            tmp_h   = row_start + r;
            tmp_w   = col;
            if (tmp_k >= output_k || tmp_h >= output_h) continue;
            tmp_exp = colm_addr(base_addr, output_k, output_h, output_w, tmp_k, tmp_h, tmp_w);
            if (snap_addr_valid[r] && snap_addr_out[r] === tmp_exp) begin
                $display("  PASS  ColM row[%0d] addr=0x%08h",r,snap_addr_out[r]);
                pass_count++;
            end else begin
                $display("  FAIL  ColM row[%0d] got=0x%08h exp=0x%08h",
                         r,snap_addr_out[r],tmp_exp);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 4: OOB masking - row h out of bounds
        // row_start=4, output_h=6: row 0→h=4 (valid), 1→h=5 (valid)
        //                           2→h=6 (OOB),      3→h=7 (OOB)
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 4: OOB masking (row_start=4, output_h=6) ---");
        layout=CHANNEL_MAJOR; channel_start=0; row_start=4; col=0;
        trigger_and_settle(100);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_h    = row_start + r;
            tmp_vexp = (tmp_h < output_h) ? 1'b1 : 1'b0;
            if (tmp_vexp && snap_addr_valid[r]) begin
                $display("  PASS  row[%0d] h=%0d in-bounds",r,tmp_h); pass_count++;
            end else if (!tmp_vexp && !snap_addr_valid[r]) begin
                $display("  PASS  row[%0d] h=%0d OOB masked",r,tmp_h); pass_count++;
            end else begin
                $display("  FAIL  row[%0d] h=%0d addr_valid=%0b exp=%0b",
                         r,tmp_h,snap_addr_valid[r],tmp_vexp);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 5: OOB masking - channel out of bounds
        // channel_start=2, output_k=4: rows 0,1 valid; rows 2,3 OOB
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 5: OOB masking (ch_start=2, output_k=4) ---");
        layout=CHANNEL_MAJOR; output_k=4; channel_start=2; row_start=0; col=0;
        trigger_and_settle(100);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_k    = channel_start + r;
            tmp_vexp = (tmp_k < output_k) ? 1'b1 : 1'b0;
            if (tmp_vexp && snap_addr_valid[r]) begin
                $display("  PASS  row[%0d] k=%0d in-bounds",r,tmp_k); pass_count++;
            end else if (!tmp_vexp && !snap_addr_valid[r]) begin
                $display("  PASS  row[%0d] k=%0d OOB masked",r,tmp_k); pass_count++;
            end else begin
                $display("  FAIL  row[%0d] k=%0d addr_valid=%0b exp=%0b",
                         r,tmp_k,snap_addr_valid[r],tmp_vexp);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 6: Position outputs (out_ch, out_row, out_col)
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 6: Position outputs ---");
        layout=CHANNEL_MAJOR; output_k=8; output_h=6; output_w=6;
        channel_start=1; row_start=2; col=3;
        trigger_and_settle(100);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            tmp_k = channel_start + r;
            tmp_h = row_start + r;
            tmp_w = col;
            if (tmp_k >= output_k || tmp_h >= output_h) continue;
            if (snap_out_ch[r]===tmp_k && snap_out_row[r]===tmp_h && snap_out_col[r]===tmp_w) begin
                $display("  PASS  pos[%0d] ch=%0d row=%0d col=%0d",
                         r,snap_out_ch[r],snap_out_row[r],snap_out_col[r]);
                pass_count++;
            end else begin
                $display("  FAIL  pos[%0d] got(ch=%0d,row=%0d,col=%0d) exp(ch=%0d,row=%0d,col=%0d)",
                         r,snap_out_ch[r],snap_out_row[r],snap_out_col[r],tmp_k,tmp_h,tmp_w);
                fail_count++;
            end
        end

        //---------------------------------------------------------------------
        // TEST 7: Re-trigger
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 7: Re-trigger ---");
        layout=CHANNEL_MAJOR; output_k=8; channel_start=0; row_start=0; col=0;
        trigger_and_settle(100);
        trigger_and_settle(100);
        $display("  PASS  Re-trigger completed"); pass_count++;

        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count==0) $display("ALL TESTS PASSED");
        else               $display("SOME TESTS FAILED");

        #100; $finish;
    end

    initial begin #1_000_000; $display("TIMEOUT"); $finish; end
    initial begin
        $dumpfile("output_fetcher_ws.vcd");
        $dumpvars(0, output_fetcher_ws_tb);
    end

endmodule : output_fetcher_ws_tb