//=============================================================================
// Module: weight_fetcher_ws_tb  (FIXED v2 - XSim compatible + snapshot)
//
// WHAT WAS WRONG IN TEST 7
// -------------------------
// The DUT's addr_valid is a 1-cycle pulse in the DONE state.
// The testbench used repeat(4) @posedge after triggering, which waited
// PAST the DONE state into IDLE, where addr_valid was already cleared.
// So even in-bounds rows (rows 0,1) showed addr_valid=0.
//
// FIX IN DUT (weight_fetcher_ws.sv):
//   addr_valid now HOLDS in DONE state (not cleared by else branch).
//   Only cleared on reset or when FSM returns to IDLE.
//
// FIX IN TESTBENCH:
//   snapshot always_ff captures addr_out and addr_valid exactly when
//   fetch_done fires (DONE state posedge). Main sequence reads snap_*
//   which remains stable after 2 settle cycles. Same pattern as
//   output_fetcher_ws_tb and output_fetcher_is_tb.
//
// XSim "automatic" fix:
//   All variables that were "automatic" inside initial/begin blocks
//   moved to module scope.
//=============================================================================

`timescale 1ns/1ps

module weight_fetcher_ws_tb;

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
    logic [15:0]           weight_k, weight_c, weight_kh, weight_kw;
    logic [DATA_WIDTH-1:0] base_addr;
    logic [15:0]           output_ch_start, input_ch, kernel_row, kernel_col_start;
    logic                  fetch_enable, fetch_done;

    logic [DATA_WIDTH-1:0]                    addr_out   [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    weight_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .layout           (layout),
        .weight_k         (weight_k),
        .weight_c         (weight_c),
        .weight_kh        (weight_kh),
        .weight_kw        (weight_kw),
        .base_addr        (base_addr),
        .output_ch_start  (output_ch_start),
        .input_ch         (input_ch),
        .kernel_row       (kernel_row),
        .kernel_col_start (kernel_col_start),
        .fetch_enable     (fetch_enable),
        .fetch_done       (fetch_done),
        .addr_out         (addr_out),
        .addr_valid       (addr_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Snapshot: latch addr_out and addr_valid on fetch_done posedge
    // This avoids any timing race - main sequence reads snap_* safely.
    //=========================================================================
    logic [DATA_WIDTH-1:0]                    snap_addr_out  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] snap_addr_valid;

    always_ff @(posedge clk) begin
        if (fetch_done) begin
            for (int r = 0; r < ARRAY_HEIGHT; r++)
                for (int c = 0; c < ARRAY_WIDTH; c++)
                    snap_addr_out[r][c] <= addr_out[r][c];
            snap_addr_valid <= addr_valid;
        end
    end

    //=========================================================================
    // Module-scope variables (XSim: no "automatic" in initial/begin blocks)
    //=========================================================================
    int pass_count;
    int fail_count;
    int trigger_cnt;

    // Test 3 (addr_valid on all PEs via while loop)
    int t3_found   [ARRAY_HEIGHT][ARRAY_WIDTH];
    int t3_timeout;

    // Address check temporaries
    logic [31:0] tmp_exp;
    logic [15:0] tmp_k, tmp_kw;

    //=========================================================================
    // Tasks
    //=========================================================================
    task automatic reset_dut();
        rst_n            = 0;
        fetch_enable     = 0;
        layout           = CHANNEL_MAJOR;
        weight_k         = 4;  weight_c   = 1;
        weight_kh        = 1;  weight_kw  = 4;
        base_addr        = 32'h2000_0000;
        output_ch_start  = 0;  input_ch   = 0;
        kernel_row       = 0;  kernel_col_start = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);  rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // Trigger, wait for fetch_done, then wait 2 settle cycles for snapshot NBA
    task automatic trigger_and_settle(input int max_cyc, input string name,
                                       input bit print_done);
        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;
        trigger_cnt = 0;
        while (!fetch_done && trigger_cnt < max_cyc) begin
            @(posedge clk);  trigger_cnt++;
        end
        if (print_done) begin
            if (fetch_done || trigger_cnt < max_cyc) begin
                $display("  PASS  fetch_done: %s (~%0d cyc)", name, trigger_cnt);
                pass_count++;
            end else begin
                $display("  FAIL  timeout: %s", name);
                fail_count++;
            end
        end
        repeat(2) @(posedge clk);   // NBA settle for snapshot always_ff
    endtask

    task automatic trigger(input int max_cyc, input string name);
        trigger_and_settle(max_cyc, name, 1'b1);
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        pass_count = 0;  fail_count = 0;

        $display("========================================");
        $display("weight_fetcher_ws testbench (FIXED v2)");
        $display("========================================");
        $display("WS: 2D addr_out[%0d][%0d], one weight per PE.",
                 ARRAY_HEIGHT, ARRAY_WIDTH);
        $display("FSM: IDLE -> FETCH (1 cycle) -> DONE -> IDLE");
        $display("Snapshot latched on fetch_done for race-free address checks.");
        $display("");
        reset_dut();

        //---------------------------------------------------------------------
        // TEST 1: fetch_done - CHANNEL_MAJOR, 4 filters, 1×4 kernel
        //---------------------------------------------------------------------
        $display("--- Test 1: CHANNEL_MAJOR  4 filters  1x4 kernel ---");
        layout=CHANNEL_MAJOR;  weight_k=4;  weight_c=1;
        weight_kh=1;  weight_kw=4;  kernel_col_start=0;
        trigger(200, "CM 4f 1x4");

        //---------------------------------------------------------------------
        // TEST 2: fetch_done - CHANNEL_MAJOR, 4ch, 3×3 kernel
        //---------------------------------------------------------------------
        $display("--- Test 2: CHANNEL_MAJOR  4ch  3x3 kernel ---");
        weight_k=4;  weight_c=4;  weight_kh=3;  weight_kw=3;
        kernel_row=0;  kernel_col_start=0;  input_ch=0;
        trigger(200, "CM 4ch 3x3");

        //---------------------------------------------------------------------
        // TEST 3: addr_valid seen on ALL [ARRAY_HEIGHT][ARRAY_WIDTH] PEs
        // Uses a while loop so it catches the 1-cycle addr_valid pulse
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 3: addr_valid on all [%0d][%0d] PEs ---",
                 ARRAY_HEIGHT, ARRAY_WIDTH);
        weight_k=4;  weight_c=1;  weight_kh=1;  weight_kw=4;
        output_ch_start=0;  kernel_col_start=0;
        for (int r=0; r<ARRAY_HEIGHT; r++)
            for (int c=0; c<ARRAY_WIDTH; c++)  t3_found[r][c] = 0;
        t3_timeout = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t3_timeout < 200) begin
            @(posedge clk);  t3_timeout++;
            for (int r=0; r<ARRAY_HEIGHT; r++)
                for (int c=0; c<ARRAY_WIDTH; c++)
                    if (addr_valid[r][c])  t3_found[r][c] = 1;
            if (fetch_done) break;
        end
        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                if (t3_found[r][c]) begin
                    $display("  PASS  PE[%0d][%0d] addr_valid seen  addr=0x%08h",
                             r, c, addr_out[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  PE[%0d][%0d] addr_valid NEVER seen", r, c);
                    fail_count++;
                end
            end
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 4: CHANNEL_MAJOR address values (read from SNAPSHOT)
        // weight_k=4, weight_c=1, kh=1, kw=4
        // output_ch_start=0, input_ch=0, kernel_row=0, kernel_col_start=0
        // PE[r][c]: k=r, c=0, kh=0, kw=c
        // CM offset = r*(C*KH*KW) + 0*(KH*KW) + 0*KW + c = r*4 + c
        // Expected: base + 4r + c
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 4: CHANNEL_MAJOR address values (from snapshot) ---");
        layout=CHANNEL_MAJOR;  weight_k=4;  weight_c=1;
        weight_kh=1;  weight_kw=4;
        base_addr=32'h2000_0000;  output_ch_start=0;
        input_ch=0;  kernel_row=0;  kernel_col_start=0;

        trigger_and_settle(200, "CM addr check", 1'b0);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                tmp_exp = base_addr + 4*r + c;
                if (snap_addr_out[r][c] === tmp_exp) begin
                    $display("  PASS  CM PE[%0d][%0d] addr=0x%08h",
                             r, c, snap_addr_out[r][c]);
                    pass_count++;
                end else begin
                    $display("  FAIL  CM PE[%0d][%0d] got=0x%08h  exp=0x%08h",
                             r, c, snap_addr_out[r][c], tmp_exp);
                    fail_count++;
                end
            end
        end

        //---------------------------------------------------------------------
        // TEST 5: ROW_MAJOR
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 5: ROW_MAJOR ---");
        layout=ROW_MAJOR;  weight_c=2;  weight_kh=3;  weight_kw=3;
        trigger(200, "RM 2ch 3x3");

        //---------------------------------------------------------------------
        // TEST 6: COLUMN_MAJOR
        //---------------------------------------------------------------------
        $display("--- Test 6: COLUMN_MAJOR ---");
        layout=COLUMN_MAJOR;  weight_c=2;
        trigger(200, "ColM 2ch 3x3");

        //---------------------------------------------------------------------
        // TEST 7: OOB masking - output channel beyond weight_k (from SNAPSHOT)
        // weight_k=2, output_ch_start=0:
        //   rows 0,1 (out_ch 0,1): valid   (0,1 < weight_k=2)
        //   rows 2,3 (out_ch 2,3): OOB     (2,3 >= weight_k=2)
        // All columns same validity since kw=0..3 < weight_kw=4 for valid rows.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 7: OOB masking (weight_k=2, output_ch_start=0) ---");
        layout=CHANNEL_MAJOR;  weight_k=2;  weight_c=1;
        weight_kh=1;  weight_kw=4;
        output_ch_start=0;  kernel_col_start=0;

        trigger_and_settle(200, "OOB masking", 1'b0);

        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            for (int c=0; c<ARRAY_WIDTH; c++) begin
                // out_ch = 0 + r;  valid if r < weight_k = 2
                if (r < 2) begin
                    if (snap_addr_valid[r][c]) begin
                        $display("  PASS  PE[%0d][%0d] k=%0d in-bounds addr_valid",
                                 r, c, r);
                        pass_count++;
                    end else begin
                        $display("  FAIL  PE[%0d][%0d] k=%0d should be valid (was OOB-masked)",
                                 r, c, r);
                        fail_count++;
                    end
                end else begin
                    if (!snap_addr_valid[r][c]) begin
                        $display("  PASS  PE[%0d][%0d] k=%0d OOB correctly masked",
                                 r, c, r);
                        pass_count++;
                    end else begin
                        $display("  FAIL  PE[%0d][%0d] k=%0d should be masked",
                                 r, c, r);
                        fail_count++;
                    end
                end
            end
        end

        //---------------------------------------------------------------------
        // TEST 8: Re-trigger
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 8: Re-trigger ---");
        layout=CHANNEL_MAJOR;  weight_k=4;  weight_c=1;
        weight_kh=1;  weight_kw=4;
        trigger(200, "Re-trigger #1");
        trigger(200, "Re-trigger #2");

        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0)  $display("ALL TESTS PASSED");
        else                   $display("SOME TESTS FAILED");

        #100;  $finish;
    end

    initial begin #5_000_000;  $display("TIMEOUT");  $finish;  end

    initial begin
        $dumpfile("weight_fetcher_ws.vcd");
        $dumpvars(0, weight_fetcher_ws_tb);
    end

endmodule : weight_fetcher_ws_tb