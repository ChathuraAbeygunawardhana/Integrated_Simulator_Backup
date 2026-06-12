//=============================================================================
// Module: input_fetcher_is_tb  (FIXED v3 - XSim compatible + run-all safe)
//
// WHY THE SIMULATION STOPPED EARLY
// ----------------------------------
// Vivado auto-generates a TCL batch file with "run 1000ns" for every
// testbench.  At 100 MHz that is only 100 clock cycles.
// Tests 1-4 use ~95 cycles total, so the simulation stops exactly at
// the end of Test 4 / beginning of Test 5, printing no summary.
// The testbench code itself is correct.
//
// PERMANENT FIX IN VIVADO (do this once, applies to all simulations):
//   Flow → Settings → Simulation → Simulation tab
//   Set "xsim.simulate.runtime" to:  all
//   Click OK, then re-launch simulation.
//
// ALTERNATIVE (per-launch, no settings change needed):
//   After clicking "Run Simulation", in the Tcl Console type:
//     run all
//   This runs until the testbench $finish is reached.
//
// XSim "automatic" keyword fix (from v2):
//   All local variables moved to module scope - XSim rejects "automatic"
//   on variables inside initial/begin blocks (only legal on task/function).
//=============================================================================

`timescale 1ns/1ps

module input_fetcher_is_tb;

    import input_stationary_pkg::*;

    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic                   clk, rst_n;
    mem_layout_t            layout;
    logic [15:0]            input_c, input_h, input_w;
    logic [15:0]            kernel_h, kernel_w;
    logic [DATA_WIDTH-1:0]  base_addr;
    logic [15:0]            output_row, output_col_start;
    logic                   fetch_enable;
    logic                   fetch_done;
    logic [ARRAY_WIDTH-1:0] load_input;
    logic [DATA_WIDTH-1:0]  addr_out  [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    input_fetcher_is #(
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .layout           (layout),
        .input_c          (input_c),
        .input_h          (input_h),
        .input_w          (input_w),
        .kernel_h         (kernel_h),
        .kernel_w         (kernel_w),
        .base_addr        (base_addr),
        .output_row       (output_row),
        .output_col_start (output_col_start),
        .fetch_enable     (fetch_enable),
        .fetch_done       (fetch_done),
        .load_input       (load_input),
        .addr_out         (addr_out),
        .addr_valid       (addr_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Module-scope variables (XSim: no "automatic" inside initial/begin)
    //=========================================================================
    int pass_count;
    int fail_count;

    // Test 5
    int t5_found_av  [ARRAY_WIDTH];
    int t5_found_li  [ARRAY_WIDTH];
    int t5_mismatch  [ARRAY_WIDTH];
    int t5_timeout;
    int t5_all_sim;

    // Test 6
    int          t6_timeout;
    int          t6_checked;
    logic [31:0] t6_exp;

    // Test 7
    int          t7_timeout;
    logic [31:0] t7_exp;

    // Test 8
    int          t8_timeout;
    logic [31:0] t8_exp;

    // Test 9
    int t9_timeout;
    int t9_pulse_count   [ARRAY_WIDTH];
    int t9_expected_pulses;

    //=========================================================================
    // Tasks
    //=========================================================================
    task automatic reset_dut();
        rst_n            = 0;
        fetch_enable     = 0;
        layout           = CHANNEL_MAJOR;
        input_c          = 1;
        input_h          = 8;  input_w  = 8;
        kernel_h         = 1;  kernel_w = 1;
        base_addr        = 32'h1000_0000;
        output_row       = 0;  output_col_start = 0;
        repeat(5) @(posedge clk);
        @(negedge clk);  rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    task automatic trigger_fetch(input int max_cycles, output int done_seen);
        int cnt;
        @(negedge clk);
        fetch_enable = 1;
        @(posedge clk);
        @(negedge clk);
        fetch_enable = 0;
        cnt = 0;  done_seen = 0;
        while (cnt < max_cycles) begin
            @(posedge clk);  cnt++;
            if (fetch_done) begin done_seen = 1; break; end
        end
    endtask

    task automatic assert_done(input string test_name);
        int done_seen;
        trigger_fetch(50000, done_seen);
        if (done_seen) begin
            $display("  PASS  fetch_done asserted: %s", test_name);
            pass_count++;
        end else begin
            $display("  FAIL  fetch_done NEVER asserted (timeout): %s", test_name);
            fail_count++;
        end
        repeat(5) @(posedge clk);
    endtask

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        pass_count = 0;  fail_count = 0;

        $display("========================================");
        $display("input_fetcher_is testbench");
        $display("========================================");
        $display("IS: ALL %0d columns served simultaneously each cycle.", ARRAY_WIDTH);
        $display("Total FETCH cycles = input_c x kernel_h x kernel_w");
        $display("NOTE: Set Vivado 'xsim.simulate.runtime' to 'all'");
        $display("      or type 'run all' in the Tcl Console.");
        $display("");
        reset_dut();

        //---------------------------------------------------------------------
        // TEST 1: fetch_done fires - CHANNEL_MAJOR, 1ch, 1x1 kernel
        // Expected FETCH cycles = 1x1x1 = 1
        //---------------------------------------------------------------------
        $display("--- Test 1: CHANNEL_MAJOR  1ch  1x1 kernel ---");
        layout   = CHANNEL_MAJOR;
        input_c  = 1;  input_h = 8;  input_w  = 8;
        kernel_h = 1;  kernel_w = 1;
        output_row = 0;  output_col_start = 0;
        assert_done("CM 1ch 1x1");

        //---------------------------------------------------------------------
        // TEST 2: fetch_done fires - 1ch, 3x3 kernel
        // Expected FETCH cycles = 1x3x3 = 9
        //---------------------------------------------------------------------
        $display("--- Test 2: CHANNEL_MAJOR  1ch  3x3 kernel ---");
        kernel_h = 3;  kernel_w = 3;  input_c = 1;
        assert_done("CM 1ch 3x3");

        //---------------------------------------------------------------------
        // TEST 3: fetch_done fires - ROW_MAJOR, 2ch, 3x3 kernel
        // Expected FETCH cycles = 2x3x3 = 18
        //---------------------------------------------------------------------
        $display("--- Test 3: ROW_MAJOR  2ch  3x3 kernel ---");
        layout  = ROW_MAJOR;  input_c = 2;
        assert_done("RM 2ch 3x3");

        //---------------------------------------------------------------------
        // TEST 4: fetch_done fires - COLUMN_MAJOR, 3ch, 3x3 kernel
        // Expected FETCH cycles = 3x3x3 = 27
        //---------------------------------------------------------------------
        $display("--- Test 4: COLUMN_MAJOR  3ch  3x3 kernel ---");
        layout  = COLUMN_MAJOR;  input_c = 3;
        assert_done("ColM 3ch 3x3");

        //---------------------------------------------------------------------
        // TEST 5: addr_valid and load_input simultaneously on ALL columns
        // For 1ch 1x1: only 1 FETCH cycle, so exactly 1 pulse per column
        // and all 4 columns must be valid in the SAME cycle (IS property).
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 5: addr_valid & load_input simultaneous on all columns ---");
        layout   = CHANNEL_MAJOR;
        input_c  = 1;  kernel_h = 1;  kernel_w = 1;
        output_row = 0;  output_col_start = 0;

        for (int j = 0; j < ARRAY_WIDTH; j++) begin
            t5_found_av[j] = 0;
            t5_found_li[j] = 0;
            t5_mismatch[j] = 0;
        end
        t5_timeout = 0;  t5_all_sim = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t5_timeout < 500) begin
            @(posedge clk);  t5_timeout++;
            if (addr_valid === {ARRAY_WIDTH{1'b1}}) t5_all_sim = 1;
            for (int j = 0; j < ARRAY_WIDTH; j++) begin
                if (addr_valid[j])                  t5_found_av[j] = 1;
                if (load_input[j])                  t5_found_li[j] = 1;
                if (addr_valid[j] !== load_input[j]) t5_mismatch[j] = 1;
            end
            if (fetch_done) break;
        end

        if (t5_all_sim) begin
            $display("  PASS  All %0d columns addr_valid simultaneously in 1 cycle",
                     ARRAY_WIDTH);
            pass_count++;
        end else begin
            $display("  FAIL  Columns NOT all valid in same cycle (IS requirement)");
            fail_count++;
        end
        for (int j = 0; j < ARRAY_WIDTH; j++) begin
            if (t5_found_av[j]) begin
                $display("  PASS  col[%0d] addr_valid seen", j);  pass_count++;
            end else begin
                $display("  FAIL  col[%0d] addr_valid NEVER seen", j);  fail_count++;
            end
            if (t5_found_li[j]) begin
                $display("  PASS  col[%0d] load_input seen", j);  pass_count++;
            end else begin
                $display("  FAIL  col[%0d] load_input NEVER seen", j);  fail_count++;
            end
            if (!t5_mismatch[j]) begin
                $display("  PASS  col[%0d] load_input == addr_valid always", j);  pass_count++;
            end else begin
                $display("  FAIL  col[%0d] load_input != addr_valid at some cycle", j);  fail_count++;
            end
        end
        repeat(5) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 6: CHANNEL_MAJOR address values
        // input_h=8, input_w=8, input_c=1, kernel 1x1
        // output_row=1, output_col_start=2
        // Col j: ch=0, h=1+0=1, w=2+j+0=2+j
        // CM offset = 0*(8*8) + 1*8 + (2+j) = 10+j
        // Expected addr = base + 10 + j
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 6: CHANNEL_MAJOR address values ---");
        layout           = CHANNEL_MAJOR;
        input_c          = 1;  input_h = 8;  input_w  = 8;
        kernel_h         = 1;  kernel_w = 1;
        output_row       = 1;  output_col_start = 2;
        base_addr        = 32'h1000_0000;
        t6_timeout = 0;  t6_checked = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t6_timeout < 500) begin
            @(posedge clk);  t6_timeout++;
            if (addr_valid !== '0) begin
                for (int j = 0; j < ARRAY_WIDTH; j++) begin
                    if (addr_valid[j]) begin
                        t6_exp = base_addr + 10 + j;
                        if (addr_out[j] === t6_exp) begin
                            $display("  PASS  CM col[%0d] addr=0x%08h", j, addr_out[j]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  CM col[%0d] got=0x%08h exp=0x%08h",
                                     j, addr_out[j], t6_exp);
                            fail_count++;
                        end
                        t6_checked++;
                    end
                end
            end
            if (fetch_done) break;
        end
        if (t6_checked == 0) begin
            $display("  FAIL  No addr_valid pulses observed in Test 6");  fail_count++;
        end
        repeat(5) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 7: ROW_MAJOR address values
        // H=8, W=8, C=1, output_row=0, output_col_start=0, kernel 1x1
        // Col j: h=0, w=j, c=0
        // RM offset = 0*(8*1) + j*1 + 0 = j
        // Expected addr = base + j
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 7: ROW_MAJOR address values ---");
        layout           = ROW_MAJOR;
        input_c          = 1;  input_h = 8;  input_w  = 8;
        kernel_h         = 1;  kernel_w = 1;
        output_row       = 0;  output_col_start = 0;
        base_addr        = 32'h2000_0000;
        t7_timeout = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t7_timeout < 500) begin
            @(posedge clk);  t7_timeout++;
            if (addr_valid !== '0) begin
                for (int j = 0; j < ARRAY_WIDTH; j++) begin
                    if (addr_valid[j]) begin
                        t7_exp = base_addr + j;
                        if (addr_out[j] === t7_exp) begin
                            $display("  PASS  RM col[%0d] addr=0x%08h", j, addr_out[j]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  RM col[%0d] got=0x%08h exp=0x%08h",
                                     j, addr_out[j], t7_exp);
                            fail_count++;
                        end
                    end
                end
            end
            if (fetch_done) break;
        end
        repeat(5) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 8: COLUMN_MAJOR address values
        // H=8, W=8, C=1, output_row=0, output_col_start=0, kernel 1x1
        // Col j: w=j, h=0, c=0
        // ColM offset = j*(8*1) + 0*1 + 0 = 8*j
        // Expected addr = base + 8*j
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 8: COLUMN_MAJOR address values ---");
        layout           = COLUMN_MAJOR;
        input_c          = 1;  input_h = 8;  input_w  = 8;
        kernel_h         = 1;  kernel_w = 1;
        output_row       = 0;  output_col_start = 0;
        base_addr        = 32'h3000_0000;
        t8_timeout = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t8_timeout < 500) begin
            @(posedge clk);  t8_timeout++;
            if (addr_valid !== '0) begin
                for (int j = 0; j < ARRAY_WIDTH; j++) begin
                    if (addr_valid[j]) begin
                        t8_exp = base_addr + 8*j;
                        if (addr_out[j] === t8_exp) begin
                            $display("  PASS  ColM col[%0d] addr=0x%08h", j, addr_out[j]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  ColM col[%0d] got=0x%08h exp=0x%08h",
                                     j, addr_out[j], t8_exp);
                            fail_count++;
                        end
                    end
                end
            end
            if (fetch_done) break;
        end
        repeat(5) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 9: Pulse count check - multi-channel, multi-kernel
        // input_c=2, kernel 2x2 → 2*2*2=8 FETCH cycles → 8 addr_valid pulses
        // per column (all simultaneously each cycle, IS property verified)
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 9: Pulse count check (2ch  2x2 kernel) ---");
        layout           = CHANNEL_MAJOR;
        input_c          = 2;  input_h = 8;  input_w  = 8;
        kernel_h         = 2;  kernel_w = 2;
        output_row       = 0;  output_col_start = 0;
        base_addr        = 32'h1000_0000;
        t9_timeout         = 0;
        t9_expected_pulses = 2*2*2;    // input_c * kernel_h * kernel_w
        for (int j = 0; j < ARRAY_WIDTH; j++) t9_pulse_count[j] = 0;

        @(negedge clk);  fetch_enable = 1;
        @(posedge clk);  @(negedge clk);  fetch_enable = 0;

        while (t9_timeout < 500) begin
            @(posedge clk);  t9_timeout++;
            for (int j = 0; j < ARRAY_WIDTH; j++)
                if (addr_valid[j]) t9_pulse_count[j]++;
            if (fetch_done) break;
        end

        for (int j = 0; j < ARRAY_WIDTH; j++) begin
            if (t9_pulse_count[j] == t9_expected_pulses) begin
                $display("  PASS  col[%0d] %0d pulses (expected %0d)",
                         j, t9_pulse_count[j], t9_expected_pulses);
                pass_count++;
            end else begin
                $display("  FAIL  col[%0d] %0d pulses (expected %0d)",
                         j, t9_pulse_count[j], t9_expected_pulses);
                fail_count++;
            end
        end
        repeat(5) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 10: Re-trigger - FSM returns to IDLE and fires again correctly
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 10: Re-trigger ---");
        layout   = CHANNEL_MAJOR;
        input_c  = 1;  kernel_h = 1;  kernel_w = 1;
        base_addr = 32'h1000_0000;
        output_row = 0;  output_col_start = 0;
        assert_done("Re-trigger #1");
        assert_done("Re-trigger #2");

        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    // Global watchdog - runs simulation to completion if TCL has "run all"
    // (5ms @ 100MHz = 500,000 cycles, far more than any test needs)
    initial begin
        #5_000_000;
        $display("ERROR: Global watchdog timeout - simulation took too long.");
        $finish;
    end

    initial begin
        $dumpfile("input_fetcher_is.vcd");
        $dumpvars(0, input_fetcher_is_tb);
    end

endmodule : input_fetcher_is_tb