//=============================================================================
// Module: input_fetcher_tb  (FIXED)
// Description: Testbench for input_fetcher (Output-Stationary dataflow)
//
// APPLIED TO TESTBENCH:
//   1. Simulation window: `run 1000ns` → `run 100000ns` in TCL (or use $finish)
//      The TB already calls $finish, so the TCL must use `run -all`.
//      ** Change your Vivado TCL to: `run -all`  (or `run 500us`) **
//
//   2. Test 2 address sampling: the original TB samples addr_valid[c] per
//      cycle and counts `found`. But since each column fires for exactly 1
//      cycle, multiple valid pulses for the same column (impossible once fixed)
//      are no longer an issue.  However col[0] firing twice in the broken
//      version caused found to reach 2 falsely showing 2 PASSes.
//      Fixed: use a `seen[c]` flag so each column is checked exactly once.
//
//   3. Test 2 expected address calculation: the TB uses `automatic int exp`.
//      In some simulators `int` is 32-bit signed; the comparison with a 32-bit
//      unsigned addr may give X. Changed to `logic [31:0]`.
//
//   4. After Test 1 fetch_done is already asserted when the `if(fetch_done)`
//      check runs immediately after wait_done. This is fine - kept as-is.
//=============================================================================

`timescale 1ns/1ps

module input_fetcher_tb;

    import systolic_array_pkg::*;

    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;

    mem_layout_t           layout;
    logic [15:0]           input_c, input_h, input_w;
    logic [15:0]           kernel_h, kernel_w;
    logic [DATA_WIDTH-1:0] base_addr;

    logic [15:0]           output_row, output_col_start;
    logic                  fetch_enable;
    logic                  fetch_done;

    logic [DATA_WIDTH-1:0] addr_out [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    input_fetcher #(
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
        .addr_out         (addr_out),
        .addr_valid       (addr_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Helpers
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

    task automatic reset_dut();
        rst_n          = 0;
        fetch_enable   = 0;
        layout         = CHANNEL_MAJOR;
        input_c        = 1; input_h = 8; input_w = 8;
        kernel_h       = 1; kernel_w = 1;
        base_addr      = 32'h1000_0000;
        output_row     = 0; output_col_start = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // Wait for fetch_done with a timeout
    task automatic wait_done(input int max_cycles);
        int cnt = 0;
        while (!fetch_done && cnt < max_cycles) begin
            @(posedge clk);
            cnt++;
        end
        if (!fetch_done) begin
            $display("  FAIL  fetch_done never asserted (timeout %0d cycles)", max_cycles);
            fail_count++;
        end else begin
            $display("  INFO  fetch_done after ~%0d cycles", cnt);
        end
    endtask

    // Trigger one fetch
    task automatic trigger_fetch();
        @(posedge clk);
        fetch_enable = 1;
        @(posedge clk);
        fetch_enable = 0;
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        $display("========================================");
        $display("input_fetcher testbench (OS) - FIXED");
        $display("========================================");
        reset_dut();

        // -----------------------------------------------------------------
        // TEST 1: CHANNEL_MAJOR, 1 channel, 3x3 kernel, fetch_done check
        // -----------------------------------------------------------------
        $display("\n--- Test 1: CHANNEL_MAJOR 1ch 3x3 kernel - fetch_done ---");
        layout         = CHANNEL_MAJOR;
        input_c        = 1;
        input_h        = 8; input_w = 8;
        kernel_h       = 3; kernel_w = 3;
        base_addr      = 32'h1000_0000;
        output_row     = 0;
        output_col_start = 0;

        trigger_fetch();
        wait_done(2000);
        if (fetch_done) begin
            $display("  PASS  fetch_done asserted (CHANNEL_MAJOR 3x3)");
            pass_count++;
        end
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 2: Address value check - CHANNEL_MAJOR
        //   C=1 H=8 W=8, output_row=0, output_col_start=0, 1x1 kernel
        //   Col 0: base + c*H*W + h*W + w = base + 0 + 0 + 0 = base+0
        //   Col 1: base + 0 + 0 + 1 = base+1
        //   Col 2: base + 0 + 0 + 2 = base+2
        //   Col 3: base + 0 + 0 + 3 = base+3
        // -----------------------------------------------------------------
        $display("\n--- Test 2: CHANNEL_MAJOR address values ---");
        layout         = CHANNEL_MAJOR;
        input_c        = 1;
        input_h        = 8; input_w = 8;
        kernel_h       = 1; kernel_w = 1;
        base_addr      = 32'h1000_0000;
        output_row     = 0;
        output_col_start = 0;

        trigger_fetch();

        // use seen[] flags so each column is checked exactly once
        begin
            automatic bit   seen[ARRAY_WIDTH];
            automatic int   found   = 0;
            automatic int   timeout = 0;
            automatic logic [31:0] exp_addr;

            for (int k = 0; k < ARRAY_WIDTH; k++) seen[k] = 0;

            while (found < ARRAY_WIDTH && timeout < 500) begin
                @(posedge clk);
                timeout++;
                for (int c = 0; c < ARRAY_WIDTH; c++) begin
                    if (addr_valid[c] && !seen[c]) begin
                        seen[c]   = 1;
                        exp_addr  = 32'h1000_0000 + c;   // FIX: logic[31:0]
                        if (addr_out[c] === exp_addr) begin
                            $display("  PASS  col[%0d] addr=0x%08h", c, addr_out[c]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  col[%0d] addr=0x%08h expected=0x%08h",
                                     c, addr_out[c], exp_addr);
                            fail_count++;
                        end
                        found++;
                    end
                end
            end
            if (found < ARRAY_WIDTH) begin
                $display("  FAIL  only %0d of %0d column addresses seen", found, ARRAY_WIDTH);
                fail_count++;
            end
        end
        wait_done(500);
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 3: ROW_MAJOR fetch_done
        // -----------------------------------------------------------------
        $display("\n--- Test 3: ROW_MAJOR 2ch 3x3 - fetch_done ---");
        layout         = ROW_MAJOR;
        input_c        = 2;
        input_h        = 8; input_w = 8;
        kernel_h       = 3; kernel_w = 3;
        output_row     = 1;
        output_col_start = 0;

        trigger_fetch();
        wait_done(5000);
        if (fetch_done) begin
            $display("  PASS  fetch_done (ROW_MAJOR)");
            pass_count++;
        end
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 4: COLUMN_MAJOR fetch_done
        // -----------------------------------------------------------------
        $display("\n--- Test 4: COLUMN_MAJOR 3ch 3x3 - fetch_done ---");
        layout         = COLUMN_MAJOR;
        input_c        = 3;
        input_h        = 8; input_w = 8;
        kernel_h       = 3; kernel_w = 3;
        output_row     = 0;
        output_col_start = 2;

        trigger_fetch();
        wait_done(10000);
        if (fetch_done) begin
            $display("  PASS  fetch_done (COLUMN_MAJOR)");
            pass_count++;
        end
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 5: Re-trigger fetch
        // -----------------------------------------------------------------
        $display("\n--- Test 5: Re-trigger fetch ---");
        layout         = CHANNEL_MAJOR;
        input_c        = 1;
        input_h        = 4; input_w = 4;
        kernel_h       = 1; kernel_w = 1;

        trigger_fetch();
        wait_done(500);
        @(posedge clk); @(posedge clk);

        trigger_fetch();
        wait_done(500);
        if (fetch_done) begin
            $display("  PASS  re-trigger successful");
            pass_count++;
        end else begin
            $display("  FAIL  re-trigger failed");
            fail_count++;
        end

        // Summary
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #100; $finish;
    end

    // FIX: generous timeout - 10ms simulation time is plenty for all tests
    initial begin #10000000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("input_fetcher.vcd");
        $dumpvars(0, input_fetcher_tb);
    end

endmodule : input_fetcher_tb