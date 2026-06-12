//============================================================================= Dilshan
// Module: weight_fetcher_tb
// Description: Testbench for weight_fetcher (Output-Stationary dataflow)
//
// The weight_fetcher pre-loads weights into ARRAY_HEIGHT PE rows.
// Each PE row[i] gets weights for output channel (output_ch_start + i).
//
// Tests:
//   1. fetch_done asserts – CHANNEL_MAJOR 1×1 kernel
//   2. fetch_done asserts – 3×3 kernel, 4 channels
//   3. addr_valid asserts per PE row during preload
//   4. ROW_MAJOR fetch_done
//   5. COLUMN_MAJOR fetch_done
//   6. Re-trigger fetch
//=============================================================================

`timescale 1ns/1ps

module weight_fetcher_tb;

    import systolic_array_pkg::*;

    localparam int ARRAY_HEIGHT = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;

    mem_layout_t           layout;
    logic [15:0]           weight_k, weight_c, weight_kh, weight_kw;
    logic [DATA_WIDTH-1:0] base_addr;
    logic [15:0]           output_ch_start;
    logic                  fetch_enable;
    logic                  fetch_done;

    logic [DATA_WIDTH-1:0]    addr_out     [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0]  addr_valid;
    logic [15:0]              weight_idx_out [ARRAY_HEIGHT];

    //=========================================================================
    // DUT
    //=========================================================================
    weight_fetcher #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .layout          (layout),
        .weight_k        (weight_k),
        .weight_c        (weight_c),
        .weight_kh       (weight_kh),
        .weight_kw       (weight_kw),
        .base_addr       (base_addr),
        .output_ch_start (output_ch_start),
        .fetch_enable    (fetch_enable),
        .fetch_done      (fetch_done),
        .addr_out        (addr_out),
        .addr_valid      (addr_valid),
        .weight_idx_out  (weight_idx_out)
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
        weight_k       = 4; weight_c = 1;
        weight_kh      = 1; weight_kw = 1;
        base_addr      = 32'h2000_0000;
        output_ch_start = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    task automatic trigger_fetch();
        @(posedge clk);
        fetch_enable = 1;
        @(posedge clk);
        fetch_enable = 0;
    endtask

    task automatic wait_done(input int max_cycles, input string test_name);
        int cnt = 0;
        while (!fetch_done && cnt < max_cycles) begin
            @(posedge clk);
            cnt++;
        end
        if (fetch_done) begin
            $display("  PASS  fetch_done: %s (~%0d cycles)", test_name, cnt);
            pass_count++;
        end else begin
            $display("  FAIL  timeout: %s (waited %0d cycles)", test_name, max_cycles);
            fail_count++;
        end
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        $display("========================================");
        $display("weight_fetcher testbench (OS)");
        $display("========================================");
        reset_dut();

        // -----------------------------------------------------------------
        // TEST 1: CHANNEL_MAJOR 1×1 kernel, 1 channel
        // -----------------------------------------------------------------
        $display("\n--- Test 1: CHANNEL_MAJOR 1×1 kernel, 1ch ---");
        layout         = CHANNEL_MAJOR;
        weight_k       = 4; weight_c = 1;
        weight_kh      = 1; weight_kw = 1;
        output_ch_start = 0;

        trigger_fetch();
        wait_done(200, "CHANNEL_MAJOR 1×1");
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 2: CHANNEL_MAJOR 3×3 kernel, 4 channels
        // -----------------------------------------------------------------
        $display("\n--- Test 2: CHANNEL_MAJOR 3×3 kernel, 4ch ---");
        layout         = CHANNEL_MAJOR;
        weight_k       = 8; weight_c = 4;
        weight_kh      = 3; weight_kw = 3;
        output_ch_start = 0;

        trigger_fetch();
        wait_done(5000, "CHANNEL_MAJOR 3×3 4ch");
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 3: addr_valid asserted for each PE row during 1×1, 1ch
        // -----------------------------------------------------------------
        $display("\n--- Test 3: addr_valid pulse check (4 PE rows) ---");
        layout         = CHANNEL_MAJOR;
        weight_k       = 4; weight_c = 1;
        weight_kh      = 1; weight_kw = 1;
        output_ch_start = 0;

        trigger_fetch();
        begin
            automatic int found [ARRAY_HEIGHT] = '{default:0};
            automatic int timeout = 0;
            while (timeout < 500) begin
                @(posedge clk);
                timeout++;
                for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) found[r] = 1;
                end
                if (fetch_done) break;
            end
            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                if (found[r]) begin
                    $display("  PASS  PE row[%0d] addr_valid seen", r);
                    pass_count++;
                end else begin
                    $display("  FAIL  PE row[%0d] addr_valid NEVER seen", r);
                    fail_count++;
                end
            end
        end
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 4: ROW_MAJOR
        // -----------------------------------------------------------------
        $display("\n--- Test 4: ROW_MAJOR 3×3 kernel, 2ch ---");
        layout         = ROW_MAJOR;
        weight_k       = 4; weight_c = 2;
        weight_kh      = 3; weight_kw = 3;
        output_ch_start = 0;

        trigger_fetch();
        wait_done(3000, "ROW_MAJOR 3×3 2ch");
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 5: COLUMN_MAJOR
        // -----------------------------------------------------------------
        $display("\n--- Test 5: COLUMN_MAJOR 3×3 kernel, 2ch ---");
        layout         = COLUMN_MAJOR;
        weight_k       = 4; weight_c = 2;
        weight_kh      = 3; weight_kw = 3;
        output_ch_start = 0;

        trigger_fetch();
        wait_done(3000, "COLUMN_MAJOR 3×3 2ch");
        repeat(5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 6: Re-trigger
        // -----------------------------------------------------------------
        $display("\n--- Test 6: Re-trigger fetch ---");
        layout         = CHANNEL_MAJOR;
        weight_k       = 4; weight_c = 1;
        weight_kh      = 1; weight_kw = 1;
        output_ch_start = 4;  // second tile of output channels

        trigger_fetch();
        wait_done(200, "Re-trigger ch_start=4");
        @(posedge clk); @(posedge clk);

        trigger_fetch();
        wait_done(200, "Re-trigger #2");
        repeat(5) @(posedge clk);

        // Summary
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #100; $finish;
    end

    initial begin #3000000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("weight_fetcher.vcd");
        $dumpvars(0, weight_fetcher_tb);
    end

endmodule : weight_fetcher_tb