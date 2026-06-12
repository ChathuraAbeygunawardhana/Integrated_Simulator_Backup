//=============================================================================
// Module: input_fetcher_ws_tb  (FIXED - XSim compatible)
//
// WS INPUT FETCHER BEHAVIOUR
// ---------------------------
//   ALL ARRAY_HEIGHT rows served SIMULTANEOUSLY each FETCH cycle.
//   Total FETCH cycles = kernel_h × kernel_w
//   (input_channel is external, no inner channel loop here)
//
// BUGS FIXED
// -----------
// BUG-3 [XSim]: "automatic" inside initial blocks → moved to module scope.
// BUG-trigger: task trigger() used blocking int inside initial block
//              (automatic int cnt = 0) → XSim error.
//              Fixed: cnt declared at module scope.
//
// Set Vivado runtime to "all":
//   set_property -name {xsim.simulate.runtime} -value {all} \
//                -objects [get_filesets sim_1]
//=============================================================================

`timescale 1ns/1ps

module input_fetcher_ws_tb;

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
    logic [15:0]           input_c, input_h, input_w;
    logic [15:0]           kernel_h, kernel_w;
    logic [DATA_WIDTH-1:0] base_addr;
    logic [15:0]           output_row_start, output_col;
    logic [15:0]           input_channel;
    logic                  fetch_enable, fetch_done;
    logic [DATA_WIDTH-1:0] addr_out   [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    input_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
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
        .output_row_start (output_row_start),
        .output_col       (output_col),
        .input_channel    (input_channel),
        .fetch_enable     (fetch_enable),
        .fetch_done       (fetch_done),
        .addr_out         (addr_out),
        .addr_valid       (addr_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Module-scope variables (XSim: no "automatic" in initial/begin blocks)
    //=========================================================================
    int pass_count;
    int fail_count;
    int trigger_cnt;     // replaces automatic int cnt in trigger task

    // Test 4 variables
    int t4_found   [ARRAY_HEIGHT];
    int t4_timeout;

    // Test 5 variables (address check)
    int          t5_timeout;
    int          t5_checked;
    logic [31:0] t5_exp;

    // Test 6 variables
    int          t6_timeout;
    logic [31:0] t6_exp;

    // Test 7 variables
    int          t7_timeout;
    logic [31:0] t7_exp;

    // Test 8 pulse count
    int t8_timeout;
    int t8_pulse_count [ARRAY_HEIGHT];
    int t8_expected;

    //=========================================================================
    // Tasks
    //=========================================================================
    task automatic reset_dut();
        rst_n             = 0;
        fetch_enable      = 0;
        layout            = CHANNEL_MAJOR;
        input_c           = 1; input_h = 8; input_w = 8;
        kernel_h          = 1; kernel_w = 1;
        base_addr         = 32'h1000_0000;
        output_row_start  = 0; output_col = 0;
        input_channel     = 0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    task automatic trigger(input int max_cyc, input string name);
        @(negedge clk); fetch_enable = 1;
        @(posedge clk); @(negedge clk); fetch_enable = 0;
        trigger_cnt = 0;
        while (!fetch_done && trigger_cnt < max_cyc) begin
            @(posedge clk); trigger_cnt++;
        end
        if (fetch_done) begin
            $display("  PASS  fetch_done: %s", name); pass_count++;
        end else begin
            $display("  FAIL  timeout: %s", name); fail_count++;
        end
        repeat(2) @(posedge clk);
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        pass_count = 0; fail_count = 0;

        $display("========================================");
        $display("input_fetcher_ws testbench (FIXED)");
        $display("========================================");
        $display("WS: ALL %0d rows served simultaneously each FETCH cycle.", ARRAY_HEIGHT);
        $display("Total FETCH cycles = kernel_h x kernel_w");
        $display("");
        reset_dut();

        //---------------------------------------------------------------------
        // TEST 1: CHANNEL_MAJOR, 1×1 kernel → 1 FETCH cycle
        //---------------------------------------------------------------------
        $display("--- Test 1: CHANNEL_MAJOR 1ch 1x1 kernel ---");
        layout=CHANNEL_MAJOR; input_c=1; kernel_h=1; kernel_w=1;
        trigger(200, "CM 1ch 1x1");

        //---------------------------------------------------------------------
        // TEST 2: ROW_MAJOR, 3×3 kernel → 9 FETCH cycles
        //---------------------------------------------------------------------
        $display("--- Test 2: ROW_MAJOR 1ch 3x3 kernel ---");
        layout=ROW_MAJOR; input_c=1; kernel_h=3; kernel_w=3;
        trigger(200, "RM 1ch 3x3");

        //---------------------------------------------------------------------
        // TEST 3: COLUMN_MAJOR, 3×3 kernel → 9 FETCH cycles
        //---------------------------------------------------------------------
        $display("--- Test 3: COLUMN_MAJOR 1ch 3x3 kernel ---");
        layout=COLUMN_MAJOR; input_c=1; kernel_h=3; kernel_w=3;
        trigger(200, "ColM 1ch 3x3");

        //---------------------------------------------------------------------
        // TEST 4: addr_valid seen on ALL ARRAY_HEIGHT rows simultaneously
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 4: addr_valid seen on all %0d rows simultaneously ---", ARRAY_HEIGHT);
        layout=CHANNEL_MAJOR; input_c=1; kernel_h=1; kernel_w=1;
        for (int r=0; r<ARRAY_HEIGHT; r++) t4_found[r] = 0;
        t4_timeout = 0;

        @(negedge clk); fetch_enable=1;
        @(posedge clk); @(negedge clk); fetch_enable=0;

        while (t4_timeout < 200) begin
            @(posedge clk); t4_timeout++;
            // Check all rows valid in the SAME cycle (WS broadcast property)
            if (addr_valid === {ARRAY_HEIGHT{1'b1}}) begin
                $display("  PASS  All %0d rows addr_valid simultaneously", ARRAY_HEIGHT);
                pass_count++;
            end
            for (int r=0; r<ARRAY_HEIGHT; r++)
                if (addr_valid[r]) t4_found[r] = 1;
            if (fetch_done) break;
        end
        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            if (t4_found[r]) begin
                $display("  PASS  row[%0d] addr_valid seen", r); pass_count++;
            end else begin
                $display("  FAIL  row[%0d] addr_valid NEVER seen", r); fail_count++;
            end
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 5: CHANNEL_MAJOR address values
        // input_h=8, input_w=8, C=1, kernel 1x1
        // output_row_start=1, output_col=2, input_channel=0
        // Row r: ch=0, h=1+r, w=2
        // CM offset = 0*(8*8) + (1+r)*8 + 2 = 8+8r+2 = 10+8r
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 5: CHANNEL_MAJOR address values ---");
        layout=CHANNEL_MAJOR; input_c=1; input_h=8; input_w=8;
        kernel_h=1; kernel_w=1; output_row_start=1; output_col=2;
        input_channel=0; base_addr=32'h1000_0000;
        t5_timeout=0; t5_checked=0;

        @(negedge clk); fetch_enable=1;
        @(posedge clk); @(negedge clk); fetch_enable=0;

        while (t5_timeout < 200) begin
            @(posedge clk); t5_timeout++;
            if (addr_valid !== '0) begin
                for (int r=0; r<ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        t5_exp = base_addr + 10 + 8*r;  // 0*(64) + (1+r)*8 + 2
                        if (addr_out[r] === t5_exp) begin
                            $display("  PASS  CM row[%0d] addr=0x%08h",r,addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  CM row[%0d] got=0x%08h exp=0x%08h",
                                     r,addr_out[r],t5_exp);
                            fail_count++;
                        end
                        t5_checked++;
                    end
                end
            end
            if (fetch_done) break;
        end
        if (t5_checked==0) begin
            $display("  FAIL  No addr_valid pulses in Test 5"); fail_count++;
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 6: ROW_MAJOR address values
        // H=8, W=8, C=1, row_start=0, col=0, kernel 1x1, ch=0
        // Row r: h=r, w=0, c=0
        // RM offset = r*(8*1) + 0*1 + 0 = 8r
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 6: ROW_MAJOR address values ---");
        layout=ROW_MAJOR; input_c=1; input_h=8; input_w=8;
        kernel_h=1; kernel_w=1; output_row_start=0; output_col=0;
        input_channel=0; base_addr=32'h2000_0000;
        t6_timeout=0;

        @(negedge clk); fetch_enable=1;
        @(posedge clk); @(negedge clk); fetch_enable=0;

        while (t6_timeout < 200) begin
            @(posedge clk); t6_timeout++;
            if (addr_valid !== '0) begin
                for (int r=0; r<ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        t6_exp = base_addr + 8*r;
                        if (addr_out[r] === t6_exp) begin
                            $display("  PASS  RM row[%0d] addr=0x%08h",r,addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  RM row[%0d] got=0x%08h exp=0x%08h",
                                     r,addr_out[r],t6_exp);
                            fail_count++;
                        end
                    end
                end
            end
            if (fetch_done) break;
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 7: COLUMN_MAJOR address values
        // H=8, W=8, C=1, row_start=0, col=0, kernel 1x1, ch=0
        // Row r: w=0, h=r, c=0
        // ColM offset = 0*(8*1) + r*1 + 0 = r
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 7: COLUMN_MAJOR address values ---");
        layout=COLUMN_MAJOR; input_c=1; input_h=8; input_w=8;
        kernel_h=1; kernel_w=1; output_row_start=0; output_col=0;
        input_channel=0; base_addr=32'h3000_0000;
        t7_timeout=0;

        @(negedge clk); fetch_enable=1;
        @(posedge clk); @(negedge clk); fetch_enable=0;

        while (t7_timeout < 200) begin
            @(posedge clk); t7_timeout++;
            if (addr_valid !== '0) begin
                for (int r=0; r<ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        t7_exp = base_addr + r;
                        if (addr_out[r] === t7_exp) begin
                            $display("  PASS  ColM row[%0d] addr=0x%08h",r,addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  ColM row[%0d] got=0x%08h exp=0x%08h",
                                     r,addr_out[r],t7_exp);
                            fail_count++;
                        end
                    end
                end
            end
            if (fetch_done) break;
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 8: Pulse count - 3x3 kernel → 9 pulses per row
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 8: Pulse count (1ch 3x3 kernel, exp=9 per row) ---");
        layout=CHANNEL_MAJOR; input_c=1; kernel_h=3; kernel_w=3;
        output_row_start=0; output_col=0; input_channel=0;
        base_addr=32'h1000_0000;
        t8_timeout=0; t8_expected=3*3;
        for (int r=0; r<ARRAY_HEIGHT; r++) t8_pulse_count[r]=0;

        @(negedge clk); fetch_enable=1;
        @(posedge clk); @(negedge clk); fetch_enable=0;

        while (t8_timeout < 200) begin
            @(posedge clk); t8_timeout++;
            for (int r=0; r<ARRAY_HEIGHT; r++)
                if (addr_valid[r]) t8_pulse_count[r]++;
            if (fetch_done) break;
        end
        for (int r=0; r<ARRAY_HEIGHT; r++) begin
            if (t8_pulse_count[r] == t8_expected) begin
                $display("  PASS  row[%0d] %0d pulses (exp %0d)",r,t8_pulse_count[r],t8_expected);
                pass_count++;
            end else begin
                $display("  FAIL  row[%0d] %0d pulses (exp %0d)",r,t8_pulse_count[r],t8_expected);
                fail_count++;
            end
        end
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // TEST 9: Re-trigger
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 9: Re-trigger ---");
        layout=CHANNEL_MAJOR; input_c=1; kernel_h=1; kernel_w=1;
        trigger(200, "Re-trigger #1");
        trigger(200, "Re-trigger #2");

        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count==0) $display("ALL TESTS PASSED");
        else               $display("SOME TESTS FAILED");

        #100; $finish;
    end

    initial begin #5_000_000; $display("TIMEOUT"); $finish; end
    initial begin
        $dumpfile("input_fetcher_ws.vcd");
        $dumpvars(0, input_fetcher_ws_tb);
    end

endmodule : input_fetcher_ws_tb