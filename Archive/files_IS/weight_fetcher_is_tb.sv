//============================================================================= Dilshan
// Module: weight_fetcher_is_tb
// Description: Testbench for weight_fetcher_is (Input-Stationary dataflow)
//
// KEY BEHAVIOUR understood from source:
//   - Very simple 3-state FSM:  IDLE → FETCH → DONE  (one cycle per state).
//   - In the FETCH state, address calculation is purely combinational per row.
//   - addr_out[r] and addr_valid[r] are registered on the rising edge of FETCH.
//   - fetch_done is registered on the DONE state (fires one cycle after FETCH).
//   - Outputs ARRAY_HEIGHT addresses simultaneously – one per row (output ch).
//   - Row r → out_ch = output_ch_start + r.
//   - Bounds check:  out_ch < weight_k, input_ch < weight_c,
//                    kernel_row < weight_kh, kernel_col < weight_kw.
//   - CHANNEL_MAJOR offset: out_ch*(C*KH*KW) + input_ch*(KH*KW)
//                            + kernel_row*KW  + kernel_col
//   - ROW_MAJOR     offset: out_ch*(KH*KW*C)  + kernel_row*(KW*C)
//                            + kernel_col*C    + input_ch
//   - COLUMN_MAJOR  offset: out_ch*(KW*KH*C)  + kernel_col*(KH*C)
//                            + kernel_row*C    + input_ch
//
// Tests
//   1.  fetch_done asserts after exactly ~2 cycles – CHANNEL_MAJOR
//   2.  addr_valid asserts on all ARRAY_HEIGHT rows in FETCH cycle
//   3.  CHANNEL_MAJOR address values (all 4 rows)
//   4.  ROW_MAJOR   address values
//   5.  COLUMN_MAJOR address values
//   6.  Out-of-bounds masking (out_ch >= weight_k → addr_valid[r] = 0)
//   7.  Out-of-bounds: input_ch >= weight_c → all rows masked
//   8.  Re-trigger: returns to IDLE and fires again
//=============================================================================

`timescale 1ns/1ps

module weight_fetcher_is_tb;

    import input_stationary_pkg::*;

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam int ARRAY_HEIGHT = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;

    // Configuration
    mem_layout_t           layout;
    logic [15:0]           weight_k, weight_c, weight_kh, weight_kw;
    logic [DATA_WIDTH-1:0] base_addr;

    // Control
    logic [15:0]           output_ch_start;
    logic [15:0]           input_ch;
    logic [15:0]           kernel_row;
    logic [15:0]           kernel_col;
    logic                  fetch_enable;
    logic                  fetch_done;

    // Outputs
    logic [DATA_WIDTH-1:0]    addr_out   [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0]  addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    weight_fetcher_is #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .layout         (layout),
        .weight_k       (weight_k),
        .weight_c       (weight_c),
        .weight_kh      (weight_kh),
        .weight_kw      (weight_kw),
        .base_addr      (base_addr),
        .output_ch_start(output_ch_start),
        .input_ch       (input_ch),
        .kernel_row     (kernel_row),
        .kernel_col     (kernel_col),
        .fetch_enable   (fetch_enable),
        .fetch_done     (fetch_done),
        .addr_out       (addr_out),
        .addr_valid     (addr_valid)
    );

    //=========================================================================
    // Clock
    //=========================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Helpers
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

    task automatic reset_dut();
        rst_n           = 0;
        fetch_enable    = 0;
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;
        weight_c        = 4;
        weight_kh       = 3;
        weight_kw       = 3;
        base_addr       = 32'h2000_0000;
        output_ch_start = 0;
        input_ch        = 0;
        kernel_row      = 0;
        kernel_col      = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // Trigger one fetch and wait up to max_cycles for fetch_done.
    // Because the FSM is IDLE→FETCH→DONE (3 states), fetch_done fires
    // within ~3 clock cycles.  200 cycles is a very generous timeout.
    task automatic do_fetch(input int max_cycles, input string name, output int done_seen);
        int cnt;
        @(posedge clk);
        fetch_enable = 1;
        @(posedge clk);
        fetch_enable = 0;

        cnt       = 0;
        done_seen = 0;
        while (cnt < max_cycles) begin
            @(posedge clk);
            cnt++;
            if (fetch_done) begin done_seen = 1; break; end
        end

        if (done_seen)
            $display("  PASS  fetch_done: %s  (after %0d cycles)", name, cnt);
        else
            $display("  FAIL  fetch_done TIMEOUT: %s", name);
    endtask

    // CHANNEL_MAJOR weight offset: k*(C*KH*KW) + c*(KH*KW) + kh*KW + kw
    function automatic logic [31:0] cm_w_offset(
        input logic [15:0] K, C, KH, KW, k, c, kh, kw);
        return k*(C*KH*KW) + c*(KH*KW) + kh*KW + kw;
    endfunction

    // ROW_MAJOR weight offset: k*(KH*KW*C) + kh*(KW*C) + kw*C + c
    function automatic logic [31:0] rm_w_offset(
        input logic [15:0] K, C, KH, KW, k, c, kh, kw);
        return k*(KH*KW*C) + kh*(KW*C) + kw*C + c;
    endfunction

    // COLUMN_MAJOR weight offset: k*(KW*KH*C) + kw*(KH*C) + kh*C + c
    function automatic logic [31:0] colm_w_offset(
        input logic [15:0] K, C, KH, KW, k, c, kh, kw);
        return k*(KW*KH*C) + kw*(KH*C) + kh*C + c;
    endfunction

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("weight_fetcher_is testbench");
        $display("========================================");
        reset_dut();

        // ------------------------------------------------------------------
        // TEST 1: fetch_done asserts – CHANNEL_MAJOR
        // ------------------------------------------------------------------
        $display("\n--- Test 1: fetch_done – CHANNEL_MAJOR ---");
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;  weight_c  = 4;
        weight_kh       = 3;  weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 0;  kernel_row = 0; kernel_col = 0;
        begin
            int ds;
            do_fetch(200, "CM basic", ds);
            if (ds) pass_count++; else fail_count++;
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 2: addr_valid asserts on ALL ARRAY_HEIGHT rows
        // ------------------------------------------------------------------
        $display("\n--- Test 2: addr_valid on all %0d rows ---", ARRAY_HEIGHT);
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;  weight_c  = 4;
        weight_kh       = 3;  weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 0;  kernel_row = 0; kernel_col = 0;

        begin
            automatic int found    [ARRAY_HEIGHT];
            automatic int timeout  = 0;
            for (int r = 0; r < ARRAY_HEIGHT; r++) found[r] = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout < 200) begin
                @(posedge clk); timeout++;
                for (int r = 0; r < ARRAY_HEIGHT; r++)
                    if (addr_valid[r]) found[r] = 1;
                if (fetch_done) break;
            end

            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                if (found[r]) begin
                    $display("  PASS  row[%0d] addr_valid seen  addr=0x%08h",
                             r, addr_out[r]);
                    pass_count++;
                end else begin
                    $display("  FAIL  row[%0d] addr_valid NEVER seen", r);
                    fail_count++;
                end
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 3: CHANNEL_MAJOR address values
        //   weight_k=8, C=4, KH=3, KW=3, base=0x2000_0000
        //   output_ch_start=0, input_ch=1, kernel_row=2, kernel_col=1
        //   Row r → out_ch = 0+r
        //   CM offset = r*(4*3*3) + 1*(3*3) + 2*3 + 1 = 36r + 9 + 6 + 1 = 36r+16
        // ------------------------------------------------------------------
        $display("\n--- Test 3: CHANNEL_MAJOR address values ---");
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;   weight_c  = 4;
        weight_kh       = 3;   weight_kw = 3;
        base_addr       = 32'h2000_0000;
        output_ch_start = 0;
        input_ch        = 1;   kernel_row = 2;  kernel_col = 1;

        begin
            automatic int timeout3 = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout3 < 200) begin
                @(posedge clk); timeout3++;
                for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        automatic logic [15:0] k_idx = output_ch_start + r;
                        automatic logic [31:0] exp   =
                            base_addr + cm_w_offset(weight_k, weight_c,
                                                    weight_kh, weight_kw,
                                                    k_idx, input_ch,
                                                    kernel_row, kernel_col);
                        if (addr_out[r] === exp) begin
                            $display("  PASS  CM row[%0d] out_ch=%0d  addr=0x%08h",
                                     r, k_idx, addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  CM row[%0d] got=0x%08h exp=0x%08h",
                                     r, addr_out[r], exp);
                            fail_count++;
                        end
                    end
                end
                if (fetch_done) break;
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 4: ROW_MAJOR address values
        //   weight_k=8, C=4, KH=3, KW=3
        //   output_ch_start=0, input_ch=0, kernel_row=1, kernel_col=2
        //   RM offset = r*(3*3*4) + 1*(3*4) + 2*4 + 0 = 36r + 12 + 8 = 36r+20
        // ------------------------------------------------------------------
        $display("\n--- Test 4: ROW_MAJOR address values ---");
        layout          = ROW_MAJOR;
        weight_k        = 8;   weight_c  = 4;
        weight_kh       = 3;   weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 0;   kernel_row = 1;  kernel_col = 2;

        begin
            automatic int timeout4 = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout4 < 200) begin
                @(posedge clk); timeout4++;
                for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        automatic logic [15:0] k_idx = output_ch_start + r;
                        automatic logic [31:0] exp   =
                            base_addr + rm_w_offset(weight_k, weight_c,
                                                    weight_kh, weight_kw,
                                                    k_idx, input_ch,
                                                    kernel_row, kernel_col);
                        if (addr_out[r] === exp) begin
                            $display("  PASS  RM row[%0d] out_ch=%0d  addr=0x%08h",
                                     r, k_idx, addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  RM row[%0d] got=0x%08h exp=0x%08h",
                                     r, addr_out[r], exp);
                            fail_count++;
                        end
                    end
                end
                if (fetch_done) break;
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 5: COLUMN_MAJOR address values
        //   output_ch_start=0, input_ch=2, kernel_row=0, kernel_col=1
        //   ColM offset = r*(3*3*4) + 1*(3*4) + 0*4 + 2 = 36r + 12 + 0 + 2 = 36r+14
        // ------------------------------------------------------------------
        $display("\n--- Test 5: COLUMN_MAJOR address values ---");
        layout          = COLUMN_MAJOR;
        weight_k        = 8;   weight_c  = 4;
        weight_kh       = 3;   weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 2;   kernel_row = 0;  kernel_col = 1;

        begin
            automatic int timeout5 = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout5 < 200) begin
                @(posedge clk); timeout5++;
                for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                    if (addr_valid[r]) begin
                        automatic logic [15:0] k_idx = output_ch_start + r;
                        automatic logic [31:0] exp   =
                            base_addr + colm_w_offset(weight_k, weight_c,
                                                      weight_kh, weight_kw,
                                                      k_idx, input_ch,
                                                      kernel_row, kernel_col);
                        if (addr_out[r] === exp) begin
                            $display("  PASS  ColM row[%0d] out_ch=%0d  addr=0x%08h",
                                     r, k_idx, addr_out[r]);
                            pass_count++;
                        end else begin
                            $display("  FAIL  ColM row[%0d] got=0x%08h exp=0x%08h",
                                     r, addr_out[r], exp);
                            fail_count++;
                        end
                    end
                end
                if (fetch_done) break;
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 6: Out-of-bounds masking – out_ch >= weight_k
        //   weight_k=4, output_ch_start=2  → rows 0,1 valid; rows 2,3 OOB
        // ------------------------------------------------------------------
        $display("\n--- Test 6: OOB masking (out_ch >= weight_k) ---");
        layout          = CHANNEL_MAJOR;
        weight_k        = 4;   weight_c  = 4;
        weight_kh       = 1;   weight_kw = 1;
        output_ch_start = 2;   // ch 2,3 valid; ch 4,5 OOB
        input_ch        = 0;   kernel_row = 0;  kernel_col = 0;

        begin
            automatic int timeout6 = 0;
            automatic int saw_valid [ARRAY_HEIGHT];
            for (int r = 0; r < ARRAY_HEIGHT; r++) saw_valid[r] = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout6 < 200) begin
                @(posedge clk); timeout6++;
                for (int r = 0; r < ARRAY_HEIGHT; r++)
                    if (addr_valid[r]) saw_valid[r] = 1;
                if (fetch_done) break;
            end

            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                automatic logic [15:0] out_ch = output_ch_start + r;
                automatic bit   should = (out_ch < weight_k);
                if (should && saw_valid[r]) begin
                    $display("  PASS  row[%0d] ch=%0d in-bounds addr_valid seen",
                             r, out_ch);
                    pass_count++;
                end else if (!should && !saw_valid[r]) begin
                    $display("  PASS  row[%0d] ch=%0d OOB correctly masked",
                             r, out_ch);
                    pass_count++;
                end else if (should && !saw_valid[r]) begin
                    $display("  FAIL  row[%0d] ch=%0d in-bounds but addr_valid NOT seen",
                             r, out_ch);
                    fail_count++;
                end else begin
                    $display("  FAIL  row[%0d] ch=%0d OOB but addr_valid WAS seen",
                             r, out_ch);
                    fail_count++;
                end
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 7: Out-of-bounds masking – input_ch >= weight_c
        //   All rows masked because the bounds check fails on input_ch.
        // ------------------------------------------------------------------
        $display("\n--- Test 7: OOB masking (input_ch >= weight_c) ---");
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;   weight_c  = 4;
        weight_kh       = 3;   weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 4;   // OOB: weight_c=4, so input_ch==4 is invalid
        kernel_row      = 0;   kernel_col = 0;

        begin
            automatic int timeout7 = 0;
            automatic int any_valid = 0;

            @(posedge clk); fetch_enable = 1;
            @(posedge clk); fetch_enable = 0;

            while (timeout7 < 200) begin
                @(posedge clk); timeout7++;
                if (addr_valid !== '0) any_valid = 1;
                if (fetch_done) break;
            end

            if (!any_valid) begin
                $display("  PASS  All rows masked when input_ch OOB");
                pass_count++;
            end else begin
                $display("  FAIL  Some addr_valid seen even with input_ch OOB (addr_valid=0x%0h)",
                         addr_valid);
                fail_count++;
            end
        end
        repeat(5) @(posedge clk);

        // ------------------------------------------------------------------
        // TEST 8: Re-trigger
        // ------------------------------------------------------------------
        $display("\n--- Test 8: Re-trigger ---");
        layout          = CHANNEL_MAJOR;
        weight_k        = 8;   weight_c  = 4;
        weight_kh       = 3;   weight_kw = 3;
        output_ch_start = 0;
        input_ch        = 0;   kernel_row = 0; kernel_col = 0;
        begin
            int ds1, ds2;
            do_fetch(200, "Re-trigger #1", ds1);
            if (ds1) pass_count++; else fail_count++;
            repeat(3) @(posedge clk);
            do_fetch(200, "Re-trigger #2", ds2);
            if (ds2) pass_count++; else fail_count++;
        end

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    // Watchdog
    initial begin
        #1_000_000;
        $display("ERROR: Global testbench timeout!");
        $finish;
    end

    // Waveform dump
    initial begin
        $dumpfile("weight_fetcher_is.vcd");
        $dumpvars(0, weight_fetcher_is_tb);
    end

endmodule : weight_fetcher_is_tb