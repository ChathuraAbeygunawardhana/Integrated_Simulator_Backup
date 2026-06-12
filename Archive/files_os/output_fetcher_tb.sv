//============================================================================= Dilshan
// Module: output_fetcher_tb  (system aligned)
//
// TIMING MODEL - identical discipline to weight_fetcher_tb and
// address_generator_tb:
//
//   trigger_and_wait:
//     E1: fetch_enable=1
//     E2: fetch_enable=0; FSM IDLE→GENERATE
//     E3: FSM GENERATE→DONE; addr_out registered
//     E4: FSM DONE→IDLE; fetch_done=1, addr_valid=1  ← while-loop exits HERE
//         Read outputs immediately at this edge - no extra clock needed.
//     E5: fetch_done=0, addr_valid=0 (IDLE clears it)
//
// This matches weight_fetcher_tb which also reads addr_valid immediately
// when fetch_done is observed, without extra clock cycles.
//=============================================================================

`timescale 1ns/1ps

module output_fetcher_tb;

    import systolic_array_pkg::*;

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
    logic                  fetch_enable;
    logic                  fetch_done;

    logic [DATA_WIDTH-1:0]                    addr_out   [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid;
    logic [15:0]                              out_ch     [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              out_row    [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [15:0]                              out_col    [ARRAY_HEIGHT][ARRAY_WIDTH];

    //=========================================================================
    // DUT
    //=========================================================================
    output_fetcher #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .layout(layout),
        .output_k(output_k), .output_h(output_h), .output_w(output_w),
        .base_addr(base_addr),
        .channel_start(channel_start), .row_start(row_start), .col_start(col_start),
        .fetch_enable(fetch_enable), .fetch_done(fetch_done),
        .addr_out(addr_out), .addr_valid(addr_valid),
        .out_ch(out_ch), .out_row(out_row), .out_col(out_col)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_count = 0;
    int fail_count = 0;

    //=========================================================================
    // Helpers
    //=========================================================================
    task automatic reset_dut();
        rst_n = 0; fetch_enable = 0;
        layout = CHANNEL_MAJOR;
        output_k = 16; output_h = 6; output_w = 6;
        base_addr = 32'h3000_0000;
        channel_start = 0; row_start = 0; col_start = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // Trigger fetch and wait for fetch_done.
    // Reads addr_out/addr_valid on the SAME cycle fetch_done is seen -
    // exactly like weight_fetcher_tb's wait_done pattern.
    // NO extra @posedge after the while loop.
    task automatic trigger_and_wait(input int max_cyc);
        int cnt;
        @(posedge clk); fetch_enable = 1;
        @(posedge clk); fetch_enable = 0;
        cnt = 0;
        while (!fetch_done && cnt < max_cyc) begin
            @(posedge clk);
            cnt++;
        end
        if (!fetch_done)
            $display("  WARN  fetch_done timeout after %0d cycles", max_cyc);
        // addr_valid and addr_out are stable RIGHT HERE (DONE state)
        // Do NOT advance another clock - addr_valid drops next cycle
    endtask

    // Address formula functions - identical to address_generator formulas
    // CHANNEL_MAJOR: base + ch*(H*W) + h*W + w
    function automatic logic [31:0] cm_addr(
        input logic [31:0] ba, input logic [15:0] K,H,W,k,h,w);
        return ba + 32'(k)*(32'(H)*32'(W)) + 32'(h)*32'(W) + 32'(w);
    endfunction

    // ROW_MAJOR: base + h*(W*K) + w*K + k
    function automatic logic [31:0] rm_addr(
        input logic [31:0] ba, input logic [15:0] K,H,W,k,h,w);
        return ba + 32'(h)*(32'(W)*32'(K)) + 32'(w)*32'(K) + 32'(k);
    endfunction

    // COLUMN_MAJOR: base + w*(H*K) + h*K + k
    function automatic logic [31:0] colm_addr(
        input logic [31:0] ba, input logic [15:0] K,H,W,k,h,w);
        return ba + 32'(w)*(32'(H)*32'(K)) + 32'(h)*32'(K) + 32'(k);
    endfunction

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("output_fetcher testbench (OS) - final");
        $display("========================================");
        reset_dut();

        // -----------------------------------------------------------------
        // TEST 1: CHANNEL_MAJOR
        //   K=8 H=6 W=6, ch_start=0 row=0 col=0
        //   cell[r][c]: ch=r, h=0, w=c
        //   expected: base + r*(6*6) + 0*6 + c = base + 36*r + c
        // -----------------------------------------------------------------
        $display("\n--- Test 1: CHANNEL_MAJOR address check ---");
        layout = CHANNEL_MAJOR;
        output_k = 8; output_h = 6; output_w = 6;
        base_addr = 32'h3000_0000;
        channel_start = 0; row_start = 0; col_start = 0;

        trigger_and_wait(20);

        if (fetch_done) begin
            $display("  PASS  fetch_done asserted"); pass_count++;
        end else begin
            $display("  FAIL  fetch_done NOT asserted"); fail_count++;
        end

        for (int rr = 0; rr < ARRAY_HEIGHT; rr++) begin
            for (int cc = 0; cc < ARRAY_WIDTH; cc++) begin
                automatic logic [15:0] k_i = channel_start + rr;
                automatic logic [15:0] h_i = row_start;
                automatic logic [15:0] w_i = col_start + cc;
                automatic logic [31:0] exp = cm_addr(base_addr,output_k,output_h,output_w,k_i,h_i,w_i);
                if (addr_valid[rr][cc] && addr_out[rr][cc] === exp) begin
                    $display("  PASS  CM [%0d][%0d]  addr=0x%08h", rr,cc,addr_out[rr][cc]);
                    pass_count++;
                end else begin
                    $display("  FAIL  CM [%0d][%0d]  got=0x%08h exp=0x%08h valid=%0b",
                             rr,cc,addr_out[rr][cc],exp,addr_valid[rr][cc]);
                    fail_count++;
                end
            end
        end

        // -----------------------------------------------------------------
        // TEST 2: ROW_MAJOR
        //   ch_start=0 row=2 col=1
        //   cell[r][c]: ch=r, h=2, w=1+c
        //   expected: base + h*(W*K) + w*K + k
        // -----------------------------------------------------------------
        $display("\n--- Test 2: ROW_MAJOR address check ---");
        layout = ROW_MAJOR;
        channel_start = 0; row_start = 2; col_start = 1;

        trigger_and_wait(20);

        if (fetch_done) begin
            $display("  PASS  fetch_done (ROW_MAJOR)"); pass_count++;
        end else begin
            $display("  FAIL  fetch_done NOT (ROW_MAJOR)"); fail_count++;
        end

        for (int rr = 0; rr < ARRAY_HEIGHT; rr++) begin
            for (int cc = 0; cc < ARRAY_WIDTH; cc++) begin
                automatic logic [15:0] k_i = channel_start + rr;
                automatic logic [15:0] h_i = row_start;
                automatic logic [15:0] w_i = col_start + cc;
                automatic logic [31:0] exp;
                if (k_i < output_k && w_i < output_w) begin
                    exp = rm_addr(base_addr,output_k,output_h,output_w,k_i,h_i,w_i);
                    if (addr_valid[rr][cc] && addr_out[rr][cc] === exp) begin
                        $display("  PASS  RM [%0d][%0d]  addr=0x%08h", rr,cc,addr_out[rr][cc]);
                        pass_count++;
                    end else begin
                        $display("  FAIL  RM [%0d][%0d]  got=0x%08h exp=0x%08h valid=%0b",
                                 rr,cc,addr_out[rr][cc],exp,addr_valid[rr][cc]);
                        fail_count++;
                    end
                end
            end
        end

        // -----------------------------------------------------------------
        // TEST 3: COLUMN_MAJOR
        //   ch_start=0 row=0 col=0
        //   cell[r][c]: ch=r, h=0, w=c
        //   expected: base + w*(H*K) + h*K + ch
        // -----------------------------------------------------------------
        $display("\n--- Test 3: COLUMN_MAJOR address check ---");
        layout = COLUMN_MAJOR;
        channel_start = 0; row_start = 0; col_start = 0;

        trigger_and_wait(20);

        if (fetch_done) begin
            $display("  PASS  fetch_done (COLUMN_MAJOR)"); pass_count++;
        end else begin
            $display("  FAIL  fetch_done NOT (COLUMN_MAJOR)"); fail_count++;
        end

        for (int rr = 0; rr < ARRAY_HEIGHT; rr++) begin
            for (int cc = 0; cc < ARRAY_WIDTH; cc++) begin
                automatic logic [15:0] k_i = channel_start + rr;
                automatic logic [15:0] h_i = row_start;
                automatic logic [15:0] w_i = col_start + cc;
                automatic logic [31:0] exp;
                if (k_i < output_k && w_i < output_w) begin
                    exp = colm_addr(base_addr,output_k,output_h,output_w,k_i,h_i,w_i);
                    if (addr_valid[rr][cc] && addr_out[rr][cc] === exp) begin
                        $display("  PASS  ColM [%0d][%0d]  addr=0x%08h", rr,cc,addr_out[rr][cc]);
                        pass_count++;
                    end else begin
                        $display("  FAIL  ColM [%0d][%0d]  got=0x%08h exp=0x%08h valid=%0b",
                                 rr,cc,addr_out[rr][cc],exp,addr_valid[rr][cc]);
                        fail_count++;
                    end
                end
            end
        end

        // -----------------------------------------------------------------
        // TEST 4: Out-of-bounds masking
        //   channel_start=6, output_k=8
        //   row 0 → ch=6 (valid), row 1 → ch=7 (valid)
        //   row 2 → ch=8 (OOB),   row 3 → ch=9 (OOB)
        // -----------------------------------------------------------------
        $display("\n--- Test 4: Out-of-bounds masking ---");
        layout = CHANNEL_MAJOR;
        output_k = 8;
        channel_start = 6; row_start = 0; col_start = 0;

        trigger_and_wait(20);

        for (int rr = 0; rr < ARRAY_HEIGHT; rr++) begin
            automatic logic [15:0] k_i       = channel_start + rr;
            automatic bit          should_v   = (k_i < output_k);
            if (should_v && addr_valid[rr][0]) begin
                $display("  PASS  row[%0d] ch=%0d valid as expected",   rr, k_i); pass_count++;
            end else if (!should_v && !addr_valid[rr][0]) begin
                $display("  PASS  row[%0d] ch=%0d correctly masked",    rr, k_i); pass_count++;
            end else begin
                $display("  FAIL  row[%0d] ch=%0d valid=%0b expected=%0b",
                         rr, k_i, addr_valid[rr][0], should_v);             fail_count++;
            end
        end

        // -----------------------------------------------------------------
        // TEST 5: addr_out holds after addr_valid drops
        //   After trigger_and_wait returns (fetch_done seen, addr_valid was 1),
        //   advance 2 more clocks - addr_valid must be 0, addr_out must hold.
        // -----------------------------------------------------------------
        $display("\n--- Test 5: addr_out holds after valid drops ---");
        layout = CHANNEL_MAJOR; output_k = 8;
        channel_start = 0; row_start = 0; col_start = 0;

        trigger_and_wait(20);
        // Capture expected values while addr_valid is high
        begin
            automatic logic [31:0] held [ARRAY_HEIGHT][ARRAY_WIDTH];
            for (int rr=0; rr<ARRAY_HEIGHT; rr++)
                for (int cc=0; cc<ARRAY_WIDTH; cc++)
                    held[rr][cc] = addr_out[rr][cc];

            // Advance 2 clocks - addr_valid should now be 0
            @(posedge clk); @(posedge clk);

            if (addr_valid === '0) begin
                $display("  PASS  addr_valid cleared after fetch_done"); pass_count++;
            end else begin
                $display("  FAIL  addr_valid not cleared (got 0x%04b)", addr_valid); fail_count++;
            end

            // addr_out should still hold
            for (int rr=0; rr<ARRAY_HEIGHT; rr++) begin
                for (int cc=0; cc<ARRAY_WIDTH; cc++) begin
                    if (addr_out[rr][cc] === held[rr][cc]) begin
                        $display("  PASS  addr_out[%0d][%0d] holds=0x%08h", rr,cc,addr_out[rr][cc]);
                        pass_count++;
                    end else begin
                        $display("  FAIL  addr_out[%0d][%0d] changed to 0x%08h (was 0x%08h)",
                                 rr,cc,addr_out[rr][cc],held[rr][cc]);
                        fail_count++;
                    end
                end
            end
        end

        // -----------------------------------------------------------------
        // TEST 6: Re-trigger
        // -----------------------------------------------------------------
        $display("\n--- Test 6: Re-trigger ---");
        layout = CHANNEL_MAJOR; output_k = 8; channel_start = 0;

        trigger_and_wait(20);
        if (fetch_done) begin
            $display("  PASS  first trigger");  pass_count++;
        end else begin
            $display("  FAIL  first trigger");  fail_count++;
        end

        @(posedge clk); @(posedge clk);   // let FSM settle to IDLE

        trigger_and_wait(20);
        if (fetch_done) begin
            $display("  PASS  re-trigger successful"); pass_count++;
        end else begin
            $display("  FAIL  re-trigger failed");     fail_count++;
        end

        // Summary
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #100; $finish;
    end

    initial begin #500000; $display("TIMEOUT"); $finish; end
    initial begin
        $dumpfile("output_fetcher.vcd");
        $dumpvars(0, output_fetcher_tb);
    end

endmodule : output_fetcher_tb