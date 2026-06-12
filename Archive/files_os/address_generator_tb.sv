//============================================================================= Dilshan
// Module: address_generator_tb
// Description: Testbench for address_generator (Output-Stationary pkg)
//
// Tests:
//   1. CHANNEL_MAJOR (C,H,W) address calculation
//   2. ROW_MAJOR (H,W,C) address calculation
//   3. COLUMN_MAJOR (W,H,C) address calculation
//   4. addr_valid pipeline (one-cycle latency)
//   5. idx_valid=0 suppresses output
//   6. Base address offset correctness
//=============================================================================

`timescale 1ns/1ps

module address_generator_tb;

    import systolic_array_pkg::*;

    localparam int DATA_WIDTH = 32;
    localparam int CLK_PERIOD = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic                  clk, rst_n;
    mem_layout_t           layout;
    logic [15:0]           shape_c, shape_h, shape_w;
    logic [DATA_WIDTH-1:0] base_addr;
    logic [15:0]           idx_0, idx_1, idx_2;
    logic                  idx_valid;
    logic [DATA_WIDTH-1:0] linear_addr;
    logic                  addr_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    address_generator #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .layout     (layout),
        .shape_c    (shape_c),
        .shape_h    (shape_h),
        .shape_w    (shape_w),
        .base_addr  (base_addr),
        .idx_0      (idx_0),
        .idx_1      (idx_1),
        .idx_2      (idx_2),
        .idx_valid  (idx_valid),
        .linear_addr(linear_addr),
        .addr_valid (addr_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Helpers
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

   task automatic reset_dut();
    rst_n     = 0;
    idx_valid = 0;
    idx_0     = 0; idx_1 = 0; idx_2 = 0;
    repeat(4) @(posedge clk);
    @(negedge clk);   // ← drive rst_n away from posedge
    rst_n = 1;
    repeat(2) @(posedge clk);
endtask

    task automatic drive_and_check(
    input string        test_name,
    input mem_layout_t  lay,
    input logic [15:0]  sc, sh, sw,
    input logic [31:0]  ba,
    input logic [15:0]  i0, i1, i2,
    input logic [31:0]  expected_offset
);
    logic [31:0] exp_addr;
    exp_addr  = ba + expected_offset;

    // Drive inputs safely away from posedge
    @(negedge clk);
    layout    = lay;
    shape_c   = sc; shape_h = sh; shape_w = sw;
    base_addr = ba;
    idx_0     = i0; idx_1 = i1; idx_2 = i2;
    idx_valid = 1;

    // posedge 1: DUT registers: linear_addr = base+offset, addr_valid = 1
    @(posedge clk);

    // Deassert valid at negedge (between edges, safe)
    @(negedge clk);
    idx_valid = 0;

    //Sample HERE - addr_valid is still 1, linear_addr is stable
    //    DO NOT wait for another posedge first!
    if (addr_valid && linear_addr === exp_addr) begin
        $display("  PASS  %-50s  addr=0x%08h", test_name, linear_addr);
        pass_count++;
    end else begin
        $display("  FAIL  %-50s  got=0x%08h  expected=0x%08h  valid=%0b",
                 test_name, linear_addr, exp_addr, addr_valid);
        fail_count++;
    end
endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        $display("========================================");
        $display("address_generator testbench (OS)");
        $display("========================================");
        reset_dut();

        // CHANNEL_MAJOR  C=4 H=8 W=8  base=0x1000
        // offset = c*H*W + h*W + w
        $display("\n--- CHANNEL_MAJOR tests (C=4, H=8, W=8, base=0x1000) ---");
        drive_and_check("CM (0,0,0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 0, 0, 0);
        drive_and_check("CM (0,1,0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 1, 0, 8);   // 1*8
        drive_and_check("CM (0,0,3)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 0, 3, 3);
        drive_and_check("CM (1,0,0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 1, 0, 0, 64);  // 1*8*8
        drive_and_check("CM (2,3,5)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 2, 3, 5,
                        2*64 + 3*8 + 5);  // 157

        // ROW_MAJOR  H=8 W=8 C=4  base=0x2000
        // offset = h*W*C + w*C + c
        $display("\n--- ROW_MAJOR tests (C=4, H=8, W=8, base=0x2000) ---");
        drive_and_check("RM (0,0,0)", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 0, 0, 0);
        // idx=(h,w,c)
        drive_and_check("RM h=1,w=0,c=0", ROW_MAJOR, 4, 8, 8, 32'h2000, 1, 0, 0, 32); // 1*8*4
        drive_and_check("RM h=0,w=2,c=0", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 2, 0, 8);  // 2*4
        drive_and_check("RM h=0,w=0,c=3", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 0, 3, 3);
        drive_and_check("RM h=2,w=3,c=1", ROW_MAJOR, 4, 8, 8, 32'h2000, 2, 3, 1,
                        2*32 + 3*4 + 1);  // 77

        // COLUMN_MAJOR  W=8 H=8 C=4  base=0x3000
        // offset = w*H*C + h*C + c
        $display("\n--- COLUMN_MAJOR tests (C=4, H=8, W=8, base=0x3000) ---");
        drive_and_check("ColM (0,0,0)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 0, 0, 0);
        // idx=(w,h,c)
        drive_and_check("ColM w=1", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 1, 0, 0, 32); // 1*8*4
        drive_and_check("ColM h=2", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 2, 0, 8);  // 2*4
        drive_and_check("ColM c=3", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 0, 3, 3);
        drive_and_check("ColM w=3,h=1,c=2", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 3, 1, 2,
                        3*32 + 1*4 + 2);  // 102

        // -----------------------------------------------------------------
        // TEST: idx_valid=0 → addr_valid must stay 0
        // -----------------------------------------------------------------
        $display("\n--- Test: idx_valid=0 suppresses output ---");
        idx_valid = 0;
        @(posedge clk); @(posedge clk);
        if (!addr_valid) begin
            $display("  PASS  addr_valid=0 when idx_valid=0");
            pass_count++;
        end else begin
            $display("  FAIL  addr_valid should be 0 when idx_valid=0, got=%0b", addr_valid);
            fail_count++;
        end

        // Summary
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else                  $display("SOME TESTS FAILED");

        #50; $finish;
    end

    initial begin #200000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("address_generator.vcd");
        $dumpvars(0, address_generator_tb);
    end

endmodule : address_generator_tb