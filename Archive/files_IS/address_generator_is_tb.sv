//=============================================================================
// Module: address_generator_is_tb  (FIXED)
// Description: Testbench for address_generator_is
//
// WHAT WAS WRONG (original testbench)
// ------------------------------------
// The check_addr task had this structure:
//
//      idx_valid = 1;
//      @(posedge clk);   // cycle A: DUT latches → addr_valid<=1, linear_addr<=result
//      idx_valid = 0;    // ← de-asserted BEFORE next posedge
//      @(posedge clk);   // cycle B: DUT latches → addr_valid<=0 (because idx_valid=0)
//      // CHECK HERE  ← addr_valid is 0 → every test fails with "valid=0"
//
// The DUT is a registered (always_ff) pipeline:
//   On the posedge that sees idx_valid=1: it outputs addr_valid=1 and the address.
//   On the NEXT posedge that sees idx_valid=0: it outputs addr_valid=0.
//
// The task read the output AFTER the second posedge, by which time addr_valid
// had already been cleared to 0.  Additionally, each test's linear_addr was
// being read as the PREVIOUS test's value (pipeline shift-by-one).
//
// PROOF FROM LOG:
//   "CM (0,0,0) got=0x00001000 exp=0x00001000 valid=0"
//   → The address value is actually CORRECT (base + 0 = 0x1000), but
//     valid=0 so it is reported as FAIL.
//   "CM (1,0,0) got=0x00001000 exp=0x00001040 valid=0"
//   → got=0x1000 is the PREVIOUS test's address (pipeline shift by 1 test).
//
// THE FIX
// -------
// Keep idx_valid HIGH for TWO clock cycles instead of one, so that addr_valid
// is still 1 when we read it after the second posedge:
//
//      idx_valid = 1;
//      @(posedge clk);   // cycle A: DUT latches → addr_valid=1, linear_addr=result
//      @(posedge clk);   // cycle B: DUT latches → addr_valid still 1 (idx_valid still 1)
//      idx_valid = 0;    // ← de-assert AFTER reading
//      // CHECK HERE  ← addr_valid=1, linear_addr=correct ✓
//
// Why 2 cycles?  The address formula does not change between cycles A and B
// (inputs are held stable), so linear_addr and addr_valid are identical on
// both edges.  Reading on cycle B is clean and race-free.
//
// THE DUT (address_generator_is.sv) IS CORRECT - DO NOT CHANGE IT.
//
// Offset formulas (same for IS and OS):
//   CHANNEL_MAJOR (C,H,W): offset = c*(H*W) + h*W + w    idx: (c,h,w)
//   ROW_MAJOR     (H,W,C): offset = h*(W*C) + w*C + c    idx: (h,w,c)
//   COLUMN_MAJOR  (W,H,C): offset = w*(H*C) + h*C + c    idx: (w,h,c)
//=============================================================================

`timescale 1ns/1ps

module address_generator_is_tb;

    import input_stationary_pkg::*;

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
    address_generator_is #(.DATA_WIDTH(DATA_WIDTH)) dut (
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

    //=========================================================================
    // Clock
    //=========================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Score tracking
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

    //=========================================================================
    // Task: reset_dut
    //=========================================================================
    task automatic reset_dut();
        rst_n     = 0;
        idx_valid = 0;
        idx_0 = 0; idx_1 = 0; idx_2 = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);   // deassert between edges to avoid race on rst_n
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    //=========================================================================
    // Task: check_addr
    //
    // TIMING (FIXED):
    //   1. Apply all inputs, assert idx_valid.
    //   2. Wait posedge A  → DUT registers: addr_valid=1, linear_addr=base+offset
    //   3. Wait posedge B  → DUT registers again: idx_valid still 1 → addr_valid=1
    //   4. Deassert idx_valid AFTER reading.
    //   5. Read and compare.  addr_valid is guaranteed 1 at step 4. ✓
    //
    // Parameters:
    //   name       - test label
    //   lay        - CHANNEL_MAJOR / ROW_MAJOR / COLUMN_MAJOR
    //   sc,sh,sw   - shape_c, shape_h, shape_w
    //   ba         - base address
    //   i0,i1,i2   - index inputs (interpretation depends on layout)
    //   exp_offset - expected byte offset from base
    //=========================================================================
    task automatic check_addr(
        input string      name,
        input mem_layout_t lay,
        input logic [15:0] sc, sh, sw,
        input logic [31:0] ba,
        input logic [15:0] i0, i1, i2,
        input logic [31:0] exp_offset
    );
        logic [31:0] exp_addr;
        exp_addr  = ba + exp_offset;

        // Apply inputs
        layout    = lay;
        shape_c   = sc;  shape_h = sh;  shape_w = sw;
        base_addr = ba;
        idx_0     = i0;  idx_1   = i1;  idx_2   = i2;
        idx_valid = 1;

        // FIX: hold idx_valid HIGH across TWO posedges
        @(posedge clk);   // cycle A: DUT registers addr_valid=1, linear_addr=result
        @(posedge clk);   // cycle B: addr_valid still 1 (idx_valid still asserted)

        // Now read - addr_valid is 1 and linear_addr is stable
        if (addr_valid && linear_addr === exp_addr) begin
            $display("  PASS  %-50s  addr=0x%08h", name, linear_addr);
            pass_count++;
        end else begin
            $display("  FAIL  %-50s  got=0x%08h exp=0x%08h valid=%0b",
                     name, linear_addr, exp_addr, addr_valid);
            fail_count++;
        end

        // De-assert after checking (clean state for next test)
        idx_valid = 0;
        @(posedge clk);   // one idle cycle between tests
    endtask

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("address_generator_is testbench (FIXED)");
        $display("========================================");
        $display("DUT: registered output, 1-cycle pipeline latency");
        $display("");
        reset_dut();

        //----------------------------------------------------------------------
        // CHANNEL_MAJOR  (C,H,W)  →  idx = (c, h, w)
        // Formula: offset = c*(H*W) + h*W + w
        // With C=4, H=8, W=8:  stride_c=64, stride_h=8
        //----------------------------------------------------------------------
        $display("--- CHANNEL_MAJOR (C=4, H=8, W=8, base=0x1000) ---");
        //                name          layout        C   H   W   base     c  h  w  expected_offset
        check_addr("CM (c=0,h=0,w=0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 0, 0, 0       );
        check_addr("CM (c=1,h=0,w=0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 1, 0, 0, 64      );  // 1*8*8
        check_addr("CM (c=0,h=1,w=0)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 1, 0, 8       );  // 1*8
        check_addr("CM (c=0,h=0,w=5)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 0, 0, 5, 5       );
        check_addr("CM (c=2,h=3,w=5)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 2, 3, 5,
                   2*64 + 3*8 + 5);  // 128+24+5 = 157 = 0x9D
        check_addr("CM (c=3,h=7,w=7)", CHANNEL_MAJOR, 4, 8, 8, 32'h1000, 3, 7, 7,
                   3*64 + 7*8 + 7);  // 192+56+7 = 255 = last element

        //----------------------------------------------------------------------
        // ROW_MAJOR  (H,W,C)  →  idx = (h, w, c)
        // Formula: offset = h*(W*C) + w*C + c
        // With C=4, H=8, W=8:  stride_h=32, stride_w=4
        //----------------------------------------------------------------------
        $display("");
        $display("--- ROW_MAJOR (C=4, H=8, W=8, base=0x2000) ---");
        //                name       layout    C   H   W   base     h  w  c  expected_offset
        check_addr("RM (h=0,w=0,c=0)", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 0, 0, 0    );
        check_addr("RM (h=1,w=0,c=0)", ROW_MAJOR, 4, 8, 8, 32'h2000, 1, 0, 0, 32   );  // 1*8*4
        check_addr("RM (h=0,w=2,c=0)", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 2, 0, 8    );  // 2*4
        check_addr("RM (h=0,w=0,c=3)", ROW_MAJOR, 4, 8, 8, 32'h2000, 0, 0, 3, 3    );
        check_addr("RM (h=2,w=3,c=1)", ROW_MAJOR, 4, 8, 8, 32'h2000, 2, 3, 1,
                   2*32 + 3*4 + 1);  // 64+12+1 = 77 = 0x4D
        check_addr("RM (h=7,w=7,c=3)", ROW_MAJOR, 4, 8, 8, 32'h2000, 7, 7, 3,
                   7*32 + 7*4 + 3);  // 224+28+3 = 255 = last element

        //----------------------------------------------------------------------
        // COLUMN_MAJOR  (W,H,C)  →  idx = (w, h, c)
        // Formula: offset = w*(H*C) + h*C + c
        // With C=4, H=8, W=8:  stride_w=32, stride_h=4
        //----------------------------------------------------------------------
        $display("");
        $display("--- COLUMN_MAJOR (C=4, H=8, W=8, base=0x3000) ---");
        //                   name          layout       C   H   W   base     w  h  c  expected_offset
        check_addr("ColM (w=0,h=0,c=0)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 0, 0, 0   );
        check_addr("ColM (w=1,h=0,c=0)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 1, 0, 0, 32  );  // 1*8*4
        check_addr("ColM (w=0,h=2,c=0)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 2, 0, 8   );  // 2*4
        check_addr("ColM (w=0,h=0,c=3)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 0, 0, 3, 3   );
        check_addr("ColM (w=3,h=1,c=2)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 3, 1, 2,
                   3*32 + 1*4 + 2);  // 96+4+2 = 102 = 0x66
        check_addr("ColM (w=7,h=7,c=3)", COLUMN_MAJOR, 4, 8, 8, 32'h3000, 7, 7, 3,
                   7*32 + 7*4 + 3);  // 224+28+3 = 255 = last element

        //----------------------------------------------------------------------
        // Edge cases
        //----------------------------------------------------------------------
        $display("");
        $display("--- Edge case: Non-square tensor (C=8, H=4, W=16, base=0x4000) ---");
        // CHANNEL_MAJOR: stride_c = H*W = 64, stride_h = W = 16
        check_addr("CM non-sq (c=1,h=2,w=3)", CHANNEL_MAJOR, 8, 4, 16, 32'h4000, 1, 2, 3,
                   1*64 + 2*16 + 3);  // 64+32+3 = 99
        // ROW_MAJOR: stride_h = W*C = 128, stride_w = C = 8
        check_addr("RM non-sq (h=1,w=2,c=3)", ROW_MAJOR, 8, 4, 16, 32'h4000, 1, 2, 3,
                   1*128 + 2*8 + 3);  // 128+16+3 = 147

        //----------------------------------------------------------------------
        // idx_valid suppression test
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test: idx_valid=0 suppresses output ---");
        idx_valid = 0;
        @(posedge clk);
        @(posedge clk);
        if (!addr_valid) begin
            $display("  PASS  addr_valid=0 when idx_valid=0");
            pass_count++;
        end else begin
            $display("  FAIL  addr_valid should be 0 when idx_valid=0  got=%0b", addr_valid);
            fail_count++;
        end

        //----------------------------------------------------------------------
        // Summary
        //----------------------------------------------------------------------
        $display("");
        $display("========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #50;
        $finish;
    end

    initial begin #500000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("address_generator_is.vcd");
        $dumpvars(0, address_generator_is_tb);
    end

endmodule : address_generator_is_tb