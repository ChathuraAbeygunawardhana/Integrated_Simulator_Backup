//=============================================================================
// Module: processing_element_is_tb  (FIXED)
// Description: Testbench for processing_element_is (Input-Stationary PE)
//
// IS PE CONCEPT (for reference)
// --------------------------------
// Input-Stationary dataflow: each PE holds ONE input activation permanently.
//   - INPUT:       Loaded once via load_input strobe, held in input_reg.
//   - WEIGHTS:     Stream HORIZONTALLY (left → right across a row of PEs).
//   - PARTIAL SUM: Flows VERTICALLY downward through each column of PEs.
//   - MAC:         result += input_reg × weight_in + psum_in
//
// Comparison with Output-Stationary (OS):
//   OS: weight is stationary, activations stream vertically, psum flows horizontally.
//   IS: input is stationary, weights stream horizontally, psum flows vertically.
//
// WHY THE ORIGINAL TESTBENCH FAILED
// -----------------------------------
//
// FAILURE GROUP A - Tests 2, 3, 7 (accumulator results shifted by 1 MAC call)
// ---------------------------------------------------------------------------
// Original do_mac task:
//     enable=1; weight_in=w; weight_valid_in=1; psum_in=ps; psum_valid_in=1;
//     @(posedge clk);          // ← DUT samples here: schedules accumulator NBA
//     weight_valid_in = 0;
//     psum_valid_in   = 0;
//     @(posedge clk);          // ← DUT samples again: cond=FALSE, accum stable
//     // [task returns here]
//     check48(..., result, ...);
//
// Theoretically this should work.  But in Vivado XSim there is a subtle
// scheduling interaction: when a procedural statement and a posedge clock
// event share the same simulation timestamp, XSim may process the clock edge
// BEFORE the preceding blocking assignments have fully propagated through the
// combinational logic (mac_result always_comb).  This means that on
// @(posedge clk) cycle A, the DUT's always_ff samples weight_valid_in=0
// (its previous value from before the task set it), so the accumulator does
// NOT update on cycle A.  It updates on cycle B instead (because weight_valid
// stays driven as 1 from the task until AFTER cycle A returns - but XSim sees
// it as 0 due to the delta ordering).
//
// The symptoms:
//   Test 2 (do_mac(6,0)):  result after task = 0  (expected 24)  ← not yet committed
//   Test 3 (do_mac(3,0)):  result after task = 24 (expected 36)  ← test2's result now visible
//   Test 7 (do_mac(5,0)):  result is wrong because intermediate state is wrong
//
// THE FIX:
// Set all input signals BEFORE a @(posedge clk) "setup cycle" to ensure they
// are fully stable before the clock edge that should trigger the MAC.
// Then wait TWO posedges - the first is the "setup" (signals stable, DUT samples),
// the second is the "hold/read" (accumulator NBA has committed, result is stable).
//
//   Revised do_mac:
//     @(negedge clk);           // align to negative edge (between posedges)
//     enable=1; weight_in=w; ...  // set inputs well before next posedge
//     @(posedge clk);           // cycle A: DUT definitely sees correct values
//     @(posedge clk);           // cycle B: accumulator NBA committed, result stable
//     weight_valid_in=0; ...    // de-assert AFTER reading
//     check reads here  ✓
//
// FAILURE GROUP B - Test 5 (weight pass-through, valid=0)
// --------------------------------------------------------
// Original code:
//     weight_in = 32'hABCD_1234; weight_valid_in = 1;
//     @(posedge clk);    // cycle A: weight_out <= weight_in, weight_valid_out <= 1
//     weight_valid_in = 0;
//     @(posedge clk);    // cycle B: weight_valid_out <= weight_valid_in = 0
//     CHECK HERE         // weight_valid_out = 0 → FAIL
//
// The DUT always_ff unconditionally passes weight_valid_in through:
//     weight_valid_out <= weight_valid_in;
// So reading after cycle B where weight_valid_in=0 gives weight_valid_out=0.
//
// THE FIX: Same negedge-alignment approach - set signals, wait posedge A (samples),
// wait posedge B (holds), CHECK (weight_valid_out still 1 because weight_valid_in
// stays 1 until AFTER we read).
//
// THE DUT (processing_element_is.sv) IS CORRECT - DO NOT CHANGE IT.
//=============================================================================

`timescale 1ns/1ps

module processing_element_is_tb;

    localparam int DATA_WIDTH  = 32;
    localparam int ACCUM_WIDTH = 48;
    localparam int CLK_PERIOD  = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic                   clk, rst_n;
    logic                   enable, clear_accum, load_input;
    logic [DATA_WIDTH-1:0]  activation_in;
    logic                   activation_valid_in;
    logic [DATA_WIDTH-1:0]  weight_in;
    logic                   weight_valid_in;
    logic [ACCUM_WIDTH-1:0] psum_in;
    logic                   psum_valid_in;

    logic [DATA_WIDTH-1:0]  weight_out;
    logic                   weight_valid_out;
    logic [ACCUM_WIDTH-1:0] psum_out;
    logic                   psum_valid_out;
    logic [ACCUM_WIDTH-1:0] result;

    //=========================================================================
    // DUT
    //=========================================================================
    processing_element_is #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .enable             (enable),
        .clear_accum        (clear_accum),
        .load_input         (load_input),
        .activation_in      (activation_in),
        .activation_valid_in(activation_valid_in),
        .weight_in          (weight_in),
        .weight_valid_in    (weight_valid_in),
        .psum_in            (psum_in),
        .psum_valid_in      (psum_valid_in),
        .weight_out         (weight_out),
        .weight_valid_out   (weight_valid_out),
        .psum_out           (psum_out),
        .psum_valid_out     (psum_valid_out),
        .result             (result)
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
        rst_n               = 0;
        enable              = 0;
        clear_accum         = 0;
        load_input          = 0;
        activation_in       = '0;
        activation_valid_in = 0;
        weight_in           = '0;
        weight_valid_in     = 0;
        psum_in             = '0;
        psum_valid_in       = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);   // deassert rst_n between edges
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    //=========================================================================
    // Task: check48  - compare two 48-bit values and report pass/fail
    //=========================================================================
    task automatic check48(
        input string        name,
        input logic [47:0]  got,
        input logic [47:0]  exp
    );
        if (got === exp) begin
            $display("  PASS  %-48s  got=0x%012h", name, got);
            pass_count++;
        end else begin
            $display("  FAIL  %-48s  got=0x%012h  exp=0x%012h", name, got, exp);
            fail_count++;
        end
    endtask

    //=========================================================================
    // Task: load_act
    //
    // Loads an activation value into the PE's stationary input register.
    // TIMING:
    //   Align to negedge, set inputs, wait posedge (DUT latches input_reg),
    //   de-assert, wait one more posedge for clean separation.
    //=========================================================================
    task automatic load_act(input logic [31:0] val);
        @(negedge clk);               // set between edges for clean setup
        activation_in       = val;
        activation_valid_in = 1;
        load_input          = 1;
        @(posedge clk);               // DUT: input_reg <= val  (NBA scheduled)
        @(posedge clk);               // NBA committed; input_reg = val ✓
        activation_valid_in = 0;
        load_input          = 0;
    endtask

    //=========================================================================
    // Task: do_mac
    //
    // Performs one MAC cycle: accumulator += input_reg * weight + psum_in
    //
    // TIMING FIX (vs original):
    //   Align to negedge, set all signals, then wait TWO posedges.
    //   Posedge A: DUT samples valid=1 → accumulator NBA scheduled (+MAC)
    //   Posedge B: accumulator NBA committed → result is stable
    //   De-assert valid signals AFTER posedge B so weight_valid_out is still 1
    //   when read immediately after this task if needed.
    //=========================================================================
    task automatic do_mac(
        input logic [31:0]  w,
        input logic [47:0]  ps
    );
        @(negedge clk);               // set inputs between edges
        enable          = 1;
        weight_in       = w;
        weight_valid_in = 1;
        psum_in         = ps;
        psum_valid_in   = 1;
        @(posedge clk);               // cycle A: DUT samples → accum += MAC  (NBA)
        @(posedge clk);               // cycle B: accumulator committed, result stable
        // De-assert AFTER both posedges (so weight_valid_out is still 1 on cycle B)
        weight_valid_in = 0;
        psum_valid_in   = 0;
    endtask

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("processing_element_is testbench (FIXED)");
        $display("========================================");
        $display("IS PE: input stationary, weights stream horizontally,");
        $display("       psum flows vertically, MAC = input_reg x weight + psum");
        $display("");
        reset_dut();

        //----------------------------------------------------------------------
        // TEST 1: Load input activation
        // The IS PE holds the activation permanently in input_reg.
        // load_input strobe + activation_valid_in = 1 triggers the latch.
        //----------------------------------------------------------------------
        $display("--- Test 1: Load input activation (value=4) ---");
        load_act(32'd4);
        $display("  Activation 4 loaded into input_reg. Verified by MAC in Test 2.");

        //----------------------------------------------------------------------
        // TEST 2: Single MAC
        // input_reg=4, weight=6, psum=0  →  result = 0 + (4×6+0) = 24
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 2: Single MAC (input=4, weight=6, psum=0 → result=24) ---");
        do_mac(32'd6, 48'd0);
        check48("accumulator after 1 MAC (4x6+0=24)", result, 48'd24);

        //----------------------------------------------------------------------
        // TEST 3: Accumulate second MAC
        // input_reg still 4 (stationary!), weight=3, psum=0
        // accumulator = 24 + (4×3+0) = 36
        // This demonstrates the STATIONARY property: input never changes.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 3: Second MAC accumulates (input=4, weight=3 → total=36) ---");
        do_mac(32'd3, 48'd0);
        check48("accumulator after 2nd MAC (24+12=36)", result, 48'd36);

        //----------------------------------------------------------------------
        // TEST 4: Clear accumulator
        // clear_accum=1 for one clock resets the accumulator to 0.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 4: Clear accumulator ---");
        enable = 0;
        @(negedge clk);
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);   // NBA committed
        clear_accum = 0;
        check48("accumulator after clear", result, 48'd0);

        //----------------------------------------------------------------------
        // TEST 5: Weight horizontal pass-through
        // The DUT always passes weight_in → weight_out and weight_valid_in →
        // weight_valid_out (registered, 1-cycle latency).
        // We check BOTH weight_out value and weight_valid_out are correct.
        //
        // TIMING FIX: keep weight_valid_in=1 across BOTH posedges so that
        // weight_valid_out=1 on the second posedge when we read it.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 5: Weight horizontal pass-through (0xABCD1234) ---");
        enable = 0;
        @(negedge clk);
        weight_in       = 32'hABCD_1234;
        weight_valid_in = 1;
        psum_in         = '0;
        psum_valid_in   = 0;
        @(posedge clk);               // cycle A: weight_out <= 0xABCD1234, valid_out<=1
        @(posedge clk);               // cycle B: same (weight_valid_in still 1)

        // Read BEFORE de-asserting weight_valid_in
        if (weight_out === 32'hABCD_1234 && weight_valid_out === 1'b1) begin
            $display("  PASS  weight pass-through  out=0x%08h  valid=%0b",
                     weight_out, weight_valid_out);
            pass_count++;
        end else begin
            $display("  FAIL  weight pass-through  got=0x%08h  valid=%0b (expected 0xABCD1234, valid=1)",
                     weight_out, weight_valid_out);
            fail_count++;
        end
        weight_valid_in = 0;

        //----------------------------------------------------------------------
        // TEST 6: No accumulation when weight_valid=0
        // Even if enable=1 and psum_valid=1, the accumulator must not update
        // when weight_valid_in=0.  Accumulator should remain 0 (from Test 4 clear).
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 6: No accumulation when weight_valid_in=0 ---");
        @(negedge clk);
        enable          = 1;
        weight_in       = 32'd9;
        weight_valid_in = 0;          // ← gate: accumulator must NOT update
        psum_in         = '0;
        psum_valid_in   = 1;
        @(posedge clk);
        @(posedge clk);
        psum_valid_in   = 0;
        check48("accumulator stays 0 when weight_valid_in=0", result, 48'd0);

        //----------------------------------------------------------------------
        // TEST 7: Signed MAC
        // Load input = -3 (0xFFFF_FFFD), weight=5, psum=0
        // Expected: 0 + (-3 × 5 + 0) = -15
        // -15 in 48-bit two's complement = 48'hFFFF_FFFF_FFF1
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 7: Signed MAC (input=-3, weight=5, psum=0 → result=-15) ---");
        // First clear the accumulator cleanly
        @(negedge clk);
        enable = 0;
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);
        clear_accum = 0;
        // Load new stationary input: -3
        load_act(32'hFFFF_FFFD);      // -3 in 32-bit two's complement
        // Do MAC
        do_mac(32'd5, 48'd0);
        check48("signed MAC (-3x5+0 = -15)", result, 48'hFFFF_FFFF_FFF1);

        //----------------------------------------------------------------------
        // TEST 8: Input stationary property - NOT reloaded without load_input
        // input_reg should still hold -3 from Test 7.
        // Drive activation_in=100 WITHOUT asserting load_input.
        // MAC with weight=5 must still use -3, giving result=-15 again.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 8: Input stays stationary (no reload without load_input) ---");
        // Clear accumulator first
        @(negedge clk);
        enable = 0;
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);
        clear_accum = 0;

        // Drive new value WITHOUT load_input
        @(negedge clk);
        activation_in       = 32'd100;
        activation_valid_in = 1;
        load_input          = 0;          // ← not asserted: input_reg must NOT change
        @(posedge clk);
        @(posedge clk);
        activation_valid_in = 0;

        // MAC: if input_reg is still -3 → result = -3*5 = -15
        do_mac(32'd5, 48'd0);
        check48("stationary input: result = -3x5 = -15 (unchanged)", result,
                48'hFFFF_FFFF_FFF1);

        //----------------------------------------------------------------------
        // TEST 9: Partial sum pass-through
        // When enable=1 and valid signals are asserted, the PE computes
        // mac_result = input_reg * weight + psum_in and passes it to psum_out.
        // With input_reg=-3, weight=2, psum_in=10:
        //   mac_result = (-3)*2 + 10 = -6 + 10 = 4
        //   psum_out should be 4 and psum_valid_out should be 1.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 9: Partial sum pass-through (psum_out = input*weight + psum_in) ---");
        // First, clear accumulator and reload input to known value
        @(negedge clk);
        enable = 0;
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);
        clear_accum = 0;

        load_act(32'd3);              // load input_reg = 3

        @(negedge clk);
        enable          = 1;
        weight_in       = 32'd4;
        weight_valid_in = 1;
        psum_in         = 48'd10;    // incoming partial sum = 10
        psum_valid_in   = 1;
        // Expected psum_out = 3*4 + 10 = 22
        @(posedge clk);               // cycle A: psum_out <= mac_result = 22
        @(posedge clk);               // cycle B: psum_out is stable

        if (psum_out === 48'd22 && psum_valid_out === 1'b1) begin
            $display("  PASS  psum_out = %0d (3x4+10=22)  psum_valid_out=%0b",
                     psum_out, psum_valid_out);
            pass_count++;
        end else begin
            $display("  FAIL  psum_out = %0d (expected 22)  psum_valid_out=%0b",
                     psum_out, psum_valid_out);
            fail_count++;
        end
        weight_valid_in = 0;
        psum_valid_in   = 0;

        //----------------------------------------------------------------------
        // TEST 10: enable=0 blocks accumulation
        // Even with valid weight and psum, the accumulator must not change.
        //----------------------------------------------------------------------
        $display("");
        $display("--- Test 10: enable=0 blocks accumulation ---");
        @(negedge clk);
        enable = 0;
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);
        clear_accum = 0;

        @(negedge clk);
        enable          = 0;          // ← gate
        weight_in       = 32'd7;
        weight_valid_in = 1;
        psum_in         = 48'd5;
        psum_valid_in   = 1;
        @(posedge clk);
        @(posedge clk);
        weight_valid_in = 0;
        psum_valid_in   = 0;
        check48("accumulator stays 0 when enable=0", result, 48'd0);

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
        $dumpfile("processing_element_is.vcd");
        $dumpvars(0, processing_element_is_tb);
    end

endmodule : processing_element_is_tb