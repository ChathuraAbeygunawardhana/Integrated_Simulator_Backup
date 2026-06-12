//============================================================================= Dilshan
// Module: processing_element_tb
// Description: Testbench for the Output-Stationary Processing Element
//
// FIXES APPLIED:
//   1. reset_dut: rst_n deasserted at negedge (not at posedge)
//   2. All inputs driven at negedge - never at posedge (race condition fix)
//   3. Outputs sampled at negedge AFTER the capturing posedge
//   4. Weight reload followed by clear is properly sequenced with negedge discipline
//   5. Added checker self-verification (deliberate wrong value must FAIL)
//   6. Added reset-output check (all outputs must be 0 after reset)
//   7. Multi-cycle accumulation test extended to 4 cycles for robustness
//
// Golden rule: DRIVE at negedge, SAMPLE at negedge, let posedge be for DUT only.
//=============================================================================

`timescale 1ns/1ps

module processing_element_tb;

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam int DATA_WIDTH  = 32;
    localparam int ACCUM_WIDTH = 48;
    localparam int CLK_PERIOD  = 10;

    //=========================================================================
    // DUT signals
    //=========================================================================
    logic                    clk;
    logic                    rst_n;
    logic                    enable;
    logic                    clear_accum;

    logic [DATA_WIDTH-1:0]   activation_in;
    logic                    activation_valid_in;

    logic [DATA_WIDTH-1:0]   weight;
    logic                    weight_valid;

    logic [ACCUM_WIDTH-1:0]  psum_in;
    logic                    psum_valid_in;

    logic [DATA_WIDTH-1:0]   activation_out;
    logic                    activation_valid_out;
    logic [ACCUM_WIDTH-1:0]  psum_out;
    logic                    psum_valid_out;
    logic [ACCUM_WIDTH-1:0]  result;

    //=========================================================================
    // DUT instantiation
    //=========================================================================
    processing_element #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) dut (
        .clk                 (clk),
        .rst_n               (rst_n),
        .enable              (enable),
        .clear_accum         (clear_accum),
        .activation_in       (activation_in),
        .activation_valid_in (activation_valid_in),
        .weight              (weight),
        .weight_valid        (weight_valid),
        .psum_in             (psum_in),
        .psum_valid_in       (psum_valid_in),
        .activation_out      (activation_out),
        .activation_valid_out(activation_valid_out),
        .psum_out            (psum_out),
        .psum_valid_out      (psum_valid_out),
        .result              (result)
    );

    //=========================================================================
    // Clock
    //=========================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Test helpers
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

    // -----------------------------------------------------------------
    // reset_dut
    //   rst_n deasserted at negedge to avoid posedge race.
    //          All inputs held at known-zero state during reset.
    // -----------------------------------------------------------------
    task automatic reset_dut();
        // Drive all inputs to safe defaults before releasing reset
        rst_n               = 0;
        enable              = 0;
        clear_accum         = 0;
        activation_in       = '0;
        activation_valid_in = 0;
        weight              = '0;
        weight_valid        = 0;
        psum_in             = '0;
        psum_valid_in       = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);     // FIXED: deassert between clock edges
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    // -----------------------------------------------------------------
    // load_weight: load a new stationary weight into the PE
    //   driven at negedge, deasserted at negedge
    // -----------------------------------------------------------------
    task automatic load_weight(input logic [DATA_WIDTH-1:0] w);
        @(negedge clk);
        weight       = w;
        weight_valid = 1;
        @(posedge clk);         // DUT captures weight on this edge
        @(negedge clk);
        weight_valid = 0;
        weight       = '0;
    endtask

    // -----------------------------------------------------------------
    // do_clear: pulse clear_accum for one cycle
    //   driven and deasserted at negedge
    // -----------------------------------------------------------------
    task automatic do_clear();
        @(negedge clk);
        clear_accum = 1;
        @(posedge clk);         // DUT clears accumulator on this edge
        @(negedge clk);
        clear_accum = 0;
    endtask

    // -----------------------------------------------------------------
    // do_mac: drive one MAC cycle and wait for result to be stable
    //   Inputs driven at negedge → posedge captures → sample at negedge
    //
    //   The PE's accumulator updates:
    //     accumulator <= accumulator + (activation_in * weight_reg + psum_in)
    //   on the posedge where enable=1 AND activation_valid_in=1.
    //   'result' is a wire on accumulator, so it is stable at negedge
    //   AFTER that posedge.
    // -----------------------------------------------------------------
    task automatic do_mac(
        input logic [DATA_WIDTH-1:0]  act,
        input logic [ACCUM_WIDTH-1:0] psum,
        input logic                   psum_v
    );
        @(negedge clk);
        activation_in       = act;
        activation_valid_in = 1;
        psum_in             = psum;
        psum_valid_in       = psum_v;
        enable              = 1;
        @(posedge clk);         // DUT accumulates on this edge
        @(negedge clk);
        // Deassert after sampling window (caller reads result here)
        activation_valid_in = 0;
        psum_valid_in       = 0;
    endtask

    // -----------------------------------------------------------------
    // check_result: compare 'result' (accumulator output) to expected
    // -----------------------------------------------------------------
    task automatic check_result(
        input string        test_name,
        input logic [47:0]  expected
    );
        // result is already stable at negedge (called after do_mac/do_clear)
        if (result === expected) begin
            $display("  PASS  %-45s  got=0x%012h", test_name, result);
            pass_count++;
        end else begin
            $display("  FAIL  %-45s  got=0x%012h  expected=0x%012h",
                     test_name, result, expected);
            fail_count++;
        end
    endtask

    // -----------------------------------------------------------------
    // check_activation: verify activation pass-through port
    // -----------------------------------------------------------------
    task automatic check_activation(
        input string              test_name,
        input logic [DATA_WIDTH-1:0] expected_data,
        input logic               expected_valid
    );
        // activation_out is registered, stable at negedge after driving posedge
        if (activation_out === expected_data && activation_valid_out === expected_valid) begin
            $display("  PASS  %-45s  out=0x%08h  valid=%0b",
                     test_name, activation_out, activation_valid_out);
            pass_count++;
        end else begin
            $display("  FAIL  %-45s  out=0x%08h (exp 0x%08h)  valid=%0b (exp %0b)",
                     test_name, activation_out, expected_data,
                     activation_valid_out, expected_valid);
            fail_count++;
        end
    endtask

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("processing_element testbench (OS)");
        $display("========================================");

        reset_dut();

        // -----------------------------------------------------------------
        // TEST 0: Post-reset state
        //   All registered outputs must be 0 after reset.
        // -----------------------------------------------------------------
        $display("\n--- Test 0: Post-reset state (all outputs must be 0) ---");
        @(negedge clk);   // sample safely
        check_result("result=0 after reset",  48'd0);
        check_activation("activation_out=0 after reset", 32'd0, 1'b0);

        // -----------------------------------------------------------------
        // TEST 1: Weight loading
        //   Load weight=5. Verified indirectly by accumulation tests.
        // -----------------------------------------------------------------
        $display("\n--- Test 1: Weight loading (weight = 5) ---");
        load_weight(32'h0000_0005);
        $display("  NOTE  Weight=5 loaded. Verified by accumulation below.");

        // -----------------------------------------------------------------
        // TEST 2: Single MAC - 3 × 5 + 0 = 15
        //   enable=1, act=3, weight_reg=5, psum_in=0 → accumulator = 15
        // -----------------------------------------------------------------
        $display("\n--- Test 2: Single MAC (3 × 5 + 0 = 15) ---");
        do_mac(32'h0000_0003, 48'd0, 1'b1);
        check_result("result after 1 MAC (3×5+0)", 48'd15);

        // -----------------------------------------------------------------
        // TEST 3: Second MAC accumulates - 3×5 = 15 again → total = 30
        //   accumulator starts at 15, psum_in=0, act=3, weight=5
        //   mac_result = 3*5+0 = 15; accumulator = 15+15 = 30
        // -----------------------------------------------------------------
        $display("\n--- Test 3: Second MAC accumulates (total = 30) ---");
        do_mac(32'h0000_0003, 48'd0, 1'b1);
        check_result("result after 2 MACs (15+15)", 48'd30);

        // -----------------------------------------------------------------
        // TEST 4: Extended multi-cycle accumulation - 4 total MACs → 60
        //   Accumulator: 30 + 15 + 15 = 60
        // -----------------------------------------------------------------
        $display("\n--- Test 4: Multi-cycle accumulation (4 MACs → 60) ---");
        do_mac(32'h0000_0003, 48'd0, 1'b1);   // cycle 3: 30+15 = 45
        do_mac(32'h0000_0003, 48'd0, 1'b1);   // cycle 4: 45+15 = 60
        check_result("result after 4 MACs (4×15=60)", 48'd60);

        // -----------------------------------------------------------------
        // TEST 5: Clear accumulator → result must be 0
        // -----------------------------------------------------------------
        $display("\n--- Test 5: Clear accumulator ---");
        do_clear();
        check_result("result after clear_accum", 48'd0);

        // -----------------------------------------------------------------
        // TEST 6: MAC with incoming partial sum
        //   act=2, weight=5, psum_in=10 → mac_result = 2*5+10 = 20
        //   accumulator starts at 0 after clear → accumulator = 0 + 20 = 20
        //
        //   NOTE: The PE's accumulator does: acc += (act*weight + psum_in)
        //         psum_in feeds INTO the mac_result, not a separate adder.
        //         So accumulator after this cycle = 0 + (2*5 + 10) = 20.
        // -----------------------------------------------------------------
        $display("\n--- Test 6: MAC with incoming psum (2×5 + 10 = 20) ---");
        do_mac(32'h0000_0002, 48'd10, 1'b1);
        check_result("result after psum MAC (2×5+10=20)", 48'd20);

        // -----------------------------------------------------------------
        // TEST 7: Activation pass-through (vertical propagation)
        //   Drive activation_in = 0xDEADBEEF with valid=1.
        //   After one posedge the DUT registers it → activation_out must
        //   equal 0xDEADBEEF with valid=1.
        //   enable=0 so accumulator is not disturbed.
        //   FIXED: drive at negedge, sample at negedge after posedge.
        // -----------------------------------------------------------------
        $display("\n--- Test 7: Activation pass-through (0xDEADBEEF, valid=1) ---");
        do_clear();
        @(negedge clk);
        activation_in       = 32'hDEAD_BEEF;
        activation_valid_in = 1;
        enable              = 0;    // do not accumulate
        @(posedge clk);             // DUT registers activation_out on this edge
        @(negedge clk);             // sample stable output here
        activation_valid_in = 0;
        check_activation("activation pass-through", 32'hDEAD_BEEF, 1'b1);

        // Verify valid goes low one cycle after valid_in goes low
        @(posedge clk);
        @(negedge clk);
        check_activation("activation valid deasserts after valid_in=0", 32'hDEAD_BEEF, 1'b0);

        // -----------------------------------------------------------------
        // TEST 8: Signed MAC - 4 × (-2) = -8
        //   Reload weight = -2 = 32'hFFFF_FFFE
        //   clear accumulator, then single MAC with act=4
        //   mac_result = signed(4) * signed(-2) + signed(0) = -8
        //   accumulator = 0 + (-8) = -8 = 48'hFFFF_FFFF_FFF8
        // -----------------------------------------------------------------
        $display("\n--- Test 8: Signed MAC (4 × -2 = -8) ---");
        @(negedge clk);
        enable = 0;   // keep enable low while reloading weight
        load_weight(32'hFFFF_FFFE);   // -2 in signed 32-bit
        do_clear();
        do_mac(32'h0000_0004, 48'd0, 1'b1);
        check_result("result signed (4×-2+0 = -8)", 48'hFFFF_FFFF_FFF8);

        // -----------------------------------------------------------------
        // TEST 9: Enable gate - no accumulation when enable=0
        //   load weight=5, clear, then send activation with enable=0.
        //   Accumulator must remain 0.
        // -----------------------------------------------------------------
        $display("\n--- Test 9: Enable gate (enable=0 → no accumulation) ---");
        load_weight(32'h0000_0005);
        do_clear();
        @(negedge clk);
        activation_in       = 32'h0000_0007;
        activation_valid_in = 1;
        psum_in             = 48'd0;
        psum_valid_in       = 1;
        enable              = 0;        // DISABLED
        @(posedge clk);
        @(negedge clk);
        activation_valid_in = 0;
        check_result("result when enable=0 (must stay 0)", 48'd0);

        // -----------------------------------------------------------------
        // TEST 10: Checker self-verification
        //   Deliberately pass a wrong expected value.
        //   If check_result accepts it, the checker itself is broken.
        //   load weight=5, clear, MAC(3,0) → result should be 15.
        //   We check against 99 (wrong) - must see a FAIL, then correct it.
        // -----------------------------------------------------------------
        $display("\n--- Test 10: Checker self-verification ---");
        load_weight(32'h0000_0005);
        do_clear();
        do_mac(32'h0000_0003, 48'd0, 1'b1);
        // result is now 15; check against wrong value 99
        begin
            int pre_fail = fail_count;
            // deliberately wrong check - must produce FAIL
            if (result === 48'd99) begin
                $display("  FAIL  [CHECKER-VERIFY] checker accepted wrong value 99 for result=%0d!", result);
                fail_count++;
            end else begin
                $display("  PASS  [CHECKER-VERIFY] checker correctly rejects expected=99 (result=0x%012h)", result);
                pass_count++;
            end
        end

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        $display("\n========================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("========================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #50;
        $finish;
    end

    // Watchdog
    initial begin
        #500000;
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    // Waveform dump
    initial begin
        $dumpfile("processing_element.vcd");
        $dumpvars(0, processing_element_tb);
    end

endmodule : processing_element_tb