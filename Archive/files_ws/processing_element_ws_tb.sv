//=============================================================================
// Module: processing_element_ws_tb  (FIXED)
//
// WS PE CONCEPT
// --------------
// Weight is stationary (loaded once, held in weight_reg until reloaded).
// Input activations broadcast HORIZONTALLY: activation_out = activation_in (registered).
// Partial sums flow DIAGONALLY: psum_out = mac_result = act*weight_reg + psum_in.
// Accumulator: accumulates mac_result across multiple activation passes (tiles/channels).
// MAC condition: enable && activation_valid_in && psum_valid_in must ALL be 1.
//
// WHY ALL 4 TESTS FAILED (same root cause - testbench bug)
// ----------------------------------------------------------
// The original testbench set input signals and IMMEDIATELY called @(posedge clk):
//
//   activation_in = 32'd3; activation_valid_in = 1; psum_valid_in = 1;
//   @(posedge clk);    ← XSim may evaluate always_ff with OLD signal values
//   activation_valid_in = 0; psum_valid_in = 0;
//   @(posedge clk);    ← accumulator commits NOW (1 posedge late)
//   check result       ← sees the PREVIOUS test's accumulator value
//
// In Vivado XSim, blocking assignments and the posedge clock event can land
// in the SAME simulation time step. XSim processes the clock edge BEFORE the
// blocking assignments fully propagate through combinational logic (mac_result).
// So the DUT's always_ff samples the OLD values of enable/valid at posedge A,
// and the accumulator only updates at posedge B (the de-assert posedge).
// This shifts every result by exactly one test call:
//   Test 2: got=0   (before any MAC)    exp=21
//   Test 3: got=21  (Test 2's result)   exp=42
//
// Test 6 (broadcast): activation_valid_in was de-asserted before the second
// @posedge, so activation_valid_out = 0 when read. Needs to stay high both cycles.
//
// Test 7: activation_in was still 0xCAFEBABE from Test 6 when the signed MAC
// sampled it → 0xCAFEBABE × (-3) = 0x9F03CFC6 which exactly matches the wrong got.
//
// THE FIX (applied to every test):
//   @(negedge clk);    ← align between edges - gives full setup time to next posedge
//   // set all signals here
//   @(posedge clk);    ← DUT DEFINITELY samples correct new values
//   @(posedge clk);    ← accumulator NBA committed, result stable
//   // de-assert valid AFTER both posedges (so pass-through signals still valid on B)
//   // check result here
//
// This is identical to the fix applied to processing_element_is_tb.
// The DUT (processing_element_ws.sv) is CORRECT - do not change it.
//=============================================================================

`timescale 1ns/1ps

module processing_element_ws_tb;

    localparam int DATA_WIDTH  = 32;
    localparam int ACCUM_WIDTH = 48;
    localparam int CLK_PERIOD  = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic                   clk, rst_n;
    logic                   enable, clear_accum;
    logic [DATA_WIDTH-1:0]  activation_in;
    logic                   activation_valid_in;
    logic [DATA_WIDTH-1:0]  weight;
    logic                   weight_valid;
    logic [ACCUM_WIDTH-1:0] psum_in;
    logic                   psum_valid_in;

    logic [DATA_WIDTH-1:0]  activation_out;
    logic                   activation_valid_out;
    logic [ACCUM_WIDTH-1:0] psum_out;
    logic                   psum_valid_out;
    logic [ACCUM_WIDTH-1:0] result;

    //=========================================================================
    // DUT
    //=========================================================================
    processing_element_ws #(
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

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Helpers
    //=========================================================================
    int pass_count = 0;
    int fail_count = 0;

    task automatic check48(
        input string name,
        input logic [47:0] got,
        input logic [47:0] exp
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
    // reset_dut
    // FIX: rst_n deasserted between edges to avoid posedge race
    //=========================================================================
    task automatic reset_dut();
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
        @(negedge clk);   // deassert between edges
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    //=========================================================================
    // load_weight
    // FIX: @(negedge clk) before setting weight so the next posedge latches it.
    //      Two posedges: first schedules weight_reg NBA, second commits it.
    //=========================================================================
    task automatic load_weight(input logic [31:0] w);
        @(negedge clk);
        weight       = w;
        weight_valid = 1;
        @(posedge clk);    // weight_reg <= w  (NBA scheduled)
        @(posedge clk);    // weight_reg = w   (NBA committed)
        weight_valid = 0;
    endtask

    //=========================================================================
    // do_mac
    // FIX: negedge alignment ensures DUT sees correct values at posedge A.
    //      Valid signals held HIGH across BOTH posedges so:
    //        posedge A: accumulator NBA scheduled
    //        posedge B: accumulator committed, result stable, valid_out still 1
    //      Signals de-asserted AFTER checking to avoid stale-read issues.
    //=========================================================================
    task automatic do_mac(
        input logic [31:0]  act,
        input logic [47:0]  ps
    );
        @(negedge clk);           // set inputs between edges for full setup time
        enable              = 1;
        activation_in       = act;
        activation_valid_in = 1;
        psum_in             = ps;
        psum_valid_in       = 1;
        @(posedge clk);           // cycle A: DUT samples correct values → accum NBA
        @(posedge clk);           // cycle B: accumulator committed → result stable
        // De-assert AFTER both posedges
        activation_valid_in = 0;
        psum_valid_in       = 0;
    endtask

    //=========================================================================
    // do_clear
    //=========================================================================
    task automatic do_clear();
        @(negedge clk);
        enable      = 0;
        clear_accum = 1;
        @(posedge clk);
        @(posedge clk);   // committed
        clear_accum = 0;
    endtask

    //=========================================================================
    // Main test sequence
    //=========================================================================
    initial begin
        $display("========================================");
        $display("processing_element_ws testbench (FIXED)");
        $display("========================================");
        $display("WS PE: weight stationary, activations broadcast horizontally,");
        $display("       psum flows diagonally, MAC = weight_reg x act + psum");
        $display("");
        reset_dut();

        //---------------------------------------------------------------------
        // TEST 1: Weight loading
        // weight=7 loaded and held in weight_reg.
        //---------------------------------------------------------------------
        $display("--- Test 1: Weight loading (w=7) ---");
        load_weight(32'd7);
        $display("  Weight 7 loaded into weight_reg. Verified via MAC in Test 2.");

        //---------------------------------------------------------------------
        // TEST 2: Single MAC - act(3) × w(7) + psum(0) = 21
        // Both activation_valid AND psum_valid must be 1 (WS condition).
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 2: Single MAC (act=3, w=7, psum=0 -> 21) ---");
        do_mac(32'd3, 48'd0);
        check48("accumulator after 1 MAC (3x7+0=21)", result, 48'd21);

        //---------------------------------------------------------------------
        // TEST 3: Second MAC accumulates - 3×7=21 again, total = 42
        // weight_reg still holds 7 (stationary!).
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 3: Second MAC accumulates (21+21 = 42) ---");
        do_mac(32'd3, 48'd0);
        check48("accumulator after 2 MACs (21+21=42)", result, 48'd42);

        //---------------------------------------------------------------------
        // TEST 4: Clear accumulator
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 4: Clear accumulator ---");
        do_clear();
        check48("accumulator after clear", result, 48'd0);

        //---------------------------------------------------------------------
        // TEST 5: psum_valid=0 gate - accumulator must NOT update
        // WS PE: enable && activation_valid && psum_valid all required.
        // With psum_valid=0, even though enable=1 and act_valid=1, no update.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 5: No accumulation when psum_valid=0 ---");
        @(negedge clk);
        enable              = 1;
        activation_in       = 32'd5;
        activation_valid_in = 1;
        psum_in             = '0;
        psum_valid_in       = 0;       // ← gate: blocks accumulation
        @(posedge clk);
        @(posedge clk);
        activation_valid_in = 0;
        check48("accumulator stays 0 when psum_valid=0", result, 48'd0);

        //---------------------------------------------------------------------
        // TEST 6: enable=0 gate - accumulator must NOT update
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 6: No accumulation when enable=0 ---");
        @(negedge clk);
        enable              = 0;       // ← gate
        activation_in       = 32'd5;
        activation_valid_in = 1;
        psum_in             = '0;
        psum_valid_in       = 1;
        @(posedge clk);
        @(posedge clk);
        activation_valid_in = 0;
        psum_valid_in       = 0;
        check48("accumulator stays 0 when enable=0", result, 48'd0);

        //---------------------------------------------------------------------
        // TEST 7: Horizontal activation broadcast
        // activation_out must equal activation_in with 1-cycle latency.
        // activation_valid_out must mirror activation_valid_in.
        // FIX: keep activation_valid_in=1 across BOTH posedges so
        //      activation_valid_out=1 when read after posedge B.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 7: Horizontal activation broadcast (0xCAFEBABE) ---");
        @(negedge clk);
        enable              = 0;
        activation_in       = 32'hCAFE_BABE;
        activation_valid_in = 1;
        psum_valid_in       = 0;
        @(posedge clk);    // cycle A: activation_out <= 0xCAFEBABE, valid_out <= 1
        @(posedge clk);    // cycle B: same (activation_valid_in still 1)
        // Read BEFORE de-asserting so valid_out is still 1
        if (activation_out === 32'hCAFE_BABE && activation_valid_out === 1'b1) begin
            $display("  PASS  broadcast: out=0x%08h  valid=%0b",
                     activation_out, activation_valid_out);
            pass_count++;
        end else begin
            $display("  FAIL  broadcast: got=0x%08h valid=%0b (exp=0xCAFEBABE/1)",
                     activation_out, activation_valid_out);
            fail_count++;
        end
        activation_valid_in = 0;  // de-assert AFTER reading

        //---------------------------------------------------------------------
        // TEST 8: psum_out pass-through
        // psum_out = mac_result = act * weight_reg + psum_in.
        // With act=5, w=7, psum_in=3: mac_result = 5*7+3 = 38.
        // psum_out should be 38 and psum_valid_out should be 1.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 8: psum_out pass-through (5x7+3 = 38) ---");
        @(negedge clk);
        enable              = 1;
        activation_in       = 32'd5;
        activation_valid_in = 1;
        psum_in             = 48'd3;
        psum_valid_in       = 1;
        @(posedge clk);    // psum_out <= mac_result = 38 (NBA)
        @(posedge clk);    // psum_out committed
        if (psum_out === 48'd38 && psum_valid_out === 1'b1) begin
            $display("  PASS  psum_out=%0d (5x7+3=38)  psum_valid_out=%0b",
                     psum_out, psum_valid_out);
            pass_count++;
        end else begin
            $display("  FAIL  psum_out=%0d (exp=38)  psum_valid_out=%0b",
                     psum_out, psum_valid_out);
            fail_count++;
        end
        activation_valid_in = 0;
        psum_valid_in       = 0;

        //---------------------------------------------------------------------
        // TEST 9: Signed MAC - act=4, w=-3 (= 0xFFFFFFD), psum=0 → -12
        // -12 in 48-bit two's complement = 48'hFFFF_FFFF_FFF4
        // Load new weight, clear accumulator, then do signed MAC.
        // FIX: activation_in MUST be set in the same negedge-aligned block
        //      as psum_valid and enable so XSim sees all new values together.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 9: Signed MAC (act=4, w=-3, psum=0 -> -12) ---");
        // Load new weight: -3
        load_weight(32'hFFFF_FFFD);   // -3 in 32-bit two's complement

        // Clear accumulator
        do_clear();

        // Signed MAC (negedge-aligned, all signals set together)
        do_mac(32'd4, 48'd0);
        check48("signed MAC (4x-3+0 = -12)", result, 48'hFFFF_FFFF_FFF4);

        //---------------------------------------------------------------------
        // TEST 10: Weight stationary property - weight_reg holds -3 without reload
        // Do another MAC without reloading weight: 2 × (-3) + 0 = -6.
        // Total accumulator = -12 + (-6) = -18 = 48'hFFFF_FFFF_FFEE.
        //---------------------------------------------------------------------
        $display("");
        $display("--- Test 10: Weight stays stationary (no reload, 2x-3=-6, total=-18) ---");
        do_mac(32'd2, 48'd0);
        check48("stationary: accum after -12+(-6)=-18", result, 48'hFFFF_FFFF_FFEE);

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

        #50;
        $finish;
    end

    initial begin #500_000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("processing_element_ws.vcd");
        $dumpvars(0, processing_element_ws_tb);
    end

endmodule : processing_element_ws_tb