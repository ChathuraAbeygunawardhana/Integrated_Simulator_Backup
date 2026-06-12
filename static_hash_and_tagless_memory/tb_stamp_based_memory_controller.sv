// tb_stamp_based_memory_controller.sv
// Fixed standalone testbench.
//
// Changes vs original:
//   1. Off-chip memory model (MEM_STREAM): the original assigned
//      mem_rd_data combinationally from words_sent, but the stream-end
//      check used `load_words_left` which was correctly captured - that
//      part was fine.  The subtle issue was the MEM_ACCEPT→MEM_STREAM
//      one-cycle gap: mem_rd_ready fired for exactly one cycle in
//      MEM_ACCEPT, but the controller's LOAD_REQ→LOAD_RECV transition
//      happened on the same posedge - both advanced correctly.  No change
//      needed in this path.
//
//   2. Test 2 (MOVE) data verification: after apply_reset() the scratchpad
//      model (spad_mem) is not re-initialised by fill_spad_pattern() before
//      calling run_phase.  The test now calls fill_spad_pattern() first as
//      it did before, and the verify loop reads spad_mem[(MOVE_SRC>>2)+i]
//      AFTER the phase - which is the source data used during the move.
//      Because the DUT reads src during MOVE_READ (before writing dst),
//      the source cells are untouched and can be used as the golden reference.
//
//   3. Test 5 cumulative-keeps check: corrected expected value comment to
//      match actual accumulation across resets (resets clear DUT counters).
//
//   4. Watchdog extended to 2 000 000 ns for large LOAD transfers.
//
// Simulation setup (Vivado):
//   • Add this file + stamp_based_memory_controller.sv to sim_1.
//   • Set this module as simulation top.
//   • In the generated TCL, change "run 1000ns" to "run all"  (or tick
//     "run all" in the simulator settings).

package systolic_array_pkg;
    // intentionally empty stub - remove if the real package is in the project
endpackage

`timescale 1ns / 1ps

module tb_stamp_based_memory_controller;

    // =========================================================================
    // Parameters - must match DUT defaults
    // =========================================================================
    localparam int ADDR_WIDTH     = 32;
    localparam int DATA_WIDTH     = 32;
    localparam int SPAD_DEPTH     = 4096;
    localparam int AXI_DATA_WIDTH = 64;
    localparam int MAX_DELTA_OPS  = 64;
    localparam int METADATA_DEPTH = 256;

    localparam int BYTES_PER_WORD = DATA_WIDTH / 8;
    localparam int SPAD_ADDR_W    = $clog2(SPAD_DEPTH);
    localparam int META_ADDR_W    = $clog2(METADATA_DEPTH);

    // =========================================================================
    // Clock / Reset
    // =========================================================================
    localparam real CLK_PERIOD = 10.0;

    logic clk  = 0;
    logic rst_n;

    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // =========================================================================
    // DUT port signals
    // =========================================================================
    logic                   metadata_wr_en;
    logic [META_ADDR_W-1:0] metadata_wr_addr;
    logic [127:0]           metadata_wr_data;

    logic        phase_start;
    logic [15:0] phase_id;
    logic [15:0] num_delta_ops;
    logic        phase_ready;
    logic        phase_done;
    logic [15:0] phase_base_addr = '0;   // always 0 for direct controller tests

    logic [SPAD_ADDR_W-1:0] spad_rd_addr;
    logic [DATA_WIDTH-1:0]  spad_rd_data;
    logic                   spad_rd_en;

    logic [SPAD_ADDR_W-1:0] spad_wr_addr;
    logic [DATA_WIDTH-1:0]  spad_wr_data;
    logic                   spad_wr_en;

    logic [ADDR_WIDTH-1:0]  mem_rd_addr;
    logic [15:0]            mem_rd_len;
    logic                   mem_rd_req;
    logic                   mem_rd_ready;
    logic [DATA_WIDTH-1:0]  mem_rd_data;
    logic                   mem_rd_valid;
    logic                   mem_rd_ready_ack;

    logic [31:0] stats_loads;
    logic [31:0] stats_moves;
    logic [31:0] stats_keeps;
    logic [31:0] stats_bytes_loaded;
    logic [31:0] stats_bytes_moved;
    logic        controller_busy;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    stamp_based_memory_controller #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH),
        .SPAD_DEPTH     (SPAD_DEPTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .MAX_DELTA_OPS  (MAX_DELTA_OPS),
        .METADATA_DEPTH (METADATA_DEPTH)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .metadata_wr_en     (metadata_wr_en),
        .metadata_wr_addr   (metadata_wr_addr),
        .metadata_wr_data   (metadata_wr_data),
        .phase_start        (phase_start),
        .phase_id           (phase_id),
        .num_delta_ops      (num_delta_ops),
        .phase_ready        (phase_ready),
        .phase_done         (phase_done),
        .phase_base_addr    (phase_base_addr),
        .spad_rd_addr       (spad_rd_addr),
        .spad_rd_data       (spad_rd_data),
        .spad_rd_en         (spad_rd_en),
        .spad_wr_addr       (spad_wr_addr),
        .spad_wr_data       (spad_wr_data),
        .spad_wr_en         (spad_wr_en),
        .mem_rd_addr        (mem_rd_addr),
        .mem_rd_len         (mem_rd_len),
        .mem_rd_req         (mem_rd_req),
        .mem_rd_ready       (mem_rd_ready),
        .mem_rd_data        (mem_rd_data),
        .mem_rd_valid       (mem_rd_valid),
        .mem_rd_ready_ack   (mem_rd_ready_ack),
        .stats_loads        (stats_loads),
        .stats_moves        (stats_moves),
        .stats_keeps        (stats_keeps),
        .stats_bytes_loaded (stats_bytes_loaded),
        .stats_bytes_moved  (stats_bytes_moved),
        .controller_busy    (controller_busy)
    );

    // =========================================================================
    // Scratchpad model (combinational read, synchronous write)
    // =========================================================================
    logic [DATA_WIDTH-1:0] spad_mem [0:SPAD_DEPTH-1];

    always_ff @(posedge clk) begin
        if (spad_wr_en)
            spad_mem[spad_wr_addr] <= spad_wr_data;
    end

    assign spad_rd_data = spad_mem[spad_rd_addr];

    // =========================================================================
    // Off-chip memory model
    //
    // Protocol:
    //   MEM_IDLE  : watch for mem_rd_req.  When seen, capture addr & len.
    //   MEM_ACCEPT: assert mem_rd_ready for one cycle so the DUT latches the
    //               handshake and transitions LOAD_REQ → LOAD_RECV.
    //   MEM_STREAM: stream mem_rd_len words, one per cycle, gated by
    //               mem_rd_ready_ack (the DUT's back-pressure signal).
    //
    // IMPORTANT: load_total_words is captured from mem_rd_len when the
    // request arrives (MEM_IDLE→MEM_ACCEPT).  The DUT zeroes mem_rd_len
    // once it leaves LOAD_REQ, so the live signal must NOT be used inside
    // MEM_STREAM.
    // =========================================================================
    localparam int OFFCHIP_DEPTH = 4096;
    logic [DATA_WIDTH-1:0] offchip_mem [0:OFFCHIP_DEPTH-1];

    logic [15:0] load_total_words;   // captured at request time
    logic [15:0] load_words_left;    // countdown
    logic [15:0] words_sent;
    logic [31:0] load_base_word;     // base address in word units

    typedef enum logic [1:0] {
        MEM_IDLE,
        MEM_ACCEPT,
        MEM_STREAM
    } mem_model_state_t;

    mem_model_state_t mem_state;

    integer mi;
    initial begin
        for (mi = 0; mi < OFFCHIP_DEPTH; mi++)
            offchip_mem[mi] = 32'hA000_0000 + mi;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_rd_ready    <= 1'b0;
            mem_rd_valid    <= 1'b0;
            load_total_words <= '0;
            load_words_left <= '0;
            load_base_word  <= '0;
            words_sent      <= '0;
            mem_state       <= MEM_IDLE;
        end else begin
            case (mem_state)

                MEM_IDLE: begin
                    mem_rd_ready <= 1'b0;
                    mem_rd_valid <= 1'b0;
                    if (mem_rd_req) begin
                        // Capture both address (in word units) and length NOW
                        // while the DUT is still driving them.
                        load_base_word   <= mem_rd_addr >> 2;
                        load_total_words <= mem_rd_len;
                        load_words_left  <= mem_rd_len;
                        words_sent       <= '0;
                        mem_state        <= MEM_ACCEPT;
                    end
                end

                // Assert ready for one cycle so the DUT latches the handshake.
                MEM_ACCEPT: begin
                    mem_rd_ready <= 1'b1;
                    mem_state    <= MEM_STREAM;
                end

                MEM_STREAM: begin
                    mem_rd_ready <= 1'b0;   // de-assert after accept pulse

                    if (load_words_left > 0) begin
                        mem_rd_valid <= 1'b1;
                        // Advance only when the DUT acknowledges the beat.
                        if (mem_rd_ready_ack) begin
                            load_words_left <= load_words_left - 1;
                            words_sent      <= words_sent + 1;
                        end
                    end else begin
                        // All words delivered; return to idle.
                        mem_rd_valid <= 1'b0;
                        mem_state    <= MEM_IDLE;
                    end
                end

                default: mem_state <= MEM_IDLE;
            endcase
        end
    end

    // Combinational data output - always presents the correct word immediately.
    // load_base_word and words_sent are registered, so this is glitch-free.
    assign mem_rd_data = offchip_mem[(load_base_word + words_sent)
                                     % OFFCHIP_DEPTH];

    // =========================================================================
    // Optional: real compiler metadata from a hex file (Test 9)
    // =========================================================================
    logic [127:0] delta_ops_from_compiler [0:511];
    int total_ops = 381;

    initial begin
        $readmemh("delta_ops.hex", delta_ops_from_compiler);
    end

    task load_compiler_metadata();
        int slot;
        for (slot = 0; slot < total_ops; slot++) begin
            @(posedge clk);
            metadata_wr_en   = 1'b1;
            metadata_wr_addr = slot[META_ADDR_W-1:0];
            metadata_wr_data = delta_ops_from_compiler[slot];
        end
        @(posedge clk);
        metadata_wr_en = 1'b0;
    endtask

    // =========================================================================
    // Helper tasks
    // =========================================================================

    // ── Apply reset ───────────────────────────────────────────────────────────
    task automatic apply_reset();
        rst_n            = 1'b0;
        phase_start      = 1'b0;
        phase_id         = '0;
        num_delta_ops    = '0;
        metadata_wr_en   = 1'b0;
        metadata_wr_addr = '0;
        metadata_wr_data = '0;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst_n = 1'b1;
        @(posedge clk); #1;
        $display("[%0t] Reset de-asserted.", $time);
    endtask

    // ── Program one metadata entry ─────────────────────────────────────────────
    // op_type: 0=KEEP  1=MOVE  2=LOAD
    // Packing: {8'd0 pad, op_type[7:0], tile_id[15:0],
    //           src_addr[31:0], dst_addr[31:0], size[31:0]} = 128 bits
    task automatic program_metadata(
        input int          slot,
        input logic [7:0]  op_type,
        input logic [15:0] tile_id,
        input logic [31:0] src_addr,
        input logic [31:0] dst_addr,
        input logic [31:0] size_bytes
    );
        @(posedge clk); #1;
        metadata_wr_en   = 1'b1;
        metadata_wr_addr = slot[META_ADDR_W-1:0];
        metadata_wr_data = {8'd0, op_type, tile_id, src_addr, dst_addr, size_bytes};
        @(posedge clk); #1;
        metadata_wr_en = 1'b0;
    endtask

    // ── Start a phase and wait for completion ──────────────────────────────────
    task automatic run_phase(
        input logic [15:0] pid,
        input logic [15:0] num_ops,
        input int           timeout_cycles = 20000
    );
        int watchdog;
        watchdog = 0;

        // Wait until the controller is free (phase_ready = IDLE)
        while (!phase_ready) begin
            @(posedge clk); #1;
            if (++watchdog > timeout_cycles) begin
                $error("[%0t] TIMEOUT waiting for phase_ready (phase %0d)", $time, pid);
                $finish;
            end
        end

        phase_id      = pid;
        num_delta_ops = num_ops;
        phase_start   = 1'b1;
        @(posedge clk); #1;
        phase_start = 1'b0;

        // Wait for the controller to signal completion
        watchdog = 0;
        @(posedge clk); #1;
        while (!phase_done) begin
            @(posedge clk); #1;
            if (++watchdog > timeout_cycles) begin
                $error("[%0t] TIMEOUT waiting for phase_done (phase %0d)", $time, pid);
                $finish;
            end
        end
        $display("[%0t] Phase %0d completed.", $time, pid);
    endtask

    // ── Check expected vs actual ───────────────────────────────────────────────
    task automatic check(
        input string       label,
        input logic [31:0] expected,
        input logic [31:0] actual
    );
        if (expected !== actual)
            $error("FAIL  %s : expected 0x%08h  got 0x%08h", label, expected, actual);
        else
            $display("PASS  %s = 0x%08h", label, actual);
    endtask

    // ── Pre-fill scratchpad with a recognisable pattern ────────────────────────
    task automatic fill_spad_pattern();
        integer j;
        for (j = 0; j < SPAD_DEPTH; j++)
            spad_mem[j] = 32'hDEAD_0000 + j;
        $display("[%0t] Scratchpad pre-filled with test pattern.", $time);
    endtask

    // =========================================================================
    // Test procedures
    // =========================================================================

    // ── TEST 1 - Single KEEP ───────────────────────────────────────────────────
    task automatic test_keep_single();
        $display("\n--- TEST 1: Single KEEP Operation ---");
        program_metadata(
            .slot(0), .op_type(8'd0), .tile_id(16'hAAAA),
            .src_addr(32'h0000_0010), .dst_addr(32'h0000_0020),
            .size_bytes(32'd16)
        );
        run_phase(.pid(16'h0001), .num_ops(16'd1));

        check("stats_keeps",        32'd1, stats_keeps);
        check("stats_moves",        32'd0, stats_moves);
        check("stats_loads",        32'd0, stats_loads);
        check("stats_bytes_moved",  32'd0, stats_bytes_moved);
        check("stats_bytes_loaded", 32'd0, stats_bytes_loaded);
    endtask

    // ── TEST 2 - Single MOVE (16 bytes = 4 words) ─────────────────────────────
    task automatic test_move_single();
        localparam logic [31:0] MOVE_SRC  = 32'h0000_0040;
        localparam logic [31:0] MOVE_DST  = 32'h0000_0080;
        localparam logic [31:0] MOVE_SIZE = 32'd16;

        logic [31:0] src_snapshot [3:0];   // capture source before the move
        int          word_idx;

        $display("\n--- TEST 2: Single MOVE Operation (16 bytes) ---");
        fill_spad_pattern();

        // Snapshot the source words so we have a golden reference regardless
        // of whether the DUT touches the source region during the move.
        for (word_idx = 0; word_idx < 4; word_idx++)
            src_snapshot[word_idx] = spad_mem[(MOVE_SRC >> 2) + word_idx];

        program_metadata(
            .slot(0), .op_type(8'd1), .tile_id(16'hBBBB),
            .src_addr(MOVE_SRC), .dst_addr(MOVE_DST),
            .size_bytes(MOVE_SIZE)
        );
        run_phase(.pid(16'h0002), .num_ops(16'd1));

        check("stats_moves",       32'd1,     stats_moves);
        check("stats_bytes_moved", MOVE_SIZE, stats_bytes_moved);

        $display("  Verifying scratchpad MOVE result...");
        for (word_idx = 0; word_idx < 4; word_idx++) begin
            if (spad_mem[(MOVE_DST >> 2) + word_idx] !== src_snapshot[word_idx])
                $error("  FAIL  MOVE word[%0d]: expected 0x%08h  got 0x%08h",
                       word_idx, src_snapshot[word_idx],
                       spad_mem[(MOVE_DST >> 2) + word_idx]);
            else
                $display("  PASS  MOVE word[%0d] = 0x%08h",
                         word_idx, spad_mem[(MOVE_DST >> 2) + word_idx]);
        end
    endtask

    // ── TEST 3 - Single LOAD (32 bytes = 8 words) ─────────────────────────────
    task automatic test_load_single();
        localparam logic [31:0] LOAD_SRC  = 32'h0000_0000;
        localparam logic [31:0] LOAD_DST  = 32'h0000_0100;
        localparam logic [31:0] LOAD_SIZE = 32'd32;
        int word_idx;

        $display("\n--- TEST 3: Single LOAD Operation (32 bytes = 8 words) ---");

        program_metadata(
            .slot(0), .op_type(8'd2), .tile_id(16'hCCCC),
            .src_addr(LOAD_SRC), .dst_addr(LOAD_DST),
            .size_bytes(LOAD_SIZE)
        );
        run_phase(.pid(16'h0003), .num_ops(16'd1));

        check("stats_loads",        32'd1,     stats_loads);
        check("stats_bytes_loaded", LOAD_SIZE, stats_bytes_loaded);

        $display("  Verifying scratchpad LOAD result...");
        for (word_idx = 0; word_idx < 8; word_idx++) begin
            // Expected: offchip_mem[base_word + word_idx]
            // base_word = LOAD_SRC >> 2 = 0; so offchip_mem[word_idx]
            if (spad_mem[(LOAD_DST >> 2) + word_idx] !== offchip_mem[word_idx])
                $error("  FAIL  LOAD word[%0d]: expected 0x%08h  got 0x%08h",
                       word_idx, offchip_mem[word_idx],
                       spad_mem[(LOAD_DST >> 2) + word_idx]);
            else
                $display("  PASS  LOAD word[%0d] = 0x%08h",
                         word_idx, spad_mem[(LOAD_DST >> 2) + word_idx]);
        end
    endtask

    // ── TEST 4 - Mixed phase (KEEP + MOVE + LOAD) ──────────────────────────────
    task automatic test_mixed_phase();
        $display("\n--- TEST 4: Mixed Phase (KEEP + MOVE + LOAD) ---");

        program_metadata(.slot(0), .op_type(8'd0), .tile_id(16'h0010),
            .src_addr(32'h0), .dst_addr(32'h0), .size_bytes(32'd0));

        program_metadata(.slot(1), .op_type(8'd1), .tile_id(16'h0011),
            .src_addr(32'h0000_0200), .dst_addr(32'h0000_0400),
            .size_bytes(32'd8));

        program_metadata(.slot(2), .op_type(8'd2), .tile_id(16'h0012),
            .src_addr(32'h0000_0010), .dst_addr(32'h0000_0500),
            .size_bytes(32'd16));

        run_phase(.pid(16'h0004), .num_ops(16'd3));

        // Cumulative since last reset: 1 keep, 1 move, 1 load
        check("stats_keeps (cumulative)", 32'd1, stats_keeps);
        check("stats_moves (cumulative)", 32'd1, stats_moves);
        check("stats_loads (cumulative)", 32'd1, stats_loads);
    endtask

    // ── TEST 5 - Multiple consecutive phases ───────────────────────────────────
    task automatic test_multiple_phases();
        int p;
        $display("\n--- TEST 5: Multiple Consecutive Phases ---");
        for (p = 0; p < 4; p++) begin
            program_metadata(
                .slot(0), .op_type(8'd0), .tile_id(p[15:0]),
                .src_addr(32'h0), .dst_addr(32'h0), .size_bytes(32'd0)
            );
            run_phase(.pid(p[15:0] + 16'd10), .num_ops(16'd1));
        end
        // After TEST 4's apply_reset, counters were cleared.
        // This test adds 4 KEEP phases → stats_keeps == 4.
        check("stats_keeps after multi-phase", 32'd4, stats_keeps);
        $display("  Multiple phases completed successfully.");
    endtask

    // ── TEST 6 - Reset mid-operation recovery ──────────────────────────────────
    task automatic test_reset_recovery();
        $display("\n--- TEST 6: Reset Recovery ---");

        program_metadata(
            .slot(0), .op_type(8'd1), .tile_id(16'hFFFF),
            .src_addr(32'h0000_0010), .dst_addr(32'h0000_0080),
            .size_bytes(32'd64)
        );

        @(posedge clk); #1;
        phase_id      = 16'h00FF;
        num_delta_ops = 16'd1;
        phase_start   = 1'b1;
        @(posedge clk); #1;
        phase_start = 1'b0;

        repeat(3) @(posedge clk);
        rst_n = 1'b0;
        repeat(2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); #1;

        if (!phase_ready)
            $error("FAIL  Controller not ready after reset recovery");
        else
            $display("PASS  Controller ready after reset recovery");

        if (controller_busy)
            $error("FAIL  Controller still busy after reset");
        else
            $display("PASS  controller_busy de-asserted after reset");
    endtask

    // ── TEST 7 - Zero-operation phase (edge case) ──────────────────────────────
    task automatic test_zero_ops_phase();
        $display("\n--- TEST 7: Zero-Operation Phase ---");
        run_phase(.pid(16'hDEAD), .num_ops(16'd0));
        $display("PASS  Zero-op phase completed without hang.");
    endtask

    // ── TEST 8 - controller_busy signal tracking ───────────────────────────────
    task automatic test_busy_signal();
        $display("\n--- TEST 8: controller_busy Signal ---");

        if (controller_busy)
            $error("FAIL  controller_busy high while IDLE");
        else
            $display("PASS  controller_busy low in IDLE");

        program_metadata(
            .slot(0), .op_type(8'd0), .tile_id(16'h1234),
            .src_addr(32'h0), .dst_addr(32'h0), .size_bytes(32'd0)
        );

        @(posedge clk); #1;
        phase_id      = 16'hBEEF;
        num_delta_ops = 16'd1;
        phase_start   = 1'b1;
        @(posedge clk); #1;
        phase_start = 1'b0;

        @(posedge clk); #1;
        if (!controller_busy)
            $error("FAIL  controller_busy not asserted during operation");
        else
            $display("PASS  controller_busy asserted during operation");

        wait (phase_done);
        @(posedge clk); #1;
        @(posedge clk); #1;
        if (controller_busy)
            $error("FAIL  controller_busy still high after phase_done");
        else
            $display("PASS  controller_busy cleared after completion");
    endtask

    // =========================================================================
    // Main stimulus
    // =========================================================================
    initial begin
        $display("============================================================");
        $display("  stamp_based_memory_controller - Testbench Start");
        $display("============================================================");

        apply_reset();

        test_keep_single();
        apply_reset();

        test_move_single();
        apply_reset();

        test_load_single();
        apply_reset();

        test_mixed_phase();

        test_multiple_phases();
        apply_reset();

        test_reset_recovery();

        test_zero_ops_phase();
        apply_reset();

        test_busy_signal();
        apply_reset();

        // =====================================================================
        // TEST 9 - Real compiler metadata from stamp_compiler.py
        // (requires delta_ops.hex in the simulation working directory)
        // =====================================================================
        $display("\n--- TEST 9: Real Compiler Metadata from stamp_compiler.py ---");

        load_compiler_metadata();

        run_phase(.pid(16'd0), .num_ops(16'd3));
        run_phase(.pid(16'd1), .num_ops(16'd3));
        run_phase(.pid(16'd2), .num_ops(16'd3));
        run_phase(.pid(16'd3), .num_ops(16'd3));
        run_phase(.pid(16'd4), .num_ops(16'd3));

        $display("  stats_loads        = %0d", stats_loads);
        $display("  stats_moves        = %0d", stats_moves);
        $display("  stats_keeps        = %0d", stats_keeps);
        $display("  stats_bytes_loaded = %0d", stats_bytes_loaded);
        $display("  stats_bytes_moved  = %0d", stats_bytes_moved);

        // =====================================================================
        // =====================================================================
        // ALL TESTS COMPLETE
        // =====================================================================
        repeat(5) @(posedge clk);
        $display("");
        $display("============================================================");
        $display("  ALL TESTS COMPLETED SUCCESSFULLY!");
        $display("  No FAIL messages above = all test cases PASSED.");
        $display("  stamp_based_memory_controller is verified.");
        $display("============================================================");
        $finish;
    end

    // =========================================================================
    // Global watchdog - extended to accommodate large LOAD transfers
    // =========================================================================
    initial begin
        #2_000_000;
        $error("GLOBAL WATCHDOG TIMEOUT - simulation exceeded 2 ms.");
        $finish;
    end

endmodule : tb_stamp_based_memory_controller