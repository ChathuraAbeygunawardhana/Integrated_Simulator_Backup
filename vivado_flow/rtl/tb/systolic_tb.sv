//=============================================================================
// systolic_tb.sv
// ==============
// Parameterised testbench for the output-stationary systolic array.
//
// Includes
// --------
//   1. configs/generated/optimal_config.vh  (optimizer picks everything)
//   2. configs/generated/user_config.vh     (user overrides; written last)
//
// Both headers use `define macros so later definitions win automatically.
//
// Golden reference
// ----------------
// The testbench uses $readmemh to load:
//   tb_data/input_hex.mem    – flattened input activations
//   tb_data/weight_hex.mem   – flattened weight tensor
//   tb_data/golden_hex.mem   – expected output (written by generate_golden.py)
//
// Pass/Fail
// ---------
//   • All output elements must be within `ERROR_TOLERANCE of golden.
//   • A timeout watchdog fires at `SIM_TIMEOUT cycles.
//   • The testbench writes "TEST_PASS" or "TEST_FAIL" to sim_result.txt
//     so the TCL script can grep it without parsing $display output.
//=============================================================================

// ── Include generated headers (order matters: user_config.vh overrides) ──
`include "configs/generated/optimal_config.vh"
`include "configs/generated/user_config.vh"

// ── Safe-defaults: macros that may not be set if no JSON was generated ────
`ifndef ARRAY_HEIGHT
  `define ARRAY_HEIGHT   4
`endif
`ifndef ARRAY_WIDTH
  `define ARRAY_WIDTH    4
`endif
`ifndef DATA_WIDTH
  `define DATA_WIDTH     32
`endif
`ifndef ACCUM_WIDTH
  `define ACCUM_WIDTH    48
`endif
`ifndef ADDR_WIDTH
  `define ADDR_WIDTH     32
`endif
`ifndef LAYER_K
  `define LAYER_K        16
`endif
`ifndef LAYER_C
  `define LAYER_C        8
`endif
`ifndef LAYER_KH
  `define LAYER_KH       3
`endif
`ifndef LAYER_KW
  `define LAYER_KW       3
`endif
`ifndef LAYER_H_IN
  `define LAYER_H_IN     30
`endif
`ifndef LAYER_W_IN
  `define LAYER_W_IN     30
`endif
`ifndef LAYER_H_OUT
  `define LAYER_H_OUT    (`LAYER_H_IN - `LAYER_KH + 1)
`endif
`ifndef LAYER_W_OUT
  `define LAYER_W_OUT    (`LAYER_W_IN - `LAYER_KW + 1)
`endif
`ifndef TILE_K
  `define TILE_K         `ARRAY_HEIGHT
`endif
`ifndef TILE_C
  `define TILE_C         `ARRAY_WIDTH
`endif
`ifndef TILE_H
  `define TILE_H         `ARRAY_HEIGHT
`endif
`ifndef TILE_W
  `define TILE_W         `ARRAY_WIDTH
`endif
`ifndef MEM_LAYOUT
  `define MEM_LAYOUT     0
`endif
`ifndef INPUT_BASE_ADDR
  `define INPUT_BASE_ADDR  32'h0000_0000
`endif
`ifndef WEIGHT_BASE_ADDR
  `define WEIGHT_BASE_ADDR 32'h0001_0000
`endif
`ifndef OUTPUT_BASE_ADDR
  `define OUTPUT_BASE_ADDR 32'h0002_0000
`endif
`ifndef SIM_TIMEOUT
  `define SIM_TIMEOUT    200000
`endif
`ifndef DUMP_WAVEFORMS
  `define DUMP_WAVEFORMS 1
`endif
`ifndef VERBOSE
  `define VERBOSE        0
`endif
`ifndef GOLDEN_CHECK
  `define GOLDEN_CHECK   1
`endif
`ifndef ERROR_TOLERANCE
  `define ERROR_TOLERANCE 0
`endif

// ── Package import (required by DUT) ─────────────────────────────────────
import systolic_array_pkg::*;

//=============================================================================
// Testbench module
//=============================================================================
module systolic_tb;

    // ── Parameters from header macros ─────────────────────────────────────
    localparam int AH   = `ARRAY_HEIGHT;
    localparam int AW   = `ARRAY_WIDTH;
    localparam int DW   = `DATA_WIDTH;
    localparam int ACCW = `ACCUM_WIDTH;
    localparam int ADDRW= `ADDR_WIDTH;

    localparam int K    = `LAYER_K;
    localparam int C    = `LAYER_C;
    localparam int KH   = `LAYER_KH;
    localparam int KW   = `LAYER_KW;
    localparam int HIN  = `LAYER_H_IN;
    localparam int WIN  = `LAYER_W_IN;
    localparam int HOUT = `LAYER_H_OUT;
    localparam int WOUT = `LAYER_W_OUT;

    localparam int TILE_K = `TILE_K;
    localparam int TILE_C = `TILE_C;
    localparam int TILE_H = `TILE_H;
    localparam int TILE_W = `TILE_W;

    // Memory sizes (flat element counts)
    localparam int INPUT_ELEMS  = HIN  * WIN  * C;
    localparam int WEIGHT_ELEMS = K    * C    * KH * KW;
    localparam int OUTPUT_ELEMS = HOUT * WOUT * K;

    // ── Clock & reset ──────────────────────────────────────────────────────
    logic clk  = 0;
    logic rst_n= 0;
    always #5 clk = ~clk;       // 100 MHz

    // ── DUT interface signals ──────────────────────────────────────────────
    mem_layout_t           mem_layout;

    logic [15:0]           input_channels, input_height, input_width;
    logic [15:0]           weight_k, weight_c, weight_kh, weight_kw;
    logic [15:0]           output_channels, output_height, output_width;

    logic [ADDRW-1:0]      input_base_addr, weight_base_addr, output_base_addr;
    logic                  start;
    logic [15:0]           tile_row, tile_col_start, tile_ch_start;
    logic                  done, busy;

    // Memory arrays — DUT drives addresses; TB drives data back
    logic [ADDRW-1:0]      input_addr  [AW];
    logic [AW-1:0]         input_addr_valid;
    logic [DW-1:0]         input_data  [AW];
    logic [AW-1:0]         input_data_valid;

    logic [ADDRW-1:0]      weight_addr  [AH];
    logic [AH-1:0]         weight_addr_valid;
    logic [DW-1:0]         weight_data  [AH];
    logic [AH-1:0]         weight_data_valid;

    logic [ADDRW-1:0]      output_addr  [AH][AW];
    logic [AH-1:0][AW-1:0] output_addr_valid;
    logic [DW-1:0]         output_data  [AH][AW];
    logic [AH-1:0][AW-1:0] output_data_valid;

    // ── Flat memory arrays (model SRAM) ────────────────────────────────────
    logic [DW-1:0] input_mem  [INPUT_ELEMS];
    logic [DW-1:0] weight_mem [WEIGHT_ELEMS];
    logic [DW-1:0] golden_mem [OUTPUT_ELEMS];
    logic [DW-1:0] output_mem [OUTPUT_ELEMS];   // captured from DUT

    // ── Result tracking ────────────────────────────────────────────────────
    int  errors    = 0;
    int  cycle_cnt = 0;
    int  result_fd;

    // ── DUT instantiation ──────────────────────────────────────────────────
    systolic_array_top #(
        .ARRAY_HEIGHT (`ARRAY_HEIGHT),
        .ARRAY_WIDTH  (`ARRAY_WIDTH),
        .DATA_WIDTH   (`DATA_WIDTH),
        .ACCUM_WIDTH  (`ACCUM_WIDTH),
        .ADDR_WIDTH   (`ADDR_WIDTH)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .mem_layout       (mem_layout),
        .input_channels   (input_channels),
        .input_height     (input_height),
        .input_width      (input_width),
        .weight_k         (weight_k),
        .weight_c         (weight_c),
        .weight_kh        (weight_kh),
        .weight_kw        (weight_kw),
        .output_channels  (output_channels),
        .output_height    (output_height),
        .output_width     (output_width),
        .input_base_addr  (input_base_addr),
        .weight_base_addr (weight_base_addr),
        .output_base_addr (output_base_addr),
        .start            (start),
        .tile_row         (tile_row),
        .tile_col_start   (tile_col_start),
        .tile_ch_start    (tile_ch_start),
        .done             (done),
        .busy             (busy),
        .input_addr       (input_addr),
        .input_addr_valid (input_addr_valid),
        .input_data       (input_data),
        .input_data_valid (input_data_valid),
        .weight_addr      (weight_addr),
        .weight_addr_valid(weight_addr_valid),
        .weight_data      (weight_data),
        .weight_data_valid(weight_data_valid),
        .output_addr      (output_addr),
        .output_addr_valid(output_addr_valid),
        .output_data      (output_data),
        .output_data_valid(output_data_valid)
    );

    //=========================================================================
    // Memory response logic (combinatorial SRAM model, 0-cycle latency)
    //=========================================================================

    // Input SRAM
    always_comb begin
        for (int col = 0; col < AW; col++) begin
            if (input_addr_valid[col]) begin
                automatic int addr = int'(input_addr[col]) / (DW/8);
                input_data[col]       = (addr < INPUT_ELEMS) ? input_mem[addr] : '0;
                input_data_valid[col] = 1'b1;
            end else begin
                input_data[col]       = '0;
                input_data_valid[col] = 1'b0;
            end
        end
    end

    // Weight SRAM
    always_comb begin
        for (int row = 0; row < AH; row++) begin
            if (weight_addr_valid[row]) begin
                automatic int addr = int'(weight_addr[row]) / (DW/8);
                weight_data[row]       = (addr < WEIGHT_ELEMS) ? weight_mem[addr] : '0;
                weight_data_valid[row] = 1'b1;
            end else begin
                weight_data[row]       = '0;
                weight_data_valid[row] = 1'b0;
            end
        end
    end

    // Output capture: store results when DUT drives output_data_valid
    always_ff @(posedge clk) begin
        for (int r = 0; r < AH; r++) begin
            for (int c = 0; c < AW; c++) begin
                if (output_data_valid[r][c]) begin
                    automatic int addr = int'(output_addr[r][c]) / (DW/8);
                    if (addr < OUTPUT_ELEMS) begin
                        output_mem[addr] <= output_data[r][c];
                        `ifdef VERBOSE
                        $display("[TB] cycle=%0d  output[%0d]=%0h",
                                 cycle_cnt, addr, output_data[r][c]);
                        `endif
                    end
                end
            end
        end
    end

    //=========================================================================
    // Cycle counter & watchdog
    //=========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n)
            cycle_cnt <= 0;
        else
            cycle_cnt <= cycle_cnt + 1;
    end

    //=========================================================================
    // Tiled workload sequencer
    //=========================================================================
    // For the testbench we drive the DUT tile-by-tile through the 6-D loop
    // nest.  Each tile fires one 'start' pulse and waits for 'done'.
    // The outermost sweep is K → C → KH → KW → H → W (can be reordered to
    // match the configured `LOOP_ORDER but the testbench correctness does not
    // depend on it — it just needs all tiles to run).
    //=========================================================================

    // ── Waveform dump ─────────────────────────────────────────────────────
    initial begin
        `ifdef DUMP_WAVEFORMS
        $dumpfile("sim_waveforms.vcd");
        $dumpvars(0, systolic_tb);
        `endif
    end

    // ── Main test sequence ────────────────────────────────────────────────
    initial begin : test_seq
        automatic int pass_count = 0;
        automatic int fail_count = 0;

        // ─── Load memory ─────────────────────────────────────
        $readmemh("tb_data/input_hex.mem",  input_mem);
        $readmemh("tb_data/weight_hex.mem", weight_mem);
        $readmemh("tb_data/golden_hex.mem", golden_mem);

        $display("[TB] Loaded input  (%0d elems)", INPUT_ELEMS);
        $display("[TB] Loaded weights(%0d elems)", WEIGHT_ELEMS);
        $display("[TB] Loaded golden (%0d elems)", OUTPUT_ELEMS);

        // ─── Drive static config signals ─────────────────────
        mem_layout      = (`MEM_LAYOUT == 0) ? mem_layout_t'(0) : mem_layout_t'(1);
        input_channels  = 16'(C);
        input_height    = 16'(HIN);
        input_width     = 16'(WIN);
        weight_k        = 16'(K);
        weight_c        = 16'(C);
        weight_kh       = 16'(KH);
        weight_kw       = 16'(KW);
        output_channels = 16'(K);
        output_height   = 16'(HOUT);
        output_width    = 16'(WOUT);
        input_base_addr = `INPUT_BASE_ADDR;
        weight_base_addr= `WEIGHT_BASE_ADDR;
        output_base_addr= `OUTPUT_BASE_ADDR;
        start           = 1'b0;

        // ─── Reset sequence ───────────────────────────────────
        @(posedge clk); rst_n = 1'b0;
        repeat(4) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        $display("[TB] Reset complete — starting tiled sweep …");
        $display("[TB] K=%0d C=%0d KH=%0d KW=%0d HIN=%0d WIN=%0d",
                  K, C, KH, KW, HIN, WIN);
        $display("[TB] Tile sizes: TILE_K=%0d TILE_C=%0d TILE_H=%0d TILE_W=%0d",
                  TILE_K, TILE_C, TILE_H, TILE_W);

        // ─── Tiled loop nest ──────────────────────────────────
        // Iterates over all tiles of the 6-D convolution.
        // Each iteration drives (tile_row, tile_col_start, tile_ch_start)
        // and pulses start for one cycle, then waits for done.
        for (int k0 = 0; k0 < K;   k0 += TILE_K) begin
            for (int c0 = 0; c0 < C;   c0 += TILE_C) begin
                for (int kh_i = 0; kh_i < KH; kh_i++) begin
                    for (int kw_i = 0; kw_i < KW; kw_i++) begin
                        for (int h0 = 0; h0 < HIN; h0 += TILE_H) begin
                            for (int w0 = 0; w0 < WIN; w0 += TILE_W) begin

                                tile_row       = 16'(h0 / TILE_H);
                                tile_col_start = 16'(w0);
                                tile_ch_start  = 16'(k0);

                                // Pulse start
                                @(posedge clk);
                                start = 1'b1;
                                @(posedge clk);
                                start = 1'b0;

                                // Wait for done (with watchdog)
                                fork
                                    begin : wait_done
                                        @(posedge done);
                                        disable watchdog_tile;
                                    end
                                    begin : watchdog_tile
                                        repeat(`SIM_TIMEOUT) @(posedge clk);
                                        $error("[TB] WATCHDOG: tile (%0d,%0d,%0d,%0d,%0d,%0d) timed out at cycle %0d",
                                               k0, c0, kh_i, kw_i, h0, w0, cycle_cnt);
                                        disable wait_done;
                                        fail_count++;
                                    end
                                join

                                @(posedge clk);  // gap between tiles

                            end
                        end
                    end
                end
            end
        end

        $display("[TB] Tiled sweep done at cycle %0d", cycle_cnt);

        // ─── Golden comparison ─────────────────────────────────
        `ifdef GOLDEN_CHECK
        errors = 0;
        for (int i = 0; i < OUTPUT_ELEMS; i++) begin
            automatic logic signed [DW-1:0] got  = signed'(output_mem[i]);
            automatic logic signed [DW-1:0] ref_ = signed'(golden_mem[i]);
            automatic int diff = int'(got) - int'(ref_);
            if (diff < 0) diff = -diff;
            if (diff > `ERROR_TOLERANCE) begin
                if (errors < 20)  // suppress after 20 mismatches
                    $error("[TB] MISMATCH [%0d]: got=%0h  ref=%0h  diff=%0d",
                           i, got, ref_, diff);
                errors++;
            end
        end

        if (errors == 0) begin
            $display("[TB] ✓ GOLDEN CHECK PASSED  (%0d elements checked)", OUTPUT_ELEMS);
            pass_count++;
        end else begin
            $error("[TB] ✗ GOLDEN CHECK FAILED  (%0d/%0d elements wrong)",
                   errors, OUTPUT_ELEMS);
            fail_count++;
        end
        `else
        $display("[TB] Golden check skipped (GOLDEN_CHECK=0)");
        pass_count++;
        `endif

        // ─── Write result file ────────────────────────────────
        result_fd = $fopen("sim_result.txt", "w");
        if (fail_count == 0 && pass_count > 0) begin
            $fdisplay(result_fd, "TEST_PASS");
            $display("[TB] ═══ TEST PASSED ═══");
        end else begin
            $fdisplay(result_fd, "TEST_FAIL errors=%0d", errors + fail_count);
            $display("[TB] ═══ TEST FAILED  errors=%0d ═══", errors + fail_count);
        end
        $fclose(result_fd);

        $display("[TB] Total simulation cycles: %0d", cycle_cnt);
        $finish;
    end

    //=========================================================================
    // Global watchdog (catches the case where the tile loop hangs entirely)
    //=========================================================================
    initial begin : global_watchdog
        repeat(`SIM_TIMEOUT * 3) @(posedge clk);
        $error("[TB] GLOBAL WATCHDOG timeout at cycle %0d", cycle_cnt);
        // Write failure before $finish
        result_fd = $fopen("sim_result.txt", "w");
        $fdisplay(result_fd, "TEST_FAIL errors=TIMEOUT");
        $fclose(result_fd);
        $finish;
    end

endmodule : systolic_tb
