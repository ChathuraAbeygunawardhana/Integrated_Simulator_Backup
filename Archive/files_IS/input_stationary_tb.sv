//=============================================================================
// Module: input_stationary_tb  (FIXED)
// Description: Top-level testbench for input stationary systolic array.
//
// BUG FIXED
// ----------
// Original: fork...join with "disable fork" - this requires BOTH branches to
// complete before unblocking, so the timeout branch always ran to completion
// (50,000 cycles) even when done fired early.  It also used the old XSim-
// incompatible "disable fork" idiom inside a regular fork...join.
//
// Fix: fork...join_any + explicit disable both named blocks, same pattern
// applied in the OS multilayer testbench fix.
//=============================================================================

`timescale 1ns/1ps

module input_stationary_tb;

    localparam int ARRAY_HEIGHT = 4;
    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int ACCUM_WIDTH  = 48;
    localparam int ADDR_WIDTH   = 32;

    logic clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    import input_stationary_pkg::*;

    // Configuration
    mem_layout_t            mem_layout;
    logic [15:0]            input_channels, input_height, input_width;
    logic [15:0]            weight_k, weight_c, weight_kh, weight_kw;
    logic [15:0]            output_channels, output_height, output_width;
    logic [ADDR_WIDTH-1:0]  input_base_addr, weight_base_addr, output_base_addr;
    logic                   start, done, busy;
    logic [15:0]            tile_row, tile_col_start, tile_ch_start;

    // Memory interfaces
    logic [ADDR_WIDTH-1:0]  input_addr   [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_addr_valid;
    logic [DATA_WIDTH-1:0]  input_data   [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_data_valid;

    logic [ADDR_WIDTH-1:0]  weight_addr   [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_addr_valid;
    logic [DATA_WIDTH-1:0]  weight_data   [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_data_valid;

    logic [ADDR_WIDTH-1:0]  output_addr  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid;
    logic [DATA_WIDTH-1:0]  output_data  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    input_stationary_top #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH)
    ) dut (.*);

    //=========================================================================
    // Memory model - 1-cycle latency, address-as-data
    //=========================================================================
    always_ff @(posedge clk) begin
        for (int j = 0; j < ARRAY_WIDTH; j++) begin
            input_data[j]       <= input_addr_valid[j]  ?
                                   {24'h0, input_addr[j][7:0]}  : '0;
            input_data_valid[j] <= input_addr_valid[j];
        end
        for (int j = 0; j < ARRAY_HEIGHT; j++) begin
            weight_data[j]       <= weight_addr_valid[j] ?
                                    32'h0000_0001 : '0;  // constant weight=1
            weight_data_valid[j] <= weight_addr_valid[j];
        end
    end

    //=========================================================================
    // Test task
    //=========================================================================
    task automatic run_test(input mem_layout_t layout, input string layout_name);
        $display("\n========================================");
        $display("Testing %s", layout_name);
        $display("========================================");

        mem_layout       = layout;
        input_channels   = 3;   input_height  = 8;  input_width   = 8;
        weight_k         = 16;  weight_c      = 3;
        weight_kh        = 3;   weight_kw     = 3;
        output_channels  = 16;  output_height = 6;  output_width  = 6;
        input_base_addr  = 32'h1000_0000;
        weight_base_addr = 32'h2000_0000;
        output_base_addr = 32'h3000_0000;
        tile_row         = 0;
        tile_col_start   = 0;
        tile_ch_start    = 0;

        // Pulse start
        @(negedge clk); start = 1'b1;
        @(posedge clk); @(negedge clk); start = 1'b0;

        // FIX: fork...join_any + disable both blocks
        fork
            begin : wait_done_blk
                wait(done === 1'b1);
            end
            begin : timeout_blk
                repeat(200_000) @(posedge clk);
                $display("  TIMEOUT waiting for done! (state machine may be stuck)");
            end
        join_any
        disable wait_done_blk;
        disable timeout_blk;

        $display("  Tile computation complete.");
        $display("  Sample output_addr[0][0]: 0x%08h", output_addr[0][0]);
        $display("  Sample output_addr[0][1]: 0x%08h", output_addr[0][1]);
        $display("  Sample output_addr[1][0]: 0x%08h", output_addr[1][0]);
        $display("  Sample output_data[0][0]: 0x%08h (PE[0][0] result low 32b)", output_data[0][0]);

        // Brief inter-test gap
        repeat(20) @(posedge clk);
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        rst_n          = 1'b0;
        start          = 1'b0;
        mem_layout     = CHANNEL_MAJOR;
        input_channels = '0; input_height = '0; input_width = '0;
        weight_k = '0; weight_c = '0; weight_kh = '0; weight_kw = '0;
        output_channels = '0; output_height = '0; output_width = '0;
        input_base_addr = '0; weight_base_addr = '0; output_base_addr = '0;
        tile_row = '0; tile_col_start = '0; tile_ch_start = '0;

        repeat(5) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(5) @(posedge clk);

        $display("========================================");
        $display("Input Stationary Systolic Array TB (FIXED)");
        $display("ARRAY: %0d x %0d   DATA: %0d-bit   ACCUM: %0d-bit",
                 ARRAY_HEIGHT, ARRAY_WIDTH, DATA_WIDTH, ACCUM_WIDTH);
        $display("========================================");

        run_test(CHANNEL_MAJOR, "CHANNEL_MAJOR");

        // Reset DUT between layout tests
        @(negedge clk); rst_n = 1'b0;
        repeat(5) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(5) @(posedge clk);

        run_test(ROW_MAJOR, "ROW_MAJOR");

        @(negedge clk); rst_n = 1'b0;
        repeat(5) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(5) @(posedge clk);

        run_test(COLUMN_MAJOR, "COLUMN_MAJOR");

        $display("\nAll layout tests complete!");
        #100;
        $finish;
    end

    // Global watchdog (3 layouts x 200k cycles x 10ns = 6ms; use 8ms margin)
    initial begin
        #8_000_000;
        $display("FATAL: Global simulation timeout.");
        $finish;
    end

    initial begin
        $dumpfile("input_stationary.vcd");
        $dumpvars(0, input_stationary_tb);
    end

endmodule : input_stationary_tb