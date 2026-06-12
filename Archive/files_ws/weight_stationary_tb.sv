//============================================================================= Dilshan
// Module: weight_stationary_tb
// Description: Top-level testbench for Weight-Stationary systolic array
//              Mirrors the structure of systolic_array_tb.sv (OS version)
//
// The WS top has a different port map than OS:
//   - input_addr  [ARRAY_HEIGHT]          (one stream per row, not column)
//   - weight_addr [ARRAY_HEIGHT][ARRAY_WIDTH] (2D — one per PE)
//   - output_addr [ARRAY_HEIGHT]          (one stream per row)
//   - Extra control: input_ch, kernel_row, kernel_col_start
//
// Tests:
//   1. CHANNEL_MAJOR layout – full tile run
//   2. ROW_MAJOR    layout – full tile run
//   3. COLUMN_MAJOR layout – full tile run
//=============================================================================

`timescale 1ns/1ps

module weight_stationary_tb
    import weight_stationary_pkg::*;
();

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam int ARRAY_HEIGHT = 4;
    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int ACCUM_WIDTH  = 48;
    localparam int ADDR_WIDTH   = 32;
    localparam int CLK_PERIOD   = 10;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;

    // Configuration
    mem_layout_t mem_layout;
    logic [15:0] input_channels, input_height, input_width;
    logic [15:0] weight_k, weight_c, weight_kh, weight_kw;
    logic [15:0] output_channels, output_height, output_width;
    logic [ADDR_WIDTH-1:0] input_base_addr, weight_base_addr, output_base_addr;

    // Control
    logic        start, done, busy;
    logic [15:0] tile_row_start, tile_col, tile_ch_start;
    logic [15:0] input_ch, kernel_row, kernel_col_start;

    // Memory interfaces – WS
    logic [ADDR_WIDTH-1:0] input_addr  [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] input_addr_valid;
    logic [DATA_WIDTH-1:0]   input_data  [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] input_data_valid;

    logic [ADDR_WIDTH-1:0] weight_addr [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_addr_valid;
    logic [DATA_WIDTH-1:0]  weight_data [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_data_valid;

    logic [ADDR_WIDTH-1:0] output_addr [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] output_addr_valid;
    logic [DATA_WIDTH-1:0]   output_data [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] output_data_valid;

    //=========================================================================
    // DUT
    //=========================================================================
    weight_stationary_top #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH)
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
        .tile_row_start   (tile_row_start),
        .tile_col         (tile_col),
        .tile_ch_start    (tile_ch_start),
        .input_ch         (input_ch),
        .kernel_row       (kernel_row),
        .kernel_col_start (kernel_col_start),
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
    // Clock
    //=========================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Simple memory models
    //=========================================================================
    // Input memory: return address low-byte as data (deterministic pattern)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_data       <= '{default: '0};
            input_data_valid <= '0;
        end else begin
            for (int i = 0; i < ARRAY_HEIGHT; i++) begin
                if (input_addr_valid[i]) begin
                    input_data[i]       <= {24'h0, input_addr[i][7:0]};
                    input_data_valid[i] <= 1'b1;
                end else begin
                    input_data_valid[i] <= 1'b0;
                end
            end
        end
    end

    // Weight memory: constant 1 for all weights
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_data       <= '{default: '0};
            weight_data_valid <= '0;
        end else begin
            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                for (int c = 0; c < ARRAY_WIDTH; c++) begin
                    if (weight_addr_valid[r][c]) begin
                        weight_data[r][c]       <= 32'h0000_0001;
                        weight_data_valid[r][c] <= 1'b1;
                    end else begin
                        weight_data_valid[r][c] <= 1'b0;
                    end
                end
            end
        end
    end

    //=========================================================================
    // Test tasks
    //=========================================================================
    task automatic reset_dut();
        rst_n             = 0;
        start             = 0;
        tile_row_start    = 0; tile_col = 0; tile_ch_start = 0;
        input_ch          = 0; kernel_row = 0; kernel_col_start = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
    endtask

    task automatic configure(
        input mem_layout_t lay,
        input int in_c, in_h, in_w,
        input int w_k, w_c, w_kh, w_kw,
        input int out_k, out_h, out_w
    );
        mem_layout        = lay;
        input_channels    = in_c;  input_height = in_h;  input_width = in_w;
        weight_k          = w_k;   weight_c     = w_c;
        weight_kh         = w_kh;  weight_kw    = w_kw;
        output_channels   = out_k; output_height = out_h; output_width = out_w;
        input_base_addr   = 32'h1000_0000;
        weight_base_addr  = 32'h2000_0000;
        output_base_addr  = 32'h3000_0000;
        $display("[%0t] Config: layout=%s In(%0d,%0d,%0d) W(%0d,%0d,%0d,%0d) Out(%0d,%0d,%0d)",
                 $time, lay.name(), in_c, in_h, in_w,
                 w_k, w_c, w_kh, w_kw, out_k, out_h, out_w);
    endtask

    task automatic run_tile(
        input int row_s, col, ch_s, in_ch, k_row, k_col_s
    );
        tile_row_start    = row_s;
        tile_col          = col;
        tile_ch_start     = ch_s;
        input_ch          = in_ch;
        kernel_row        = k_row;
        kernel_col_start  = k_col_s;

        $display("[%0t] Tile start: row=%0d col=%0d ch=%0d", $time, row_s, col, ch_s);
        @(posedge clk); start = 1;
        @(posedge clk); start = 0;

        fork
            begin : wait_done
                @(posedge done);
                $display("[%0t] Tile done!", $time);
            end
            begin : to
                repeat(200000) @(posedge clk);
                $display("[%0t] ERROR: tile timeout", $time);
                disable wait_done;
            end
        join_any
        disable fork;

        $display("  Sample input_addr[0] = 0x%08h", input_addr[0]);
        $display("  Sample weight_addr[0][0] = 0x%08h", weight_addr[0][0]);
        $display("  Sample output_addr[0] = 0x%08h", output_addr[0]);
    endtask

    task automatic test_layout(input mem_layout_t lay);
        $display("\n========================================");
        $display("Testing %s", lay.name());
        $display("========================================");
        configure(lay, 3, 8, 8, 16, 3, 3, 3, 16, 6, 6);
        run_tile(0, 0, 0, 0, 0, 0);
        #200;
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin
        $display("========================================");
        $display("Weight-Stationary Top Testbench");
        $display("========================================");

        reset_dut();

        test_layout(CHANNEL_MAJOR);
        test_layout(ROW_MAJOR);
        test_layout(COLUMN_MAJOR);

        $display("\n========================================");
        $display("All layout tests complete!");
        $display("========================================");
        #200;
        $finish;
    end

    initial begin #5000000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("weight_stationary.vcd");
        $dumpvars(0, weight_stationary_tb);
    end

endmodule : weight_stationary_tb