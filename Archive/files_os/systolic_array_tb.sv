//=============================================================================
// Module: systolic_array_tb
// Description: Testbench for output stationary systolic array
//              Tests all three memory layouts
//=============================================================================

`timescale 1ns/1ps

module systolic_array_tb
    import systolic_array_pkg::*;
();

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam int ARRAY_HEIGHT = 4;
    localparam int ARRAY_WIDTH  = 4;
    localparam int DATA_WIDTH   = 32;
    localparam int ACCUM_WIDTH  = 48;
    localparam int ADDR_WIDTH   = 32;
    
    localparam int CLK_PERIOD   = 10;  // 100 MHz
    
    //=========================================================================
    // Signals
    //=========================================================================
    logic clk;
    logic rst_n;
    
    // Configuration
    mem_layout_t mem_layout;
    logic [15:0] input_channels;
    logic [15:0] input_height;
    logic [15:0] input_width;
    logic [15:0] weight_k;
    logic [15:0] weight_c;
    logic [15:0] weight_kh;
    logic [15:0] weight_kw;
    logic [15:0] output_channels;
    logic [15:0] output_height;
    logic [15:0] output_width;
    logic [ADDR_WIDTH-1:0] input_base_addr;
    logic [ADDR_WIDTH-1:0] weight_base_addr;
    logic [ADDR_WIDTH-1:0] output_base_addr;
    
    // Control
    logic start;
    logic [15:0] tile_row;
    logic [15:0] tile_col_start;
    logic [15:0] tile_ch_start;
    logic done;
    logic busy;
    
    // Memory interfaces
    logic [ADDR_WIDTH-1:0] input_addr [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_addr_valid;
    logic [DATA_WIDTH-1:0] input_data [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_data_valid;
    
    logic [ADDR_WIDTH-1:0] weight_addr [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_addr_valid;
    logic [DATA_WIDTH-1:0] weight_data [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_data_valid;
    
    logic [ADDR_WIDTH-1:0] output_addr [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid;
    logic [DATA_WIDTH-1:0] output_data [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid;
    
    //=========================================================================
    // DUT Instantiation
    //=========================================================================
    systolic_array_top #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mem_layout(mem_layout),
        .input_channels(input_channels),
        .input_height(input_height),
        .input_width(input_width),
        .weight_k(weight_k),
        .weight_c(weight_c),
        .weight_kh(weight_kh),
        .weight_kw(weight_kw),
        .output_channels(output_channels),
        .output_height(output_height),
        .output_width(output_width),
        .input_base_addr(input_base_addr),
        .weight_base_addr(weight_base_addr),
        .output_base_addr(output_base_addr),
        .start(start),
        .tile_row(tile_row),
        .tile_col_start(tile_col_start),
        .tile_ch_start(tile_ch_start),
        .done(done),
        .busy(busy),
        .input_addr(input_addr),
        .input_addr_valid(input_addr_valid),
        .input_data(input_data),
        .input_data_valid(input_data_valid),
        .weight_addr(weight_addr),
        .weight_addr_valid(weight_addr_valid),
        .weight_data(weight_data),
        .weight_data_valid(weight_data_valid),
        .output_addr(output_addr),
        .output_addr_valid(output_addr_valid),
        .output_data(output_data),
        .output_data_valid(output_data_valid)
    );
    
    //=========================================================================
    // Clock Generation
    //=========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //=========================================================================
    // Memory Models (Simple Response)
    //=========================================================================
    
    // Input memory model
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_data <= '{default: '0};
            input_data_valid <= '0;
        end else begin
            for (int i = 0; i < ARRAY_WIDTH; i++) begin
                if (input_addr_valid[i]) begin
                    // Simple pattern: use address as data
                    input_data[i] <= input_addr[i][7:0];
                    input_data_valid[i] <= 1'b1;
                end else begin
                    input_data_valid[i] <= 1'b0;
                end
            end
        end
    end
    
    // Weight memory model
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_data <= '{default: '0};
            weight_data_valid <= '0;
        end else begin
            for (int i = 0; i < ARRAY_HEIGHT; i++) begin
                if (weight_addr_valid[i]) begin
                    // Simple pattern: constant weights
                    weight_data[i] <= 32'h0000_0001;  // Weight = 1
                    weight_data_valid[i] <= 1'b1;
                end else begin
                    weight_data_valid[i] <= 1'b0;
                end
            end
        end
    end
    
    //=========================================================================
    // Test Tasks
    //=========================================================================
    
    task automatic reset_dut();
        begin
            rst_n = 0;
            start = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(2) @(posedge clk);
            $display("[%0t] Reset complete", $time);
        end
    endtask
    
    task automatic configure_test(
        input mem_layout_t layout,
        input int in_c, in_h, in_w,
        input int w_k, w_c, w_kh, w_kw,
        input int out_k, out_h, out_w
    );
        begin
            mem_layout = layout;
            
            input_channels = in_c;
            input_height = in_h;
            input_width = in_w;
            
            weight_k = w_k;
            weight_c = w_c;
            weight_kh = w_kh;
            weight_kw = w_kw;
            
            output_channels = out_k;
            output_height = out_h;
            output_width = out_w;
            
            input_base_addr  = 32'h1000_0000;
            weight_base_addr = 32'h2000_0000;
            output_base_addr = 32'h3000_0000;
            
            $display("[%0t] Configuration:", $time);
            $display("  Layout: %s", layout.name());
            $display("  Input: C=%0d, H=%0d, W=%0d", in_c, in_h, in_w);
            $display("  Weight: K=%0d, C=%0d, KH=%0d, KW=%0d", w_k, w_c, w_kh, w_kw);
            $display("  Output: K=%0d, H=%0d, W=%0d", out_k, out_h, out_w);
        end
    endtask
    
    task automatic run_tile(
        input int row,
        input int col_start,
        input int ch_start
    );
        begin
            $display("[%0t] Starting tile computation:", $time);
            $display("  Row=%0d, Col_start=%0d, Ch_start=%0d", row, col_start, ch_start);
            
            tile_row = row;
            tile_col_start = col_start;
            tile_ch_start = ch_start;
            
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            // Wait for completion
            wait(done);
            @(posedge clk);
            
            $display("[%0t] Tile computation complete", $time);
        end
    endtask
    
    task automatic test_layout(input mem_layout_t layout);
        begin
            $display("\n========================================");
            $display("Testing %s Layout", layout.name());
            $display("========================================");
            
            configure_test(
                .layout(layout),
                .in_c(3), .in_h(8), .in_w(8),
                .w_k(16), .w_c(3), .w_kh(3), .w_kw(3),
                .out_k(16), .out_h(6), .out_w(6)
            );
            
            run_tile(.row(0), .col_start(0), .ch_start(0));
            
            // Check some addresses
            $display("\nSample Input Addresses:");
            for (int i = 0; i < 4; i++) begin
                $display("  Column[%0d]: 0x%08h", i, input_addr[i]);
            end
            
            $display("\nSample Weight Addresses:");
            for (int i = 0; i < 4; i++) begin
                $display("  PE Row[%0d]: 0x%08h", i, weight_addr[i]);
            end
            
            $display("\nSample Output Addresses:");
            for (int r = 0; r < 2; r++) begin
                for (int c = 0; c < 2; c++) begin
                    $display("  Output[%0d][%0d]: 0x%08h", r, c, output_addr[r][c]);
                end
            end
        end
    endtask
    
    //=========================================================================
    // Main Test
    //=========================================================================
    initial begin
        $display("========================================");
        $display("Systolic Array Testbench");
        $display("========================================");
        
        // Initialize signals
        rst_n = 0;
        start = 0;
        
        // Reset
        reset_dut();
        
        // Test all three layouts
        test_layout(CHANNEL_MAJOR);
        #100;
        
        test_layout(ROW_MAJOR);
        #100;
        
        test_layout(COLUMN_MAJOR);
        #100;
        
        $display("\n========================================");
        $display("All Tests Complete!");
        $display("========================================");
        
        #100;
        $finish;
    end
    
    //=========================================================================
    // Timeout Watchdog
    //=========================================================================
    initial begin
        #1000000;  // 1ms timeout
        $display("ERROR: Testbench timeout!");
        $finish;
    end
    
    //=========================================================================
    // Waveform Dump
    //=========================================================================
    initial begin
        $dumpfile("systolic_array.vcd");
        $dumpvars(0, systolic_array_tb);
    end

endmodule : systolic_array_tb
