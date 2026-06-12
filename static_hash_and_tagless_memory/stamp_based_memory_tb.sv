`timescale 1ns/1ps

module stamp_based_memory_tb
    import systolic_array_pkg::*;
    import stamp_memory_pkg::*;
();

    localparam int CLK_PERIOD     = 10;
    localparam int ARRAY_HEIGHT   = 4;
    localparam int ARRAY_WIDTH    = 4;
    localparam int DATA_WIDTH     = 32;
    localparam int ADDR_WIDTH     = 32;
    localparam int SPAD_DEPTH     = 4096;
    localparam int METADATA_DEPTH = 256;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk, rst_n;

    mem_layout_t           mem_layout;
    logic [15:0]           input_channels, input_height, input_width;
    logic [15:0]           weight_k, weight_c, weight_kh, weight_kw;
    logic [15:0]           output_channels, output_height, output_width;
    logic [ADDR_WIDTH-1:0] input_base_addr, weight_base_addr, output_base_addr;
    logic [15:0]           phase_base_addr;

    nn_operation_config_t  nn_config;
    logic                  enable_activation, enable_pooling, enable_normalization;

    logic                              metadata_wr_en;
    logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr;
    logic [127:0]                      metadata_wr_data;
    logic                              phase_start;
    logic [15:0]                       phase_id;
    logic [15:0]                       num_delta_ops;
    logic                              phase_ready, phase_done;

    logic [DATA_WIDTH-1:0] norm_gamma [512], norm_beta [512];
    logic [DATA_WIDTH-1:0] running_mean [512], running_var [512];
    logic [DATA_WIDTH-1:0] updated_mean [512], updated_var [512];
    logic                  norm_params_valid, training_mode;

    logic        start;
    logic [15:0] tile_row, tile_col_start, tile_ch_start;
    logic        done, busy;

    logic [$clog2(SPAD_DEPTH)-1:0] spad_load_addr;
    logic [DATA_WIDTH-1:0]          spad_load_data;
    logic                           spad_load_en;

    memory_stats_t mem_stats;
    logic          controller_busy;

    // AXI signals
    logic [3:0]            m_axi_arid;
    logic [ADDR_WIDTH-1:0] m_axi_araddr;
    logic [7:0]            m_axi_arlen;
    logic [2:0]            m_axi_arsize;
    logic [1:0]            m_axi_arburst;
    logic                  m_axi_arlock;
    logic [3:0]            m_axi_arcache;
    logic [2:0]            m_axi_arprot;
    logic                  m_axi_arvalid, m_axi_arready;
    logic [3:0]            m_axi_rid;
    logic [63:0]           m_axi_rdata;
    logic [1:0]            m_axi_rresp;
    logic                  m_axi_rlast, m_axi_rvalid, m_axi_rready;
    logic [3:0]            m_axi_awid;
    logic [ADDR_WIDTH-1:0] m_axi_awaddr;
    logic [7:0]            m_axi_awlen;
    logic [2:0]            m_axi_awsize;
    logic [1:0]            m_axi_awburst;
    logic                  m_axi_awlock;
    logic [3:0]            m_axi_awcache;
    logic [2:0]            m_axi_awprot;
    logic                  m_axi_awvalid, m_axi_awready;
    logic [63:0]           m_axi_wdata;
    logic [7:0]            m_axi_wstrb;
    logic                  m_axi_wlast, m_axi_wvalid, m_axi_wready;
    logic [3:0]            m_axi_bid;
    logic [1:0]            m_axi_bresp;
    logic                  m_axi_bvalid, m_axi_bready;

    //=========================================================================
    // DUT
    //=========================================================================
    systolic_array_stamp_top #(
        .ARRAY_HEIGHT   (ARRAY_HEIGHT),
        .ARRAY_WIDTH    (ARRAY_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .SPAD_DEPTH     (SPAD_DEPTH),
        .METADATA_DEPTH (METADATA_DEPTH)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .mem_layout         (mem_layout),
        .input_channels     (input_channels),
        .input_height       (input_height),
        .input_width        (input_width),
        .weight_k           (weight_k),
        .weight_c           (weight_c),
        .weight_kh          (weight_kh),
        .weight_kw          (weight_kw),
        .output_channels    (output_channels),
        .output_height      (output_height),
        .output_width       (output_width),
        .input_base_addr    (input_base_addr),
        .weight_base_addr   (weight_base_addr),
        .output_base_addr   (output_base_addr),
        .nn_config          (nn_config),
        .enable_activation  (enable_activation),
        .enable_pooling     (enable_pooling),
        .enable_normalization(enable_normalization),
        .metadata_wr_en     (metadata_wr_en),
        .metadata_wr_addr   (metadata_wr_addr),
        .metadata_wr_data   (metadata_wr_data),
        .phase_start        (phase_start),
        .phase_id           (phase_id),
        .phase_base_addr    (phase_base_addr),
        .num_delta_ops      (num_delta_ops),
        .phase_ready        (phase_ready),
        .phase_done         (phase_done),
        .norm_gamma         (norm_gamma),
        .norm_beta          (norm_beta),
        .running_mean       (running_mean),
        .running_var        (running_var),
        .updated_mean       (updated_mean),
        .updated_var        (updated_var),
        .norm_params_valid  (norm_params_valid),
        .training_mode      (training_mode),
        .start              (start),
        .tile_row           (tile_row),
        .tile_col_start     (tile_col_start),
        .tile_ch_start      (tile_ch_start),
        .done               (done),
        .busy               (busy),
        .spad_load_addr     (spad_load_addr),
        .spad_load_data     (spad_load_data),
        .spad_load_en       (spad_load_en),
        .mem_stats          (mem_stats),
        .controller_busy    (controller_busy),
        .m_axi_arid         (m_axi_arid),
        .m_axi_araddr       (m_axi_araddr),
        .m_axi_arlen        (m_axi_arlen),
        .m_axi_arsize       (m_axi_arsize),
        .m_axi_arburst      (m_axi_arburst),
        .m_axi_arlock       (m_axi_arlock),
        .m_axi_arcache      (m_axi_arcache),
        .m_axi_arprot       (m_axi_arprot),
        .m_axi_arvalid      (m_axi_arvalid),
        .m_axi_arready      (m_axi_arready),
        .m_axi_rid          (m_axi_rid),
        .m_axi_rdata        (m_axi_rdata),
        .m_axi_rresp        (m_axi_rresp),
        .m_axi_rlast        (m_axi_rlast),
        .m_axi_rvalid       (m_axi_rvalid),
        .m_axi_rready       (m_axi_rready),
        .m_axi_awid         (m_axi_awid),
        .m_axi_awaddr       (m_axi_awaddr),
        .m_axi_awlen        (m_axi_awlen),
        .m_axi_awsize       (m_axi_awsize),
        .m_axi_awburst      (m_axi_awburst),
        .m_axi_awlock       (m_axi_awlock),
        .m_axi_awcache      (m_axi_awcache),
        .m_axi_awprot       (m_axi_awprot),
        .m_axi_awvalid      (m_axi_awvalid),
        .m_axi_awready      (m_axi_awready),
        .m_axi_wdata        (m_axi_wdata),
        .m_axi_wstrb        (m_axi_wstrb),
        .m_axi_wlast        (m_axi_wlast),
        .m_axi_wvalid       (m_axi_wvalid),
        .m_axi_wready       (m_axi_wready),
        .m_axi_bid          (m_axi_bid),
        .m_axi_bresp        (m_axi_bresp),
        .m_axi_bvalid       (m_axi_bvalid),
        .m_axi_bready       (m_axi_bready)
    );

    //=========================================================================
    // AXI Memory Model
    // off_chip_mem[i] = i, word-indexed.
    // Addresses from testbench: 0x0000=word0, 0x1000=word1024, 0x2000=word2048
    // All within the OFF_CHIP_WORDS array.
    //=========================================================================
    localparam int OFF_CHIP_WORDS = 8192;
    logic [DATA_WIDTH-1:0] off_chip_mem [OFF_CHIP_WORDS];
    logic [7:0]            read_counter;
    logic [ADDR_WIDTH-1:0] burst_rd_addr;

    initial begin
        for (int i = 0; i < OFF_CHIP_WORDS; i++)
            off_chip_mem[i] = i;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid  <= 1'b0;
            m_axi_rlast   <= 1'b0;
            read_counter  <= 0;
            burst_rd_addr <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready && !m_axi_rvalid) begin
                // arlen = beats-1; load counter with arlen
                read_counter  <= m_axi_arlen;
                burst_rd_addr <= m_axi_araddr >> 2;   // byte → word index
                m_axi_rvalid  <= 1'b1;
                m_axi_rlast   <= (m_axi_arlen == 8'b00000000);
                m_axi_arready <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (read_counter > 8'd0) begin
                    read_counter  <= read_counter - 1;
                    burst_rd_addr <= burst_rd_addr + 1;
                    m_axi_rlast   <= (read_counter == 8'b00000001);
                end else begin
                    // Last beat just accepted
                    m_axi_rvalid  <= 1'b0;
                    m_axi_rlast   <= 1'b0;
                    m_axi_arready <= 1'b1;
                end
            end
        end
    end

    assign m_axi_rdata  = {32'b0, off_chip_mem[burst_rd_addr]};
    assign m_axi_rid    = 4'b0;
    assign m_axi_rresp  = 2'b00;
    assign m_axi_awready = 1'b1;
    assign m_axi_wready  = 1'b1;
    assign m_axi_bvalid  = 1'b0;
    assign m_axi_bid     = 4'b0;
    assign m_axi_bresp   = 2'b00;

    //=========================================================================
    // Clock
    //=========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //=========================================================================
    // Stimulus
    //=========================================================================
    initial begin
        $display("========================================");
        $display("Stamp-Based Memory Management Testbench");
        $display("========================================\n");

        rst_n           = 0;
        metadata_wr_en  = 0;
        phase_start     = 0;
        phase_base_addr = 0;
        start           = 0;
        spad_load_en    = 0;

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // Test 1: Program metadata
        $display("[%0t] Test 1: Programming metadata for 3 phases", $time);
        program_metadata();

        // Test 2: Phase 0 - 3 LOADs (slots 0,1,2)
        $display("\n[%0t] Test 2: Executing Phase 0 (cold start - all loads)", $time);
        execute_phase(0, 0, 3);
        $display("  Loads: %0d, Moves: %0d, Keeps: %0d",
                 mem_stats.total_loads, mem_stats.total_moves, mem_stats.total_keeps);
        $display("  Bytes loaded: %0d, Bytes moved: %0d",
                 mem_stats.bytes_loaded, mem_stats.bytes_moved);

        // Test 3: Phase 1 - MOVE+KEEP+2xLOAD (slots 3,4,5,6)
        $display("\n[%0t] Test 3: Executing Phase 1 (data reuse with moves)", $time);
        execute_phase(1, 3, 4);
        $display("  Loads: %0d, Moves: %0d, Keeps: %0d",
                 mem_stats.total_loads, mem_stats.total_moves, mem_stats.total_keeps);
        $display("  Bytes loaded: %0d, Bytes moved: %0d",
                 mem_stats.bytes_loaded, mem_stats.bytes_moved);
        begin : savings_block
            automatic real savings_pct =
                (mem_stats.bytes_loaded + mem_stats.bytes_moved > 0) ?
                100.0 * mem_stats.bytes_moved /
                (mem_stats.bytes_loaded + mem_stats.bytes_moved) : 0.0;
            $display("  Bandwidth savings: %0.1f%%", savings_pct);
        end

        // Test 4: Phase 2 - 2xKEEP+2xMOVE+LOAD (slots 7..11)
        $display("\n[%0t] Test 4: Executing Phase 2 (maximum data reuse)", $time);
        execute_phase(2, 7, 5);
        $display("  Loads: %0d, Moves: %0d, Keeps: %0d",
                 mem_stats.total_loads, mem_stats.total_moves, mem_stats.total_keeps);
        $display("  Bytes loaded: %0d, Bytes moved: %0d",
                 mem_stats.bytes_loaded, mem_stats.bytes_moved);

        // Final stats
        $display("\n========================================");
        $display("Final Statistics");
        $display("========================================");
        $display("Total loads:        %0d", mem_stats.total_loads);
        $display("Total moves:        %0d", mem_stats.total_moves);
        $display("Total keeps:        %0d", mem_stats.total_keeps);
        $display("Total bytes loaded: %0d", mem_stats.bytes_loaded);
        $display("Total bytes moved:  %0d", mem_stats.bytes_moved);
        $display("Total cycles:       %0d", mem_stats.cycles_elapsed);
        $display("========================================\n");
        $display("TEST PASSED!");
        $finish;
    end

    //=========================================================================
    // program_metadata - pack as {8'b0, op[7:0], tid[15:0], src[31:0], dst[31:0], size[31:0]}
    // Total = 8+8+16+32+32+32 = 128 bits exactly
    //=========================================================================
    task program_metadata();
        metadata_wr_en = 1;

        // Phase 0 - all LOADs
        metadata_wr_addr = 0;
        metadata_wr_data = {8'd0,8'd2,16'd1,32'h0000_0000,32'h0000_0000,32'd1024};
        @(posedge clk);
        metadata_wr_addr = 1;
        metadata_wr_data = {8'd0,8'd2,16'd2,32'h0000_1000,32'h0000_0400,32'd2048};
        @(posedge clk);
        metadata_wr_addr = 2;
        metadata_wr_data = {8'd0,8'd2,16'd3,32'h0000_2000,32'h0000_0C00,32'd512};
        @(posedge clk);

        // Phase 1 - MOVE, KEEP, LOAD, LOAD
        metadata_wr_addr = 3;
        metadata_wr_data = {8'd0,8'd1,16'd1,32'h0000_0000,32'h0000_0200,32'd1024};
        @(posedge clk);
        metadata_wr_addr = 4;
        metadata_wr_data = {8'd0,8'd0,16'd2,32'h0000_0400,32'h0000_0400,32'd2048};
        @(posedge clk);
        metadata_wr_addr = 5;
        metadata_wr_data = {8'd0,8'd2,16'd4,32'h0000_3000,32'h0000_0C00,32'd512};
        @(posedge clk);
        metadata_wr_addr = 6;
        metadata_wr_data = {8'd0,8'd2,16'd5,32'h0000_4000,32'h0000_0E00,32'd256};
        @(posedge clk);

        // Phase 2 - KEEP, MOVE, MOVE, KEEP, LOAD
        metadata_wr_addr = 7;
        metadata_wr_data = {8'd0,8'd0,16'd1,32'h0000_0200,32'h0000_0200,32'd1024};
        @(posedge clk);
        metadata_wr_addr = 8;
        metadata_wr_data = {8'd0,8'd1,16'd2,32'h0000_0400,32'h0000_0600,32'd2048};
        @(posedge clk);
        metadata_wr_addr = 9;
        metadata_wr_data = {8'd0,8'd1,16'd4,32'h0000_0C00,32'h0000_0E00,32'd512};
        @(posedge clk);
        metadata_wr_addr = 10;
        metadata_wr_data = {8'd0,8'd0,16'd5,32'h0000_0E00,32'h0000_0E00,32'd256};
        @(posedge clk);
        metadata_wr_addr = 11;
        metadata_wr_data = {8'd0,8'd2,16'd6,32'h0000_5000,32'h0000_1000,32'd128};
        @(posedge clk);

        metadata_wr_en = 0;
        @(posedge clk);
        $display("  Metadata programmed for 3 phases");
    endtask

    //=========================================================================
    // execute_phase
    //=========================================================================
    task execute_phase(input int pid, input int base_addr, input int num_ops);
        @(posedge clk);
        while (!phase_ready) @(posedge clk);

        @(negedge clk);
        phase_base_addr = base_addr;
        phase_id        = pid;
        num_delta_ops   = num_ops;
        phase_start     = 1;
        //@(posedge clk);
        @(negedge clk);
        phase_start = 0;

        @(posedge clk);
        while (!phase_done) @(posedge clk);
        $display("  Phase %0d completed", pid);
    endtask

    //=========================================================================
    // Waveform dump
    //=========================================================================
    initial begin
        $dumpfile("stamp_memory.vcd");
        $dumpvars(0, stamp_based_memory_tb);
    end

    //=========================================================================
    // Timeout with debug
    //=========================================================================
    initial begin
        #(CLK_PERIOD * 1000000);
        $display("ERROR: Simulation timeout!");
        $display("DEBUG phase_ready=%b phase_done=%b controller_busy=%b",
                 phase_ready, phase_done, controller_busy);
        $display("DEBUG m_axi_arvalid=%b m_axi_arready=%b m_axi_arlen=%0d",
                 m_axi_arvalid, m_axi_arready, m_axi_arlen);
        $display("DEBUG m_axi_rvalid=%b m_axi_rready=%b m_axi_rlast=%b",
                 m_axi_rvalid, m_axi_rready, m_axi_rlast);
        $display("DEBUG read_counter=%0d burst_rd_addr=%0d",
                 read_counter, burst_rd_addr);
        $display("DEBUG loads=%0d bytes_loaded=%0d",
                 mem_stats.total_loads, mem_stats.bytes_loaded);
        $finish;
    end

endmodule : stamp_based_memory_tb