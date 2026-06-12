 `timescale 1ns/1ps
module systolic_array_stamp_top
    import systolic_array_pkg::*;
    import stamp_memory_pkg::*;
#(
    parameter int ARRAY_HEIGHT      = 4,
    parameter int ARRAY_WIDTH       = 4,
    parameter int DATA_WIDTH        = 32,
    parameter int ACCUM_WIDTH       = 48,
    parameter int ADDR_WIDTH        = 32,
    parameter int AXI_ID_WIDTH      = 4,
    parameter int AXI_DATA_WIDTH    = 64,
    parameter int SPAD_DEPTH        = 4096,
    parameter int MAX_CHANNELS      = 512,
    parameter int METADATA_DEPTH    = 256
) (
    input  logic clk,
    input  logic rst_n,

    input  mem_layout_t           mem_layout,
    input  logic [15:0]           input_channels,
    input  logic [15:0]           input_height,
    input  logic [15:0]           input_width,
    input  logic [15:0]           weight_k,
    input  logic [15:0]           weight_c,
    input  logic [15:0]           weight_kh,
    input  logic [15:0]           weight_kw,
    input  logic [15:0]           output_channels,
    input  logic [15:0]           output_height,
    input  logic [15:0]           output_width,
    input  logic [ADDR_WIDTH-1:0] input_base_addr,
    input  logic [ADDR_WIDTH-1:0] weight_base_addr,
    input  logic [ADDR_WIDTH-1:0] output_base_addr,

    input  nn_operation_config_t  nn_config,
    input  logic                  enable_activation,
    input  logic                  enable_pooling,
    input  logic                  enable_normalization,

    input  logic                              metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]                      metadata_wr_data,

    input  logic        phase_start,
    input  logic [15:0] phase_id,
    input  logic [15:0] num_delta_ops,
    input  logic [15:0] phase_base_addr,
    output logic        phase_ready,
    output logic        phase_done,

    input  logic [DATA_WIDTH-1:0] norm_gamma   [MAX_CHANNELS],
    input  logic [DATA_WIDTH-1:0] norm_beta    [MAX_CHANNELS],
    input  logic [DATA_WIDTH-1:0] running_mean [MAX_CHANNELS],
    input  logic [DATA_WIDTH-1:0] running_var  [MAX_CHANNELS],
    output logic [DATA_WIDTH-1:0] updated_mean [MAX_CHANNELS],
    output logic [DATA_WIDTH-1:0] updated_var  [MAX_CHANNELS],
    input  logic                  norm_params_valid,
    input  logic                  training_mode,

    input  logic        start,
    input  logic [15:0] tile_row,
    input  logic [15:0] tile_col_start,
    input  logic [15:0] tile_ch_start,
    output logic        done,
    output logic        busy,

    input  logic [$clog2(SPAD_DEPTH)-1:0] spad_load_addr,
    input  logic [DATA_WIDTH-1:0]          spad_load_data,
    input  logic                           spad_load_en,

    output memory_stats_t mem_stats,
    output logic          controller_busy,

    output logic [AXI_ID_WIDTH-1:0]     m_axi_arid,
    output logic [ADDR_WIDTH-1:0]       m_axi_araddr,
    output logic [7:0]                  m_axi_arlen,
    output logic [2:0]                  m_axi_arsize,
    output logic [1:0]                  m_axi_arburst,
    output logic                        m_axi_arlock,
    output logic [3:0]                  m_axi_arcache,
    output logic [2:0]                  m_axi_arprot,
    output logic                        m_axi_arvalid,
    input  logic                        m_axi_arready,

    input  logic [AXI_ID_WIDTH-1:0]     m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]   m_axi_rdata,
    input  logic [1:0]                  m_axi_rresp,
    input  logic                        m_axi_rlast,
    input  logic                        m_axi_rvalid,
    output logic                        m_axi_rready,

    output logic [AXI_ID_WIDTH-1:0]     m_axi_awid,
    output logic [ADDR_WIDTH-1:0]       m_axi_awaddr,
    output logic [7:0]                  m_axi_awlen,
    output logic [2:0]                  m_axi_awsize,
    output logic [1:0]                  m_axi_awburst,
    output logic                        m_axi_awlock,
    output logic [3:0]                  m_axi_awcache,
    output logic [2:0]                  m_axi_awprot,
    output logic                        m_axi_awvalid,
    input  logic                        m_axi_awready,

    output logic [AXI_DATA_WIDTH-1:0]   m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output logic                        m_axi_wlast,
    output logic                        m_axi_wvalid,
    input  logic                        m_axi_wready,

    input  logic [AXI_ID_WIDTH-1:0] m_axi_bid,
    input  logic [1:0]              m_axi_bresp,
    input  logic                    m_axi_bvalid,
    output logic                    m_axi_bready
);

    //=========================================================================
    // Internal wires
    //=========================================================================
    logic [$clog2(SPAD_DEPTH)-1:0] spad_rd_addr_ctrl, spad_rd_addr_comp;
    logic [DATA_WIDTH-1:0]         spad_rd_data;
    logic                          spad_rd_en_ctrl, spad_rd_en_comp;
    logic [$clog2(SPAD_DEPTH)-1:0] spad_wr_addr_ctrl, spad_wr_addr_comp;
    logic [DATA_WIDTH-1:0]         spad_wr_data_ctrl, spad_wr_data_comp;
    logic                          spad_wr_en_ctrl, spad_wr_en_comp;

    logic [ADDR_WIDTH-1:0] mem_rd_addr;
    logic [15:0]           mem_rd_len;
    logic                  mem_rd_req;
    logic                  mem_rd_ready;
    logic [DATA_WIDTH-1:0] mem_rd_data;
    logic                  mem_rd_valid;

    logic compute_done, compute_busy;

    //=========================================================================
    // Scratchpad - combinational read
    //=========================================================================
    logic [DATA_WIDTH-1:0] scratchpad [SPAD_DEPTH];

    always_ff @(posedge clk) begin
        if (spad_load_en)
            scratchpad[spad_load_addr] <= spad_load_data;
        else if (spad_wr_en_ctrl)
            scratchpad[spad_wr_addr_ctrl] <= spad_wr_data_ctrl;
        if (spad_wr_en_comp)
            scratchpad[spad_wr_addr_comp] <= spad_wr_data_comp;
    end

    assign spad_rd_data = spad_rd_en_ctrl ? scratchpad[spad_rd_addr_ctrl] :
                          spad_rd_en_comp ? scratchpad[spad_rd_addr_comp] :
                          '0;

    //=========================================================================
    // Memory controller instance
    //=========================================================================
    stamp_based_memory_controller #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH),
        .SPAD_DEPTH     (SPAD_DEPTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .METADATA_DEPTH (METADATA_DEPTH)
    ) u_ctrl (
        .clk              (clk),
        .rst_n            (rst_n),
        .metadata_wr_en   (metadata_wr_en),
        .metadata_wr_addr (metadata_wr_addr),
        .metadata_wr_data (metadata_wr_data),
        .phase_start      (phase_start),
        .phase_id         (phase_id),
        .num_delta_ops    (num_delta_ops),
        .phase_base_addr  (phase_base_addr),
        .phase_ready      (phase_ready),
        .phase_done       (phase_done),
        .spad_rd_addr     (spad_rd_addr_ctrl),
        .spad_rd_data     (spad_rd_data),
        .spad_rd_en       (spad_rd_en_ctrl),
        .spad_wr_addr     (spad_wr_addr_ctrl),
        .spad_wr_data     (spad_wr_data_ctrl),
        .spad_wr_en       (spad_wr_en_ctrl),
        .mem_rd_addr      (mem_rd_addr),
        .mem_rd_len       (mem_rd_len),
        .mem_rd_req       (mem_rd_req),
        .mem_rd_ready     (mem_rd_ready),
        .mem_rd_data      (mem_rd_data),
        .mem_rd_valid     (mem_rd_valid),
        .stats_loads      (mem_stats.total_loads),
        .stats_moves      (mem_stats.total_moves),
        .stats_keeps      (mem_stats.total_keeps),
        .stats_bytes_loaded (mem_stats.bytes_loaded),
        .stats_bytes_moved  (mem_stats.bytes_moved),
        .controller_busy  (controller_busy)
    );

    //=========================================================================
    // AXI Read Adapter
    //
    // mem_rd_ready is COMBINATIONAL: asserted when AXI_IDLE and mem_rd_req=1.
    // Both controller and AXI adapter see it simultaneously.
    // Controller: transitions LOAD_REQ → LOAD_RECV on this cycle.
    // AXI adapter: latches addr/len and transitions AXI_IDLE → AXI_ADDR
    //              on the same posedge. No deadlock - both advance together.
    //=========================================================================
// AXI Read Adapter
//=========================================================================
typedef enum logic [1:0] {
    AXI_IDLE,
    AXI_ADDR,
    AXI_DATA
} axi_state_t;

axi_state_t axi_state;

logic [ADDR_WIDTH-1:0] rd_addr_reg;
logic [15:0]           rd_len_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        axi_state  <= AXI_IDLE;
        rd_addr_reg <= '0;
        rd_len_reg  <= '0;
    end
    else begin
        case (axi_state)

            //-------------------------------------------------------------
            // Wait for controller request
            //-------------------------------------------------------------
            AXI_IDLE: begin
                if (mem_rd_req) begin
                    rd_addr_reg <= mem_rd_addr;
                    rd_len_reg  <= mem_rd_len;
                    axi_state   <= AXI_ADDR;
                end
            end

            //-------------------------------------------------------------
            // Send AR channel
            //-------------------------------------------------------------
            AXI_ADDR: begin
                if (m_axi_arvalid && m_axi_arready) begin
                    axi_state <= AXI_DATA;
                end
            end

            //-------------------------------------------------------------
            // Receive R channel
            //-------------------------------------------------------------
            AXI_DATA: begin
                if (m_axi_rvalid && m_axi_rready && m_axi_rlast) begin
                    axi_state <= AXI_IDLE;
                end
            end

        endcase
    end
end

//-------------------------------------------------------------
// Controller handshake
//-------------------------------------------------------------
assign mem_rd_ready = (axi_state == AXI_IDLE);

//-------------------------------------------------------------
// AXI AR channel
//-------------------------------------------------------------
assign m_axi_arvalid = (axi_state == AXI_ADDR);

assign m_axi_araddr  = rd_addr_reg;

assign m_axi_arlen =
    (rd_len_reg == 0) ? 8'd0 :
    (rd_len_reg - 1);

assign m_axi_arsize  = 3'b010; // 4 bytes
assign m_axi_arburst = 2'b01;
assign m_axi_arlock  = 1'b0;
assign m_axi_arcache = 4'b0011;
assign m_axi_arprot  = 3'b000;
assign m_axi_arid    = '0;

//-------------------------------------------------------------
// AXI R channel
//-------------------------------------------------------------
assign m_axi_rready = (axi_state == AXI_DATA);

assign mem_rd_valid = m_axi_rvalid;
assign mem_rd_data  = m_axi_rdata[31:0];


//-------------------------------------------------------------
// ACK to controller
//-------------------------------------------------------------


    // Write channel unused
    assign m_axi_awvalid = 1'b0; assign m_axi_awaddr  = '0;
    assign m_axi_awlen   = '0;   assign m_axi_awsize  = '0;
    assign m_axi_awburst = '0;   assign m_axi_awlock  = '0;
    assign m_axi_awcache = '0;   assign m_axi_awprot  = '0;
    assign m_axi_awid    = '0;   assign m_axi_wvalid  = 1'b0;
    assign m_axi_wdata   = '0;   assign m_axi_wstrb   = '0;
    assign m_axi_wlast   = 1'b0; assign m_axi_bready  = 1'b1;

    //=========================================================================
    // Placeholder compute engine
    //=========================================================================
    logic        compute_start;
    logic [15:0] compute_cycles;

    assign compute_start = start && phase_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_busy   <= 1'b0;
            compute_done   <= 1'b0;
            compute_cycles <= 0;
        end else begin
            if (compute_start) begin
                compute_busy   <= 1'b1;
                compute_done   <= 1'b0;
                compute_cycles <= 16'd100;
            end else if (compute_busy) begin
                if (compute_cycles > 0)
                    compute_cycles <= compute_cycles - 1;
                else begin
                    compute_busy <= 1'b0;
                    compute_done <= 1'b1;
                end
            end else
                compute_done <= 1'b0;
        end
    end

    assign spad_rd_en_comp   = compute_busy;
    assign spad_rd_addr_comp = compute_cycles[$clog2(SPAD_DEPTH)-1:0];
    assign spad_wr_en_comp   = 1'b0;
    assign spad_wr_addr_comp = '0;
    assign spad_wr_data_comp = '0;

    //=========================================================================
    // Outputs
    //=========================================================================
    assign done = compute_done && phase_done;
    assign busy = compute_busy || controller_busy;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mem_stats.cycles_elapsed <= 0;
        else if (busy) mem_stats.cycles_elapsed <= mem_stats.cycles_elapsed + 1;
    end

endmodule : systolic_array_stamp_top