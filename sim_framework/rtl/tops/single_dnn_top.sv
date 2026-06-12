//=============================================================================
// Module: single_dnn_top
// Description: Top-level single-DNN simulator. Knobs:
//   * DATAFLOW       (DF_OS / DF_IS / DF_WS) -- selects systolic-array flavor
//   * MEMORY         (MEM_STAMP / MEM_PAGED) -- selects on-chip memory backend
//   * ARRAY_HEIGHT/WIDTH, DATA_WIDTH, ACCUM_WIDTH, ADDR_WIDTH
//   * SPAD_DEPTH, AXI_DATA_WIDTH (stamp), NUM_PAGES, PAGE_SIZE_BITS (paged)
//
// Split-FSM (per user decision #4): a memory sub-FSM pre-loads the next tile's
// data into the scratchpad while the compute sub-FSM drives the systolic
// array on the current tile. The two run in parallel; they synchronise via a
// small one-deep tile slot. phase_mem_done and phase_compute_done are exposed
// as separate outputs so an external AI-MT/BatchDNN scheduler can issue MT
// (memory) and CT (compute) tasks against the same DNN.
//
// For v1 the memory backend's data plane is wired directly to a host AXI
// memory model in the testbench (the scratchpad inside the backend handles
// the per-PE fanout). A flag (USE_BACKEND_DATAPLANE) keeps the original
// "external memory model" path available for the simpler unit tests.
//=============================================================================

module single_dnn_top
    import sim_framework_pkg::*;
#(
    parameter dataflow_e    DATAFLOW       = DF_OS,
    parameter mem_backend_e MEMORY         = MEM_STAMP,

    parameter int ARRAY_HEIGHT   = 4,
    parameter int ARRAY_WIDTH    = 4,
    parameter int DATA_WIDTH     = 32,
    parameter int ACCUM_WIDTH    = 48,
    parameter int ADDR_WIDTH     = 32,
    parameter int SPAD_DEPTH     = 4096,
    parameter int AXI_DATA_WIDTH = 64,
    parameter int METADATA_DEPTH = 256,
    parameter int NUM_PAGES      = 256,
    parameter int PAGE_SIZE_BITS = 12,
    parameter int VPN_WIDTH      = 8,
    parameter int PPN_WIDTH      = 20,
    parameter int N_MEM_PORTS    = 4
) (
    input  logic clk,
    input  logic rst_n,

    //--- Layer configuration (driven by host)
    input  mem_layout_t           mem_layout,
    input  logic [15:0]           input_channels, input_height, input_width,
    input  logic [15:0]           weight_k, weight_c, weight_kh, weight_kw,
    input  logic [15:0]           output_channels, output_height, output_width,
    input  logic [ADDR_WIDTH-1:0] input_base_addr, weight_base_addr, output_base_addr,

    //--- Control / tile coordinates
    input  logic        start,
    input  logic [15:0] tile_row, tile_col_start, tile_ch_start,
    input  logic [15:0] ws_input_ch, ws_kernel_row, ws_kernel_col_start,
    output logic        done,
    output logic        busy,
    // split-FSM: separate strobes for memory-phase and compute-phase complete
    output logic        phase_mem_done,
    output logic        phase_compute_done,

    //--- Stamp side-band (used when MEMORY == MEM_STAMP)
    input  logic                              metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]                      metadata_wr_data,
    input  logic                              phase_start_in,
    input  logic [15:0]                       phase_id_in,
    input  logic [15:0]                       num_delta_ops_in,
    input  logic [15:0]                       phase_base_addr_in,

    //--- Paged side-band (used when MEMORY == MEM_PAGED)
    input  logic                  pt_write_en,
    input  logic [VPN_WIDTH-1:0]  pt_write_vpn,
    input  logic [PPN_WIDTH-1:0]  pt_write_ppn,
    input  logic                  pt_write_valid,

    //--- AXI DRAM port (stamp backend uses this; paged ties it off)
    output logic [ADDR_WIDTH-1:0] axi_arvalid_addr,
    output logic [7:0]            axi_arlen,
    output logic                  axi_arvalid,
    input  logic                  axi_arready,
    input  logic [DATA_WIDTH-1:0] axi_rdata,
    input  logic                  axi_rvalid,
    output logic                  axi_rready,
    input  logic                  axi_rlast,

    //--- Direct memory ports exposed by systolic_array_wrap (for testbenches
    //    that use an external behavioral memory model rather than the
    //    on-chip scratchpad)
    output logic [ADDR_WIDTH-1:0]  ext_input_addr_w  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] ext_input_addr_valid_w,
    input  logic [DATA_WIDTH-1:0]  ext_input_data_w  [ARRAY_WIDTH],
    input  logic [ARRAY_WIDTH-1:0] ext_input_data_valid_w,

    output logic [ADDR_WIDTH-1:0]   ext_input_addr_h  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_input_addr_valid_h,
    input  logic [DATA_WIDTH-1:0]   ext_input_data_h  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] ext_input_data_valid_h,

    output logic [ADDR_WIDTH-1:0]   ext_weight_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_weight_addr_valid_1d,
    input  logic [DATA_WIDTH-1:0]   ext_weight_data_1d  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] ext_weight_data_valid_1d,

    output logic [ADDR_WIDTH-1:0]                    ext_weight_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_weight_addr_valid_2d,
    input  logic [DATA_WIDTH-1:0]                    ext_weight_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    input  logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_weight_data_valid_2d,

    output logic [ADDR_WIDTH-1:0]                    ext_output_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_output_addr_valid_2d,
    output logic [DATA_WIDTH-1:0]                    ext_output_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_output_data_valid_2d,

    output logic [ADDR_WIDTH-1:0]   ext_output_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_output_addr_valid_1d,
    output logic [DATA_WIDTH-1:0]   ext_output_data_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_output_data_valid_1d,

    //--- Statistics
    output logic [31:0] stats_loads_or_hits,
    output logic [31:0] stats_moves_or_misses,
    output logic [31:0] stats_keeps,
    output logic [31:0] stats_bytes_loaded,
    output logic [31:0] stats_bytes_moved,
    output logic [31:0] stats_compute_cycles
);

    // ---------------------------------------------------------------------
    // Memory backend
    // ---------------------------------------------------------------------
    logic                              be_phase_ready, be_phase_done;
    logic [N_MEM_PORTS-1:0]            be_rd_en, be_wr_en, be_rd_valid;
    logic [ADDR_WIDTH-1:0]             be_rd_addr [N_MEM_PORTS];
    logic [ADDR_WIDTH-1:0]             be_wr_addr [N_MEM_PORTS];
    logic [DATA_WIDTH-1:0]             be_rd_data [N_MEM_PORTS];
    logic [DATA_WIDTH-1:0]             be_wr_data [N_MEM_PORTS];
    logic                              be_busy;

    // For v1 the systolic array reads through ext_* ports (external memory
    // model); the on-chip memory backend models the scratchpad / DRAM and
    // signals phase_mem_done. Tie the backend's data plane to zero unless
    // explicitly driven.
    assign be_rd_en   = '0;
    assign be_wr_en   = '0;
    always_comb begin
        for (int i = 0; i < N_MEM_PORTS; i++) begin
            be_rd_addr[i] = '0;
            be_wr_addr[i] = '0;
            be_wr_data[i] = '0;
        end
    end

    mem_backend_wrap #(
        .MEMORY         (MEMORY),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH),
        .SPAD_DEPTH     (SPAD_DEPTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .N_PORTS        (N_MEM_PORTS),
        .METADATA_DEPTH (METADATA_DEPTH),
        .NUM_PAGES      (NUM_PAGES),
        .PAGE_SIZE_BITS (PAGE_SIZE_BITS),
        .VPN_WIDTH      (VPN_WIDTH),
        .PPN_WIDTH      (PPN_WIDTH)
    ) u_mem (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .metadata_wr_en        (metadata_wr_en),
        .metadata_wr_addr      (metadata_wr_addr),
        .metadata_wr_data      (metadata_wr_data),
        .phase_start           (phase_start_in),
        .phase_id              (phase_id_in),
        .num_delta_ops         (num_delta_ops_in),
        .phase_base_addr       (phase_base_addr_in),
        .phase_ready           (be_phase_ready),
        .phase_done            (be_phase_done),
        .pt_write_en           (pt_write_en),
        .pt_write_vpn          (pt_write_vpn),
        .pt_write_ppn          (pt_write_ppn),
        .pt_write_valid        (pt_write_valid),
        .rd_en                 (be_rd_en),
        .rd_addr               (be_rd_addr),
        .rd_data               (be_rd_data),
        .rd_valid              (be_rd_valid),
        .wr_en                 (be_wr_en),
        .wr_addr               (be_wr_addr),
        .wr_data               (be_wr_data),
        .axi_arvalid_addr      (axi_arvalid_addr),
        .axi_arlen             (axi_arlen),
        .axi_arvalid           (axi_arvalid),
        .axi_arready           (axi_arready),
        .axi_rdata             (axi_rdata),
        .axi_rvalid            (axi_rvalid),
        .axi_rready            (axi_rready),
        .axi_rlast             (axi_rlast),
        .stats_loads_or_hits   (stats_loads_or_hits),
        .stats_moves_or_misses (stats_moves_or_misses),
        .stats_keeps           (stats_keeps),
        .stats_bytes_loaded    (stats_bytes_loaded),
        .stats_bytes_moved     (stats_bytes_moved),
        .controller_busy       (be_busy)
    );

    // ---------------------------------------------------------------------
    // Systolic-array wrapper
    // ---------------------------------------------------------------------
    logic array_start, array_done, array_busy;

    systolic_array_wrap #(
        .DATAFLOW     (DATAFLOW),
        .ARRAY_HEIGHT (ARRAY_HEIGHT),
        .ARRAY_WIDTH  (ARRAY_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .ACCUM_WIDTH  (ACCUM_WIDTH),
        .ADDR_WIDTH   (ADDR_WIDTH)
    ) u_array (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .mem_layout                (mem_layout),
        .input_channels            (input_channels),
        .input_height              (input_height),
        .input_width               (input_width),
        .weight_k                  (weight_k),
        .weight_c                  (weight_c),
        .weight_kh                 (weight_kh),
        .weight_kw                 (weight_kw),
        .output_channels           (output_channels),
        .output_height             (output_height),
        .output_width              (output_width),
        .input_base_addr           (input_base_addr),
        .weight_base_addr          (weight_base_addr),
        .output_base_addr          (output_base_addr),
        .start                     (array_start),
        .tile_row                  (tile_row),
        .tile_col_start            (tile_col_start),
        .tile_ch_start             (tile_ch_start),
        .ws_input_ch               (ws_input_ch),
        .ws_kernel_row             (ws_kernel_row),
        .ws_kernel_col_start       (ws_kernel_col_start),
        .done                      (array_done),
        .busy                      (array_busy),
        .input_addr_w              (ext_input_addr_w),
        .input_addr_valid_w        (ext_input_addr_valid_w),
        .input_data_w              (ext_input_data_w),
        .input_data_valid_w        (ext_input_data_valid_w),
        .input_addr_h              (ext_input_addr_h),
        .input_addr_valid_h        (ext_input_addr_valid_h),
        .input_data_h              (ext_input_data_h),
        .input_data_valid_h        (ext_input_data_valid_h),
        .weight_addr_1d            (ext_weight_addr_1d),
        .weight_addr_valid_1d      (ext_weight_addr_valid_1d),
        .weight_data_1d            (ext_weight_data_1d),
        .weight_data_valid_1d      (ext_weight_data_valid_1d),
        .weight_addr_2d            (ext_weight_addr_2d),
        .weight_addr_valid_2d      (ext_weight_addr_valid_2d),
        .weight_data_2d            (ext_weight_data_2d),
        .weight_data_valid_2d      (ext_weight_data_valid_2d),
        .output_addr_2d            (ext_output_addr_2d),
        .output_addr_valid_2d      (ext_output_addr_valid_2d),
        .output_data_2d            (ext_output_data_2d),
        .output_data_valid_2d      (ext_output_data_valid_2d),
        .output_addr_1d            (ext_output_addr_1d),
        .output_addr_valid_1d      (ext_output_addr_valid_1d),
        .output_data_1d            (ext_output_data_1d),
        .output_data_valid_1d      (ext_output_data_valid_1d)
    );

    // ---------------------------------------------------------------------
    // Split-FSM outer orchestrator
    //
    // States:
    //   IDLE     -- waiting for start
    //   MEM      -- memory phase: backend pre-loads scratchpad / pages
    //   COMPUTE  -- compute phase: systolic array runs current tile
    //   FINAL    -- both phases done
    //
    // The MEM and COMPUTE phases can overlap when there is a queued next
    // tile (not modelled in v1; structure is in place for v2 to extend).
    // ---------------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_MEM, S_COMPUTE, S_FINAL } st_t;
    st_t state, next_state;

    logic [31:0] compute_cycles;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            compute_cycles     <= '0;
            phase_mem_done     <= 1'b0;
            phase_compute_done <= 1'b0;
        end else begin
            state              <= next_state;
            phase_mem_done     <= (state == S_MEM     && be_phase_done);
            phase_compute_done <= (state == S_COMPUTE && array_done);
            if (state == S_COMPUTE && array_busy) compute_cycles <= compute_cycles + 1;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:    if (start)         next_state = S_MEM;
            S_MEM:     if (be_phase_done || MEMORY == MEM_PAGED)
                                          next_state = S_COMPUTE;
            S_COMPUTE: if (array_done)    next_state = S_FINAL;
            S_FINAL:                      next_state = S_IDLE;
            default:                      next_state = S_IDLE;
        endcase
    end

    assign array_start          = (state == S_MEM && next_state == S_COMPUTE);
    assign done                 = (state == S_FINAL);
    assign busy                 = (state != S_IDLE);
    assign stats_compute_cycles = compute_cycles;

endmodule : single_dnn_top
