//=============================================================================
// Module: stamp_memory_backend
// Description: Stamp-based on-chip memory backend that presents the canonical
//              N-port data-plane interface (mem_if-shaped, but expanded here
//              to plain signals to avoid Verilator-interface-in-port limits).
//
// Wraps:
//   * stamp_based_memory_controller (delta-op-driven scratchpad pre-loader)
//   * scratchpad_ram                 (multi-port BRAM-style SRAM)
//   * axi_read_adapter               (single-outstanding AXI bridge to host
//                                     memory model)
//
// Side-band ports (metadata write, phase start, stats) stay at the top of the
// module so that single_dnn_top can mux them with the paged backend's
// side-band ports based on the MEMORY parameter.
//=============================================================================

module stamp_memory_backend #(
    parameter int ADDR_WIDTH     = 32,
    parameter int DATA_WIDTH     = 32,
    parameter int SPAD_DEPTH     = 4096,
    parameter int AXI_DATA_WIDTH = 64,
    parameter int N_PORTS        = 4,
    parameter int METADATA_DEPTH = 256
) (
    input  logic clk,
    input  logic rst_n,

    //--- side-band: metadata write port (host loads the delta-op table)
    input  logic                              metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]                      metadata_wr_data,

    //--- side-band: phase control
    input  logic        phase_start,
    input  logic [15:0] phase_id,
    input  logic [15:0] num_delta_ops,
    input  logic [15:0] phase_base_addr,
    output logic        phase_ready,
    output logic        phase_done,

    //--- canonical data plane: read ports from the systolic-array wrapper
    input  logic [N_PORTS-1:0]                rd_en,
    input  logic [$clog2(SPAD_DEPTH)-1:0]     rd_addr  [N_PORTS],
    output logic [DATA_WIDTH-1:0]             rd_data  [N_PORTS],
    output logic [N_PORTS-1:0]                rd_valid,

    //--- canonical data plane: write ports for writeback
    input  logic [N_PORTS-1:0]                wr_en,
    input  logic [$clog2(SPAD_DEPTH)-1:0]     wr_addr  [N_PORTS],
    input  logic [DATA_WIDTH-1:0]             wr_data  [N_PORTS],

    //--- downstream AXI to host DRAM model
    output logic [ADDR_WIDTH-1:0] axi_arvalid_addr,
    output logic [7:0]            axi_arlen,
    output logic                  axi_arvalid,
    input  logic                  axi_arready,
    input  logic [DATA_WIDTH-1:0] axi_rdata,
    input  logic                  axi_rvalid,
    output logic                  axi_rready,
    input  logic                  axi_rlast,

    //--- statistics
    output logic [31:0] stats_loads,
    output logic [31:0] stats_moves,
    output logic [31:0] stats_keeps,
    output logic [31:0] stats_bytes_loaded,
    output logic [31:0] stats_bytes_moved,
    output logic        controller_busy
);

    // The stamp controller drives a dedicated read/write port (port 0) on
    // the scratchpad while loading/moving tiles. The systolic array uses
    // the remaining ports 1..N_PORTS-1 for runtime data reads/writes.

    logic [$clog2(SPAD_DEPTH)-1:0] sc_spad_rd_addr;
    logic [DATA_WIDTH-1:0]         sc_spad_rd_data;
    logic                          sc_spad_rd_en;

    logic [$clog2(SPAD_DEPTH)-1:0] sc_spad_wr_addr;
    logic [DATA_WIDTH-1:0]         sc_spad_wr_data;
    logic                          sc_spad_wr_en;

    logic [ADDR_WIDTH-1:0] mem_rd_addr_int;
    logic [15:0]           mem_rd_len_int;
    logic                  mem_rd_req_int, mem_rd_ready_int;
    logic [DATA_WIDTH-1:0] mem_rd_data_int;
    logic                  mem_rd_valid_int;

    stamp_based_memory_controller #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH),
        .SPAD_DEPTH     (SPAD_DEPTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .METADATA_DEPTH (METADATA_DEPTH)
    ) u_stamp_ctrl (
        .clk                (clk),
        .rst_n              (rst_n),
        .metadata_wr_en     (metadata_wr_en),
        .metadata_wr_addr   (metadata_wr_addr),
        .metadata_wr_data   (metadata_wr_data),
        .phase_start        (phase_start),
        .phase_id           (phase_id),
        .num_delta_ops      (num_delta_ops),
        .phase_base_addr    (phase_base_addr),
        .phase_ready        (phase_ready),
        .phase_done         (phase_done),
        .spad_rd_addr       (sc_spad_rd_addr),
        .spad_rd_data       (sc_spad_rd_data),
        .spad_rd_en         (sc_spad_rd_en),
        .spad_wr_addr       (sc_spad_wr_addr),
        .spad_wr_data       (sc_spad_wr_data),
        .spad_wr_en         (sc_spad_wr_en),
        .mem_rd_addr        (mem_rd_addr_int),
        .mem_rd_len         (mem_rd_len_int),
        .mem_rd_req         (mem_rd_req_int),
        .mem_rd_ready       (mem_rd_ready_int),
        .mem_rd_data        (mem_rd_data_int),
        .mem_rd_valid       (mem_rd_valid_int),
        .stats_loads        (stats_loads),
        .stats_moves        (stats_moves),
        .stats_keeps        (stats_keeps),
        .stats_bytes_loaded (stats_bytes_loaded),
        .stats_bytes_moved  (stats_bytes_moved),
        .controller_busy    (controller_busy)
    );

    axi_read_adapter #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_axi_read (
        .clk              (clk),
        .rst_n            (rst_n),
        .up_rd_addr       (mem_rd_addr_int),
        .up_rd_len        (mem_rd_len_int),
        .up_rd_req        (mem_rd_req_int),
        .up_rd_ready      (mem_rd_ready_int),
        .up_rd_data       (mem_rd_data_int),
        .up_rd_valid      (mem_rd_valid_int),
        .axi_arvalid_addr (axi_arvalid_addr),
        .axi_arlen        (axi_arlen),
        .axi_arvalid      (axi_arvalid),
        .axi_arready      (axi_arready),
        .axi_rdata        (axi_rdata),
        .axi_rvalid       (axi_rvalid),
        .axi_rready       (axi_rready),
        .axi_rlast        (axi_rlast)
    );

    //--- scratchpad with N_PORTS+1 ports (extra port reserved for the stamp ctrl)
    logic [N_PORTS:0]                       spad_rd_en_arr;
    logic [$clog2(SPAD_DEPTH)-1:0]          spad_rd_addr_arr [N_PORTS+1];
    logic [DATA_WIDTH-1:0]                  spad_rd_data_arr [N_PORTS+1];
    logic [N_PORTS:0]                       spad_rd_valid_arr;
    logic [N_PORTS:0]                       spad_wr_en_arr;
    logic [$clog2(SPAD_DEPTH)-1:0]          spad_wr_addr_arr [N_PORTS+1];
    logic [DATA_WIDTH-1:0]                  spad_wr_data_arr [N_PORTS+1];

    // Port 0 is the stamp controller; ports 1..N are the wrapper's ports.
    always_comb begin
        spad_rd_en_arr[0]    = sc_spad_rd_en;
        spad_rd_addr_arr[0]  = sc_spad_rd_addr;
        spad_wr_en_arr[0]    = sc_spad_wr_en;
        spad_wr_addr_arr[0]  = sc_spad_wr_addr;
        spad_wr_data_arr[0]  = sc_spad_wr_data;
        for (int i = 0; i < N_PORTS; i++) begin
            spad_rd_en_arr[i+1]   = rd_en[i];
            spad_rd_addr_arr[i+1] = rd_addr[i];
            spad_wr_en_arr[i+1]   = wr_en[i];
            spad_wr_addr_arr[i+1] = wr_addr[i];
            spad_wr_data_arr[i+1] = wr_data[i];
        end
    end

    scratchpad_ram #(
        .DATA_WIDTH (DATA_WIDTH),
        .DEPTH      (SPAD_DEPTH),
        .N_PORTS    (N_PORTS + 1)
    ) u_spad (
        .clk      (clk),
        .rst_n    (rst_n),
        .rd_en    (spad_rd_en_arr),
        .rd_addr  (spad_rd_addr_arr),
        .rd_data  (spad_rd_data_arr),
        .rd_valid (spad_rd_valid_arr),
        .wr_en    (spad_wr_en_arr),
        .wr_addr  (spad_wr_addr_arr),
        .wr_data  (spad_wr_data_arr)
    );

    // Route port-0 results back to the stamp controller and the rest back to
    // the wrapper.
    assign sc_spad_rd_data = spad_rd_data_arr[0];

    always_comb begin
        for (int i = 0; i < N_PORTS; i++) begin
            rd_data[i]  = spad_rd_data_arr[i+1];
            rd_valid[i] = spad_rd_valid_arr[i+1];
        end
    end

endmodule : stamp_memory_backend
