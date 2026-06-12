//=============================================================================
// Module: mem_backend_wrap
// Description: Compile-time mux between the stamp and paged memory backends.
//              The MEMORY parameter (MEM_STAMP or MEM_PAGED) picks the
//              backend; the unused branch is dead-code-eliminated by Verilator
//              and synth tools.
//
// Presents the union of both backends' side-band ports at the top. The
// caller (single_dnn_top) drives only the relevant subset; the other one is
// tied off.
//=============================================================================

module mem_backend_wrap
    import sim_framework_pkg::*;
#(
    parameter mem_backend_e MEMORY        = MEM_STAMP,
    parameter int           ADDR_WIDTH    = 32,
    parameter int           DATA_WIDTH    = 32,
    parameter int           SPAD_DEPTH    = 4096,
    parameter int           AXI_DATA_WIDTH = 64,
    parameter int           N_PORTS       = 4,
    parameter int           METADATA_DEPTH = 256,
    parameter int           NUM_PAGES     = 256,
    parameter int           PAGE_SIZE_BITS = 12,
    parameter int           VPN_WIDTH     = 8,
    parameter int           PPN_WIDTH     = 20
) (
    input  logic clk,
    input  logic rst_n,

    //--- side-band: stamp metadata + phase ports (used when MEMORY==MEM_STAMP)
    input  logic                              metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]                      metadata_wr_data,
    input  logic                              phase_start,
    input  logic [15:0]                       phase_id,
    input  logic [15:0]                       num_delta_ops,
    input  logic [15:0]                       phase_base_addr,
    output logic                              phase_ready,
    output logic                              phase_done,

    //--- side-band: paged page-table update (used when MEMORY==MEM_PAGED)
    input  logic                   pt_write_en,
    input  logic [VPN_WIDTH-1:0]   pt_write_vpn,
    input  logic [PPN_WIDTH-1:0]   pt_write_ppn,
    input  logic                   pt_write_valid,

    //--- canonical data plane (shared by both backends)
    input  logic [N_PORTS-1:0]                rd_en,
    input  logic [ADDR_WIDTH-1:0]             rd_addr  [N_PORTS],
    output logic [DATA_WIDTH-1:0]             rd_data  [N_PORTS],
    output logic [N_PORTS-1:0]                rd_valid,
    input  logic [N_PORTS-1:0]                wr_en,
    input  logic [ADDR_WIDTH-1:0]             wr_addr  [N_PORTS],
    input  logic [DATA_WIDTH-1:0]             wr_data  [N_PORTS],

    //--- AXI DRAM port (stamp only; paged version ties off)
    output logic [ADDR_WIDTH-1:0] axi_arvalid_addr,
    output logic [7:0]            axi_arlen,
    output logic                  axi_arvalid,
    input  logic                  axi_arready,
    input  logic [DATA_WIDTH-1:0] axi_rdata,
    input  logic                  axi_rvalid,
    output logic                  axi_rready,
    input  logic                  axi_rlast,

    //--- stats (stamp's set; paged uses page_hits/misses but reuses fields)
    output logic [31:0] stats_loads_or_hits,
    output logic [31:0] stats_moves_or_misses,
    output logic [31:0] stats_keeps,
    output logic [31:0] stats_bytes_loaded,
    output logic [31:0] stats_bytes_moved,
    output logic        controller_busy
);

    generate
        if (MEMORY == MEM_STAMP) begin : g_stamp
            // Scratchpad-relative addresses come pre-narrowed from the wrapper.
            logic [$clog2(SPAD_DEPTH)-1:0] s_rd_addr [N_PORTS];
            logic [$clog2(SPAD_DEPTH)-1:0] s_wr_addr [N_PORTS];
            always_comb begin
                for (int i = 0; i < N_PORTS; i++) begin
                    s_rd_addr[i] = rd_addr[i][$clog2(SPAD_DEPTH)-1:0];
                    s_wr_addr[i] = wr_addr[i][$clog2(SPAD_DEPTH)-1:0];
                end
            end

            stamp_memory_backend #(
                .ADDR_WIDTH     (ADDR_WIDTH),
                .DATA_WIDTH     (DATA_WIDTH),
                .SPAD_DEPTH     (SPAD_DEPTH),
                .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
                .N_PORTS        (N_PORTS),
                .METADATA_DEPTH (METADATA_DEPTH)
            ) u_stamp (
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
                .rd_en              (rd_en),
                .rd_addr            (s_rd_addr),
                .rd_data            (rd_data),
                .rd_valid           (rd_valid),
                .wr_en              (wr_en),
                .wr_addr            (s_wr_addr),
                .wr_data            (wr_data),
                .axi_arvalid_addr   (axi_arvalid_addr),
                .axi_arlen          (axi_arlen),
                .axi_arvalid        (axi_arvalid),
                .axi_arready        (axi_arready),
                .axi_rdata          (axi_rdata),
                .axi_rvalid         (axi_rvalid),
                .axi_rready         (axi_rready),
                .axi_rlast          (axi_rlast),
                .stats_loads        (stats_loads_or_hits),
                .stats_moves        (stats_moves_or_misses),
                .stats_keeps        (stats_keeps),
                .stats_bytes_loaded (stats_bytes_loaded),
                .stats_bytes_moved  (stats_bytes_moved),
                .controller_busy    (controller_busy)
            );
        end else begin : g_paged
            // Paged backend: no phase/metadata, no AXI -- tie off.
            assign phase_ready         = 1'b1;
            assign phase_done          = 1'b0;
            assign axi_arvalid_addr    = '0;
            assign axi_arlen           = '0;
            assign axi_arvalid         = 1'b0;
            assign axi_rready          = 1'b0;
            assign stats_keeps         = '0;
            assign stats_bytes_loaded  = '0;
            assign stats_bytes_moved   = '0;
            assign controller_busy     = 1'b0;

            paged_memory_backend #(
                .ADDR_WIDTH     (ADDR_WIDTH),
                .DATA_WIDTH     (DATA_WIDTH),
                .SPAD_DEPTH     (SPAD_DEPTH),
                .N_PORTS        (N_PORTS),
                .VADDR_WIDTH    (ADDR_WIDTH),
                .PADDR_WIDTH    (ADDR_WIDTH),
                .PAGE_SIZE_BITS (PAGE_SIZE_BITS),
                .NUM_PAGES      (NUM_PAGES),
                .VPN_WIDTH      (VPN_WIDTH),
                .PPN_WIDTH      (PPN_WIDTH)
            ) u_paged (
                .clk               (clk),
                .rst_n             (rst_n),
                .pt_write_en       (pt_write_en),
                .pt_write_vpn      (pt_write_vpn),
                .pt_write_ppn      (pt_write_ppn),
                .pt_write_valid    (pt_write_valid),
                .rd_en             (rd_en),
                .rd_addr           (rd_addr),
                .rd_data           (rd_data),
                .rd_valid          (rd_valid),
                .wr_en             (wr_en),
                .wr_addr           (wr_addr),
                .wr_data           (wr_data),
                .stats_page_hits   (stats_loads_or_hits),
                .stats_page_misses (stats_moves_or_misses)
            );
        end
    endgenerate

endmodule : mem_backend_wrap
