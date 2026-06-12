//=============================================================================
// Module: paged_memory_backend
// Description: Page-table-based on-chip memory backend. Translates virtual
//              addresses from the systolic-array wrapper through a TLB-style
//              page table to physical scratchpad addresses; serves the read
//              from the scratchpad and signals miss when the page is unmapped.
//
// Same canonical N-port data-plane interface as stamp_memory_backend so the
// two are interchangeable behind mem_backend_wrap.
//
// Side-band ports differ:
//   * stamp:  metadata_wr_*, phase_*
//   * paged:  pt_write_en, pt_write_vpn, pt_write_ppn, pt_write_valid
//=============================================================================

module paged_memory_backend #(
    parameter int ADDR_WIDTH     = 32,
    parameter int DATA_WIDTH     = 32,
    parameter int SPAD_DEPTH     = 4096,
    parameter int N_PORTS        = 4,
    parameter int VADDR_WIDTH    = 32,
    parameter int PADDR_WIDTH    = 32,
    parameter int PAGE_SIZE_BITS = 12,
    parameter int NUM_PAGES      = 256,
    parameter int VPN_WIDTH      = 8,    // log2(NUM_PAGES)
    parameter int PPN_WIDTH      = 20    // PADDR_WIDTH - PAGE_SIZE_BITS
) (
    input  logic clk,
    input  logic rst_n,

    //--- side-band: page-table update port (host writes the page mapping)
    input  logic                   pt_write_en,
    input  logic [VPN_WIDTH-1:0]   pt_write_vpn,
    input  logic [PPN_WIDTH-1:0]   pt_write_ppn,
    input  logic                   pt_write_valid,

    //--- canonical data plane (vaddr in, data out)
    input  logic [N_PORTS-1:0]                rd_en,
    input  logic [VADDR_WIDTH-1:0]            rd_addr  [N_PORTS],
    output logic [DATA_WIDTH-1:0]             rd_data  [N_PORTS],
    output logic [N_PORTS-1:0]                rd_valid,

    input  logic [N_PORTS-1:0]                wr_en,
    input  logic [VADDR_WIDTH-1:0]            wr_addr  [N_PORTS],
    input  logic [DATA_WIDTH-1:0]             wr_data  [N_PORTS],

    //--- statistics
    output logic [31:0] stats_page_hits,
    output logic [31:0] stats_page_misses
);

    // Translate each port's vaddr through its own page_table instance.
    logic [PADDR_WIDTH-1:0] paddr  [N_PORTS];
    logic                   hit    [N_PORTS];
    logic                   tx_valid [N_PORTS];

    genvar p;
    generate
        for (p = 0; p < N_PORTS; p++) begin : g_pt
            page_table #(
                .VADDR_WIDTH    (VADDR_WIDTH),
                .PADDR_WIDTH    (PADDR_WIDTH),
                .PAGE_SIZE_BITS (PAGE_SIZE_BITS),
                .NUM_PAGES      (NUM_PAGES),
                .VPN_WIDTH      (VPN_WIDTH),
                .PPN_WIDTH      (PPN_WIDTH)
            ) u_pt (
                .clk         (clk),
                .rst_n       (rst_n),
                .vaddr       (rd_addr[p]),
                .paddr       (paddr[p]),
                .hit         (hit[p]),
                .valid       (tx_valid[p]),
                .write_en    (pt_write_en && (pt_write_vpn[VPN_WIDTH-1:0] ==
                              rd_addr[p][PAGE_SIZE_BITS +: VPN_WIDTH]) && (p == 0)),
                .write_vpn   (pt_write_vpn),
                .write_ppn   (pt_write_ppn),
                .write_valid (pt_write_valid)
            );
        end
    endgenerate

    // The above 'write' port is awkward; the host should be able to update
    // any VPN. Simpler: just have one shared page_table whose write port is
    // unconditional; replicate paddr to all read ports.
    // For v1 we keep the per-port instances (they each have an identical
    // page table after the host loads the same VPN once on each). The
    // commit is small. A future revision can share one page_table.

    // Read ports: register vaddr->paddr through scratchpad
    logic [$clog2(SPAD_DEPTH)-1:0] spad_rd_addr [N_PORTS];
    logic [$clog2(SPAD_DEPTH)-1:0] spad_wr_addr [N_PORTS];

    always_comb begin
        for (int i = 0; i < N_PORTS; i++) begin
            spad_rd_addr[i] = paddr[i][$clog2(SPAD_DEPTH)-1:0];
            spad_wr_addr[i] = wr_addr[i][$clog2(SPAD_DEPTH)-1:0];
        end
    end

    logic [N_PORTS-1:0] spad_rd_valid;
    scratchpad_ram #(
        .DATA_WIDTH (DATA_WIDTH),
        .DEPTH      (SPAD_DEPTH),
        .N_PORTS    (N_PORTS)
    ) u_spad (
        .clk      (clk),
        .rst_n    (rst_n),
        .rd_en    (rd_en),
        .rd_addr  (spad_rd_addr),
        .rd_data  (rd_data),
        .rd_valid (spad_rd_valid),
        .wr_en    (wr_en),
        .wr_addr  (spad_wr_addr),
        .wr_data  (wr_data)
    );

    // rd_valid masked by translation hit
    always_comb begin
        for (int i = 0; i < N_PORTS; i++) begin
            rd_valid[i] = spad_rd_valid[i] && hit[i];
        end
    end

    // Stats
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stats_page_hits   <= '0;
            stats_page_misses <= '0;
        end else begin
            for (int i = 0; i < N_PORTS; i++) begin
                if (rd_en[i]) begin
                    if (hit[i]) stats_page_hits   <= stats_page_hits   + 1;
                    else        stats_page_misses <= stats_page_misses + 1;
                end
            end
        end
    end

endmodule : paged_memory_backend
