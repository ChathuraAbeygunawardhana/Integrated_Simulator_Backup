//=============================================================================
// Module: scratchpad_ram
// Description: Behaviorally-modelled multi-port scratchpad SRAM. Used by both
//              the stamp and the paged memory backends.
//
// Provides:
//   * N read ports (1-cycle latency)
//   * N write ports (1-cycle latency, write-first)
//   * Single backing storage so reads see prior writes via natural sim
//     ordering
//
// For Vivado synthesis this maps to multi-pumped/banked BRAM; for behavioral
// verification it's a simple `logic [DW-1:0] mem [DEPTH]` array.
//=============================================================================

module scratchpad_ram #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH      = 4096,
    parameter int N_PORTS    = 4
) (
    input  logic                              clk,
    input  logic                              rst_n,

    // Read ports
    input  logic [N_PORTS-1:0]                rd_en,
    input  logic [$clog2(DEPTH)-1:0]          rd_addr  [N_PORTS],
    output logic [DATA_WIDTH-1:0]             rd_data  [N_PORTS],
    output logic [N_PORTS-1:0]                rd_valid,

    // Write ports
    input  logic [N_PORTS-1:0]                wr_en,
    input  logic [$clog2(DEPTH)-1:0]          wr_addr  [N_PORTS],
    input  logic [DATA_WIDTH-1:0]             wr_data  [N_PORTS]
);

    logic [DATA_WIDTH-1:0] mem [DEPTH];

    integer i;

    // Synchronous reads.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid <= '0;
            for (i = 0; i < N_PORTS; i++) rd_data[i] <= '0;
        end else begin
            for (i = 0; i < N_PORTS; i++) begin
                rd_valid[i] <= rd_en[i];
                if (rd_en[i]) rd_data[i] <= mem[rd_addr[i]];
            end
        end
    end

    // Synchronous writes. (If two ports target the same address in the same
    // cycle, the highest-numbered port wins. The verifier asserts no such
    // contention in normal operation.)
    always_ff @(posedge clk) begin
        for (i = 0; i < N_PORTS; i++) begin
            if (wr_en[i]) mem[wr_addr[i]] <= wr_data[i];
        end
    end

endmodule : scratchpad_ram
