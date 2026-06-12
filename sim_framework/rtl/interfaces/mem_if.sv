//=============================================================================
// Interface: mem_if
// Description: Canonical data-plane interface between a memory backend
//              (stamp or paged) and a systolic-array wrapper. Read/write
//              handshake; the backend serves N parallel reads per cycle from
//              its scratchpad (the systolic array's per-PE fetchers all hit
//              the same backend through this interface).
//
// Notes:
//   * Backend-specific *configuration* paths (stamp's metadata bus, paged's
//     page-table write port) are NOT in this interface -- they remain on the
//     top-level module's port list and are muxed by the MEMORY parameter.
//     This interface carries only the runtime data plane.
//   * `wide_rd_data` is the full word-line of the backend scratchpad (N
//     elements per cycle); the wrapper distributes it to the systolic array's
//     per-PE fetch ports.
//=============================================================================

interface mem_if #(
    parameter int AW = 32,
    parameter int DW = 32,
    parameter int N  = 4    // backend bandwidth: words served per cycle
);

    // Read channel (systolic array -> backend)
    logic [AW-1:0] rd_addr  [N];
    logic [N-1:0]  rd_req;
    logic [N-1:0]  rd_ready;

    // Read response (backend -> systolic array)
    logic [DW-1:0] rd_data  [N];
    logic [N-1:0]  rd_valid;

    // Write channel (systolic array -> backend; used for writeback)
    logic [AW-1:0] wr_addr  [N];
    logic [DW-1:0] wr_data  [N];
    logic [N-1:0]  wr_en;

    modport master (
        output rd_addr, rd_req, wr_addr, wr_data, wr_en,
        input  rd_data, rd_valid, rd_ready
    );

    modport slave (
        input  rd_addr, rd_req, wr_addr, wr_data, wr_en,
        output rd_data, rd_valid, rd_ready
    );

endinterface : mem_if
