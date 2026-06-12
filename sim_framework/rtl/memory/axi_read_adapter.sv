//=============================================================================
// Module: axi_read_adapter
// Description: Bridges a single-outstanding read-request channel (used by the
//              stamp controller) to an AXI4-Lite-ish read interface presented
//              to the testbench AXI memory model.
//
//              Single-outstanding is documented in the framework plan as a
//              v1 limitation. The reaadapter accepts (addr, len) and returns a
//              burst of `len` words at one word per cycle.
//=============================================================================

module axi_read_adapter #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,

    // Upstream request port (stamp ctrl -> adapter)
    input  logic [ADDR_WIDTH-1:0] up_rd_addr,
    input  logic [15:0]           up_rd_len,
    input  logic                  up_rd_req,
    output logic                  up_rd_ready,
    output logic [DATA_WIDTH-1:0] up_rd_data,
    output logic                  up_rd_valid,

    // Downstream AXI master (adapter -> mem model)
    output logic [ADDR_WIDTH-1:0] axi_arvalid_addr,
    output logic [7:0]            axi_arlen,
    output logic                  axi_arvalid,
    input  logic                  axi_arready,
    input  logic [DATA_WIDTH-1:0] axi_rdata,
    input  logic                  axi_rvalid,
    output logic                  axi_rready,
    input  logic                  axi_rlast
);

    typedef enum logic [1:0] { IDLE, AR, R } st_t;
    st_t state;

    logic [15:0] beats_left;
    logic [15:0] beats_total;

    assign axi_arvalid_addr = up_rd_addr;
    // AXI arlen is 0-based: a burst of N beats has arlen = N-1.
    assign axi_arlen        = (up_rd_len > 0) ? (up_rd_len[7:0] - 8'd1) : 8'd0;
    assign axi_arvalid      = (state == AR);
    assign axi_rready       = (state == R);

    assign up_rd_data  = axi_rdata;
    assign up_rd_valid = (state == R) && axi_rvalid;
    assign up_rd_ready = (state == IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            beats_left  <= '0;
            beats_total <= '0;
        end else begin
            case (state)
                IDLE: if (up_rd_req) begin
                    beats_total <= up_rd_len;
                    beats_left  <= up_rd_len;
                    state       <= AR;
                end
                AR: if (axi_arready) state <= R;
                R: if (axi_rvalid) begin
                    if (beats_left == 16'd1 || axi_rlast) state <= IDLE;
                    else beats_left <= beats_left - 16'd1;
                end
            endcase
        end
    end

endmodule : axi_read_adapter
