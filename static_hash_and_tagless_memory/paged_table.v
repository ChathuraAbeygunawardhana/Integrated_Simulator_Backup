`timescale 1ns / 1ps

module page_table #(
    parameter VADDR_WIDTH    = 32,
    parameter PADDR_WIDTH    = 32,
    parameter PAGE_SIZE_BITS = 12,
    parameter NUM_PAGES      = 256,
    // FIX: promote to parameters so they can be used in port declarations
    parameter VPN_WIDTH      = 8,   // log2(NUM_PAGES); keep in sync with NUM_PAGES
    parameter PPN_WIDTH      = 20   // PADDR_WIDTH - PAGE_SIZE_BITS
)(
    input  wire clk,
    input  wire rst_n,

    // Address translation interface
    input  wire [VADDR_WIDTH-1:0] vaddr,
    output reg  [PADDR_WIDTH-1:0] paddr,
    output reg                    hit,
    output reg                    valid,

    // Page table update interface
    input  wire                   write_en,
    input  wire [VPN_WIDTH-1:0]   write_vpn,
    input  wire [PPN_WIDTH-1:0]   write_ppn,
    input  wire                   write_valid
);

    localparam OFFSET_WIDTH = PAGE_SIZE_BITS;

    // Extract VPN and page offset from virtual address
    wire [VPN_WIDTH-1:0]   vpn    = vaddr[PAGE_SIZE_BITS +: VPN_WIDTH];
    wire [OFFSET_WIDTH-1:0] offset = vaddr[OFFSET_WIDTH-1:0];

    // Page table storage
    reg [PPN_WIDTH-1:0] page_table [0:NUM_PAGES-1];
    reg                 valid_bits [0:NUM_PAGES-1];

    integer i;

    // Synchronous write + reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_PAGES; i = i + 1) begin
                page_table[i] <= {PPN_WIDTH{1'b0}};
                valid_bits[i] <= 1'b0;
            end
        end else if (write_en) begin
            page_table[write_vpn] <= write_ppn;
            valid_bits[write_vpn] <= write_valid;
        end
    end

    // Combinational address translation
    wire [PPN_WIDTH-1:0] ppn        = page_table[vpn];
    wire                 page_valid = valid_bits[vpn];

    always @(*) begin
        if (page_valid) begin
            paddr = {ppn, offset};
            hit   = 1'b1;
            valid = 1'b1;
        end else begin
            paddr = {PADDR_WIDTH{1'b0}};
            hit   = 1'b0;
            valid = 1'b0;
        end
    end

endmodule