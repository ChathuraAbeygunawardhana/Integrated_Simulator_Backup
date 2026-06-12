//=============================================================================
// Module: address_generator
// Description: Generates linear addresses from multi-dimensional indices
//              based on configurable memory layout
//=============================================================================

module address_generator
    import systolic_array_pkg::*;
#(
    parameter int DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,
    
    // Configuration
    input  mem_layout_t           layout,
    input  logic [15:0]           shape_c,      // no of Channels
    input  logic [15:0]           shape_h,      // Height
    input  logic [15:0]           shape_w,      // Width
    input  logic [DATA_WIDTH-1:0] base_addr,    // starting memory address
    
    // Input indices (interpretation depends on layout)
    input  logic [15:0]           idx_0,
    input  logic [15:0]           idx_1,
    input  logic [15:0]           idx_2,
    input  logic                  idx_valid,
    
    // Output address
    output logic [DATA_WIDTH-1:0] linear_addr,
    output logic                  addr_valid
);

    // Internal signals
    logic [31:0] offset;                // distance from base address
    logic [31:0] stride_0, stride_1;    // how much to jump when index increases

    // moving one row → jump W
    // moving one channel → jump H×W


    logic [15:0] c_idx, h_idx, w_idx;
    
    // Calculate strides and map indices based on layout
    always_comb begin       // pure combinational logic (no clock)
        // Default values
        stride_0 = 0;
        stride_1 = 0;
        c_idx = 0;
        h_idx = 0;
        w_idx = 0;
        
        case (layout)
            CHANNEL_MAJOR: begin
                // Layout: C, H, W (channel varies slowest)
                // Indices are (c, h, w)
                c_idx = idx_0;
                h_idx = idx_1;
                w_idx = idx_2;
                
                stride_0 = shape_h * shape_w;  // Stride for C
                stride_1 = shape_w;             // Stride for H (skip 1 row = width)
                offset = c_idx * stride_0 + h_idx * stride_1 + w_idx;
            end
            
            ROW_MAJOR: begin
                // Layout: H, W, C (height varies slowest)
                // Indices are (h, w, c)
                h_idx = idx_0;
                w_idx = idx_1;
                c_idx = idx_2;
                
                stride_0 = shape_w * shape_c;   // Stride for H 
                stride_1 = shape_c;             // Stride for W 
                offset = h_idx * stride_0 + w_idx * stride_1 + c_idx;
            end
            
            COLUMN_MAJOR: begin
                // Layout: W, H, C (width varies slowest)
                // Indices are (w, h, c)
                w_idx = idx_0;
                h_idx = idx_1;
                c_idx = idx_2;
                
                stride_0 = shape_h * shape_c;   // Stride for W
                stride_1 = shape_c;             // Stride for H
                offset = w_idx * stride_0 + h_idx * stride_1 + c_idx;
            end
            
            default: begin
                offset = 0;
            end
        endcase
    end
    
    // Generate output address
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            linear_addr <= '0;
            addr_valid  <= 1'b0;
        end else begin
            addr_valid  <= idx_valid;
            if (idx_valid) begin
                linear_addr <= base_addr + offset;
            end
        end
    end

endmodule : address_generator