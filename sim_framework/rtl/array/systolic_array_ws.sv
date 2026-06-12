//=============================================================================
// Module: weight_stationary_top
// Description: Top-level weight stationary systolic array with integrated fetchers
//              Fully parameterized for flexible configuration
//=============================================================================

module weight_stationary_top
    import weight_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int ACCUM_WIDTH  = 48,
    parameter int ADDR_WIDTH   = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,
    
    //=========================================================================
    // Configuration Interface
    //=========================================================================
    input  mem_layout_t           mem_layout,
    
    // Input tensor shape
    input  logic [15:0]           input_channels,
    input  logic [15:0]           input_height,
    input  logic [15:0]           input_width,
    
    // Weight tensor shape
    input  logic [15:0]           weight_k,        // Output channels (filters)
    input  logic [15:0]           weight_c,        // Input channels
    input  logic [15:0]           weight_kh,       // Kernel height
    input  logic [15:0]           weight_kw,       // Kernel width
    
    // Output tensor shape
    input  logic [15:0]           output_channels,
    input  logic [15:0]           output_height,
    input  logic [15:0]           output_width,
    
    // Base addresses
    input  logic [ADDR_WIDTH-1:0] input_base_addr,
    input  logic [ADDR_WIDTH-1:0] weight_base_addr,
    input  logic [ADDR_WIDTH-1:0] output_base_addr,
    
    //=========================================================================
    // Control Interface
    //=========================================================================
    input  logic                  start,
    input  logic [15:0]           tile_row_start,  // Output row start for tile
    input  logic [15:0]           tile_col,        // Output column
    input  logic [15:0]           tile_ch_start,   // Output channel start
    input  logic [15:0]           input_ch,        // Current input channel
    input  logic [15:0]           kernel_row,      // Kernel row position
    input  logic [15:0]           kernel_col_start,// Kernel column start
    output logic                  done,
    output logic                  busy,
    
    //=========================================================================
    // Memory Interface - Input
    //=========================================================================
    output logic [ADDR_WIDTH-1:0] input_addr [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] input_addr_valid,
    input  logic [DATA_WIDTH-1:0] input_data [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] input_data_valid,
    
    //=========================================================================
    // Memory Interface - Weight
    //=========================================================================
    output logic [ADDR_WIDTH-1:0] weight_addr [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_addr_valid,
    input  logic [DATA_WIDTH-1:0] weight_data [ARRAY_HEIGHT][ARRAY_WIDTH],
    input  logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] weight_data_valid,
    
    //=========================================================================
    // Memory Interface - Output
    //=========================================================================
    output logic [ADDR_WIDTH-1:0] output_addr [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] output_addr_valid,
    output logic [DATA_WIDTH-1:0] output_data [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] output_data_valid
);

    //=========================================================================
    // Internal Signals
    //=========================================================================
    
    // Fetcher control signals
    logic input_fetch_enable, input_fetch_done;
    logic weight_fetch_enable, weight_fetch_done;
    logic output_fetch_enable, output_fetch_done;
    
    // PE array signals
    // Activations broadcast horizontally (left to right)
    logic [DATA_WIDTH-1:0]  pe_activation [ARRAY_HEIGHT][ARRAY_WIDTH+1];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH:0] pe_activation_valid;
    
    // Partial sums flow diagonally
    logic [ACCUM_WIDTH-1:0] pe_psum [ARRAY_HEIGHT+1][ARRAY_WIDTH+1];
    logic [ARRAY_HEIGHT:0][ARRAY_WIDTH:0] pe_psum_valid;
    
    // Weights (stationary in each PE)
    logic [DATA_WIDTH-1:0]  pe_weight [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_weight_valid;
    
    // Control signals for PEs
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_enable;
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_clear;
    logic [ACCUM_WIDTH-1:0] pe_result [ARRAY_HEIGHT][ARRAY_WIDTH];
    
    // FSM
    typedef enum logic [3:0] {
        IDLE,
        FETCH_WEIGHTS,
        WAIT_WEIGHTS,
        COMPUTE,
        WAIT_COMPUTE,
        WRITEBACK,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    //=========================================================================
    // Fetcher Instances
    //=========================================================================
    
    // Input Fetcher (broadcasts horizontally)
    input_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(ADDR_WIDTH)
    ) input_fetch_inst (
        .clk(clk),
        .rst_n(rst_n),
        .layout(mem_layout),
        .input_c(input_channels),
        .input_h(input_height),
        .input_w(input_width),
        .kernel_h(weight_kh),
        .kernel_w(weight_kw),
        .base_addr(input_base_addr),
        .output_row_start(tile_row_start),
        .output_col(tile_col),
        .input_channel(input_ch),
        .fetch_enable(input_fetch_enable),
        .fetch_done(input_fetch_done),
        .addr_out(input_addr),
        .addr_valid(input_addr_valid)
    );
    
    // Weight Fetcher (one weight per PE)
    weight_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(ADDR_WIDTH)
    ) weight_fetch_inst (
        .clk(clk),
        .rst_n(rst_n),
        .layout(mem_layout),
        .weight_k(weight_k),
        .weight_c(weight_c),
        .weight_kh(weight_kh),
        .weight_kw(weight_kw),
        .base_addr(weight_base_addr),
        .output_ch_start(tile_ch_start),
        .input_ch(input_ch),
        .kernel_row(kernel_row),
        .kernel_col_start(kernel_col_start),
        .fetch_enable(weight_fetch_enable),
        .fetch_done(weight_fetch_done),
        .addr_out(weight_addr),
        .addr_valid(weight_addr_valid)
    );
    
    // Output Fetcher
    output_fetcher_ws #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(ADDR_WIDTH)
    ) output_fetch_inst (
        .clk(clk),
        .rst_n(rst_n),
        .layout(mem_layout),
        .output_k(output_channels),
        .output_h(output_height),
        .output_w(output_width),
        .base_addr(output_base_addr),
        .channel_start(tile_ch_start),
        .row_start(tile_row_start),
        .col(tile_col),
        .fetch_enable(output_fetch_enable),
        .fetch_done(output_fetch_done),
        .addr_out(output_addr),
        .addr_valid(output_addr_valid),
        .out_ch(),     // Not used in this version
        .out_row(),
        .out_col()
    );
    
    //=========================================================================
    // Processing Element Array
    //=========================================================================
    
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_pe_rows
            for (col = 0; col < ARRAY_WIDTH; col++) begin : gen_pe_cols
                
                processing_element_ws #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACCUM_WIDTH(ACCUM_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(pe_enable[row][col]),
                    .clear_accum(pe_clear[row][col]),
                    .activation_in(pe_activation[row][col]),
                    .activation_valid_in(pe_activation_valid[row][col]),
                    .weight(pe_weight[row][col]),
                    .weight_valid(pe_weight_valid[row][col]),
                    .psum_in(pe_psum[row][col]),
                    .psum_valid_in(pe_psum_valid[row][col]),
                    .activation_out(pe_activation[row][col+1]),
                    .activation_valid_out(pe_activation_valid[row][col+1]),
                    .psum_out(pe_psum[row+1][col+1]),
                    .psum_valid_out(pe_psum_valid[row+1][col+1]),
                    .result(pe_result[row][col])
                );
                
            end
        end
    endgenerate
    
    // Connect input data to left side of array (broadcast horizontally)
    always_comb begin
        for (int r = 0; r < ARRAY_HEIGHT; r++) begin
            pe_activation[r][0] = input_data[r];
            pe_activation_valid[r][0] = input_data_valid[r];
        end
    end
    
    // Connect weights from fetcher
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_weight <= '{default: '0};
            pe_weight_valid <= '{default: 1'b0};
        end else begin
            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                for (int c = 0; c < ARRAY_WIDTH; c++) begin
                    if (weight_data_valid[r][c]) begin
                        pe_weight[r][c] <= weight_data[r][c];
                        pe_weight_valid[r][c] <= 1'b1;
                    end else begin
                        pe_weight_valid[r][c] <= 1'b0;
                    end
                end
            end
        end
    end
    
    // Initialize partial sums (top and left edges)
    always_comb begin
        // Top row
        for (int c = 0; c <= ARRAY_WIDTH; c++) begin
            pe_psum[0][c] = '0;
            pe_psum_valid[0][c] = 1'b1;
        end
        // Left column
        for (int r = 0; r <= ARRAY_HEIGHT; r++) begin
            pe_psum[r][0] = '0;
            pe_psum_valid[r][0] = 1'b1;
        end
    end
    
    //=========================================================================
    // Control FSM
    //=========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start)
                    next_state = FETCH_WEIGHTS;
            end
            
            FETCH_WEIGHTS: begin
                next_state = WAIT_WEIGHTS;
            end
            
            WAIT_WEIGHTS: begin
                if (weight_fetch_done)
                    next_state = COMPUTE;
            end
            
            COMPUTE: begin
                next_state = WAIT_COMPUTE;
            end
            
            WAIT_COMPUTE: begin
                if (input_fetch_done)
                    next_state = WRITEBACK;
            end
            
            WRITEBACK: begin
                if (output_fetch_done)
                    next_state = DONE_STATE;
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Control signals
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_fetch_enable <= 1'b0;
            input_fetch_enable  <= 1'b0;
            output_fetch_enable <= 1'b0;
            pe_enable <= '{default: 1'b0};
            pe_clear  <= '{default: 1'b0};
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            // Default values
            weight_fetch_enable <= 1'b0;
            input_fetch_enable  <= 1'b0;
            output_fetch_enable <= 1'b0;
            done <= 1'b0;
            
            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    pe_clear <= '{default: 1'b1};
                end
                
                FETCH_WEIGHTS: begin
                    busy <= 1'b1;
                    weight_fetch_enable <= 1'b1;
                    pe_clear <= '{default: 1'b0};
                end
                
                WAIT_WEIGHTS: begin
                    busy <= 1'b1;
                end
                
                COMPUTE: begin
                    busy <= 1'b1;
                    input_fetch_enable <= 1'b1;
                    pe_enable <= '{default: 1'b1};
                end
                
                WAIT_COMPUTE: begin
                    busy <= 1'b1;
                    pe_enable <= '{default: 1'b1};
                end
                
                WRITEBACK: begin
                    busy <= 1'b1;
                    output_fetch_enable <= 1'b1;
                    pe_enable <= '{default: 1'b0};
                end
                
                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Output data from rightmost column of PEs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= '{default: '0};
            output_data_valid <= '{default: 1'b0};
        end else if (state == WRITEBACK) begin
            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                // Take result from last PE in each row
                output_data[r] <= pe_result[r][ARRAY_WIDTH-1][DATA_WIDTH-1:0];
                output_data_valid[r] <= 1'b1;
            end
        end else begin
            output_data_valid <= '{default: 1'b0};
        end
    end

endmodule : weight_stationary_top
