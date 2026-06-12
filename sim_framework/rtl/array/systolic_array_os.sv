//=============================================================================
// Module: systolic_array_top  (FIXED)
// Description: Top-level output stationary systolic array
//
// FIX: The original FSM checked fetch_done in the combinatorial next_state
//      block but drove the control signals (weight_fetch_enable etc.) in a
//      separate always_ff block that was one cycle behind.  Because
//      weight_fetcher / input_fetcher now pulse fetch_done for exactly 1 cycle,
//      the original code could *miss* the pulse if the registered control block
//      hadn't yet de-asserted fetch_enable on that same cycle.
//
//      Changes:
//        1. The FSM transition from WAIT_WEIGHTS→COMPUTE now uses the
//           registered fetch_done correctly (no change to combinatorial
//           next_state - it already used the signal; the fix is in the
//           enable/disable ordering below).
//        2. weight_fetch_enable / input_fetch_enable / output_fetch_enable
//           are now de-asserted in the *same* clock cycle they are sampled
//           by the sub-modules (i.e. they are gated to the FETCH_xxx states
//           only) so there is no double-enable on the following cycle.
//        3. A registered "fetch_done_r" latch is added to make the FSM
//           robust against 1-cycle pulses even if the registered path adds
//           a cycle of delay.
//=============================================================================

module systolic_array_top
    import systolic_array_pkg::*;
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

    input  logic [15:0]           input_channels,
    input  logic [15:0]           input_height,
    input  logic [15:0]           input_width,

    input  logic [15:0]           weight_k,
    input  logic [15:0]           weight_c,
    input  logic [15:0]           weight_kh,
    input  logic [15:0]           weight_kw,

    input  logic [15:0]           output_channels,
    input  logic [15:0]           output_height,
    input  logic [15:0]           output_width,

    input  logic [ADDR_WIDTH-1:0] input_base_addr,
    input  logic [ADDR_WIDTH-1:0] weight_base_addr,
    input  logic [ADDR_WIDTH-1:0] output_base_addr,

    //=========================================================================
    // Control Interface
    //=========================================================================
    input  logic                  start,
    input  logic [15:0]           tile_row,
    input  logic [15:0]           tile_col_start,
    input  logic [15:0]           tile_ch_start,
    output logic                  done,
    output logic                  busy,

    //=========================================================================
    // Memory Interface - Input
    //=========================================================================
    output logic [ADDR_WIDTH-1:0]  input_addr  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] input_addr_valid,
    input  logic [DATA_WIDTH-1:0]  input_data  [ARRAY_WIDTH],
    input  logic [ARRAY_WIDTH-1:0] input_data_valid,

    //=========================================================================
    // Memory Interface - Weight
    //=========================================================================
    output logic [ADDR_WIDTH-1:0]   weight_addr  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] weight_addr_valid,
    input  logic [DATA_WIDTH-1:0]   weight_data  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] weight_data_valid,

    //=========================================================================
    // Memory Interface - Output
    //=========================================================================
    output logic [ADDR_WIDTH-1:0]                    output_addr  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid,
    output logic [DATA_WIDTH-1:0]                    output_data  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid
);

    //=========================================================================
    // Internal Signals
    //=========================================================================
    logic input_fetch_enable,  input_fetch_done;
    logic weight_fetch_enable, weight_fetch_done;
    logic output_fetch_enable, output_fetch_done;

    // Sticky latches - capture the 1-cycle fetch_done pulses so the FSM
    // cannot miss them even if it is in the wrong pipeline stage.
    logic weight_done_latch, input_done_latch, output_done_latch;

    logic [DATA_WIDTH-1:0]  pe_activation [ARRAY_HEIGHT+1][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT:0][ARRAY_WIDTH-1:0] pe_activation_valid;
    logic [ACCUM_WIDTH-1:0] pe_psum [ARRAY_HEIGHT][ARRAY_WIDTH+1];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH:0] pe_psum_valid;
    logic [DATA_WIDTH-1:0]  pe_weight [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_weight_valid;
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_enable;
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] pe_clear;
    logic [ACCUM_WIDTH-1:0] pe_result [ARRAY_HEIGHT][ARRAY_WIDTH];

    typedef enum logic [3:0] {
        IDLE,
        FETCH_WEIGHTS,
        WAIT_WEIGHTS,
        COMPUTE,
        WAIT_COMPUTE,
        WRITEBACK,
        WAIT_WRITEBACK,
        DONE_STATE
    } state_t;

    state_t state, next_state;

    //=========================================================================
    // Sticky latch: set when fetch_done pulses, cleared when FSM advances past
    // the corresponding WAIT state.
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_done_latch <= 1'b0;
            input_done_latch  <= 1'b0;
            output_done_latch <= 1'b0;
        end else begin
            // Set on pulse
            if (weight_fetch_done) weight_done_latch <= 1'b1;
            if (input_fetch_done)  input_done_latch  <= 1'b1;
            if (output_fetch_done) output_done_latch <= 1'b1;

            // Clear when leaving the corresponding WAIT state
            if (state == WAIT_WEIGHTS && weight_done_latch)
                weight_done_latch <= 1'b0;
            if (state == WAIT_COMPUTE && input_done_latch)
                input_done_latch  <= 1'b0;
            if (state == WAIT_WRITEBACK && output_done_latch)
                output_done_latch <= 1'b0;
        end
    end

    //=========================================================================
    // FSM state register
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:           if (start)              next_state = FETCH_WEIGHTS;
            FETCH_WEIGHTS:                          next_state = WAIT_WEIGHTS;
            WAIT_WEIGHTS:   if (weight_done_latch)  next_state = COMPUTE;
            COMPUTE:                                next_state = WAIT_COMPUTE;
            WAIT_COMPUTE:   if (input_done_latch)   next_state = WRITEBACK;
            WRITEBACK:                              next_state = WAIT_WRITEBACK;
            WAIT_WRITEBACK: if (output_done_latch)  next_state = DONE_STATE;
            DONE_STATE:                             next_state = IDLE;
            default:                                next_state = IDLE;
        endcase
    end

    //=========================================================================
    // Control signals
    //=========================================================================
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
            // Default - de-assert everything each cycle
            weight_fetch_enable <= 1'b0;
            input_fetch_enable  <= 1'b0;
            output_fetch_enable <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    busy     <= 1'b0;
                    pe_clear <= '{default: 1'b1};
                end

                FETCH_WEIGHTS: begin
                    busy                <= 1'b1;
                    weight_fetch_enable <= 1'b1;   // single-cycle pulse to sub-module
                    pe_clear            <= '{default: 1'b0};
                end

                WAIT_WEIGHTS: begin
                    busy <= 1'b1;
                end

                COMPUTE: begin
                    busy               <= 1'b1;
                    input_fetch_enable <= 1'b1;    // single-cycle pulse
                    pe_enable          <= '{default: 1'b1};
                end

                WAIT_COMPUTE: begin
                    busy      <= 1'b1;
                    pe_enable <= '{default: 1'b1};
                end

                WRITEBACK: begin
                    busy                <= 1'b1;
                    output_fetch_enable <= 1'b1;   // single-cycle pulse
                    pe_enable           <= '{default: 1'b0};
                end

                WAIT_WRITEBACK: begin
                    busy <= 1'b1;
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end

    //=========================================================================
    // Fetcher Instances
    //=========================================================================
    input_fetcher #(
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(ADDR_WIDTH)
    ) input_fetch_inst (
        .clk(clk), .rst_n(rst_n),
        .layout(mem_layout),
        .input_c(input_channels), .input_h(input_height), .input_w(input_width),
        .kernel_h(weight_kh), .kernel_w(weight_kw),
        .base_addr(input_base_addr),
        .output_row(tile_row), .output_col_start(tile_col_start),
        .fetch_enable(input_fetch_enable),
        .fetch_done(input_fetch_done),
        .addr_out(input_addr),
        .addr_valid(input_addr_valid)
    );

    weight_fetcher #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .DATA_WIDTH(ADDR_WIDTH)
    ) weight_fetch_inst (
        .clk(clk), .rst_n(rst_n),
        .layout(mem_layout),
        .weight_k(weight_k), .weight_c(weight_c),
        .weight_kh(weight_kh), .weight_kw(weight_kw),
        .base_addr(weight_base_addr),
        .output_ch_start(tile_ch_start),
        .fetch_enable(weight_fetch_enable),
        .fetch_done(weight_fetch_done),
        .addr_out(weight_addr),
        .addr_valid(weight_addr_valid),
        .weight_idx_out()
    );

    output_fetcher #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .DATA_WIDTH(ADDR_WIDTH)
    ) output_fetch_inst (
        .clk(clk), .rst_n(rst_n),
        .layout(mem_layout),
        .output_k(output_channels),
        .output_h(output_height),
        .output_w(output_width),
        .base_addr(output_base_addr),
        .channel_start(tile_ch_start),
        .row_start(tile_row),
        .col_start(tile_col_start),
        .fetch_enable(output_fetch_enable),
        .fetch_done(output_fetch_done),
        .addr_out(output_addr),
        .addr_valid(output_addr_valid),
        .out_ch(), .out_row(), .out_col()
    );

    //=========================================================================
    // Processing Element Array
    //=========================================================================
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_pe_rows
            for (col = 0; col < ARRAY_WIDTH; col++) begin : gen_pe_cols
                processing_element #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACCUM_WIDTH(ACCUM_WIDTH)
                ) pe_inst (
                    .clk(clk), .rst_n(rst_n),
                    .enable(pe_enable[row][col]),
                    .clear_accum(pe_clear[row][col]),
                    .activation_in(pe_activation[row][col]),
                    .activation_valid_in(pe_activation_valid[row][col]),
                    .weight(pe_weight[row][col]),
                    .weight_valid(pe_weight_valid[row][col]),
                    .psum_in(pe_psum[row][col]),
                    .psum_valid_in(pe_psum_valid[row][col]),
                    .activation_out(pe_activation[row+1][col]),
                    .activation_valid_out(pe_activation_valid[row+1][col]),
                    .psum_out(pe_psum[row][col+1]),
                    .psum_valid_out(pe_psum_valid[row][col+1]),
                    .result(pe_result[row][col])
                );
            end
        end
    endgenerate

    // Connect input data to top of array
    always_comb begin
        for (int c = 0; c < ARRAY_WIDTH; c++) begin
            pe_activation[0][c]       = input_data[c];
            pe_activation_valid[0][c] = input_data_valid[c];
        end
    end

    // Connect weights from fetcher
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_weight       <= '{default: '0};
            pe_weight_valid <= '{default: 1'b0};
        end else begin
            for (int r = 0; r < ARRAY_HEIGHT; r++) begin
                for (int c = 0; c < ARRAY_WIDTH; c++) begin
                    if (weight_data_valid[r]) begin
                        pe_weight[r][c]       <= weight_data[r];
                        pe_weight_valid[r][c] <= 1'b1;
                    end else begin
                        pe_weight_valid[r][c] <= 1'b0;
                    end
                end
            end
        end
    end

    // Initialize partial sums (leftmost column)
    always_comb begin
        for (int r = 0; r < ARRAY_HEIGHT; r++) begin
            pe_psum[r][0]       = '0;
            pe_psum_valid[r][0] = 1'b1;
        end
    end

    // Output data from PEs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data       <= '{default: '0};
            output_data_valid <= '{default: 1'b0};
        end else if (state == WAIT_WRITEBACK || state == WRITEBACK) begin
            for (int r = 0; r < ARRAY_HEIGHT; r++)
                for (int c = 0; c < ARRAY_WIDTH; c++) begin
                    output_data[r][c]       <= pe_result[r][c][DATA_WIDTH-1:0];
                    output_data_valid[r][c] <= 1'b1;
                end
        end else begin
            output_data_valid <= '{default: 1'b0};
        end
    end

endmodule : systolic_array_top