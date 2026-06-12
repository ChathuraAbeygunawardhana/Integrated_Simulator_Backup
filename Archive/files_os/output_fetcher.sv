//============================================================================= Dilshan
// Module: output_fetcher  (system aligned)
// Description: Manages output addresses for output stationary systolic array
//              Generates writeback addresses for completed output tile
//
// DESIGN PATTERN - aligned with input_fetcher and weight_fetcher:
//   - addr_valid pulses HIGH for exactly 1 cycle (same cycle as fetch_done)
//   - addr_out holds its last value after addr_valid drops (not cleared)
//   - fetch_done pulses HIGH for 1 cycle in DONE state
//   - address formula is identical to address_generator math
//
// CYCLE TIMELINE:
//   T0: fetch_enable=1 sampled           → IDLE→GENERATE
//   T1: state=GENERATE                   → addr_out[r][c] registered for all cells
//                                          state→DONE
//   T2: state=DONE                       → addr_valid SET HIGH (all valid cells)
//                                          fetch_done SET HIGH
//                                          state→IDLE
//   T3: state=IDLE                       → addr_valid CLEARED
//                                          fetch_done CLEARED
//                                          addr_out HOLDS last value
//
// The system controller reads addr_out/addr_valid on the same cycle that
// fetch_done is observed (T2), exactly as with input_fetcher/weight_fetcher.
//=============================================================================

`timescale 1ns/1ps

module output_fetcher
    import systolic_array_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // Configuration
    input  mem_layout_t           layout,
    input  logic [15:0]           output_k,       // Output channels (filters)
    input  logic [15:0]           output_h,       // Output height
    input  logic [15:0]           output_w,       // Output width
    input  logic [DATA_WIDTH-1:0] base_addr,

    // Control
    input  logic [15:0]           channel_start,  // First output channel in tile
    input  logic [15:0]           row_start,      // Output row for this tile
    input  logic [15:0]           col_start,      // First output column in tile
    input  logic                  fetch_enable,
    output logic                  fetch_done,

    // Address outputs - one per PE in the 2-D tile
    // addr_valid pulses HIGH for 1 cycle (same cycle as fetch_done)
    // addr_out holds its value until next fetch
    output logic [DATA_WIDTH-1:0]                    addr_out   [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid,

    // Position outputs - which tensor element each PE writes back to
    output logic [15:0]           out_ch  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [15:0]           out_row [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [15:0]           out_col [ARRAY_HEIGHT][ARRAY_WIDTH]
);

    // =========================================================================
    // FSM - 3 states
    // =========================================================================
    typedef enum logic [1:0] { IDLE, GENERATE, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        case (state)
            IDLE:     next_state = fetch_enable ? GENERATE : IDLE;
            GENERATE: next_state = DONE;
            DONE:     next_state = IDLE;
            default:  next_state = IDLE;
        endcase
    end

    // fetch_done: pulses HIGH for exactly 1 cycle in DONE state
    // (registered from DONE state, cleared when state returns to IDLE)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

    // =========================================================================
    // Per-cell address computation
    //
    // Each [r][c] maps to output tensor position:
    //   channel = channel_start + r
    //   height  = row_start
    //   width   = col_start + c
    //
    // Address formula (identical to address_generator):
    //   CHANNEL_MAJOR:  base + ch*(H*W)  + h*W  + w
    //   ROW_MAJOR:      base + h*(W*K)   + w*K  + ch
    //   COLUMN_MAJOR:   base + w*(H*K)   + h*K  + ch
    //
    // Each cell has its own private wires - no shared drivers between cells.
    // =========================================================================
    genvar r, c;
    generate
        for (r = 0; r < ARRAY_HEIGHT; r++) begin : gen_rows
            for (c = 0; c < ARRAY_WIDTH; c++) begin : gen_cols

                // Private combinational signals for this cell
                logic [15:0]           cell_ch, cell_h, cell_w;
                logic                  cell_valid;
                logic [31:0]           cell_offset;
                logic [DATA_WIDTH-1:0] cell_addr;

                always_comb begin
                    cell_ch = channel_start + 16'(r);
                    cell_h  = row_start;
                    cell_w  = col_start + 16'(c);

                    // Position bounds check
                    cell_valid = (cell_ch < output_k) &&
                                 (cell_h  < output_h) &&
                                 (cell_w  < output_w);

                    // Address offset - same formula as address_generator
                    case (layout)
                        CHANNEL_MAJOR:
                            cell_offset = 32'(cell_ch) * (32'(output_h) * 32'(output_w))
                                        + 32'(cell_h)  *  32'(output_w)
                                        + 32'(cell_w);
                        ROW_MAJOR:
                            cell_offset = 32'(cell_h)  * (32'(output_w) * 32'(output_k))
                                        + 32'(cell_w)  *  32'(output_k)
                                        + 32'(cell_ch);
                        COLUMN_MAJOR:
                            cell_offset = 32'(cell_w)  * (32'(output_h) * 32'(output_k))
                                        + 32'(cell_h)  *  32'(output_k)
                                        + 32'(cell_ch);
                        default:
                            cell_offset = '0;
                    endcase

                    cell_addr = cell_valid ? (base_addr + cell_offset) : '0;
                end

                // --------------------------------------------------------
                // Register addr_out in GENERATE (1 cycle before fetch_done)
                // This gives addr_out time to be stable when fetch_done fires.
                //
                // addr_valid follows the same pulse pattern as weight_fetcher
                // and input_fetcher: HIGH in DONE (same cycle as fetch_done),
                // LOW in IDLE.
                //
                // addr_out is NOT cleared in IDLE - it holds its last computed
                // value, matching input_fetcher/weight_fetcher hold behavior.
                // --------------------------------------------------------
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        addr_out  [r][c] <= '0;
                        addr_valid[r][c] <= 1'b0;
                        out_ch    [r][c] <= '0;
                        out_row   [r][c] <= '0;
                        out_col   [r][c] <= '0;
                    end else begin
                        case (state)
                            GENERATE: begin
                                // Latch address and position - stable in DONE
                                addr_out[r][c] <= cell_addr;
                                out_ch  [r][c] <= cell_ch;
                                out_row [r][c] <= cell_h;
                                out_col [r][c] <= cell_w;
                                // addr_valid will be driven by DONE state below
                            end
                            DONE: begin
                                // Pulse addr_valid HIGH - matches fetch_done cycle
                                addr_valid[r][c] <= cell_valid;
                            end
                            IDLE: begin
                                // Clear addr_valid (pulse complete)
                                // addr_out intentionally holds last value
                                addr_valid[r][c] <= 1'b0;
                            end
                            default: addr_valid[r][c] <= 1'b0;
                        endcase
                    end
                end

            end // gen_cols
        end // gen_rows
    endgenerate

endmodule : output_fetcher