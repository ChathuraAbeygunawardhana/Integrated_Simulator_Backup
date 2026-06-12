//=============================================================================
// Module: weight_fetcher_ws  (FIXED - uses address_generator_ws)
// Description: Fetches weights for weight stationary systolic array.
//              Each PE[row][col] gets one unique weight that stays stationary.
//
// WS WEIGHT CONCEPT
// ------------------
// In WS, weights are loaded ONCE before computation starts and held forever.
// The 2D weight fetcher maps (row, col) → weight tensor position:
//   PE[row][col]:  output_channel = output_ch_start + row
//                  kernel_col     = kernel_col_start + col
//                  (kernel_row and input_ch are provided externally)
//
// For a 4×4 array with output_ch_start=0 and kernel_col_start=0:
//   PE[0][0]: weight[0][ch][kh][0]   PE[0][1]: weight[0][ch][kh][1]
//   PE[1][0]: weight[1][ch][kh][0]   PE[1][1]: weight[1][ch][kh][1]
//   etc.
//
// The FSM is IDLE → FETCH (1 cycle) → DONE - all 4×4=16 addresses in 1 cycle.
//
// FIX (BUG-5)
// ------------
// Original used address_generator (systolic_array_pkg type).
// Now uses address_generator_ws (weight_stationary_pkg type) to avoid
// type mismatch on mem_layout_t port.
//=============================================================================

module weight_fetcher_ws
    import weight_stationary_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int ARRAY_WIDTH  = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Configuration
    input  mem_layout_t            layout,
    input  logic [15:0]            weight_k,
    input  logic [15:0]            weight_c,
    input  logic [15:0]            weight_kh,
    input  logic [15:0]            weight_kw,
    input  logic [DATA_WIDTH-1:0]  base_addr,

    // Current position (from top-level loop)
    input  logic [15:0]            output_ch_start,
    input  logic [15:0]            input_ch,
    input  logic [15:0]            kernel_row,
    input  logic [15:0]            kernel_col_start,

    // Control
    input  logic                   fetch_enable,
    output logic                   fetch_done,

    // 2D address outputs: one per PE
    output logic [DATA_WIDTH-1:0]                    addr_out   [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] addr_valid
);

    //=========================================================================
    // FSM: IDLE → FETCH (1 cycle) → DONE → IDLE
    //=========================================================================
    typedef enum logic [1:0] { IDLE, FETCH, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        case (state)
            IDLE:  next_state = fetch_enable ? FETCH : IDLE;
            FETCH: next_state = DONE;
            DONE:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    //=========================================================================
    // Per-PE address generation (all 4×4 run in parallel)
    //=========================================================================
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_HEIGHT; row++) begin : gen_rows
            for (col = 0; col < ARRAY_WIDTH; col++) begin : gen_cols

                logic [15:0]           idx_0_pe, idx_1_pe, idx_2_pe, idx_3_pe;
                logic                  idx_valid_pe;
                logic [DATA_WIDTH-1:0] gen_addr_pe;
                logic                  gen_valid_pe;
                logic [15:0]           out_ch, kw_pos;
                logic                  position_valid;

                // Weight address computation uses 4D tensor: (K, C, KH, KW)
                // address_generator_ws handles 3D, so we fold the 4D to 3D
                // by computing the offset manually and using shape_c=1 trick,
                // OR we use the 3D generator with pre-folded indices.
                //
                // APPROACH: Use direct offset calculation for the 4D weight tensor.
                // The address_generator_ws computes: base + idx0*(H*W) + idx1*W + idx2
                // For weight tensor [K, C, KH, KW]:
                //   CHANNEL_MAJOR: offset = k*(C*KH*KW) + c*(KH*KW) + kh*KW + kw
                //   We map: idx_0=k, idx_1=c*KH+kh (folded), idx_2=kw
                //   But this only works if KH is fixed. Safer to compute full offset.
                //
                // Use a dedicated always_comb for 4D weight offset.

                logic [31:0] weight_offset;

                always_comb begin
                    out_ch = output_ch_start + row[15:0];
                    kw_pos = kernel_col_start + col[15:0];

                    position_valid = (out_ch     < weight_k)  &&
                                     (input_ch   < weight_c)  &&
                                     (kernel_row < weight_kh) &&
                                     (kw_pos     < weight_kw);

                    idx_valid_pe = (state == FETCH) && position_valid;

                    // 4D weight offset formulas
                    case (layout)
                        // CHANNEL_MAJOR: [K, C, KH, KW]
                        CHANNEL_MAJOR:
                            weight_offset = out_ch * (32'(weight_c) * weight_kh * weight_kw)
                                          + input_ch   * (32'(weight_kh) * weight_kw)
                                          + kernel_row *  32'(weight_kw)
                                          + kw_pos;
                        // ROW_MAJOR: [K, KH, KW, C]
                        ROW_MAJOR:
                            weight_offset = out_ch * (32'(weight_kh) * weight_kw * weight_c)
                                          + kernel_row * (32'(weight_kw) * weight_c)
                                          + kw_pos     *  32'(weight_c)
                                          + input_ch;
                        // COLUMN_MAJOR: [K, KW, KH, C]
                        COLUMN_MAJOR:
                            weight_offset = out_ch * (32'(weight_kw) * weight_kh * weight_c)
                                          + kw_pos     * (32'(weight_kh) * weight_c)
                                          + kernel_row *  32'(weight_c)
                                          + input_ch;
                        default:
                            weight_offset = '0;
                    endcase
                end

                // Register outputs.
                // addr_valid is set in FETCH and held through DONE.
                // Only cleared when FSM returns to IDLE (ready for next fetch)
                // or on reset.
                //
                // WHY THE ORIGINAL FAILED:
                //   The "else" branch cleared addr_valid in BOTH DONE and IDLE.
                //   addr_valid was only high for the ONE posedge where DONE's NBA
                //   committed, then immediately cleared by the next DONE-state
                //   posedge. Any code waiting more than 1 cycle after fetch_done
                //   saw addr_valid=0 - causing Test 7 OOB masking to FAIL for
                //   valid rows because they appeared masked.
                //
                // FIX: hold addr_valid in DONE state, clear only in IDLE.
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        addr_out[row][col]   <= '0;
                        addr_valid[row][col] <= 1'b0;
                    end else if (state == FETCH && position_valid) begin
                        // Latch address and assert valid
                        addr_out[row][col]   <= base_addr + weight_offset;
                        addr_valid[row][col] <= 1'b1;
                    end else if (state == FETCH && !position_valid) begin
                        // OOB position - assert invalid
                        addr_valid[row][col] <= 1'b0;
                    end else if (state == IDLE) begin
                        // Returning to IDLE - clear for next fetch cycle
                        addr_valid[row][col] <= 1'b0;
                    end
                    // DONE state: no branch - addr_valid holds its value
                end

            end
        end
    endgenerate

    //=========================================================================
    // fetch_done - 1-cycle pulse in DONE state
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : weight_fetcher_ws