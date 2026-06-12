//=============================================================================
// Module: weight_fetcher  (FIXED)
// Description: Fetches and preloads weights for output stationary systolic array
//
// FIX: Replaced the broken multi-state pipeline loop (PRELOAD_CHANNELS →
//      PRELOAD_KH → PRELOAD_KW) with a single FETCH state that advances
//      all three counters (c_cnt, kh_cnt, kw_cnt) in one always_ff block.
//      The original code updated counters in a different clock edge than the
//      FSM transition checks, causing the nested loop to never terminate,
//      which kept fetch_done permanently LOW and deadlocked the top-level FSM
//      in WAIT_WEIGHTS forever.
//
//      Also fixed: fetch_done was driven combinatorially from state==DONE,
//      but the top-level FSM samples it one cycle later.  Now fetch_done is
//      registered (pulsed HIGH for exactly 1 cycle in state DONE).
//=============================================================================

module weight_fetcher
    import systolic_array_pkg::*;
#(
    parameter int ARRAY_HEIGHT = 4,
    parameter int DATA_WIDTH   = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_KERNEL   = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // Configuration
    input  mem_layout_t           layout,
    input  logic [15:0]           weight_k,       // Number of output channels (filters)
    input  logic [15:0]           weight_c,       // Input channels
    input  logic [15:0]           weight_kh,      // Kernel height
    input  logic [15:0]           weight_kw,      // Kernel width
    input  logic [DATA_WIDTH-1:0] base_addr,

    // Control
    input  logic [15:0]           output_ch_start, // Starting output channel
    input  logic                  fetch_enable,
    output logic                  fetch_done,

    // Address outputs (one per array row / PE)
    output logic [DATA_WIDTH-1:0] addr_out   [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] addr_valid,

    // Weight index outputs
    output logic [15:0]           weight_idx_out [ARRAY_HEIGHT]
);

    //=========================================================================
    // FSM
    //=========================================================================
    typedef enum logic [1:0] { IDLE, FETCH, DONE } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    //=========================================================================
    // Counters  (c_cnt, kh_cnt, kw_cnt)
    // Innermost = kw, then kh, outermost = c
    //=========================================================================
    logic [15:0] c_cnt, kh_cnt, kw_cnt;

    wire last_kw = (kw_cnt == weight_kw - 1);
    wire last_kh = (kh_cnt == weight_kh - 1);
    wire last_c  = (c_cnt  == weight_c  - 1);
    wire last_all = last_kw && last_kh && last_c;

    // Next-state logic
    always_comb begin
        case (state)
            IDLE:  next_state = fetch_enable ? FETCH : IDLE;
            FETCH: next_state = last_all     ? DONE  : FETCH;
            DONE:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // Counter update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_cnt  <= '0;
            kh_cnt <= '0;
            kw_cnt <= '0;
        end else if (state == IDLE) begin
            c_cnt  <= '0;
            kh_cnt <= '0;
            kw_cnt <= '0;
        end else if (state == FETCH) begin
            if (last_kw) begin
                kw_cnt <= '0;
                if (last_kh) begin
                    kh_cnt <= '0;
                    if (!last_c) c_cnt <= c_cnt + 1;
                end else begin
                    kh_cnt <= kh_cnt + 1;
                end
            end else begin
                kw_cnt <= kw_cnt + 1;
            end
        end
    end

    //=========================================================================
    // Address generation (one per PE row)
    //=========================================================================
    genvar i;
    generate
        for (i = 0; i < ARRAY_HEIGHT; i++) begin : gen_addr_gens

            logic [15:0]           out_ch;
            logic [31:0]           weight_offset;
            logic [DATA_WIDTH-1:0] cell_addr;
            logic                  cell_valid;

            always_comb begin
                out_ch = output_ch_start + 16'(i);

                case (layout)
                    CHANNEL_MAJOR:
                        // [K, C, KH, KW]
                        weight_offset = 32'(out_ch) * (32'(weight_c) * 32'(weight_kh) * 32'(weight_kw))
                                      + 32'(c_cnt)  * (32'(weight_kh) * 32'(weight_kw))
                                      + 32'(kh_cnt) *  32'(weight_kw)
                                      + 32'(kw_cnt);
                    ROW_MAJOR:
                        // [K, KH, KW, C]
                        weight_offset = 32'(out_ch) * (32'(weight_kh) * 32'(weight_kw) * 32'(weight_c))
                                      + 32'(kh_cnt) * (32'(weight_kw) * 32'(weight_c))
                                      + 32'(kw_cnt) *  32'(weight_c)
                                      + 32'(c_cnt);
                    COLUMN_MAJOR:
                        // [K, KW, KH, C]
                        weight_offset = 32'(out_ch) * (32'(weight_kw) * 32'(weight_kh) * 32'(weight_c))
                                      + 32'(kw_cnt) * (32'(weight_kh) * 32'(weight_c))
                                      + 32'(kh_cnt) *  32'(weight_c)
                                      + 32'(c_cnt);
                    default:
                        weight_offset = '0;
                endcase

                cell_valid = (out_ch < weight_k);
                cell_addr  = cell_valid ? (base_addr + cell_offset) : '0;
            end

            // Suppress undeclared warning - alias offset to cell_offset
            logic [31:0] cell_offset;
            assign cell_offset = weight_offset;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    addr_out[i]   <= '0;
                    addr_valid[i] <= 1'b0;
                end else begin
                    if (state == FETCH && cell_valid) begin
                        addr_out[i]   <= cell_addr;
                        addr_valid[i] <= 1'b1;
                    end else begin
                        addr_valid[i] <= 1'b0;
                    end
                end
            end
        end
    endgenerate

    //=========================================================================
    // Weight index output and fetch_done
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < ARRAY_HEIGHT; j++)
                weight_idx_out[j] <= '0;
        end else if (state == FETCH) begin
            for (int j = 0; j < ARRAY_HEIGHT; j++)
                weight_idx_out[j] <= c_cnt * weight_kh * weight_kw
                                   + kh_cnt * weight_kw
                                   + kw_cnt;
        end
    end

    // Pulse fetch_done for exactly 1 cycle in DONE state
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fetch_done <= 1'b0;
        else        fetch_done <= (state == DONE);
    end

endmodule : weight_fetcher