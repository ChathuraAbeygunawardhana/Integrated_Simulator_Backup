`timescale 1ns / 1ps

module stamp_based_memory_controller
    //import systolic_array_pkg::*;
#(
    parameter int ADDR_WIDTH     = 32,
    parameter int DATA_WIDTH     = 32,
    parameter int SPAD_DEPTH     = 4096,
    parameter int AXI_DATA_WIDTH = 64,
    parameter int MAX_DELTA_OPS  = 64,
    parameter int METADATA_DEPTH = 256
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                              metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]                      metadata_wr_data,

    input  logic        phase_start,
    input  logic [15:0] phase_id,
    input  logic [15:0] num_delta_ops,
    input  logic [15:0] phase_base_addr,
    output logic        phase_ready,
    output logic        phase_done,

    output logic [$clog2(SPAD_DEPTH)-1:0] spad_rd_addr,
    input  logic [DATA_WIDTH-1:0]          spad_rd_data,
    output logic                           spad_rd_en,

    output logic [$clog2(SPAD_DEPTH)-1:0] spad_wr_addr,
    output logic [DATA_WIDTH-1:0]          spad_wr_data,
    output logic                           spad_wr_en,

    output logic [ADDR_WIDTH-1:0] mem_rd_addr,
    output logic [15:0]           mem_rd_len,
    output logic                  mem_rd_req,
    input  logic                  mem_rd_ready,
    input  logic [DATA_WIDTH-1:0] mem_rd_data,
    input  logic                  mem_rd_valid,

    output logic [31:0] stats_loads,
    output logic [31:0] stats_moves,
    output logic [31:0] stats_keeps,
    output logic [31:0] stats_bytes_loaded,
    output logic [31:0] stats_bytes_moved,
    output logic        controller_busy
);

typedef struct packed {
    logic [7:0]  op_type;
    logic [15:0] tile_id;
    logic [31:0] src_addr;
    logic [31:0] dst_addr;
    logic [31:0] size;
} delta_op_t;

delta_op_t metadata_ram [METADATA_DEPTH];

always_ff @(posedge clk) begin
    if (metadata_wr_en) begin
        metadata_ram[metadata_wr_addr].op_type <= metadata_wr_data[119:112];
        metadata_ram[metadata_wr_addr].tile_id <= metadata_wr_data[111:96];
        metadata_ram[metadata_wr_addr].src_addr <= metadata_wr_data[95:64];
        metadata_ram[metadata_wr_addr].dst_addr <= metadata_wr_data[63:32];
        metadata_ram[metadata_wr_addr].size <= metadata_wr_data[31:0];
    end
end

typedef enum logic [3:0] {
    IDLE,
    FETCH_META,
    DO_KEEP,
    MOVE_READ,
    MOVE_WRITE,
    LOAD_REQ,
    LOAD_RECV,
    PHASE_DONE
} state_t;

state_t state;

delta_op_t cur_op;

logic [15:0] op_idx;
logic [15:0] op_end;

logic [31:0] src_addr;
logic [31:0] dst_addr;
logic [15:0] words_left;
logic [15:0] burst_words_left;

localparam int BPW = DATA_WIDTH / 8;
localparam logic [15:0] MAX_AXI_BURST_WORDS = 16'd256;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;

        op_idx <= 0;
        op_end <= 0;

        src_addr <= 0;
        dst_addr <= 0;
        words_left <= 0;
        burst_words_left <= 0;

        stats_loads <= 0;
        stats_moves <= 0;
        stats_keeps <= 0;
        stats_bytes_loaded <= 0;
        stats_bytes_moved <= 0;
    end
    else begin

        case(state)

        IDLE: begin
            if (phase_start) begin
                op_idx <= phase_base_addr;
                op_end <= phase_base_addr + num_delta_ops;
                state <= FETCH_META;
            end
        end

        FETCH_META: begin

            if (op_idx >= op_end) begin
                state <= PHASE_DONE;
            end
            else begin

                cur_op <= metadata_ram[op_idx];

                src_addr <= metadata_ram[op_idx].src_addr;
                dst_addr <= metadata_ram[op_idx].dst_addr;

                words_left <= metadata_ram[op_idx].size / BPW;

                case(metadata_ram[op_idx].op_type)

                    8'd0: begin
                        stats_keeps <= stats_keeps + 1;
                        op_idx <= op_idx + 1;
                        state <= FETCH_META;
                    end

                    8'd1: begin
                        stats_moves <= stats_moves + 1;
                        stats_bytes_moved <=
                            stats_bytes_moved + metadata_ram[op_idx].size;

                        state <= MOVE_READ;
                    end

                    8'd2: begin
                        stats_loads <= stats_loads + 1;
                        stats_bytes_loaded <=
                            stats_bytes_loaded + metadata_ram[op_idx].size;

                        state <= LOAD_REQ;
                    end

                    default: begin
                        op_idx <= op_idx + 1;
                        state <= FETCH_META;
                    end
                endcase
            end
        end

        MOVE_READ: begin
            state <= MOVE_WRITE;
        end

        MOVE_WRITE: begin

            words_left <= words_left - 1;

            src_addr <= src_addr + BPW;
            dst_addr <= dst_addr + BPW;

            if (words_left == 1) begin
                op_idx <= op_idx + 1;
                state <= FETCH_META;
            end
            else begin
                state <= MOVE_READ;
            end
        end

        LOAD_REQ: begin
            if (mem_rd_ready) begin
                // AXI4 ARLEN is 8 bits, so one burst can transfer at most
                // 256 beats. Split larger LOAD operations into multiple
                // back-to-back read bursts.
                if (words_left > MAX_AXI_BURST_WORDS)
                    burst_words_left <= MAX_AXI_BURST_WORDS;
                else
                    burst_words_left <= words_left;

                state <= LOAD_RECV;
            end
        end

        LOAD_RECV: begin

            if (mem_rd_valid) begin

                words_left       <= words_left - 1;
                burst_words_left <= burst_words_left - 1;
                src_addr         <= src_addr + BPW;
                dst_addr         <= dst_addr + BPW;

                if (words_left == 1) begin
                    op_idx <= op_idx + 1;
                    state <= FETCH_META;
                end
                else if (burst_words_left == 1) begin
                    state <= LOAD_REQ;
                end
            end
        end

        PHASE_DONE: begin
            state <= IDLE;
        end

        endcase
    end
end

//----------------------------------------------------------
// Scratchpad interface
//----------------------------------------------------------

always_comb begin

    spad_rd_en = 0;
    spad_rd_addr = 0;

    spad_wr_en = 0;
    spad_wr_addr = 0;
    spad_wr_data = 0;

    case(state)

    MOVE_READ: begin
        spad_rd_en = 1;
        spad_rd_addr = src_addr[$clog2(SPAD_DEPTH)+1:2];
    end

    MOVE_WRITE: begin
        spad_wr_en = 1;
        spad_wr_addr = dst_addr[$clog2(SPAD_DEPTH)+1:2];
        spad_wr_data = spad_rd_data;
    end

    LOAD_RECV: begin
        if (mem_rd_valid) begin
            spad_wr_en = 1;
            spad_wr_addr = dst_addr[$clog2(SPAD_DEPTH)+1:2];
            spad_wr_data = mem_rd_data;
        end
    end

    endcase
end

//----------------------------------------------------------
// Memory read interface
//----------------------------------------------------------

assign mem_rd_req  = (state == LOAD_REQ);

assign mem_rd_addr = src_addr;

assign mem_rd_len  = (words_left > MAX_AXI_BURST_WORDS) ?
                     MAX_AXI_BURST_WORDS : words_left;

//----------------------------------------------------------
// Status
//----------------------------------------------------------

assign phase_ready     = (state == IDLE);
assign phase_done      = (state == PHASE_DONE);
assign controller_busy = (state != IDLE);

endmodule