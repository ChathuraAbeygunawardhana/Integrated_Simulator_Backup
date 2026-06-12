//=============================================================================
// Module: dnn_scheduler_wrapper
// Description: Wraps the three DNN-aware (layer-granularity) schedulers --
//              AI-MT, BatchDNN, BatchDNN++ -- behind a uniform port set.
//              dnn_scheduler_select picks which one is active.
//
//   dnn_scheduler_select:
//     2'd0 = AI-MT
//     2'd1 = BatchDNN
//     2'd2 = BatchDNN++
//=============================================================================

`timescale 1ns/1ps

module dnn_scheduler_wrapper #(
    parameter MAX_LAYERS         = 32,
    parameter MAX_DNNS           = 4,
    parameter LAYER_ID_WIDTH     = 8,
    parameter DNN_ID_WIDTH       = 4,
    parameter BATCH_WIDTH        = 8,
    parameter CYCLE_WIDTH        = 32,
    parameter MEM_WIDTH          = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic [1:0]  dnn_scheduler_select,

    // Scheduling table write port (broadcast to whichever sub-module is active)
    input  logic                       st_write_en,
    input  logic [LAYER_ID_WIDTH-1:0]  st_layer_idx,
    input  logic [DNN_ID_WIDTH-1:0]    st_dnn_id,
    input  logic [LAYER_ID_WIDTH-1:0]  st_prev_layer,
    input  logic [CYCLE_WIDTH-1:0]     st_mem_cycles,
    input  logic [CYCLE_WIDTH-1:0]     st_compute_cycles,
    input  logic [MEM_WIDTH-1:0]       st_weight_fp,
    input  logic [MEM_WIDTH-1:0]       st_ifmap_fp,
    input  logic [MEM_WIDTH-1:0]       st_ofmap_fp,
    input  logic [BATCH_WIDTH-1:0]     st_batch,         // BatchDNN: initial_batch; BatchDNN++: requested_batch
    input  logic [LAYER_ID_WIDTH-1:0]  st_total_layers,

    // Runtime
    input  logic mem_access_done,
    input  logic compute_done,

    // Tagged-union output (caller inspects ct_valid + ct_batch_size)
    output logic                      mt_valid,
    output logic [LAYER_ID_WIDTH-1:0] mt_layer_id,
    output logic [DNN_ID_WIDTH-1:0]   mt_dnn_id,
    output logic                      ct_valid,
    output logic [LAYER_ID_WIDTH-1:0] ct_layer_id,
    output logic [DNN_ID_WIDTH-1:0]   ct_dnn_id,
    output logic [BATCH_WIDTH-1:0]    ct_batch_size,    // AI-MT: always 1

    output logic [CYCLE_WIDTH-1:0]    available_memory,
    output logic                      stall_detected,
    output logic [CYCLE_WIDTH-1:0]    total_stall_cycles
);

    // Per-scheduler outputs
    logic                      a_mt_valid, b_mt_valid, p_mt_valid;
    logic [LAYER_ID_WIDTH-1:0] a_mt_layer, b_mt_layer, p_mt_layer;
    logic [DNN_ID_WIDTH-1:0]   a_mt_dnn,   b_mt_dnn,   p_mt_dnn;

    logic                      a_ct_valid, b_ct_valid, p_ct_valid;
    logic [LAYER_ID_WIDTH-1:0] a_ct_layer, b_ct_layer, p_ct_layer;
    logic [DNN_ID_WIDTH-1:0]   a_ct_dnn,   b_ct_dnn,   p_ct_dnn;
    logic [BATCH_WIDTH-1:0]    b_ct_batch, p_ct_batch;

    logic [CYCLE_WIDTH-1:0]    a_avail, b_avail, p_avail;
    logic                      a_stall, b_stall, p_stall;
    logic [CYCLE_WIDTH-1:0]    a_total_stall, b_total_stall, p_total_stall;

    // Gate done strobes so only the active scheduler sees them
    logic mad_a, mad_b, mad_p, cd_a, cd_b, cd_p;
    assign mad_a = (dnn_scheduler_select == 2'd0) && mem_access_done;
    assign mad_b = (dnn_scheduler_select == 2'd1) && mem_access_done;
    assign mad_p = (dnn_scheduler_select == 2'd2) && mem_access_done;
    assign cd_a  = (dnn_scheduler_select == 2'd0) && compute_done;
    assign cd_b  = (dnn_scheduler_select == 2'd1) && compute_done;
    assign cd_p  = (dnn_scheduler_select == 2'd2) && compute_done;

    aimt_scheduler #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_aimt (
        .clk               (clk),
        .rst_n             (rst_n),
        .st_write_en       (st_write_en && (dnn_scheduler_select == 2'd0)),
        .st_layer_idx      (st_layer_idx),
        .st_dnn_id         (st_dnn_id),
        .st_prev_layer     (st_prev_layer),
        .st_mem_cycles     (st_mem_cycles),
        .st_compute_cycles (st_compute_cycles),
        .st_weight_fp      (st_weight_fp),
        .st_ifmap_fp       (st_ifmap_fp),
        .st_ofmap_fp       (st_ofmap_fp),
        .st_total_layers   (st_total_layers),
        .mem_access_done   (mad_a),
        .compute_done      (cd_a),
        .mt_valid          (a_mt_valid),
        .mt_layer_id       (a_mt_layer),
        .mt_dnn_id         (a_mt_dnn),
        .ct_valid          (a_ct_valid),
        .ct_layer_id       (a_ct_layer),
        .ct_dnn_id         (a_ct_dnn),
        .available_memory  (a_avail),
        .stall_detected    (a_stall),
        .total_stall_cycles(a_total_stall)
    );

    batchdnn_scheduler #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .BATCH_WIDTH     (BATCH_WIDTH),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_batchdnn (
        .clk               (clk),
        .rst_n             (rst_n),
        .st_write_en       (st_write_en && (dnn_scheduler_select == 2'd1)),
        .st_layer_idx      (st_layer_idx),
        .st_dnn_id         (st_dnn_id),
        .st_prev_layer     (st_prev_layer),
        .st_mem_cycles     (st_mem_cycles),
        .st_compute_cycles (st_compute_cycles),
        .st_weight_fp      (st_weight_fp),
        .st_ifmap_fp       (st_ifmap_fp),
        .st_ofmap_fp       (st_ofmap_fp),
        .st_initial_batch  (st_batch),
        .st_total_layers   (st_total_layers),
        .mem_access_done   (mad_b),
        .compute_done      (cd_b),
        .mt_valid          (b_mt_valid),
        .mt_layer_id       (b_mt_layer),
        .mt_dnn_id         (b_mt_dnn),
        .ct_valid          (b_ct_valid),
        .ct_layer_id       (b_ct_layer),
        .ct_dnn_id         (b_ct_dnn),
        .ct_batch_size     (b_ct_batch),
        .available_memory  (b_avail),
        .stall_detected    (b_stall),
        .total_stall_cycles(b_total_stall),
        .total_split_events(),
        .total_merge_events()
    );

    batchdnn_pp_scheduler #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .BATCH_WIDTH     (BATCH_WIDTH),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_batchpp (
        .clk               (clk),
        .rst_n             (rst_n),
        .st_write_en       (st_write_en && (dnn_scheduler_select == 2'd2)),
        .st_layer_idx      (st_layer_idx),
        .st_dnn_id         (st_dnn_id),
        .st_prev_layer     (st_prev_layer),
        .st_mem_cycles     (st_mem_cycles),
        .st_compute_cycles (st_compute_cycles),
        .st_weight_fp      (st_weight_fp),
        .st_ifmap_fp       (st_ifmap_fp),
        .st_ofmap_fp       (st_ofmap_fp),
        .st_requested_batch(st_batch),
        .st_total_layers   (st_total_layers),
        .mem_access_done   (mad_p),
        .compute_done      (cd_p),
        .mt_valid          (p_mt_valid),
        .mt_layer_id       (p_mt_layer),
        .mt_dnn_id         (p_mt_dnn),
        .ct_valid          (p_ct_valid),
        .ct_layer_id       (p_ct_layer),
        .ct_dnn_id         (p_ct_dnn),
        .ct_batch_size     (p_ct_batch),
        .available_memory  (p_avail),
        .stall_detected    (p_stall),
        .total_stall_cycles(p_total_stall),
        .total_split_events(),
        .total_merge_events(),
        .total_throttle_events(),
        .total_expedition_events(),
        .total_slices_executed()
    );

    // Mux
    always_comb begin
        case (dnn_scheduler_select)
            2'd0: begin
                mt_valid           = a_mt_valid;
                mt_layer_id        = a_mt_layer;
                mt_dnn_id          = a_mt_dnn;
                ct_valid           = a_ct_valid;
                ct_layer_id        = a_ct_layer;
                ct_dnn_id          = a_ct_dnn;
                ct_batch_size      = {{(BATCH_WIDTH-1){1'b0}}, 1'b1};
                available_memory   = a_avail;
                stall_detected     = a_stall;
                total_stall_cycles = a_total_stall;
            end
            2'd1: begin
                mt_valid           = b_mt_valid;
                mt_layer_id        = b_mt_layer;
                mt_dnn_id          = b_mt_dnn;
                ct_valid           = b_ct_valid;
                ct_layer_id        = b_ct_layer;
                ct_dnn_id          = b_ct_dnn;
                ct_batch_size      = b_ct_batch;
                available_memory   = b_avail;
                stall_detected     = b_stall;
                total_stall_cycles = b_total_stall;
            end
            default: begin
                mt_valid           = p_mt_valid;
                mt_layer_id        = p_mt_layer;
                mt_dnn_id          = p_mt_dnn;
                ct_valid           = p_ct_valid;
                ct_layer_id        = p_ct_layer;
                ct_dnn_id          = p_ct_dnn;
                ct_batch_size      = p_ct_batch;
                available_memory   = p_avail;
                stall_detected     = p_stall;
                total_stall_cycles = p_total_stall;
            end
        endcase
    end

endmodule : dnn_scheduler_wrapper
